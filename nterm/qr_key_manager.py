"""
QR-based API key management for nterm
"""
import os
import time
import uuid
import qrcode
import requests
import threading
from typing import Optional, Tuple
from agno.utils.log import logger


class QRKeyManager:
    """
    Manages QR-based API key insertion for nterm.
    
    This class handles:
    - Generating unique session IDs
    - Creating QR codes for the worker URL
    - Long-polling the worker API for keys
    - Setting environment variables
    """
    
    def __init__(self, worker_url: str, timeout: int = 300, poll_interval: int = 2):
        """
        Initialize the QR key manager.
        
        Args:
            worker_url: Base URL of the Cloudflare Worker (e.g., https://your-worker.workers.dev)
            timeout: Maximum time to wait for key (seconds)
            poll_interval: Time between polling attempts (seconds)
        """
        self.worker_url = worker_url.rstrip('/')
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.session_id = None
        self.stop_polling = False
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return str(uuid.uuid4()).replace('-', '')[:16]
    
    def generate_qr_code(self, session_id: str, display: bool = True) -> str:
        """
        Generate QR code for the session URL.
        
        Args:
            session_id: Unique session identifier
            display: Whether to print the QR code to terminal
            
        Returns:
            The session URL
        """
        session_url = f"{self.worker_url}/?session={session_id}"
        
        if display:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(session_url)
            qr.make(fit=True)
            
            print("\n" + "="*60)
            print("🔑 OPENAI API KEY SETUP")
            print("="*60)
            print("Scan this QR code with your phone to set your API key:")
            print()
            qr.print_ascii(invert=True)
            print()
            print(f"Or visit: {session_url}")
            print()
            print("Waiting for API key... (Press Ctrl+C to cancel)")
            print("="*60)
        
        return session_url
    
    def poll_for_key(self, session_id: str) -> Optional[str]:
        """
        Long-poll the worker API for the API key.
        
        Args:
            session_id: Session ID to poll for
            
        Returns:
            The API key if received, None if timeout or cancelled
        """
        poll_url = f"{self.worker_url}/api/poll-session?session={session_id}"
        start_time = time.time()
        
        while not self.stop_polling and (time.time() - start_time) < self.timeout:
            try:
                response = requests.get(poll_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    api_key = data.get('key')
                    if api_key:
                        logger.info(f"✅ API key received successfully!")
                        return api_key
                elif response.status_code == 404:
                    # Key not yet provided, continue polling
                    pass
                else:
                    logger.warning(f"Unexpected response status: {response.status_code}")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Polling request failed: {e}")
            except ValueError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
            
            time.sleep(self.poll_interval)
        
        return None
    
    def set_environment_variable(self, key: str, value: str) -> bool:
        """
        Set environment variable for the current process.
        
        Args:
            key: Environment variable name
            value: Environment variable value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.environ[key] = value
            logger.info(f"✅ Environment variable {key} set successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to set environment variable {key}: {e}")
            return False
    
    def setup_api_key_interactive(self) -> Tuple[bool, Optional[str]]:
        """
        Interactive setup process for API key via QR code.
        
        Returns:
            Tuple of (success, api_key)
        """
        try:
            # Generate session ID and QR code
            self.session_id = self.generate_session_id()
            session_url = self.generate_qr_code(self.session_id)
            
            # Start polling in background
            self.stop_polling = False
            
            # Poll for the key
            api_key = self.poll_for_key(self.session_id)
            
            if api_key:
                # Set the environment variable
                success = self.set_environment_variable('OPENAI_API_KEY', api_key)
                if success:
                    print("\n✅ API key setup completed successfully!")
                    print("You can now use nterm with your OpenAI API key.")
                    return True, api_key
                else:
                    print("\n❌ Failed to set environment variable.")
                    return False, api_key
            else:
                print("\n⏰ Timeout or cancelled. No API key received.")
                return False, None
                
        except KeyboardInterrupt:
            print("\n\n🛑 Setup cancelled by user.")
            self.stop_polling = True
            return False, None
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False, None
    
    def check_existing_key(self) -> bool:
        """
        Check if OPENAI_API_KEY is already set in environment.
        
        Returns:
            True if key exists, False otherwise
        """
        existing_key = os.getenv('OPENAI_API_KEY')
        if existing_key:
            print(f"✅ OPENAI_API_KEY is already set (ends with: ...{existing_key[-4:]})")
            return True
        return False
    
    def setup_key_if_needed(self, force: bool = False) -> bool:
        """
        Setup API key if not already present or if forced.
        
        Args:
            force: Force setup even if key already exists
            
        Returns:
            True if key is available (existing or newly set), False otherwise
        """
        if not force and self.check_existing_key():
            return True
        
        if force:
            print("🔄 Forcing new API key setup...")
        else:
            print("🔍 No OPENAI_API_KEY found in environment.")
            print("Let's set it up using QR code...")
        
        success, _ = self.setup_api_key_interactive()
        return success


def create_qr_key_manager(worker_url: str, **kwargs) -> QRKeyManager:
    """
    Factory function to create a QR key manager.
    
    Args:
        worker_url: Cloudflare Worker URL
        **kwargs: Additional arguments for QRKeyManager
        
    Returns:
        Configured QRKeyManager instance
    """
    return QRKeyManager(worker_url, **kwargs)