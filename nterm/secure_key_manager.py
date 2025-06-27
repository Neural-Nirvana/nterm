"""
Fixed secure QR-based API key management for nterm (matching encryption + persistence)
Uses AES-GCM encryption to match JavaScript implementation and saves keys persistently
"""
import os
import time
import uuid
import qrcode
import requests
import hashlib
import base64
import json
from typing import Optional, Tuple, Dict
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet

from agno.utils.log import logger


class SecureKeyManager:
    """
    Manages secure QR-based API key insertion for nterm with persistent storage.
    
    Features:
    - Client-side encryption before transmission (AES-GCM)
    - Temporary server storage (auto-expires)
    - Secure key derivation (PBKDF2)
    - Persistent local storage of API keys
    - Simple HTTP polling (no WebRTC complexity)
    """
    
    def __init__(self, worker_url: str, timeout: int = 300, poll_interval: int = 2):
        """
        Initialize the secure key manager.
        
        Args:
            worker_url: Base URL of the Cloudflare Worker
            timeout: Maximum time to wait for key (seconds)
            poll_interval: Time between polling attempts (seconds)
        """
        self.worker_url = worker_url.rstrip('/')
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.session_id = None
        self.encryption_key = None
        self.stop_polling = False
        
        # Setup persistent storage
        self.config_dir = Path.home() / '.nterm'
        self.config_file = self.config_dir / 'api_keys.json'
        self.master_key_file = self.config_dir / 'master.key'
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Load existing keys on initialization
        self._load_persistent_keys()
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create a master key for encrypting stored API keys."""
        if self.master_key_file.exists():
            with open(self.master_key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new master key
            master_key = Fernet.generate_key()
            with open(self.master_key_file, 'wb') as f:
                f.write(master_key)
            # Secure the file (readable only by owner)
            os.chmod(self.master_key_file, 0o600)
            return master_key
    
    def _encrypt_for_storage(self, data: str) -> str:
        """Encrypt data for persistent storage."""
        master_key = self._get_or_create_master_key()
        fernet = Fernet(master_key)
        return fernet.encrypt(data.encode()).decode()
    
    def _decrypt_from_storage(self, encrypted_data: str) -> str:
        """Decrypt data from persistent storage."""
        master_key = self._get_or_create_master_key()
        fernet = Fernet(master_key)
        return fernet.decrypt(encrypted_data.encode()).decode()
    
    def _load_persistent_keys(self):
        """Load API keys from persistent storage into environment."""
        if not self.config_file.exists():
            return
        
        try:
            with open(self.config_file, 'r') as f:
                encrypted_keys = json.load(f)
            
            for key_name, encrypted_value in encrypted_keys.items():
                try:
                    decrypted_value = self._decrypt_from_storage(encrypted_value)
                    os.environ[key_name] = decrypted_value
                    logger.debug(f"Loaded {key_name} from persistent storage")
                except Exception as e:
                    logger.warning(f"Failed to decrypt stored key {key_name}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to load persistent keys: {e}")
    
    def _save_persistent_key(self, key_name: str, key_value: str):
        """Save an API key to persistent storage."""
        try:
            # Load existing keys
            encrypted_keys = {}
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    encrypted_keys = json.load(f)
            
            # Add/update the new key
            encrypted_keys[key_name] = self._encrypt_for_storage(key_value)
            
            # Save back to file
            with open(self.config_file, 'w') as f:
                json.dump(encrypted_keys, f, indent=2)
            
            # Secure the file (readable only by owner)
            os.chmod(self.config_file, 0o600)
            
            logger.info(f"API key {key_name} saved to persistent storage")
            
        except Exception as e:
            logger.error(f"Failed to save persistent key: {e}")
    
    def _remove_persistent_key(self, key_name: str):
        """Remove an API key from persistent storage."""
        try:
            if not self.config_file.exists():
                return
            
            with open(self.config_file, 'r') as f:
                encrypted_keys = json.load(f)
            
            if key_name in encrypted_keys:
                del encrypted_keys[key_name]
                
                with open(self.config_file, 'w') as f:
                    json.dump(encrypted_keys, f, indent=2)
                
                logger.info(f"API key {key_name} removed from persistent storage")
                
        except Exception as e:
            logger.error(f"Failed to remove persistent key: {e}")
    
    def list_stored_keys(self) -> Dict[str, str]:
        """List all stored API keys with masked values."""
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                encrypted_keys = json.load(f)
            
            result = {}
            for key_name, encrypted_value in encrypted_keys.items():
                try:
                    decrypted_value = self._decrypt_from_storage(encrypted_value)
                    # Mask the value for display
                    result[key_name] = f"...{decrypted_value[-4:]}" if len(decrypted_value) > 4 else "****"
                except Exception:
                    result[key_name] = "Invalid/Corrupted"
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list stored keys: {e}")
            return {}
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return str(uuid.uuid4()).replace('-', '')[:16]
    
    def generate_encryption_key(self, session_id: str) -> bytes:
        """Generate encryption key from session ID using same method as JavaScript."""
        # Use session ID as password for key derivation (same as JS)
        password = session_id.encode('utf-8')
        
        # Create salt from session ID hash (same as JS)
        salt = hashlib.sha256(session_id.encode('utf-8')).digest()[:16]
        
        # Use PBKDF2 with same parameters as JavaScript
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits for AES-256
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password)
    
    def decrypt_key(self, encrypted_data_b64: str, session_id: str) -> str:
        """
        Decrypt the key using AES-GCM (matching JavaScript implementation).
        
        Args:
            encrypted_data_b64: Base64 encoded encrypted data (IV + ciphertext)
            session_id: Session ID used for key derivation
            
        Returns:
            Decrypted plaintext key
        """
        try:
            # Generate the same encryption key as JavaScript
            key = self.generate_encryption_key(session_id)
            
            # Decode base64
            encrypted_data = base64.b64decode(encrypted_data_b64)
            
            # Extract IV (first 12 bytes) and ciphertext (rest)
            iv = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            
            # Decrypt using AES-GCM
            aesgcm = AESGCM(key)
            plaintext = aesgcm.decrypt(iv, ciphertext, None)
            
            return plaintext.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
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
            print("🔑 API KEY SETUP")
            print("="*60)
            print("Scan this QR code with your phone to set your API key:")
            print()
            qr.print_ascii(invert=True)
            print()
            print(f"Or visit: {session_url}")
            print()
            print("Waiting for API key... (Press Ctrl+C to cancel)")
            print("🔒 Keys are encrypted before transmission")
            print("="*60)
        
        return session_url
    
    def poll_for_key(self, session_id: str) -> Optional[str]:
        """
        Poll the worker API for the encrypted API key.
        
        Args:
            session_id: Session ID to poll for
            
        Returns:
            The decrypted API key if received, None if timeout or cancelled
        """
        poll_url = f"{self.worker_url}/api/poll-session?session={session_id}"
        start_time = time.time()
        
        print("🔄 Polling for encrypted key...")
        
        while not self.stop_polling and (time.time() - start_time) < self.timeout:
            try:
                response = requests.get(poll_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    encrypted_key = data.get('key')
                    if encrypted_key:
                        try:
                            print("🔑 Encrypted key received, decrypting...")
                            # Decrypt the key using matching method
                            decrypted_key = self.decrypt_key(encrypted_key, session_id)
                            logger.info(f"✅ API key received and decrypted successfully!")
                            return decrypted_key
                        except Exception as e:
                            logger.error(f"Failed to decrypt key: {e}")
                            print(f"❌ Decryption failed: {e}")
                            return None
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
        Set environment variable for the current process AND save persistently.
        
        Args:
            key: Environment variable name
            value: Environment variable value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Set for current process
            os.environ[key] = value
            
            # Save persistently
            self._save_persistent_key(key, value)
            
            logger.info(f"✅ Environment variable {key} set successfully and saved persistently!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to set environment variable {key}: {e}")
            return False
    
    def setup_api_key_interactive(self, key_name: str = 'OPENAI_API_KEY') -> Tuple[bool, Optional[str]]:
        """
        Interactive setup process for API key via QR code.
        
        Args:
            key_name: Name of the environment variable to set
            
        Returns:
            Tuple of (success, api_key)
        """
        try:
            # Generate session ID
            self.session_id = self.generate_session_id()
            
            # Generate and display QR code
            session_url = self.generate_qr_code(self.session_id)
            
            # Start polling
            self.stop_polling = False
            
            # Poll for the key
            api_key = self.poll_for_key(self.session_id)
            
            if api_key:
                # Set the environment variable and save persistently
                success = self.set_environment_variable(key_name, api_key)
                if success:
                    print(f"\n✅ {key_name} setup completed successfully!")
                    print("🔒 API key saved securely and will persist across sessions.")
                    print("You can now use nterm with your API key.")
                    return True, api_key
                else:
                    print(f"\n❌ Failed to set environment variable {key_name}.")
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
    
    def check_existing_key(self, key_name: str = 'OPENAI_API_KEY') -> bool:
        """
        Check if the specified API key is already set in environment.
        
        Args:
            key_name: Name of the environment variable to check
            
        Returns:
            True if key exists, False otherwise
        """
        existing_key = os.getenv(key_name)
        if existing_key:
            print(f"✅ {key_name} is already configured (ends with: ...{existing_key[-4:]})")
            return True
        return False
    
    def setup_key_if_needed(self, force: bool = False, key_name: str = 'OPENAI_API_KEY') -> bool:
        """
        Setup API key if not already present or if forced.
        
        Args:
            force: Force setup even if key already exists
            key_name: Name of the environment variable to check/set
            
        Returns:
            True if key is available (existing or newly set), False otherwise
        """
        if not force and self.check_existing_key(key_name):
            return True
        
        if force:
            print("🔄 Forcing new API key setup...")
        else:
            print(f"🔍 No {key_name} found in environment.")
            print("Let's set it up using secure QR code...")
        
        success, _ = self.setup_api_key_interactive(key_name)
        return success
    
    def remove_key(self, key_name: str) -> bool:
        """
        Remove an API key from both environment and persistent storage.
        
        Args:
            key_name: Name of the environment variable to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from environment
            if key_name in os.environ:
                del os.environ[key_name]
            
            # Remove from persistent storage
            self._remove_persistent_key(key_name)
            
            print(f"✅ {key_name} removed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to remove key {key_name}: {e}")
            return False


def create_secure_key_manager(worker_url: str, **kwargs) -> SecureKeyManager:
    """
    Factory function to create a secure key manager.
    
    Args:
        worker_url: Cloudflare Worker URL
        **kwargs: Additional arguments for SecureKeyManager
        
    Returns:
        Configured SecureKeyManager instance
    """
    return SecureKeyManager(worker_url, **kwargs)