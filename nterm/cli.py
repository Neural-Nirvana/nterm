"""
Command-line interface for the reasoning agent with encryption-based API key setup
"""
import argparse
import sys
import os
from typing import Optional
from .agent import ReasoningAgent, create_nterm
from .config import DEFAULT_MODEL_ID, DEFAULT_DB_FILE, DEFAULT_TABLE_NAME, DEFAULT_HISTORY_RUNS, DEFAULT_WORKER_URL
from agno.utils.log import logger

# Import the secure key manager
try:
    from .secure_key_manager import SecureKeyManager
    SECURE_MANAGER_AVAILABLE = True
except ImportError:
    SECURE_MANAGER_AVAILABLE = False
    SecureKeyManager = None


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Reasoning Agent - A system administration and IoT assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nterm                          # Start interactive CLI
  nterm --setup-key              # Setup OpenAI API key via encrypted QR code
  nterm --setup-multiple-keys    # Setup multiple API keys interactively
  nterm --list-keys              # List all stored API keys
  nterm --remove-key OPENAI_API_KEY  # Remove a specific API key
  nterm --model gpt-4           # Use different model
  nterm --db-file ./my_data.db  # Use custom database file
  nterm --query "What OS am I running?"  # Single query mode
  nterm --worker-url https://your-worker.workers.dev  # Custom worker URL
        """
    )
    
    parser.add_argument(
        "--model", 
        default=DEFAULT_MODEL_ID,
        help=f"OpenAI model ID to use (default: {DEFAULT_MODEL_ID})"
    )
    
    parser.add_argument(
        "--db-file",
        default=DEFAULT_DB_FILE,
        help=f"SQLite database file path (default: {DEFAULT_DB_FILE})"
    )
    
    parser.add_argument(
        "--table-name",
        default=DEFAULT_TABLE_NAME,
        help=f"Database table name for sessions (default: {DEFAULT_TABLE_NAME})"
    )
    
    parser.add_argument(
        "--history-runs",
        type=int,
        default=DEFAULT_HISTORY_RUNS,
        help=f"Number of history runs to keep (default: {DEFAULT_HISTORY_RUNS})"
    )
    
    parser.add_argument(
        "--query",
        help="Single query to run (non-interactive mode)"
    )
    
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Clear the agent's session history before starting"
    )
    
    parser.add_argument(
        "--setup-key",
        action="store_true",
        help="Setup OpenAI API key via encrypted QR code"
    )
    
    parser.add_argument(
        "--setup-multiple-keys",
        action="store_true",
        help="Setup multiple API keys (OpenAI, Anthropic, etc.) via encrypted QR code"
    )
    
    parser.add_argument(
        "--force-key-setup",
        action="store_true",
        help="Force API key setup even if already exists"
    )
    
    parser.add_argument(
        "--key-name",
        default="OPENAI_API_KEY",
        help="Name of the environment variable for the API key (default: OPENAI_API_KEY)"
    )
    
    parser.add_argument(
        "--worker-url",
        default=DEFAULT_WORKER_URL,
        help=f"Cloudflare Worker URL for encrypted key setup (default: {DEFAULT_WORKER_URL})"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for encrypted key setup in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="List all stored API keys"
    )
    
    parser.add_argument(
        "--remove-key",
        help="Remove a specific API key from storage"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="nterm 0.4.0"
    )
    
    return parser


def check_api_key(key_name: str = "OPENAI_API_KEY") -> bool:
    """Check if the specified API key is available."""
    key_value = os.getenv(key_name)
    if key_value:
        print(f"✅ {key_name} found (ends with: ...{key_value[-4:]})")
        return True
    return False


def check_any_supported_key() -> tuple[bool, Optional[str]]:
    """
    Check for any supported API key.
    
    Returns:
        Tuple of (key_found, key_name)
    """
    supported_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", 
        "GOOGLE_API_KEY",
        "AZURE_OPENAI_API_KEY"
    ]
    
    for key_name in supported_keys:
        if check_api_key(key_name):
            return True, key_name
    
    return False, None


def setup_api_key(worker_url: str, key_name: str = "OPENAI_API_KEY", timeout: int = 300, force: bool = False) -> bool:
    """
    Setup API key via secure QR code.
    
    Args:
        worker_url: Cloudflare Worker URL
        key_name: Environment variable name for the API key
        timeout: Timeout in seconds
        force: Force setup even if key exists
        
    Returns:
        True if key is available, False otherwise
    """
    if not SECURE_MANAGER_AVAILABLE:
        print("❌ Secure key management requires additional dependencies.")
        print("Please install: pip install cryptography qrcode[pil] requests")
        return False
    
    try:
        print(f"\n🔑 Setting up {key_name}...")
        print("🔒 Using encrypted transmission (keys auto-expire from server)")
        
        manager = SecureKeyManager(worker_url, timeout=timeout)
        return manager.setup_key_if_needed(force=force, key_name=key_name)
        
    except Exception as e:
        print(f"❌ Failed to setup API key: {e}")
        return False


def setup_multiple_keys(worker_url: str, timeout: int = 300) -> bool:
    """
    Setup multiple API keys interactively.
    
    Args:
        worker_url: Cloudflare Worker URL
        timeout: Timeout in seconds
        
    Returns:
        True if at least one key was set up successfully
    """
    if not SECURE_MANAGER_AVAILABLE:
        print("❌ Secure key management requires additional dependencies.")
        print("Please install: pip install cryptography qrcode[pil] requests")
        return False
    
    print("\n🔧 Multiple API Key Setup")
    print("="*50)
    print("Set up API keys for different AI services.")
    print("🔒 All keys encrypted before transmission (auto-expire from server)")
    print()
    
    keys_to_setup = [
        ("OPENAI_API_KEY", "OpenAI (GPT models)"),
        ("ANTHROPIC_API_KEY", "Anthropic (Claude models)"),
        ("GOOGLE_API_KEY", "Google (Gemini models)"),
        ("AZURE_OPENAI_API_KEY", "Azure OpenAI"),
    ]
    
    success_count = 0
    
    for key_name, description in keys_to_setup:
        # Check if key already exists
        if check_api_key(key_name):
            print(f"✅ {description} - already configured")
            success_count += 1
            continue
        
        # Ask user if they want to set up this key
        print(f"\n📋 Setup {description}? (y/n/s to skip all remaining): ", end="")
        choice = input().lower().strip()
        
        if choice == 's':
            print("⏭️  Skipping remaining keys...")
            break
        elif choice in ['n', 'no']:
            print(f"⏭️  Skipping {description}")
            continue
        elif choice in ['y', 'yes', '']:
            print(f"\n🔑 Setting up {description}...")
            try:
                manager = SecureKeyManager(worker_url, timeout=timeout)
                success = manager.setup_key_if_needed(force=True, key_name=key_name)
                if success:
                    success_count += 1
                    print(f"✅ {description} configured successfully!")
                else:
                    print(f"❌ {description} setup failed or cancelled")
            except KeyboardInterrupt:
                print(f"\n🛑 Setup cancelled for {description}")
                break
            except Exception as e:
                print(f"❌ Error setting up {description}: {e}")
        else:
            print("Please enter 'y' for yes, 'n' for no, or 's' to skip remaining")
            continue
    
    print(f"\n🎉 Setup complete! {success_count} API key(s) configured.")
    return success_count > 0


def interactive_key_setup_prompt(worker_url: str, timeout: int = 300) -> bool:
    """
    Prompt user for API key setup if not available.
    
    Args:
        worker_url: Cloudflare Worker URL
        timeout: Timeout in seconds
        
    Returns:
        True if key is available, False otherwise
    """
    print("\n🔑 API Key Required")
    print("="*50)
    print("nterm requires an AI API key to function.")
    print("You can set it up using our secure encrypted QR code system.")
    print("🔒 Your API keys are encrypted before transmission and auto-expire from our servers.")
    print()
    
    while True:
        print("Setup options:")
        print("  1. Setup OpenAI API key (recommended)")
        print("  2. Setup multiple API keys (OpenAI, Anthropic, etc.)")
        print("  3. Skip setup (configure manually later)")
        print()
        
        choice = input("Choose an option (1/2/3): ").strip()
        
        if choice == '1':
            return setup_api_key(worker_url, timeout=timeout)
        elif choice == '2':
            return setup_multiple_keys(worker_url, timeout=timeout)
        elif choice == '3':
            print("\nYou can setup your API key later by running:")
            print("  nterm --setup-key")
            print("  nterm --setup-multiple-keys")
            print("\nOr set it manually:")
            print("  export OPENAI_API_KEY='your-api-key-here'")
            return False
        else:
            print("❌ Please enter 1, 2, or 3.")


def load_persistent_keys(worker_url: str) -> bool:
    """
    Load persistent API keys into environment at startup.
    
    Args:
        worker_url: Cloudflare Worker URL (for manager initialization)
        
    Returns:
        True if keys were loaded successfully, False otherwise
    """
    if not SECURE_MANAGER_AVAILABLE:
        return False
    
    try:
        # Create manager instance to trigger key loading
        manager = SecureKeyManager(worker_url)
        
        # Check if any keys were loaded
        loaded_keys = []
        supported_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "AZURE_OPENAI_API_KEY"]
        for key_name in supported_keys:
            if os.getenv(key_name):
                loaded_keys.append(key_name)
        
        if loaded_keys:
            logger.debug(f"🔑 Loaded {len(loaded_keys)} API key(s) from persistent storage: {', '.join(loaded_keys)}")
        
        return True
    except Exception as e:
        logger.debug(f"Failed to load persistent keys: {e}")
        return False
    """
    List all stored API keys.
    
    Args:
        worker_url: Cloudflare Worker URL (for manager initialization)
        
    Returns:
        True if successful, False otherwise
    """
    if not SECURE_MANAGER_AVAILABLE:
        print("❌ Secure key management requires additional dependencies.")
        print("Please install: pip install cryptography qrcode[pil] requests")
        return False
    
    try:
        manager = SecureKeyManager(worker_url)
        stored_keys = manager.list_stored_keys()
        
        if not stored_keys:
            print("📭 No API keys found in storage.")
            print("💡 Run 'nterm --setup-key' to add your first API key.")
            return True
        
        print("\n🔑 Stored API Keys:")
        print("="*50)
        
        key_type_descriptions = {
            'OPENAI_API_KEY': '🤖 OpenAI (GPT, DALL-E, Whisper)',
            'ANTHROPIC_API_KEY': '🧠 Anthropic (Claude)',
            'GOOGLE_API_KEY': '🔍 Google (Gemini, Search)',
            'AZURE_OPENAI_API_KEY': '☁️ Azure OpenAI'
        }
        
        for key_name, masked_value in stored_keys.items():
            description = key_type_descriptions.get(key_name, f'🔑 {key_name}')
            print(f"{description}")
            print(f"   └─ {masked_value}")
            print()
        
        print(f"📊 Total: {len(stored_keys)} API key(s) stored")
        print("\n💡 Commands:")
        print("  nterm --remove-key KEY_NAME  # Remove a specific key")
        print("  nterm --setup-key --force    # Replace a key")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to list keys: {e}")
        return False


def remove_stored_key(worker_url: str, key_name: str) -> bool:
    """
    Remove a stored API key.
    
    Args:
        worker_url: Cloudflare Worker URL (for manager initialization)
        key_name: Name of the key to remove
        
    Returns:
        True if successful, False otherwise
    """
    if not SECURE_MANAGER_AVAILABLE:
        print("❌ Secure key management requires additional dependencies.")
        print("Please install: pip install cryptography qrcode[pil] requests")
        return False
    
    try:
        manager = SecureKeyManager(worker_url)
        stored_keys = manager.list_stored_keys()
        
        if key_name not in stored_keys:
            print(f"❌ Key '{key_name}' not found in storage.")
            print("\n📝 Available keys:")
            for stored_key in stored_keys.keys():
                print(f"  • {stored_key}")
            return False
        
        # Confirm deletion
        print(f"🗑️  Remove API key: {key_name}")
        print(f"   Value: {stored_keys[key_name]}")
        
        confirm = input("\nAre you sure? (y/N): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("❌ Removal cancelled.")
            return False
        
        # Remove the key
        success = manager.remove_key(key_name)
        if success:
            print(f"✅ API key '{key_name}' removed successfully!")
        
        return success
        
    except Exception as e:
        print(f"❌ Failed to remove key: {e}")
        return False


def show_key_status():
    """Show current API key status."""
    print("\n🔑 API Key Status:")
    print("-" * 40)
    
    keys_to_check = [
        ("OPENAI_API_KEY", "OpenAI"),
        ("ANTHROPIC_API_KEY", "Anthropic"),
        ("GOOGLE_API_KEY", "Google"),
        ("AZURE_OPENAI_API_KEY", "Azure OpenAI"),
    ]
    
    found_keys = 0
    for key_name, service_name in keys_to_check:
        key_value = os.getenv(key_name)
        if key_value:
            print(f"✅ {service_name:<15} ...{key_value[-4:]}")
            found_keys += 1
        else:
            print(f"❌ {service_name:<15} Not set")
    
    if found_keys == 0:
        print("\n💡 No API keys found. Run 'nterm --setup-key' to get started.")
    else:
        print(f"\n🎉 {found_keys} API key(s) configured!")


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Load persistent API keys early in the startup process
        if SECURE_MANAGER_AVAILABLE:
            load_persistent_keys(args.worker_url)
        
        # Handle key management commands first
        if args.list_keys:
            success = list_stored_keys(args.worker_url)
            sys.exit(0 if success else 1)
        
        if args.remove_key:
            success = remove_stored_key(args.worker_url, args.remove_key)
            sys.exit(0 if success else 1)
        
        # Handle API key setup modes
        if args.setup_key or args.force_key_setup:
            success = setup_api_key(
                args.worker_url,
                key_name=args.key_name,
                timeout=args.timeout, 
                force=args.force_key_setup
            )
            if success:
                print(f"\n🎉 {args.key_name} setup completed! You can now use nterm.")
            else:
                print(f"\n❌ {args.key_name} setup failed or cancelled.")
                sys.exit(1)
            
            # If not in query mode, continue to normal operation
            if not args.query:
                print("\nStarting nterm...")
        
        elif args.setup_multiple_keys:
            success = setup_multiple_keys(args.worker_url, timeout=args.timeout)
            if success:
                print("\n🎉 API key setup completed! You can now use nterm.")
            else:
                print("\n❌ API key setup failed or cancelled.")
                sys.exit(1)
            
            # If not in query mode, continue to normal operation
            if not args.query:
                print("\nStarting nterm...")
        
        # Check for API key availability (after loading persistent keys)
        key_found, found_key_name = check_any_supported_key()
        
        if not key_found:
            print("🔍 No supported API keys found in environment.")
            
            # In query mode, require key to be set up
            if args.query:
                print("❌ API key required for query mode.")
                print("Please run: nterm --setup-key")
                sys.exit(1)
            
            # In interactive mode, offer to set up key
            key_available = interactive_key_setup_prompt(args.worker_url, args.timeout)
            if not key_available:
                print("❌ Cannot proceed without API key.")
                sys.exit(1)
        else:
            print(f"🔑 Using API key: {found_key_name}")
        
        # Create the reasoning agent (keys should now be loaded)
        try:
            agent = ReasoningAgent(
                model_id=args.model,
                db_file=args.db_file,
                table_name=args.table_name,
                num_history_runs=args.history_runs
            )
        except ValueError as e:
            if "API key not found" in str(e):
                print("❌ API key configuration error.")
                print("Please run: nterm --setup-key")
                sys.exit(1)
            else:
                raise
        
        # Clear history if requested
        if args.clear_history:
            agent.clear_history()
            print("✅ Session history cleared.")
        
        # Single query mode
        if args.query:
            print(f"\n🤖 Query: {args.query}")
            print("=" * 60)
            try:
                response = agent.query(args.query)
                print(response)
            except Exception as e:
                if "authentication" in str(e).lower() or "api key" in str(e).lower():
                    print("❌ API key authentication failed.")
                    print("Your API key might be invalid or expired.")
                    print("Please run: nterm --setup-key --force")
                    sys.exit(1)
                else:
                    raise e
            return
        
        # Interactive CLI mode
        print("\n🧠 Starting Nirvana Terminal Reasoning Agent...")
        print("✅ API key configured")
        print("🔒 Keys are loaded from secure storage")
        print("📱 Pro tip: You can scan QR codes to securely send encrypted API keys anytime!")
        print("Type 'exit' or 'quit' to end the session, or press Ctrl+C to exit.")
        print("=" * 60)
        
        # Show available commands
        print("\n💡 Available commands:")
        print("  /keys     - Show API key status")  
        print("  /list     - List all stored API keys")
        print("  /setup    - Setup additional API keys")
        print("  /help     - Show all commands")
        print("  /clear    - Clear conversation history")
        print()
        
        agent.run_cli()
        
    except KeyboardInterrupt:
        print("\n\n👋 Exiting nterm...")
        sys.exit(0)
    except Exception as e:
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            print(f"\n❌ API Authentication Error: {e}")
            print("Your API key might be invalid or expired.")
            print("Please run: nterm --setup-key --force")
        else:
            print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()