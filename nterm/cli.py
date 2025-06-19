"""
Command-line interface for the reasoning agent with QR-based API key setup
"""
import argparse
import sys
import os
from typing import Optional
from .agent import ReasoningAgent, create_nterm
from .config import DEFAULT_MODEL_ID, DEFAULT_DB_FILE, DEFAULT_TABLE_NAME, DEFAULT_HISTORY_RUNS, DEFAULT_WORKER_URL
from .qr_key_manager import QRKeyManager


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Reasoning Agent - A system administration and IoT assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nterm                          # Start interactive CLI
  nterm --setup-key              # Setup OpenAI API key via QR code
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
        help="Setup OpenAI API key via QR code"
    )
    
    parser.add_argument(
        "--force-key-setup",
        action="store_true",
        help="Force API key setup even if already exists"
    )
    
    parser.add_argument(
        "--worker-url",
        default=DEFAULT_WORKER_URL,
        help=f"Cloudflare Worker URL for QR key setup (default: {DEFAULT_WORKER_URL})"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for QR key setup in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="nterm 0.1.0"
    )
    
    return parser


def check_api_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv('OPENAI_API_KEY'))


def setup_api_key(worker_url: str, timeout: int = 300, force: bool = False) -> bool:
    """
    Setup OpenAI API key via QR code.
    
    Args:
        worker_url: Cloudflare Worker URL
        timeout: Timeout in seconds
        force: Force setup even if key exists
        
    Returns:
        True if key is available, False otherwise
    """
    try:
        qr_manager = QRKeyManager(worker_url, timeout=timeout)
        return qr_manager.setup_key_if_needed(force=force)
    except ImportError:
        print("❌ QR code functionality requires additional dependencies.")
        print("Please install: pip install qrcode[pil] requests")
        return False
    except Exception as e:
        print(f"❌ Failed to setup API key: {e}")
        return False


def interactive_key_setup_prompt(worker_url: str, timeout: int = 300) -> bool:
    """
    Prompt user for API key setup if not available.
    
    Args:
        worker_url: Cloudflare Worker URL
        timeout: Timeout in seconds
        
    Returns:
        True if key is available, False otherwise
    """
    print("\n🔑 OpenAI API Key Required")
    print("="*50)
    print("nterm requires an OpenAI API key to function.")
    print("You can set it up using our secure QR code system.")
    print()
    
    while True:
        choice = input("Would you like to setup your API key now? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            return setup_api_key(worker_url, timeout=timeout)
        elif choice in ['n', 'no']:
            print("\nYou can setup your API key later by running:")
            print("  nterm --setup-key")
            print("\nOr set it manually:")
            print("  export OPENAI_API_KEY='your-api-key-here'")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Handle QR key setup mode
        if args.setup_key or args.force_key_setup:
            success = setup_api_key(
                args.worker_url, 
                timeout=args.timeout, 
                force=args.force_key_setup
            )
            if success:
                print("\n🎉 API key setup completed! You can now use nterm.")
            else:
                print("\n❌ API key setup failed or cancelled.")
                sys.exit(1)
            
            # If not in query mode, continue to normal operation
            if not args.query:
                print("\nStarting nterm...")
        
        # Check for API key availability
        if not check_api_key():
            print("🔍 No OpenAI API key found in environment.")
            
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
        
        # Create the reasoning agent
        agent = ReasoningAgent(
            model_id=args.model,
            db_file=args.db_file,
            table_name=args.table_name,
            num_history_runs=args.history_runs
        )
        
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
        print("Type 'exit' or 'quit' to end the session, or press Ctrl+C to exit.")
        print("=" * 60)
        
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