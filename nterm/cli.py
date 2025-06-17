"""
Command-line interface for the reasoning agent
"""
import argparse
import sys
from typing import Optional
from .agent import ReasoningAgent, create_nterm
from .config import DEFAULT_MODEL_ID, DEFAULT_DB_FILE, DEFAULT_TABLE_NAME, DEFAULT_HISTORY_RUNS


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Reasoning Agent - A system administration and IoT assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nterm                          # Start interactive CLI
  nterm --model gpt-4           # Use different model
  nterm --db-file ./my_data.db  # Use custom database file
  nterm --query "What OS am I running?"  # Single query mode
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
        "--version",
        action="version",
        version="nterm 0.1.0"
    )
    
    return parser


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
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
            print("Session history cleared.")
        
        # Single query mode
        if args.query:
            print(f"Query: {args.query}")
            print("=" * 50)
            response = agent.query(args.query)
            print(response)
            return
        
        # Interactive CLI mode
        agent.run_cli()
        
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()