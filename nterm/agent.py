"""
Core reasoning agent implementation with environment variable support
"""
import os
from textwrap import dedent
from typing import Optional, List, Dict, Any
from agno.agent import Agent
from agno.models.openai import OpenAIChat

from agno.tools.reasoning import ReasoningTools
from agno.tools.shell import ShellTools
from agno.storage.sqlite import SqliteStorage
from agno.utils.log import logger
from agno.tools.python import PythonTools
from agno.tools.file import FileTools

from .config import DEFAULT_INSTRUCTIONS, DEFAULT_MODEL_ID


class ReasoningAgent:
    """
    A reasoning agent with system administration and IoT capabilities.
    
    This agent can understand user queries about system environments and
    use shell tools and reasoning capabilities to provide comprehensive answers.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        instructions: Optional[str] = None,
        db_file: Optional[str] = None,
        table_name: str = "nterm_sessions",
        num_history_runs: int = 3,
        custom_tools: Optional[List[Any]] = None,
        **kwargs
    ):
        """
        Initialize the reasoning agent.
        
        Args:
            model_id: OpenAI model ID to use (default from env or config)
            instructions: Custom instructions for the agent
            db_file: SQLite database file path for storage (default from env or config)
            table_name: Database table name for sessions
            num_history_runs: Number of history runs to keep
            custom_tools: Additional tools to add to the agent
            **kwargs: Additional arguments passed to the Agent
        """
        # Use environment variables or defaults
        self.model_id = model_id or os.getenv('NTERM_MODEL_ID', DEFAULT_MODEL_ID)
        self.instructions = instructions or os.getenv('NTERM_INSTRUCTIONS', DEFAULT_INSTRUCTIONS)
        self.db_file = db_file or os.getenv('NTERM_DB_FILE', "tmp/data.db")
        self.table_name = table_name
        self.num_history_runs = num_history_runs
        
        # Check for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key not found in environment. "
                "Please run 'nterm --setup-key' to configure your API key."
            )
        
        # Setup tools
        tools = [ReasoningTools(add_instructions=True), ShellTools(), PythonTools(), FileTools()]
        if custom_tools:
            tools.extend(custom_tools)
        
        # Create the agent
        try:
            self.agent = Agent(
                model=OpenAIChat(id=self.model_id, api_key=api_key),
                tools=tools,
                instructions=self.instructions,
                add_datetime_to_instructions=True,
                stream_intermediate_steps=True,
                show_tool_calls=True,
                markdown=True,
                storage=SqliteStorage(table_name=self.table_name, db_file=self.db_file),
                add_history_to_messages=True,
                num_history_runs=self.num_history_runs,
                **kwargs
            )
            
            # Log successful initialization
            if os.getenv('NTERM_DEBUG'):
                logger.info(f"✅ Agent initialized with model: {self.model_id}")
                logger.info(f"✅ Database: {self.db_file}")
                logger.info(f"✅ API Key: ***...{api_key[-4:]}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent: {e}")
            raise
    
    def run_cli(self):
        """Start the interactive CLI application."""
        logger.info("Starting Nirvana Terminal Reasoning Agent CLI. Type 'exit' or 'quit' to end the session.")
        try:
            self.agent.cli_app()
        except KeyboardInterrupt:
            print("\n👋 Session ended by user.")
        except Exception as e:
            logger.error(f"❌ CLI error: {e}")
            raise
    
    def query(self, message: str) -> str:
        """
        Send a single query to the agent and get response.
        
        Args:
            message: The query/question to ask the agent
            
        Returns:
            The agent's response as a string
        """
        try:
            response = self.agent.run(message)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            if "authentication" in str(e).lower() or "api key" in str(e).lower():
                raise ValueError(
                    f"API authentication failed: {e}. "
                    "Your API key might be invalid. Run 'nterm --setup-key --force' to update it."
                )
            raise
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get the current session history."""
        if hasattr(self.agent, 'storage') and self.agent.storage:
            return self.agent.storage.get_messages()
        return []
    
    def clear_history(self):
        """Clear the agent's session history."""
        if hasattr(self.agent, 'storage') and self.agent.storage:
            self.agent.storage.clear()
            logger.info("✅ Session history cleared")
        else:
            logger.warning("⚠️ No storage available to clear")
    
    def add_tool(self, tool):
        """
        Add a custom tool to the agent.
        
        Args:
            tool: The tool instance to add
        """
        if hasattr(self.agent, 'tools'):
            self.agent.tools.append(tool)
            logger.info(f"✅ Added tool: {tool.__class__.__name__}")
        else:
            logger.warning("⚠️ Cannot add tool - agent tools not accessible")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status and configuration.
        
        Returns:
            Dictionary with status information
        """
        api_key = os.getenv('OPENAI_API_KEY', '')
        return {
            'model_id': self.model_id,
            'db_file': self.db_file,
            'table_name': self.table_name,
            'api_key_configured': bool(api_key),
            'api_key_suffix': f"***...{api_key[-4:]}" if api_key else "Not set",
            'tools_count': len(self.agent.tools) if hasattr(self.agent, 'tools') else 0,
            'history_runs': self.num_history_runs
        }


def create_nterm(**kwargs) -> ReasoningAgent:
    """
    Factory function to create a reasoning agent with default settings.
    
    Args:
        **kwargs: Arguments passed to ReasoningAgent constructor
        
    Returns:
        Configured ReasoningAgent instance
    """
    return ReasoningAgent(**kwargs)