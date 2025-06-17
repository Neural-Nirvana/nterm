"""
Core reasoning agent implementation
"""
from textwrap import dedent
from typing import Optional, List, Dict, Any
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.tools.shell import ShellTools
from agno.storage.sqlite import SqliteStorage
from agno.utils.log import logger

from .config import DEFAULT_INSTRUCTIONS, DEFAULT_MODEL_ID


class ReasoningAgent:
    """
    A reasoning agent with system administration and IoT capabilities.
    
    This agent can understand user queries about system environments and
    use shell tools and reasoning capabilities to provide comprehensive answers.
    """
    
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        instructions: Optional[str] = None,
        db_file: str = "tmp/data.db",
        table_name: str = "nterm_sessions",
        num_history_runs: int = 3,
        custom_tools: Optional[List[Any]] = None,
        **kwargs
    ):
        """
        Initialize the reasoning agent.
        
        Args:
            model_id: OpenAI model ID to use (default: gpt-4o)
            instructions: Custom instructions for the agent
            db_file: SQLite database file path for storage
            table_name: Database table name for sessions
            num_history_runs: Number of history runs to keep
            custom_tools: Additional tools to add to the agent
            **kwargs: Additional arguments passed to the Agent
        """
        self.model_id = model_id
        self.instructions = instructions or DEFAULT_INSTRUCTIONS
        self.db_file = db_file
        self.table_name = table_name
        self.num_history_runs = num_history_runs
        
        # Setup tools
        tools = [ReasoningTools(add_instructions=True), ShellTools()]
        if custom_tools:
            tools.extend(custom_tools)
        
        # Create the agent
        self.agent = Agent(
            model=OpenAIChat(id=self.model_id),
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
    
    def run_cli(self):
        """Start the interactive CLI application."""
        logger.info("Starting interactive reasoning agent CLI. Type 'exit' or 'quit' to end the session.")
        self.agent.cli_app()
    
    def query(self, message: str) -> str:
        """
        Send a single query to the agent and get response.
        
        Args:
            message: The query/question to ask the agent
            
        Returns:
            The agent's response as a string
        """
        response = self.agent.run(message)
        return response.content if hasattr(response, 'content') else str(response)
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get the current session history."""
        if hasattr(self.agent, 'storage') and self.agent.storage:
            return self.agent.storage.get_messages()
        return []
    
    def clear_history(self):
        """Clear the agent's session history."""
        if hasattr(self.agent, 'storage') and self.agent.storage:
            self.agent.storage.clear()
    
    def add_tool(self, tool):
        """
        Add a custom tool to the agent.
        
        Args:
            tool: The tool instance to add
        """
        if hasattr(self.agent, 'tools'):
            self.agent.tools.append(tool)
        else:
            logger.warning("Cannot add tool - agent tools not accessible")


def create_nterm(**kwargs) -> ReasoningAgent:
    """
    Factory function to create a reasoning agent with default settings.
    
    Args:
        **kwargs: Arguments passed to ReasoningAgent constructor
        
    Returns:
        Configured ReasoningAgent instance
    """
    return ReasoningAgent(**kwargs)