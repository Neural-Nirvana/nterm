"""
Elegant reasoning agent
"""
import os
from typing import Optional, List, Dict, Any
from agno.agent import Agent
from agno.models.openai import OpenAIChat

from agno.tools.reasoning import ReasoningTools
from agno.tools.shell import ShellTools
from agno.storage.sqlite import SqliteStorage
from agno.utils.log import logger
from agno.tools.python import PythonTools
from agno.tools.file import FileTools

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme
from rich.align import Align
from rich.box import ROUNDED, MINIMAL
from rich.rule import Rule
from rich.columns import Columns

from .config import DEFAULT_INSTRUCTIONS, DEFAULT_MODEL_ID


class ElegantReasoningAgent:
    """
    A beautifully designed reasoning agent.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        instructions: Optional[str] = None,
        db_file: Optional[str] = None,
        table_name: str = "nterm_sessions",
        num_history_runs: int = 3,
        custom_tools: Optional[List[Any]] = None,
        show_reasoning: bool = True,
        **kwargs
    ):
        """Initialize the elegant reasoning agent."""
        
        # Use environment variables or defaults
        self.model_id = model_id or os.getenv('NTERM_MODEL_ID', DEFAULT_MODEL_ID)
        self.instructions = instructions or os.getenv('NTERM_INSTRUCTIONS', DEFAULT_INSTRUCTIONS)
        self.db_file = db_file or os.getenv('NTERM_DB_FILE', "tmp/data.db")
        self.table_name = table_name
        self.num_history_runs = num_history_runs
        self.show_reasoning = show_reasoning
        
        # Setup elegant console theme
        self.console = self._setup_console()
        
        # Check for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.console.print(
                self._create_error_panel(
                    "OpenAI API Key Missing",
                    "Please run 'nterm --setup-key' to configure your API key."
                )
            )
            raise ValueError("OpenAI API key not found in environment.")
        
        # Setup tools with reasoning
        tools = [
            ReasoningTools(
                think=True, 
                analyze=True, 
                add_instructions=True,
                add_few_shot=True
            ), 
            ShellTools(), 
            PythonTools(), 
            FileTools()
        ]
        if custom_tools:
            tools.extend(custom_tools)
        
        # Create a simple, reliable agent
        try:
            self.agent = Agent(
                model=OpenAIChat(id=self.model_id, api_key=api_key),
                tools=tools,
                instructions=self.instructions,
                add_datetime_to_instructions=True,
                show_tool_calls=True,
                markdown=True,
                storage=SqliteStorage(table_name=self.table_name, db_file=self.db_file),
                add_history_to_messages=True,
                num_history_runs=self.num_history_runs,
                **kwargs
            )
                
        except Exception as e:
            self.console.print(
                self._create_error_panel(
                    "Agent Initialization Failed",
                    str(e)
                )
            )
            raise
    
    def _setup_console(self) -> Console:
        """Setup console with elegant theme and visual hierarchy."""
        elegant_theme = Theme({
            # Primary brand colors
            "primary": "bold #00D4AA",           # Bright teal
            "primary_dim": "#00A085",            # Darker teal
            "secondary": "bold #6366F1",         # Indigo
            "secondary_dim": "#4F46E5",          # Darker indigo
            
            # Functional colors with hierarchy
            "accent": "bold #F59E0B",            # Amber - for highlights
            "success": "bold #10B981",           # Emerald
            "warning": "bold #F59E0B",           # Amber
            "error": "bold #EF4444",             # Red
            "info": "#3B82F6",                   # Blue
            
            # Text hierarchy
            "text_primary": "white",             # Main content
            "text_secondary": "#E5E7EB",         # Secondary content
            "text_muted": "#9CA3AF",             # Muted content
            "text_subtle": "#6B7280",            # Very subtle
            
            # User interaction
            "user": "bold #FF6B9D",              # Pink for user
            "ai": "bold #00D4AA",                # Teal for AI
            "system": "#F59E0B",                 # Amber for system
            
            # Reasoning and tools
            "thinking": "bold #8B5CF6",          # Purple
            "tool": "#06B6D4",                   # Cyan
            "reasoning": "bold #EC4899",         # Pink
            
            # Interface elements
            "border_primary": "#00D4AA",
            "border_secondary": "#6366F1",
            "border_muted": "#374151",
        })
        
        return Console(theme=elegant_theme, width=120, legacy_windows=False)
    
    def run_cli(self):
        """Start the elegant interactive CLI application."""
        # Clear screen and show elegant header
        self.console.clear()
        self._show_header()
        
        try:
            self._chat_loop()
        except KeyboardInterrupt:
            self._show_goodbye()
        except Exception as e:
            self.console.print(
                self._create_error_panel("Unexpected Error", str(e))
            )
            raise
    
    def _show_header(self):
        """Show elegant application header with clear visual hierarchy."""
        # Main brand header
        self.console.print()
        
        header_content = Text()
        header_content.append("◆ ", style="primary")
        header_content.append("NTERM", style="bold white")
        header_content.append(" ◆", style="primary")
        header_content.append("\n")
        header_content.append("Terminal that Thinks", style="text_secondary")
        
        header_panel = Panel(
            Align.center(header_content),
            box=ROUNDED,
            border_style="primary",
            padding=(1, 4),
            width=60
        )
        
        self.console.print(Align.center(header_panel))
        self.console.print()
        
        # Status line with elegant formatting
        status_parts = []
        status_parts.append(("Model: ", "text_muted"))
        status_parts.append((self.model_id, "accent"))
        status_parts.append((" • ", "text_muted"))
        status_parts.append(("Show Reasoning steps: ", "text_muted"))
        status_parts.append(("ON" if self.show_reasoning else "OFF", "success" if self.show_reasoning else "warning"))
        status_parts.append((" • ", "text_muted"))
        status_parts.append(("Ready", "success"))
        
        status_text = Text()
        for text, style in status_parts:
            status_text.append(text, style=style)
        
        self.console.print(Align.center(status_text))
        self.console.print()
        
        # Welcome message
        welcome = Text("Ask me anything or type ", style="text_muted")
        welcome.append("help", style="accent")
        welcome.append(" for commands", style="text_muted")
        self.console.print(Align.center(welcome))
        
        # Elegant separator
        self.console.print()
        self.console.print(Rule(style="border_muted"))
        self.console.print()
    
    def _show_goodbye(self):
        """Show elegant goodbye message."""
        self.console.print()
        self.console.print(Rule(style="border_muted"))
        self.console.print()
        
        goodbye_text = Text()
        goodbye_text.append("Session ended", style="text_secondary")
        goodbye_text.append(" • ", style="text_muted")
        goodbye_text.append("Thank you for using NTERM", style="primary")
        
        self.console.print(Align.center(goodbye_text))
        self.console.print()
    
    def _chat_loop(self):
        """Main chat interaction loop with elegant prompting."""
        exit_commands = {"exit", "quit", "bye", "goodbye"}
        
        while True:
            try:
                # Elegant prompt with visual hierarchy
                prompt_text = Text()
                prompt_text.append("▸ ", style="user")
                
                message = Prompt.ask(prompt_text, console=self.console)
                
                if message.lower().strip() in exit_commands:
                    break
                    
                # Handle special commands
                if self._handle_command(message):
                    continue
                
                # Process AI query with elegant formatting
                self._process_ai_query(message)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.console.print(
                    self._create_error_panel("Processing Error", str(e))
                )
    
    def _handle_command(self, message: str) -> bool:
        """Handle special commands with elegant feedback."""
        cmd = message.lower().strip()
        
        if cmd == "help":
            self._show_help()
            return True
        elif cmd == "status":
            self._show_status()
            return True
        elif cmd == "clear":
            self.console.clear()
            self._show_header()
            return True
        elif cmd == "history":
            self._show_history()
            return True
        elif cmd == "reasoning on":
            self.show_reasoning = True
            self.console.print(self._create_success_message("Reasoning display enabled"))
            return True
        elif cmd == "reasoning off":
            self.show_reasoning = False
            self.console.print(self._create_warning_message("Reasoning display disabled"))
            return True
        elif cmd == "test":
            self._test_agent()
            return True
        
        return False
    
    def _process_ai_query(self, message: str):
        """Process AI query with beautiful visual hierarchy."""
        try:
            # Query header with elegant design
            self._show_query_header(message)
            
            # Use the agent's built-in print_response method
            self.agent.print_response(
                message=message,
                stream=True,
                show_reasoning=self.show_reasoning,
                show_tool_calls=True,
                console=self.console,
                markdown=True
            )
            
            # Add elegant spacing after response
            self._show_response_footer()
            
        except Exception as e:
            self.console.print(
                self._create_error_panel("Query Processing Error", str(e))
            )
            
            # Try a simple non-streaming version as fallback
            try:
                self.console.print(self._create_info_message("Trying alternative mode..."))
                response = self.agent.run(message)
                
                # Format response with elegant styling
                response_content = str(response.content) if hasattr(response, 'content') else str(response)
                self._show_response_content(response_content)
                self._show_response_footer()
                
            except Exception as e2:
                self.console.print(
                    self._create_error_panel("Complete Failure", str(e2))
                )
    
    def _show_query_header(self, message: str):
        """Show elegant query header."""
        # Create query display with visual hierarchy
        query_text = Text()
        query_text.append("Query", style="text_muted")
        
        query_panel = Panel(
            message,
            title=query_text,
            title_align="left",
            border_style="user",
            box=MINIMAL,
            padding=(0, 1),
        )
        
        self.console.print()
        self.console.print(query_panel)
    
    def _show_response_content(self, content: str):
        """Show response content with elegant formatting."""
        response_panel = Panel(
            content,
            title=Text("Response", style="ai"),
            title_align="left",
            border_style="ai",
            box=ROUNDED,
            padding=(1, 2),
        )
        self.console.print(response_panel)
    
    def _show_response_footer(self):
        """Show elegant response footer."""
        self.console.print()
        self.console.print(Rule(style="border_muted", characters="·"))
        self.console.print()
    
    def _test_agent(self):
        """Test the agent with elegant feedback."""
        test_panel = Panel(
            "Testing agent functionality...",
            title=Text("System Test", style="system"),
            border_style="system",
            box=MINIMAL
        )
        self.console.print(test_panel)
        
        try:
            response = self.agent.run("What is 2+2?", stream=False)
            self.console.print(
                self._create_success_message(f"Agent working! Response: {response.content}")
            )
        except Exception as e:
            self.console.print(
                self._create_error_panel("Agent Test Failed", str(e))
            )
    
    def _create_success_message(self, message: str) -> Panel:
        """Create elegant success message."""
        return Panel(
            f"✓ {message}",
            border_style="success",
            box=MINIMAL,
            padding=(0, 1)
        )
    
    def _create_warning_message(self, message: str) -> Panel:
        """Create elegant warning message."""
        return Panel(
            f"⚠ {message}",
            border_style="warning", 
            box=MINIMAL,
            padding=(0, 1)
        )
    
    def _create_info_message(self, message: str) -> Panel:
        """Create elegant info message."""
        return Panel(
            f"ℹ {message}",
            border_style="info",
            box=MINIMAL,
            padding=(0, 1)
        )
    
    def _create_error_panel(self, title: str, message: str) -> Panel:
        """Create elegant error panel."""
        error_content = Text()
        error_content.append("✗ ", style="error")
        error_content.append(message, style="text_primary")
        
        return Panel(
            error_content,
            title=Text(title, style="error"),
            border_style="error",
            box=ROUNDED,
            padding=(1, 2)
        )
    
    def _show_help(self):
        """Show elegant help with visual hierarchy."""
        # Main commands section
        commands_content = Text()
        commands_content.append("Essential Commands\n", style="bold text_primary")
        commands_content.append("\n")
        
        commands = [
            ("help", "Show this help message"),
            ("status", "Display system information"),
            ("clear", "Clear the screen"),
            # ("history", "Show conversation history"),
            ("reasoning on/off", "Toggle reasoning display"),
            ("test", "Test agent functionality"),
            ("use nterm --model <model_id>", "Change the model before initiating the session"),
            ("exit", "End the session")
        ]
        
        for cmd, desc in commands:
            commands_content.append(f"  {cmd:<18}", style="accent")
            commands_content.append(f"{desc}\n", style="text_secondary")
        
        commands_panel = Panel(
            commands_content,
            title=Text("Commands", style="primary"),
            border_style="primary",
            box=ROUNDED,
            padding=(1, 2)
        )
        
        # Features section
        features_content = Text()
        features_content.append("Core Features\n", style="bold text_primary")
        features_content.append("\n")
        
        features = [
            "• Natural language conversation",
            "• Real-time reasoning display",
            "• Shell command execution", 
            "• Python code execution",
            "• File analysis and manipulation",
            "• Persistent conversation history"
        ]
        
        for feature in features:
            features_content.append(f"{feature}\n", style="text_secondary")
        
        features_panel = Panel(
            features_content,
            title=Text("Features", style="secondary"),
            border_style="secondary", 
            box=ROUNDED,
            padding=(1, 2)
        )
        
        # Tips section
        tips_content = Text()
        tips_content.append("Usage Tips\n", style="bold text_primary")
        tips_content.append("\n")
        
        tips = [
            "• Ask any question",
            "• request step-by-step explanations", 
            "• Use 'reasoning on' to see overall thinking",
            "• Try mathematical or analytical problems",
            "• Ask for system information or file operations. enjoy"
        ]
        
        for tip in tips:
            tips_content.append(f"{tip}\n", style="text_secondary")
        
        tips_panel = Panel(
            tips_content,
            title=Text("Tips", style="thinking"),
            border_style="thinking",
            box=ROUNDED,
            padding=(1, 2)
        )
        
        # Layout in columns for better visual hierarchy
        self.console.print()
        self.console.print(commands_panel)
        self.console.print()
        
        # Two column layout for features and tips
        columns = Columns([features_panel, tips_panel], equal=True, expand=True)
        self.console.print(columns)
        self.console.print()
    
    def _show_status(self):
        """Show elegant system status with visual hierarchy."""
        api_key = os.getenv('OPENAI_API_KEY', '')
        
        # Create status table with elegant styling
        table = Table(
            show_header=False,
            box=None,
            padding=(0, 2, 0, 0),
            collapse_padding=True
        )
        table.add_column("Component", style="text_secondary", width=25, no_wrap=True)
        table.add_column("Status", style="text_primary")
        table.add_column("Indicator", width=3, justify="center")
        
        # Add status rows with visual indicators
        status_rows = [
            ("AI Model", self.model_id, "●", "accent"),
            ("API Key", f"Configured (***{api_key[-4:]})" if api_key else "Missing", 
             "●" if api_key else "●", "success" if api_key else "error"),
            ("Database", self.db_file, "●", "info"),
            ("Tools", f"{len(self.agent.tools)} loaded", "●", "success"),
            ("History", f"{self.num_history_runs} runs", "●", "info"),
            ("Reasoning", "ON" if self.show_reasoning else "OFF", 
             "●", "success" if self.show_reasoning else "warning"),
        ]
        
        # Check ReasoningTools
        has_reasoning_tools = any(
            tool.__class__.__name__ == "ReasoningTools" 
            for tool in self.agent.tools 
            if hasattr(tool, '__class__')
        )
        status_rows.append((
            "ReasoningTools", 
            "Present" if has_reasoning_tools else "Missing",
            "●", 
            "success" if has_reasoning_tools else "error"
        ))
        
        # Test agent functionality
        try:
            test_response = self.agent.run("test", stream=False)
            agent_status = ("Agent Status", "Working", "●", "success")
        except:
            agent_status = ("Agent Status", "Error", "●", "error")
        
        status_rows.append(agent_status)
        
        for component, status, indicator, color in status_rows:
            table.add_row(
                component,
                status,
                Text(indicator, style=color)
            )
        
        # Wrap in elegant panel
        status_panel = Panel(
            table,
            title=Text("System Status", style="system"),
            border_style="system",
            box=ROUNDED,
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(status_panel)
        self.console.print()
    
    def _show_history(self):
        """Show conversation history with elegant formatting."""
        try:
            history = self.get_session_history()
            if not history:
                self.console.print(
                    Panel(
                        "No conversation history available",
                        title=Text("History", style="text_muted"),
                        border_style="border_muted",
                        box=MINIMAL
                    )
                )
                return
            
            # Format history with visual hierarchy
            history_content = Text()
            
            for i, msg in enumerate(history[-10:], 1):  # Show last 10 messages
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                
                # Truncate long content elegantly
                if len(content) > 100:
                    content = content[:97] + "..."
                
                if role == 'user':
                    history_content.append(f"{i:2d}. ", style="text_muted")
                    history_content.append("You: ", style="user")
                    history_content.append(f"{content}\n", style="text_secondary")
                elif role == 'assistant':
                    history_content.append(f"{i:2d}. ", style="text_muted")
                    history_content.append("AI: ", style="ai")
                    history_content.append(f"{content}\n", style="text_secondary")
                
                history_content.append("\n")
            
            history_panel = Panel(
                history_content,
                title=Text("Recent Conversation History", style="system"),
                border_style="system",
                box=ROUNDED,
                padding=(1, 2)
            )
            
            self.console.print()
            self.console.print(history_panel)
            self.console.print()
            
        except Exception as e:
            self.console.print(
                self._create_error_panel("History Error", str(e))
            )
    
    def query(self, message: str) -> str:
        """Send a single query to the agent and get response."""
        try:
            response = self.agent.run(message, stream=False)
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
            return self.agent.storage.get_all_sessions()
        return []
    
    def clear_history(self):
        """Clear the agent's session history."""
        if hasattr(self.agent, 'storage') and self.agent.storage:
            self.agent.storage.clear()
            self.console.print(self._create_success_message("Session history cleared"))
        else:
            self.console.print(self._create_warning_message("No storage available to clear"))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and configuration."""
        api_key = os.getenv('OPENAI_API_KEY', '')
        return {
            'model_id': self.model_id,
            'db_file': self.db_file,
            'table_name': self.table_name,
            'api_key_configured': bool(api_key),
            'api_key_suffix': f"***...{api_key[-4:]}" if api_key else "Not set",
            'tools_count': len(self.agent.tools) if hasattr(self.agent, 'tools') else 0,
            'history_runs': self.num_history_runs,
            'show_reasoning': self.show_reasoning
        }


def create_elegant_nterm(**kwargs) -> ElegantReasoningAgent:
    """
    Factory function to create an elegant reasoning agent with beautiful visual hierarchy.
    
    Args:
        **kwargs: Arguments passed to ElegantReasoningAgent constructor
        
    Returns:
        Configured ElegantReasoningAgent instance with elegant visual design
    """
    return ElegantReasoningAgent(**kwargs)