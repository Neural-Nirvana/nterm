"""
Elegant reasoning agent implementation focused on chat experience
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

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.theme import Theme
from rich.align import Align
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live
import time
import re

from .config import DEFAULT_INSTRUCTIONS, DEFAULT_MODEL_ID


class ElegantReasoningAgent:
    """
    A clean, elegant reasoning agent focused on exceptional chat experience.
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
        """Initialize the elegant reasoning agent."""
        
        # Use environment variables or defaults
        self.model_id = model_id or os.getenv('NTERM_MODEL_ID', DEFAULT_MODEL_ID)
        self.instructions = instructions or os.getenv('NTERM_INSTRUCTIONS', DEFAULT_INSTRUCTIONS)
        self.db_file = db_file or os.getenv('NTERM_DB_FILE', "tmp/data.db")
        self.table_name = table_name
        self.num_history_runs = num_history_runs
        
        # Setup elegant console theme
        self.console = self._setup_console()
        
        # Check for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.console.print(
                Panel.fit(
                    "[bold red]❌ OpenAI API key not found[/bold red]\n"
                    "[dim]Please run 'nterm --setup-key' to configure your API key.[/dim]",
                    border_style="red"
                ),
                style="bold"
            )
            raise ValueError("OpenAI API key not found in environment.")
        
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
                
        except Exception as e:
            self.console.print(
                Panel.fit(
                    f"[bold red]❌ Failed to initialize agent[/bold red]\n"
                    f"[dim]{str(e)}[/dim]",
                    border_style="red"
                )
            )
            raise
    
    def _setup_console(self) -> Console:
        """Setup console with elegant theme."""
        elegant_theme = Theme({
            "primary": "bold cyan",
            "secondary": "dim cyan", 
            "accent": "bold green",
            "text": "white",
            "dim_text": "dim white",
            "user": "bold blue",
            "ai": "bold green",
            "system": "yellow",
            "code": "bright_blue",
            "success": "green",
            "warning": "yellow",
            "error": "red"
        })
        
        return Console(theme=elegant_theme, width=100)
    
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
                Panel.fit(
                    f"[error]Unexpected error: {str(e)}[/error]",
                    border_style="red"
                )
            )
            raise
    
    def _show_header(self):
        """Show elegant application header."""
        header = """
╭──────────────────────────────────────╮
│                                      │
│            ◆ NTERM ◆                 │
│      AI Reasoning Terminal           │
│                                      │
╰──────────────────────────────────────╯
        """
        
        header_text = Text(header, style="primary")
        self.console.print(Align.center(header_text))
        
        # Elegant status line
        status_line = f"Model: {self.model_id} • Ready"
        status_text = Text(status_line, style="secondary")
        self.console.print(Align.center(status_text))
        self.console.print()
        
        # Welcome message
        welcome = Text("Ask me anything or type 'help' for commands", style="dim_text")
        self.console.print(Align.center(welcome))
        self.console.print("─" * 100, style="dim_text")
        self.console.print()
    
    def _show_goodbye(self):
        """Show elegant goodbye message."""
        self.console.print()
        self.console.print("─" * 100, style="dim_text")
        goodbye = Text("Session ended • Thank you for using NTERM", style="secondary")
        self.console.print(Align.center(goodbye))
    
    def _chat_loop(self):
        """Main chat interaction loop."""
        exit_commands = {"exit", "quit", "bye", "goodbye"}
        
        while True:
            try:
                # Elegant prompt
                message = Prompt.ask("[user]▸[/user]", console=self.console)
                
                if message.lower().strip() in exit_commands:
                    break
                    
                # Handle special commands
                if self._handle_command(message):
                    continue
                
                # Process AI query
                self._process_ai_query(message)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.console.print(
                    Panel.fit(
                        f"[error]Error: {str(e)}[/error]",
                        border_style="red"
                    )
                )
    
    def _handle_command(self, message: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
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
        
        return False
    
    def _process_ai_query(self, message: str):
        """Process AI query with elegant feedback."""
        # Show thinking indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[secondary]Thinking...[/secondary]"),
            transient=True,
            console=self.console
        ) as progress:
            progress.add_task("", total=None)
            
            try:
                response = self.agent.run(message)
                time.sleep(0.2)  # Brief pause for elegance
            except Exception as e:
                progress.stop()
                self.console.print(
                    Panel.fit(
                        f"[error]AI Error: {str(e)}[/error]",
                        border_style="red"
                    )
                )
                return
        
        # Format and display response
        self._display_ai_response(response)
    
    def _display_ai_response(self, response):
        """Display AI response with elegant formatting."""
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Check if response contains code blocks
        if self._contains_code(content):
            self._display_formatted_response(content)
        else:
            # Simple text response
            response_panel = Panel(
                content,
                title="[ai]◆ AI Assistant[/ai]",
                border_style="accent",
                padding=(1, 2)
            )
            self.console.print(response_panel)
        
        self.console.print()
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code blocks."""
        return "```" in text or "`" in text
    
    def _display_formatted_response(self, content: str):
        """Display response with proper code highlighting."""
        # Split content into text and code blocks
        parts = re.split(r'```(\w*)\n(.*?)```', content, flags=re.DOTALL)
        
        panel_content = []
        
        for i, part in enumerate(parts):
            if i % 3 == 0:  # Regular text
                if part.strip():
                    panel_content.append(part.strip())
            elif i % 3 == 1:  # Language identifier
                continue
            else:  # Code content
                lang = parts[i-1] if parts[i-1] else "text"
                try:
                    syntax = Syntax(part.strip(), lang, theme="monokai", line_numbers=True)
                    panel_content.append(syntax)
                except:
                    # Fallback to plain text if syntax highlighting fails
                    panel_content.append(f"```{lang}\n{part.strip()}\n```")
        
        if panel_content:
            # Create response panel with mixed content
            response_panel = Panel(
                "\n\n".join(str(part) for part in panel_content),
                title="[ai]◆ AI Assistant[/ai]",
                border_style="accent",
                padding=(1, 2)
            )
            self.console.print(response_panel)
        else:
            # Fallback to simple display
            response_panel = Panel(
                content,
                title="[ai]◆ AI Assistant[/ai]",
                border_style="accent",
                padding=(1, 2)
            )
            self.console.print(response_panel)
    
    def _show_help(self):
        """Show elegant help information."""
        help_content = """[bold]Available Commands[/bold]

[user]help[/user]     Show this help message
[user]status[/user]   Display system status  
[user]clear[/user]    Clear the screen
[user]history[/user]  Show conversation history
[user]exit[/user]     End the session

[bold]Features[/bold]

• Talk with your computer in your language
• Execute shell commands and Python code
• Integrated on-demand file analysis and manipulation
• Reasoning through complex problems with any LLM
• Persistent conversation history(configurable)

[bold]Tips[/bold]

• Ask questions in your language
• Request code examples or explanations  
• Ask for system information or file operations
• Use Ctrl+C to interrupt long operations
        """
        
        help_panel = Panel(
            help_content,
            title="[system]Help & Commands[/system]",
            border_style="system",
            padding=(1, 2)
        )
        
        self.console.print(help_panel)
        self.console.print()
    
    def _show_status(self):
        """Show elegant system status."""
        api_key = os.getenv('OPENAI_API_KEY', '')
        
        table = Table(show_header=False, box=None)
        table.add_column("Component", style="secondary", width=20)
        table.add_column("Status", style="text")
        
        table.add_row("AI Model", f"[accent]{self.model_id}[/accent]")
        table.add_row("API Key", f"[success]Configured[/success] (***{api_key[-4:]})" if api_key else "[error]Missing[/error]")
        table.add_row("Database", f"[accent]{self.db_file}[/accent]")
        table.add_row("Tools", f"[accent]{len(self.agent.tools) if hasattr(self.agent, 'tools') else 0} loaded[/accent]")
        table.add_row("History", f"[accent]{self.num_history_runs} runs[/accent]")
        
        status_panel = Panel(
            table,
            title="[system]System Status[/system]",
            border_style="system",
            padding=(1, 2)
        )
        
        self.console.print(status_panel)
        self.console.print()
    
    def _show_history(self):
        """Show conversation history elegantly."""
        try:
            history = self.get_session_history()
            if not history:
                self.console.print(
                    Panel.fit(
                        "[secondary]No conversation history available[/secondary]",
                        border_style="secondary"
                    )
                )
                return
            
            history_content = []
            for i, msg in enumerate(history[-10:], 1):  # Show last 10 messages
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:100] + "..." if len(msg.get('content', '')) > 100 else msg.get('content', '')
                
                if role == 'user':
                    history_content.append(f"[user]{i}. You:[/user] {content}")
                elif role == 'assistant':
                    history_content.append(f"[ai]{i}. AI:[/ai] {content}")
            
            history_panel = Panel(
                "\n".join(history_content),
                title="[system]Recent Conversation History[/system]",
                border_style="system",
                padding=(1, 2)
            )
            
            self.console.print(history_panel)
            self.console.print()
            
        except Exception as e:
            self.console.print(
                Panel.fit(
                    f"[error]Could not retrieve history: {str(e)}[/error]",
                    border_style="red"
                )
            )
    
    def query(self, message: str) -> str:
        """Send a single query to the agent and get response."""
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
            return self.agent.storage.get_all_sessions()
        return []
    
    def clear_history(self):
        """Clear the agent's session history."""
        if hasattr(self.agent, 'storage') and self.agent.storage:
            self.agent.storage.clear()
            self.console.print(
                Panel.fit(
                    "[success]✓ Session history cleared[/success]",
                    border_style="success"
                )
            )
        else:
            self.console.print(
                Panel.fit(
                    "[warning]No storage available to clear[/warning]",
                    border_style="warning"
                )
            )
    
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
            'history_runs': self.num_history_runs
        }


def create_elegant_nterm(**kwargs) -> ElegantReasoningAgent:
    """
    Factory function to create an elegant reasoning agent.
    
    Args:
        **kwargs: Arguments passed to ElegantReasoningAgent constructor
        
    Returns:
        Configured ElegantReasoningAgent instance
    """
    return ElegantReasoningAgent(**kwargs)