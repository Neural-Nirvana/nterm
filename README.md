# NTerm: Reasoning Agent Terminal Application

A powerful reasoning agent with system administration and IoT capabilities. This agent can understand queries about system environments and use shell tools combined with reasoning capabilities to provide comprehensive answers.

## Features

- 🧠 **Intelligent Reasoning**: Uses advanced AI models to understand and analyze system queries
- 🖥️ **System Administration**: Built-in shell tools for system interaction and analysis  
- 🔌 **IoT Capabilities**: Specialized tools for IoT device management and monitoring
- 💾 **Persistent Storage**: SQLite-based session storage with conversation history
- 🔄 **Interactive CLI**: Easy-to-use command-line interface
- 📚 **Programmable API**: Use as a library in your Python projects

## Installation

```bash
pip install nterm
```

## Quick Start

### Command Line Usage

Start an interactive session:
```bash
nterm
```

Run a single query:
```bash
nterm --query "What operating system am I running?"
```

Use a different model:
```bash
nterm --model gpt-4.1
```

### Python API Usage

```python
from nterm import ReasoningAgent

# Create an agent
agent = ReasoningAgent()

# Ask a question
response = agent.query("What's the current CPU usage?")
print(response)

# Start interactive mode
agent.run_cli()
```

### Advanced Usage

```python
from nterm import create_nterm

# Create with custom configuration
agent = create_nterm(
    model_id="gpt-4",
    db_file="./my_sessions.db",
    num_history_runs=5
)

# Add custom tools
from my_custom_tools import MyTool
agent.add_tool(MyTool())

# Get session history
history = agent.get_session_history()

# Clear history
agent.clear_history()
```

## Configuration

The agent can be configured with various options:

- **model_id**: OpenAI model to use (default: "gpt-4o")
- **db_file**: SQLite database file for session storage
- **table_name**: Database table name for sessions
- **num_history_runs**: Number of previous conversations to remember
- **custom_tools**: Additional tools to extend agent capabilities

## Command Line Options

```
usage: nterm [-h] [--model MODEL] [--db-file DB_FILE] 
                      [--table-name TABLE_NAME] [--history-runs HISTORY_RUNS]
                      [--query QUERY] [--clear-history] [--version]

Options:
  -h, --help            Show help message
  --model MODEL         OpenAI model ID to use (default: gpt-4o)
  --db-file DB_FILE     SQLite database file path
  --table-name TABLE_NAME Database table name for sessions
  --history-runs HISTORY_RUNS Number of history runs to keep
  --query QUERY         Single query to run (non-interactive mode)
  --clear-history       Clear session history before starting
  --version             Show version information
```

## Requirements

- Python 3.8+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)
- agno framework
- SQLite (for session storage)

## Examples

### System Information
```bash
nterm --query "Show me system information including CPU, memory, and disk usage"
```

### Process Management
```bash
nterm --query "What processes are consuming the most CPU?"
```

### Network Analysis
```bash
nterm --query "Check network connectivity and show active connections"
```

### IoT Device Management
```bash
nterm --query "Scan for IoT devices on the local network"
```

## Development

### Setting up for development

1. Clone the repository
2. Install in development mode: `pip install -e .`
3. Run tests: `python -m pytest tests/`

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues or have questions, please open an issue on the project repository.