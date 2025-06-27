# NTerm: AI Terminal Reasoning Agent for System Administration and IoT Management

![NTerm Demo](nterm-demo.gif)

**NTerm** is an intelligent command-line interface (CLI) tool that combines artificial intelligence reasoning capabilities with system administration and Internet of Things (IoT) device management. This Python-based terminal application leverages advanced AI models to understand natural language queries about system environments and provides comprehensive answers using built-in shell tools and reasoning algorithms.

## What is NTerm? Key Features and Capabilities

### 🧠 **AI-Powered System Analysis and Reasoning**
- Advanced artificial intelligence integration using OpenAI GPT models
- Natural language processing for system queries and commands
- Intelligent analysis of system performance, processes, and configurations
- Context-aware responses based on your specific system environment

### 🖥️ **Comprehensive System Administration Tools**
- Built-in shell command execution and system interaction
- Real-time system monitoring and performance analysis
- Process management and resource utilization tracking
- Network connectivity analysis and troubleshooting
- File system operations and disk usage monitoring

### 🔌 **IoT Device Management and Monitoring**
- IoT device discovery and network scanning capabilities
- Device status monitoring and health checks
- IoT sensor data analysis and interpretation
- Smart home and industrial IoT integration support

### 💾 **Persistent Session Management**
- SQLite-based conversation history and session storage
- Contextual memory across multiple interactions
- Session replay and historical query analysis
- Customizable data retention policies

### 🔄 **Interactive Command-Line Interface**
- User-friendly terminal-based interaction
- Single-query execution mode for automation
- Batch processing capabilities
- Customizable output formatting

### 📚 **Python API and Library Integration**
- Programmable interface for custom applications
- Extensible tool architecture for custom functionality
- Integration with existing Python workflows and scripts

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (required for AI functionality)
- Operating System: Windows, macOS, or Linux

### Quick Installation
```bash
pip install nterm
```

### Verify Installation
```bash
nterm --version
```

## Getting Started: Usage Examples and Tutorials

### Basic Command-Line Usage

**Start Interactive Session:**
```bash
nterm
```

**Execute Single Query:**
```bash
nterm --query "What operating system am I running and what are the current system specifications?"
```

**Use Specific AI Model:**
```bash
nterm --model gpt-4.1 --query "Analyze current CPU usage and suggest optimization strategies"
```

### Python API Integration Examples

**Basic Usage:**
```python
from nterm import ReasoningAgent

# Initialize the AI reasoning agent
agent = ReasoningAgent()

# Query system information
response = agent.query("What's the current memory usage and which processes are using the most RAM?")
print(response)

# Start interactive command-line mode
agent.run_cli()
```

**Advanced Configuration:**
```python
from nterm import create_nterm
from my_custom_tools import CustomSystemTool

# Create agent with custom settings
agent = create_nterm(
    model_id="gpt-4.1",
    db_file="./system_sessions.db",
    num_history_runs=10
)

# Add custom tools and extensions
agent.add_tool(CustomSystemTool())

# Retrieve conversation history
session_history = agent.get_session_history()

# Clear stored history
agent.clear_history()
```

## Configuration Options and Customization

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key-here"
export NTERM_DB_FILE="./custom_sessions.db"
```

### Configuration Parameters
- **model_id**: OpenAI model selection (default: "gpt-4.1")
- **db_file**: SQLite database path for session persistence
- **table_name**: Custom database table name
- **num_history_runs**: Conversation history retention limit
- **custom_tools**: Additional tool integrations

## Command-Line Arguments and Options

```bash
nterm [OPTIONS] [--query "YOUR_QUERY"]

Options:
  -h, --help                    Display help information and usage examples
  --model MODEL                 Specify OpenAI model (gpt-4o, gpt-4.1, gpt-4o-mini)
  --db-file DB_FILE            Custom SQLite database file location
  --table-name TABLE_NAME      Database table name for session storage
  --history-runs HISTORY_RUNS  Number of previous conversations to remember
  --query QUERY                Execute single query in non-interactive mode
  --clear-history              Reset conversation history before starting
  --version                    Show version and build information
```

## Real-World Use Cases and Examples

### System Administration and Monitoring
```bash
# Comprehensive system health check
nterm --query "Perform a complete system health analysis including CPU, memory, disk space, and running services"

# Network troubleshooting
nterm --query "Diagnose network connectivity issues and show active network connections"

# Security analysis
nterm --query "Check for unusual processes and potential security concerns on this system"
```

### IoT Device Management
```bash
# Device discovery
nterm --query "Scan local network for IoT devices and smart home equipment"

# IoT monitoring
nterm --query "Monitor IoT sensor data and identify any devices with connectivity issues"
```

### Performance Optimization
```bash
# Resource analysis
nterm --query "Identify processes consuming excessive resources and recommend optimization strategies"

# Disk cleanup recommendations
nterm --query "Analyze disk usage patterns and suggest cleanup strategies"
```

## Technical Requirements and Dependencies

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+, CentOS 7+)
- **Python Version**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Memory**: Minimum 512MB RAM, Recommended 2GB+
- **Storage**: 100MB free disk space

### Required Dependencies
- OpenAI API access and valid API key
- agno framework for AI agent functionality
- SQLite3 for session data persistence
- Standard Python libraries (os, sys, subprocess, sqlite3)

### Optional Dependencies
- Custom tool integrations
- Additional AI model providers
- Extended IoT protocol support

## Development and Contribution Guide

### Development Setup
```bash
# Clone repository
git clone https://github.com/Neural-Nirvana/nterm.git
cd nterm

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run test suite
python -m pytest tests/ -v
```

### Contributing Guidelines
We welcome contributions! Please follow these guidelines:
- Fork the repository and create feature branches
- Write comprehensive tests for new functionality
- Follow PEP 8 Python coding standards
- Update documentation for new features
- Submit pull requests with detailed descriptions

### Testing and Quality Assurance
```bash
# Run all tests
python -m pytest

# Run with coverage report
python -m pytest --cov=nterm

# Code quality checks
flake8 nterm/
black nterm/
```

## Troubleshooting and Support

### Common Issues and Solutions

**OpenAI API Key Issues:**
- Ensure `OPENAI_API_KEY` environment variable is set
- Verify API key validity and account credits
- Check network connectivity for API access

**Installation Problems:**
- Update pip: `pip install --upgrade pip`
- Use virtual environment to avoid conflicts
- Check Python version compatibility

**Performance Issues:**
- Adjust `num_history_runs` to reduce memory usage
- Use lighter AI models for faster responses
- Clear session history periodically

### Getting Help
- **Documentation**: Comprehensive guides and API reference
- **GitHub Issues**: Bug reports and feature requests
- **Community Support**: Discussion forums and user community
- **Professional Support**: Enterprise support options available

## License and Legal Information

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete terms and conditions.

## Version History and Changelog

- **v1.0.0**: Initial release with core AI reasoning capabilities
- **v1.1.0**: Added IoT device management features
- **v1.2.0**: Enhanced system administration tools
- **Latest**: Improved performance and stability

## Related Tools and Integrations

- **System Monitoring**: Integration with popular monitoring tools
- **DevOps Workflows**: CI/CD pipeline integration
- **IoT Platforms**: Compatible with major IoT management systems

#nterm #AIterminal #DevOpsTool #OpenSourceCLI #TerminalWithAI #ShellAutomation #AIforSysadmins #MultiAgentAI #RemoteTerminal #ContextAwareCLI #PlanningMode #SelfHostedAI #DevTool #ProductivityTool

- **AI Frameworks**: Extensible AI model support

---

*NTerm - Intelligent Terminal Assistant for System Administration and IoT Management. Powered by AI, designed for professionals.*
