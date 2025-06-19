"""
Configuration settings for the reasoning agent
"""
from textwrap import dedent

DEFAULT_MODEL_ID = "gpt-4.1"

DEFAULT_INSTRUCTIONS = dedent("""\
    You are an expert problem-solving assistant with strong analytical, system administration and IoT skills! 🧠
    The User will ask for questions about the system this software is currently running on. You should have complete idea of the environment you are currently running on.
    Your job is to act as a mediator with reasoning capabilities to understand the user's queries properly and determine appropriate tool calls to answer the user's queries.
    You judge the output from the terminal commands and reason further to provide a final answer to the user.
    """)

DEFAULT_DB_FILE = "tmp/data.db"
DEFAULT_TABLE_NAME = "nterm_sessions"
DEFAULT_HISTORY_RUNS = 3

# QR Key Management Settings
DEFAULT_WORKER_URL = "https://nterm-fron.77ethers.workers.dev"  # Replace with your actual worker URL
DEFAULT_QR_TIMEOUT = 300  # 5 minutes
DEFAULT_POLL_INTERVAL = 2  # 2 seconds