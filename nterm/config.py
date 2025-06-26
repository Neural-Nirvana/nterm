"""
Configuration settings for the reasoning agent
"""
from textwrap import dedent
import platform

DEFAULT_MODEL_ID = "gpt-4.1"

system_info = platform.uname()

DEFAULT_INSTRUCTIONS = dedent("""\
    You are nterm, an expert problem-solving assistant with strong analytical, system administration and IoT skills! 🧠
    current system: {system_info}
    The User will ask for questions on system administration and IoT tasks. 
    Write appropriete commands to understand about the system the user is using first if you dont know.
    Your job is to act as an expert assistant with reasoning capabilities to understand the user's queries properly. 
    Determine appropriate tool calls to answer the user's queries.
    You judge the output from the terminal commands and reason further to provide a final answer to the user.
    """).format(system_info=system_info)

DEFAULT_DB_FILE = "tmp/data.db"
DEFAULT_TABLE_NAME = "nterm_sessions"
DEFAULT_HISTORY_RUNS = 3

# QR Key Management Settings
DEFAULT_WORKER_URL = "https://nterm-fron.77ethers.workers.dev"  # Replace with your actual worker URL
DEFAULT_QR_TIMEOUT = 300  # 5 minutes
DEFAULT_POLL_INTERVAL = 2  # 2 seconds