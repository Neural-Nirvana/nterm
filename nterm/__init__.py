"""
Alpa - A reasoning agent with system administration and IoT capabilities

This package provides a reasoning agent with capabilities for system administration
and IoT tasks. It uses OpenAI models with shell tools and reasoning capabilities.
"""

__version__ = "0.3.0"
__author__ = "Neural Nirvana"
__email__ = "ekansh@duck.com"

from .agent import ReasoningAgent, create_nterm
from .config import (
    DEFAULT_MODEL_ID,
    DEFAULT_INSTRUCTIONS,
    DEFAULT_DB_FILE,
    DEFAULT_TABLE_NAME,
    DEFAULT_HISTORY_RUNS
)

__all__ = [
    "ReasoningAgent",
    "create_nterm",
    "DEFAULT_MODEL_ID",
    "DEFAULT_INSTRUCTIONS",
    "DEFAULT_DB_FILE",
    "DEFAULT_TABLE_NAME",
    "DEFAULT_HISTORY_RUNS",
]