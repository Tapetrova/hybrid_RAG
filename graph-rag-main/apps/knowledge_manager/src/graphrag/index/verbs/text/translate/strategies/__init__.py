"""The Indexing Engine translate strategies package root."""

from .mock import run as run_mock
from .openai import run as run_openai

__all__ = ["run_mock", "run_openai"]
