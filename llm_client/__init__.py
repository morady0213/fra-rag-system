"""
LLM Client module for FRA RAG System.
Contains Grok (xAI) API integration.
"""

from .grok_client import GrokClient, generate_answer

__all__ = ["GrokClient", "generate_answer"]
