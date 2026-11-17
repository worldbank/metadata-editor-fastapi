"""Server-side assistant for the Metadata Editor (chat over LiteLLM). See README-assistant.md."""

from src.assistant.integration import register_assistant

__all__ = ["register_assistant"]
