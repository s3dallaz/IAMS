"""
AI integration layer for the OmniRoute-powered attendance assistant.

Modules:
- `llm_client`: OpenAI-compatible HTTP client for the OmniRoute provider.
- `intent_schema`: Allowlisted, validated structured-intent boundary.
- `prompts`: Planner and answer system prompts.

The assistant reasons with the LLM but reads data exclusively through
`src.services.ai_tools`, which wraps the existing Phase 4/5 services.
"""

from src.ai.intent_schema import ToolCall, validate_intent, parse_llm_json
from src.ai.llm_client import (
    OmniRouteClient,
    LLMError,
    LLMNotConfiguredError,
    LLMUnavailableError,
    LLMRequestFailedError,
    LLMResponseFormatError,
)

__all__ = [
    "ToolCall",
    "validate_intent",
    "parse_llm_json",
    "OmniRouteClient",
    "LLMError",
    "LLMNotConfiguredError",
    "LLMUnavailableError",
    "LLMRequestFailedError",
    "LLMResponseFormatError",
]