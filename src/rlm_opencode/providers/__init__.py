"""Provider abstraction for RLM-Native.

Supports multiple LLM API providers:
- OpenAI-compatible (covers most providers)
- Anthropic
- Custom providers from opencode.json
"""

from rlm_opencode.providers.base import BaseProvider, StreamChunk
from rlm_opencode.providers.openai_compatible import OpenAICompatibleProvider
from rlm_opencode.providers.registry import ProviderRegistry, get_registry

__all__ = [
    "BaseProvider",
    "StreamChunk", 
    "OpenAICompatibleProvider",
    "ProviderRegistry",
    "get_registry",
]
