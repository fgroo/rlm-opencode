"""Base provider interface for RLM-Native."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Any


@dataclass
class StreamChunk:
    """A chunk of streaming response."""
    content: str
    finish_reason: str | None = None
    tool_calls: list[dict] | None = None
    reasoning: str = ""  # For thinking/reasoning models


@dataclass  
class ModelInfo:
    """Information about a model."""
    id: str
    provider_id: str
    name: str
    context_limit: int = 128000
    output_limit: int = 4096
    supports_streaming: bool = True
    supports_tools: bool = True
    variants: dict[str, dict] | None = None


class BaseProvider(ABC):
    """Abstract base class for LLM providers.
    
    All providers must implement:
    - stream(): Async generator for streaming completions
    - complete(): Non-streaming completion (optional, can use stream internally)
    - get_model_info(): Return model metadata
    """
    
    def __init__(self, provider_id: str, config: dict):
        """Initialize provider with config from opencode.json.
        
        Args:
            provider_id: Provider identifier (e.g., "openai", "anthropic")
            config: Provider config from opencode.json
                - npm: NPM package name
                - options: Provider options (baseURL, etc.)
                - models: Dict of model configs
        """
        self.provider_id = provider_id
        self.config = config
        self.base_url = config.get("options", {}).get("baseURL")
        self.models: dict[str, ModelInfo] = {}
        self._load_models()
    
    def _load_models(self):
        """Load model info from config."""
        models_config = self.config.get("models", {})
        for model_id, model_config in models_config.items():
            limits = model_config.get("limit", {})
            self.models[model_id] = ModelInfo(
                id=model_id,
                provider_id=self.provider_id,
                name=model_config.get("name", model_id),
                context_limit=limits.get("context", 128000),
                output_limit=limits.get("output", 4096),
                variants=model_config.get("variants"),
            )
    
    def has_model(self, model_id: str) -> bool:
        """Check if this provider has the model."""
        return model_id in self.models
    
    def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get model info."""
        return self.models.get(model_id)
    
    def list_models(self) -> list[ModelInfo]:
        """List all models for this provider."""
        return list(self.models.values())
    
    @abstractmethod
    async def stream(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a chat completion.
        
        Args:
            model_id: Model identifier (without provider prefix)
            messages: OpenAI-format messages
            temperature: Sampling temperature
            max_tokens: Max output tokens
            tools: Available tools (OpenAI format)
            **kwargs: Additional provider-specific options
        
        Yields:
            StreamChunk objects
        """
        pass
    
    async def complete(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> str:
        """Non-streaming completion (collects stream)."""
        result = []
        async for chunk in self.stream(
            model_id, messages, temperature, max_tokens, tools, **kwargs
        ):
            result.append(chunk.content)
        return "".join(result)
    
    def _get_api_key(self) -> str | None:
        """Get API key dynamically from config, environment, or opencode.json.
        
        Searches:
        1. Direct config options (apiKey / Authorization header)
        2. All provider configs in opencode.json (match by provider_id or baseURL)
        3. MCP server environment variables in opencode.json
        4. Environment variables
        """
        import os
        from pathlib import Path
        import json
        
        opencode_config_path = Path.home() / ".config" / "opencode" / "opencode.json"
        
        # 1. Direct config options
        options = self.config.get("options", {})
        if "apiKey" in options:
            return options["apiKey"]
        
        headers = options.get("headers", {})
        auth = headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
        
        # 2. Search opencode.json providers for API key
        if opencode_config_path.exists():
            try:
                with open(opencode_config_path) as f:
                    oc_config = json.load(f)
                
                providers = oc_config.get("provider", {})
                
                # Direct match by provider_id
                if self.provider_id in providers:
                    prov_opts = providers[self.provider_id].get("options", {})
                    if "apiKey" in prov_opts:
                        return prov_opts["apiKey"]
                    prov_auth = prov_opts.get("headers", {}).get("Authorization", "")
                    if prov_auth.startswith("Bearer "):
                        return prov_auth[7:]
                
                # If we have a baseURL, find any provider with the same baseURL
                if self.base_url:
                    for prov_id, prov_config in providers.items():
                        prov_url = prov_config.get("options", {}).get("baseURL", "")
                        if prov_url and prov_url == self.base_url:
                            prov_opts = prov_config.get("options", {})
                            if "apiKey" in prov_opts:
                                return prov_opts["apiKey"]
                            prov_auth = prov_opts.get("headers", {}).get("Authorization", "")
                            if prov_auth.startswith("Bearer "):
                                return prov_auth[7:]
                
                # Search all MCP server environment variables
                mcp = oc_config.get("mcp", {})
                for server_name, server_config in mcp.items():
                    env = server_config.get("environment", {})
                    for env_key, env_val in env.items():
                        if env_key.endswith("_API_KEY") and env_val:
                            # Check if this MCP key matches our provider
                            provider_upper = self.provider_id.upper().replace("-", "_")
                            if provider_upper in env_key or env_key == "API_KEY":
                                return env_val
                
            except Exception:
                pass
        
        # 3. Environment variables
        env_patterns = [
            f"{self.provider_id.upper().replace('-', '_')}_API_KEY",
            f"{self.provider_id.upper().replace('.', '_')}_API_KEY",
            f"{self.provider_id.upper()}_API_KEY",
        ]
        
        for pattern in env_patterns:
            key = os.environ.get(pattern)
            if key:
                return key
        
        return None
