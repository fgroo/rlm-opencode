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
        """Get API key from config, environment, or opencode MCP config.
        
        Checks:
        1. RLM config file (~/.config/rlm-session/config.json)
        2. opencode.json MCP configs (zai-mcp-server has Z_AI_API_KEY)
        3. config["options"]["apiKey"]
        4. config["options"]["headers"]["Authorization"] (Bearer token)
        5. Environment variable based on provider
        """
        import os
        from pathlib import Path
        import json
        
        opencode_config_path = Path.home() / ".config" / "opencode" / "opencode.json"
        
        # 1. Check RLM config file
        rlm_config_path = Path.home() / ".config" / "rlm-session" / "config.json"
        if rlm_config_path.exists():
            try:
                with open(rlm_config_path) as f:
                    rlm_config = json.load(f)
                api_keys = rlm_config.get("api_keys", {})
                if self.provider_id in api_keys:
                    return api_keys[self.provider_id]
            except:
                pass
        
        # 2. Check opencode.json MCP configs for API keys
        if opencode_config_path.exists():
            try:
                with open(opencode_config_path) as f:
                    oc_config = json.load(f)
                
                mcp = oc_config.get("mcp", {})
                
                # Provider -> MCP server mappings
                mcp_mappings = {
                    "z_ai_xhigh_coding_plan": "zai-mcp-server",
                    "zai-coding-plan": "zai-mcp-server",
                    "zhipu": "zai-mcp-server",
                }
                
                mcp_server = mcp_mappings.get(self.provider_id.lower())
                if mcp_server and mcp_server in mcp:
                    server_config = mcp[mcp_server]
                    env = server_config.get("environment", {})
                    # Check for Z_AI_API_KEY or similar
                    for key in ["Z_AI_API_KEY", "API_KEY", f"{self.provider_id.upper()}_API_KEY"]:
                        if key in env:
                            return env[key]
            except:
                pass
        
        options = self.config.get("options", {})
        
        # 3. Direct apiKey in config
        if "apiKey" in options:
            return options["apiKey"]
        
        # 4. Bearer token in headers
        headers = options.get("headers", {})
        auth = headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
        
        # 5. Environment variables
        provider_env = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "zhipu": "ZHIPU_API_KEY",
            "z_ai_xhigh_coding_plan": "Z_AI_API_KEY",
            "zai-coding-plan": "Z_AI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        
        env_patterns = [
            f"{self.provider_id.upper()}_API_KEY",
            f"{self.provider_id.upper().replace('-', '_')}_API_KEY",
            f"{self.provider_id.upper().replace('.', '_')}_API_KEY",
        ]
        
        if self.provider_id.lower() in provider_env:
            env_patterns.insert(0, provider_env[self.provider_id.lower()])
        
        for pattern in env_patterns:
            key = os.environ.get(pattern)
            if key:
                return key
        
        return None
