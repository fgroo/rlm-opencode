"""Provider registry for RLM-Native.

Discovers and manages providers from opencode.json configuration.
Uses opencode CLI for model discovery.
"""

import json
import subprocess
from pathlib import Path
from typing import Iterator

from rich.console import Console

from rlm_opencode.providers.base import BaseProvider, ModelInfo
from rlm_opencode.providers.openai_compatible import OpenAICompatibleProvider

console = Console()

OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "opencode.json"


class ProviderRegistry:
    """Registry of all available providers.
    
    Loads providers from:
    1. opencode.json configuration
    2. Model discovery via opencode CLI
    """
    
    def __init__(self):
        self.providers: dict[str, BaseProvider] = {}
        self.model_to_provider: dict[str, str] = {}  # model_id -> provider_id
        self._load_from_config()
        self._discover_models()
    
    def _load_from_config(self):
        """Load providers from opencode.json."""
        if not OPENCODE_CONFIG.exists():
            console.print("[yellow]Warning: opencode.json not found[/yellow]")
            return
        
        with open(OPENCODE_CONFIG) as f:
            config = json.load(f)
        
        providers_config = config.get("provider", {})
        
        for provider_id, provider_config in providers_config.items():
            # Skip RLM providers (avoid recursion)
            if provider_id in ("rlm", "rlm-session", "rlm-native"):
                continue
            
            # Create appropriate provider type
            provider = self._create_provider(provider_id, provider_config)
            if provider:
                self.providers[provider_id] = provider
        
        console.print(f"[green]Loaded {len(self.providers)} providers from config[/green]")
    
    def _discover_models(self):
        """Discover models using opencode CLI."""
        try:
            result = subprocess.run(
                ["opencode", "models"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode != 0:
                console.print(f"[yellow]Warning: opencode models failed[/yellow]")
                return
            
            models = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            
            for model_id in models:
                # Skip RLM models (avoid recursion)
                if model_id.startswith("rlm/") or model_id.startswith("rlm-session/"):
                    continue
                
                # Parse provider/model format
                if "/" in model_id:
                    provider_id, model_name = model_id.split("/", 1)
                    
                    # Create provider if not exists
                    if provider_id not in self.providers:
                        self._create_provider_from_model(provider_id, model_name)
                    
                    self.model_to_provider[model_id] = provider_id
                    self.model_to_provider[model_name] = provider_id
            
            console.print(f"[green]Discovered {len(self.model_to_provider)} model mappings[/green]")
            
        except Exception as e:
            console.print(f"[yellow]Warning: Model discovery failed: {e}[/yellow]")
    
    def _create_provider_from_model(self, provider_id: str, model_name: str):
        """Create a provider entry based on discovered model."""
        # Try to get config from opencode.json
        config = self._get_provider_config(provider_id)
        
        if not config:
            # Create minimal config for known providers
            config = self._get_default_config(provider_id)
        
        if config:
            provider = OpenAICompatibleProvider(provider_id, config)
            self.providers[provider_id] = provider
    
    def _get_provider_config(self, provider_id: str) -> dict | None:
        """Get provider config from opencode.json."""
        if not OPENCODE_CONFIG.exists():
            return None
        
        with open(OPENCODE_CONFIG) as f:
            config = json.load(f)
        
        return config.get("provider", {}).get(provider_id)
    
    def _get_default_config(self, provider_id: str) -> dict | None:
        """Get default config for known providers."""
        # Known provider base URLs
        known_providers = {
            "openai": {
                "options": {"baseURL": "https://api.openai.com/v1"},
                "models": {},
            },
            "anthropic": {
                "options": {"baseURL": "https://api.anthropic.com/v1"},
                "models": {},
            },
            "openrouter": {
                "options": {"baseURL": "https://openrouter.ai/api/v1"},
                "models": {},
            },
            "zhipu": {
                "options": {"baseURL": "https://open.bigmodel.cn/api/paas/v4"},
                "models": {},
            },
        }
        
        return known_providers.get(provider_id.lower())
    
    def _create_provider(self, provider_id: str, config: dict) -> BaseProvider | None:
        """Create a provider instance based on config."""
        
        # Check for baseURL (required for our providers)
        base_url = config.get("options", {}).get("baseURL")
        if not base_url:
            # Some providers might be npm-only without custom baseURL
            npm = config.get("npm", "")
            if "openai-compatible" in npm.lower():
                # Default OpenAI base URL for openai-compatible npm package
                base_url = "https://api.openai.com/v1"
            else:
                return None
        
        # Most providers use OpenAI-compatible API
        return OpenAICompatibleProvider(provider_id, config)
    
    def get_provider(self, provider_id: str) -> BaseProvider | None:
        """Get a provider by ID."""
        return self.providers.get(provider_id)
    
    def get_provider_for_model(self, model_id: str) -> BaseProvider | None:
        """Find the provider that has a given model."""
        # Check direct mapping first
        provider_id = self.model_to_provider.get(model_id)
        if provider_id:
            return self.providers.get(provider_id)
        
        # Check provider/model format
        if "/" in model_id:
            provider_id, _ = model_id.split("/", 1)
            return self.providers.get(provider_id)
        
        return None
    
    def resolve_model(self, model_id: str) -> tuple[BaseProvider, str] | None:
        """Resolve a model ID to (provider, local_model_id).
        
        Args:
            model_id: Can be:
                - "provider/model" -> (provider, "model")
                - "provider.model" -> (provider, "model") 
                - Just "model" -> search all providers
        
        Returns:
            (provider, local_model_id) or None
        """
        # Handle provider/model format (most reliable)
        if "/" in model_id:
            provider_id, model_name = model_id.split("/", 1)
            provider = self.providers.get(provider_id)
            if provider:
                return provider, model_name
        
        # Handle provider.model format
        # Try each possible split point, preferring longer provider names
        if "." in model_id:
            # Try splitting from right to left (longer provider names first)
            parts = model_id.split(".")
            for i in range(len(parts) - 1, 0, -1):
                provider_id = ".".join(parts[:i])
                model_name = ".".join(parts[i:])
                provider = self.providers.get(provider_id)
                if provider:
                    return provider, model_name
                # Also try with underscores converted to dots? No, try exact match
                provider = self.providers.get(provider_id.replace(".", "_"))
                if provider:
                    return provider, model_name
            
            # Try treating first segment as provider (old behavior as fallback)
            first_dot = model_id.index(".")
            provider_id = model_id[:first_dot]
            model_name = model_id[first_dot + 1:]
            provider = self.providers.get(provider_id)
            if provider:
                return provider, model_name
        
        # Search in mappings
        provider_id = self.model_to_provider.get(model_id)
        if provider_id:
            provider = self.providers.get(provider_id)
            if provider:
                return provider, model_id
        
        return None
    
    def list_providers(self) -> list[str]:
        """List all provider IDs."""
        return list(self.providers.keys())
    
    def list_all_models(self) -> Iterator[ModelInfo]:
        """Iterate over all models from all providers."""
        for provider in self.providers.values():
            yield from provider.list_models()
    
    def count_models(self) -> int:
        """Count total model mappings."""
        return len(self.model_to_provider)


# Global registry instance
_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def reload_registry() -> ProviderRegistry:
    """Reload the provider registry from config."""
    global _registry
    _registry = ProviderRegistry()
    return _registry
