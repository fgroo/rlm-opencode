#!/usr/bin/env python3
"""RLM-OpenCode Setup Script for OpenCode integration.

This script adds RLM-OpenCode-wrapped versions of all your opencode models.
RLM-OpenCode provides 40M+ character context for agentic workflows.

Usage:
    rlm-opencode-setup install      # Install RLM-OpenCode provider
    rlm-opencode-setup uninstall    # Remove RLM-OpenCode provider
    rlm-opencode-setup status       # Check current status
    rlm-opencode-setup serve        # Start the server

The RLM-OpenCode server must be running for models to work.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "opencode.json"
RLM_CONTEXT_LIMIT = 40000000

# Server settings
SERVER_URL = "http://localhost:8769/v1"
SERVER_PORT = 8769
PROVIDER_ID = "rlm-opencode"


def get_opencode_models():
    """Get list of all available models from opencode CLI (excludes rlm models)."""
    result = subprocess.run(
        ["opencode", "models"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"Error getting models: {result.stderr}")
        return []
    
    models = [
        line.strip() for line in result.stdout.strip().split("\n")
        if line.strip() 
        and not line.strip().startswith("rlm/")
        and not line.strip().startswith("rlm-opencode/")
        and not line.strip().startswith("rlm-native/")
    ]
    return models


def load_config():
    """Load opencode config."""
    if not OPENCODE_CONFIG.exists():
        return {}
    with open(OPENCODE_CONFIG) as f:
        return json.load(f)


def save_config(config):
    """Save opencode config."""
    OPENCODE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(OPENCODE_CONFIG, "w") as f:
        json.dump(config, f, indent=2)


def is_server_running():
    """Check if RLM-OpenCode server is running."""
    try:
        response = httpx.get(f"http://localhost:{SERVER_PORT}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def build_model_info_from_config(config: dict) -> dict:
    """Build a lookup of model info from opencode.json providers."""
    model_info = {}
    
    providers = config.get("provider", {})
    for prov_id, provider_data in providers.items():
        if prov_id in (PROVIDER_ID, "rlm", "rlm-native"):
            continue
        
        models = provider_data.get("models", {})
        for model_id, model_data in models.items():
            full_id = f"{prov_id}/{model_id}"
            model_info[full_id] = {
                "name": model_data.get("name", model_id),
                "limit": model_data.get("limit"),
                "variants": model_data.get("variants"),
            }
    
    return model_info


def create_rlm_model_entry(model_id: str, config_key: str, original_info: dict | None) -> dict:
    """Create an RLM model entry."""
    entry = {
        "name": f"RLM {config_key}",
    }
    
    output_limit = 131072
    if original_info and original_info.get("limit"):
        output_limit = original_info["limit"].get("output", 131072)
    
    entry["limit"] = {
        "context": RLM_CONTEXT_LIMIT,
        "output": output_limit,
    }
    
    if original_info and original_info.get("variants"):
        entry["variants"] = original_info["variants"]
    
    return entry


def install():
    """Install RLM-OpenCode provider with all models."""
    print("=" * 60)
    print("RLM-OpenCode Setup - Installing")
    print("=" * 60)
    print()
    
    if not is_server_running():
        print(f"RLM-OpenCode server is NOT RUNNING on port {SERVER_PORT}")
        print()
        print("Starting server...")
        os.system(f"nohup rlm-opencode serve > /tmp/rlm-opencode.log 2>&1 &")
        
        for i in range(10):
            time.sleep(1)
            if is_server_running():
                print(f"Server started on http://localhost:{SERVER_PORT}")
                break
        else:
            print("Failed to start server. Check /tmp/rlm-opencode.log")
            return 1
    else:
        print(f"Server is running on port {SERVER_PORT}")
    
    print()
    print("Fetching models from opencode...")
    all_models = get_opencode_models()
    if not all_models:
        print("Error: No models found from opencode")
        return 1
    
    print(f"Found {len(all_models)} models from opencode CLI")
    
    config = load_config()
    model_info_from_config = build_model_info_from_config(config)
    print(f"Found {len(model_info_from_config)} models with detailed info")
    print()
    
    # Ensure provider section exists
    if "provider" not in config:
        config["provider"] = {}
    
    # Clean up any legacy providers
    for legacy_id in ("rlm", "rlm-native"):
        if legacy_id in config["provider"]:
            print(f"Removing legacy provider: {legacy_id}")
            del config["provider"][legacy_id]
    
    # Check for existing provider
    existing = config["provider"].get(PROVIDER_ID, {})
    if existing:
        existing_name = existing.get("name", "unknown")
        existing_count = len(existing.get("models", {}))
        print(f"Replacing existing provider: {existing_name} ({existing_count} models)")
        print()
    
    # Build fresh model list
    rlm_models = {}
    models_with_variants = 0
    
    for model_id in all_models:
        config_key = model_id.replace("/", ".")
        original_info = model_info_from_config.get(model_id)
        entry = create_rlm_model_entry(model_id, config_key, original_info)
        rlm_models[config_key] = entry
        
        if "variants" in entry:
            models_with_variants += 1
    
    # FULLY REPLACE the provider (idempotent)
    config["provider"][PROVIDER_ID] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "RLM-OpenCode (context management)",
        "options": {
            "baseURL": SERVER_URL
        },
        "models": rlm_models
    }
    
    save_config(config)
    
    print(f"Installed {len(rlm_models)} models ({models_with_variants} with variants)")
    print()
    print("=" * 60)
    print("RLM-OpenCode is now available!")
    print("=" * 60)
    print()
    print("Usage:")
    print(f"  opencode -m {PROVIDER_ID}/<provider>.<model>")
    print()
    print("To remove: rlm-opencode-setup uninstall")
    
    return 0


def uninstall():
    """Remove RLM-OpenCode provider from config."""
    print("=" * 60)
    print("RLM-OpenCode Setup - Uninstalling")
    print("=" * 60)
    print()
    
    config = load_config()
    
    if "provider" not in config or PROVIDER_ID not in config["provider"]:
        print(f"Provider '{PROVIDER_ID}' is not installed.")
        return 0
    
    existing = config["provider"].get(PROVIDER_ID, {})
    existing_name = existing.get("name", "unknown")
    models_count = len(existing.get("models", {}))
    
    del config["provider"][PROVIDER_ID]
    save_config(config)
    
    print(f"Removed '{existing_name}' ({models_count} models)")
    print()
    print("RLM models are no longer available in opencode.")
    print("Note: Server is still running (if it was).")
    
    return 0


def status():
    """Check RLM-OpenCode status."""
    print("=" * 60)
    print("RLM-OpenCode Status")
    print("=" * 60)
    print()
    
    print(f"Config: {OPENCODE_CONFIG}")
    print()
    
    # Check server
    running = is_server_running()
    status_str = "RUNNING" if running else "NOT RUNNING"
    print(f"Server (port {SERVER_PORT}): {status_str}")
    
    if running:
        try:
            response = httpx.get(f"http://localhost:{SERVER_PORT}/health", timeout=2)
            data = response.json()
            print(f"  Version: {data.get('version')}, Mode: {data.get('mode')}")
        except:
            pass
    
    print()
    
    config = load_config()
    if "provider" in config and PROVIDER_ID in config["provider"]:
        provider = config["provider"][PROVIDER_ID]
        models = provider.get("models", {})
        provider_name = provider.get("name", "unknown")
        models_with_variants = sum(1 for m in models.values() if "variants" in m)
        
        print(f"Provider '{PROVIDER_ID}': INSTALLED")
        print(f"  Name: {provider_name}")
        print(f"  Models: {len(models)} ({models_with_variants} with variants)")
        
        if models:
            print("  Sample models:")
            for i, (model_id, model_info) in enumerate(models.items()):
                if i >= 3:
                    print(f"    ... and {len(models) - 3} more")
                    break
                print(f"    {PROVIDER_ID}/{model_id}")
        print()
    else:
        print(f"Provider '{PROVIDER_ID}': NOT INSTALLED")
    
    print()
    print("Commands:")
    print("  rlm-opencode-setup install      - Install provider")
    print("  rlm-opencode-setup uninstall    - Remove provider")
    print("  rlm-opencode-setup status       - Check status")
    print("  rlm-opencode-setup serve        - Start server")
    
    return 0


def serve():
    """Start the RLM-OpenCode server."""
    if is_server_running():
        print(f"Server is already running on port {SERVER_PORT}!")
        try:
            response = httpx.get(f"http://localhost:{SERVER_PORT}/health", timeout=2)
            data = response.json()
            print(f"URL: http://localhost:{SERVER_PORT}")
            print(f"Version: {data.get('version')}, Mode: {data.get('mode')}")
        except:
            pass
        return 0
    
    print("Starting RLM-OpenCode server...")
    
    log_file = "/tmp/rlm-opencode.log"
    cmd = f"nohup rlm-opencode serve > {log_file} 2>&1 &"
    os.system(cmd)
    
    for i in range(10):
        time.sleep(1)
        if is_server_running():
            print(f"Server started on http://localhost:{SERVER_PORT}")
            return 0
    
    print(f"Failed to start server. Check {log_file}")
    return 1


def main():
    args = sys.argv[1:]
    
    if not args:
        print(__doc__)
        return 0
    
    command = args[0].lower()
    
    if command in ("install", "add", "enable"):
        return install()
    elif command in ("uninstall", "remove", "disable"):
        return uninstall()
    elif command in ("status", "check"):
        return status()
    elif command in ("serve", "start"):
        return serve()
    else:
        print(f"Unknown command: {command}")
        print("Use: install, uninstall, status, or serve")
        return 1


if __name__ == "__main__":
    sys.exit(main())
