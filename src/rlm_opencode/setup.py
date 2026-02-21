#!/usr/bin/env python3
"""RLM Session Setup Script for OpenCode integration.

This script adds RLM-Session-wrapped versions of all your opencode models.
RLM-Session provides 40M+ character context for agentic workflows.

Usage:
    rlm-opencode-setup install           # Default mode (port 8769)
    rlm-opencode-setup install --native  # Native mode (port 8769)
    rlm-opencode-setup uninstall         # Remove proxy models
    rlm-opencode-setup uninstall --native # Remove native models
    rlm-opencode-setup status            # Check current status
    rlm-opencode-setup serve             # Start proxy server
    rlm-opencode-setup serve --native    # Start native server

Modes:
    Proxy mode (default): Wraps opencode run for context management
    Native mode (--native): Calls model APIs directly, no proxy

The RLM-Session server must be running for models to work.
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

# Mode-specific settings
PROXY_URL = "http://localhost:8769/v1"
PROXY_PROVIDER_ID = "rlm-opencode"
NATIVE_URL = "http://localhost:8769/v1"
NATIVE_PROVIDER_ID = "rlm-native"


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


def is_server_running(native: bool = False):
    """Check if RLM-Session server is running."""
    port = 8769
    try:
        response = httpx.get(f"http://localhost:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def is_rlm_server_running():
    """Check if rlm-server (port 8765) is running."""
    try:
        response = httpx.get("http://localhost:8765/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_mode_settings(native: bool = False) -> dict:
    """Get mode-specific settings."""
    if native:
        return {
            "provider_id": NATIVE_PROVIDER_ID,
            "url": NATIVE_URL,
            "port": 8769,
            "name": "RLM-Native (direct API, context management)",
            "description": "Native mode calls model APIs directly with context injection.",
        }
    return {
        "provider_id": PROXY_PROVIDER_ID,
        "url": PROXY_URL,
        "port": 8769,
        "name": "RLM-Session (proxy, context management)",
        "description": "Proxy mode wraps opencode run for context accumulation.",
    }


def build_model_info_from_config(config: dict, provider_id: str) -> dict:
    """Build a lookup of model info from opencode.json providers."""
    model_info = {}
    
    providers = config.get("provider", {})
    for prov_id, provider_data in providers.items():
        if prov_id in (provider_id, "rlm", "rlm-opencode", "rlm-native"):
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


def install(native: bool = False):
    """Install RLM-Session provider with all models."""
    mode = get_mode_settings(native)
    
    print("=" * 60)
    print(f"RLM-Session Setup - Installing ({'Native' if native else 'Proxy'} Mode)")
    print("=" * 60)
    print()
    
    if not is_server_running(native):
        print(f"RLM-Session server is NOT RUNNING on port {mode['port']}")
        print()
        print("Starting server...")
        if native:
            os.system("nohup rlm-opencode serve --native > /tmp/rlm-native.log 2>&1 &")
        else:
            os.system("nohup rlm-opencode serve > /tmp/rlm-opencode.log 2>&1 &")
        
        for i in range(10):
            time.sleep(1)
            if is_server_running(native):
                print(f"Server started on http://localhost:{mode['port']}")
                break
        else:
            print(f"Failed to start server. Check /tmp/rlm-{'native' if native else 'session'}.log")
            return 1
    else:
        print(f"Server is running on port {mode['port']}")
    
    print()
    print("Fetching models from opencode...")
    all_models = get_opencode_models()
    if not all_models:
        print("Error: No models found from opencode")
        return 1
    
    print(f"Found {len(all_models)} models from opencode CLI")
    
    config = load_config()
    model_info_from_config = build_model_info_from_config(config, mode["provider_id"])
    print(f"Found {len(model_info_from_config)} models with detailed info")
    print()
    
    # Ensure provider section exists
    if "provider" not in config:
        config["provider"] = {}
    
    # Check for existing provider
    existing = config["provider"].get(mode["provider_id"], {})
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
    config["provider"][mode["provider_id"]] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": mode["name"],
        "options": {
            "baseURL": mode["url"]
        },
        "models": rlm_models
    }
    
    save_config(config)
    
    print(f"Installed {len(rlm_models)} models ({models_with_variants} with variants)")
    print()
    print("=" * 60)
    print(f"RLM-{'Native' if native else 'Session'} is now available!")
    print("=" * 60)
    print()
    print("Usage:")
    print(f"  opencode run -m {mode['provider_id']}/<provider>.<model>")
    print()
    print("Examples:")
    print(f"  opencode run -m {mode['provider_id']}/rlm-internal.rlm-core-v1")
    print(f"  opencode run -m {mode['provider_id']}/openai.gpt-4o")
    print()
    print(mode["description"])
    print()
    print(f"To remove: rlm-opencode-setup uninstall{' --native' if native else ''}")
    
    return 0


def uninstall(native: bool = False):
    """Remove RLM provider from config."""
    mode = get_mode_settings(native)
    
    print("=" * 60)
    print(f"RLM-Session Setup - Uninstalling ({'Native' if native else 'Proxy'} Mode)")
    print("=" * 60)
    print()
    
    config = load_config()
    
    if "provider" not in config or mode["provider_id"] not in config["provider"]:
        print(f"Provider '{mode['provider_id']}' is not installed.")
        return 0
    
    existing = config["provider"].get(mode["provider_id"], {})
    existing_name = existing.get("name", "unknown")
    models_count = len(existing.get("models", {}))
    
    del config["provider"][mode["provider_id"]]
    save_config(config)
    
    print(f"Removed '{existing_name}' ({models_count} models)")
    print()
    print("RLM models are no longer available in opencode.")
    print("Note: Server is still running (if it was).")
    
    return 0


def status():
    """Check RLM-Session status."""
    print("=" * 60)
    print("RLM-Session Setup - Status")
    print("=" * 60)
    print()
    
    print(f"Config: {OPENCODE_CONFIG}")
    print()
    
    # Check servers
    print("Servers:")
    for name, port, native in [
        ("RLM-Session (Proxy)", 8767, False),
        ("RLM-Session (Native)", 8768, True),
        ("RLM-Server", 8765, None),
    ]:
        if native is None:
            running = is_rlm_server_running()
        else:
            running = is_server_running(native)
        
        status_str = "RUNNING" if running else "NOT RUNNING"
        print(f"  {name} (port {port}): {status_str}")
        
        if running:
            try:
                response = httpx.get(f"http://localhost:{port}/health", timeout=2)
                data = response.json()
                mode = data.get("mode", "unknown")
                print(f"    Version: {data.get('version')}, Mode: {mode}")
            except:
                pass
    
    print()
    
    config = load_config()
    for provider_id, display_name in [(PROXY_PROVIDER_ID, "Proxy"), (NATIVE_PROVIDER_ID, "Native")]:
        if "provider" in config and provider_id in config["provider"]:
            provider = config["provider"][provider_id]
            models = provider.get("models", {})
            provider_name = provider.get("name", "unknown")
            models_with_variants = sum(1 for m in models.values() if "variants" in m)
            
            print(f"Provider '{provider_id}' ({display_name}): INSTALLED")
            print(f"  Name: {provider_name}")
            print(f"  Models: {len(models)} ({models_with_variants} with variants)")
            
            if models:
                print("  Sample models:")
                for i, (model_id, model_info) in enumerate(models.items()):
                    if i >= 3:
                        print(f"    ... and {len(models) - 3} more")
                        break
                    variants_str = ""
                    if "variants" in model_info:
                        variants = list(model_info["variants"].keys())
                        variants_str = f" [{', '.join(variants[:3])}...]"
                    print(f"    {provider_id}/{model_id}{variants_str}")
            print()
        else:
            print(f"Provider '{provider_id}' ({display_name}): NOT INSTALLED")
    
    print()
    print("Commands:")
    print("  rlm-opencode-setup install           - Install proxy mode")
    print("  rlm-opencode-setup install --native  - Install native mode")
    print("  rlm-opencode-setup uninstall         - Remove proxy")
    print("  rlm-opencode-setup uninstall --native - Remove native")
    print("  rlm-opencode-setup serve             - Start proxy server")
    print("  rlm-opencode-setup serve --native    - Start native server")
    
    return 0


def serve(native: bool = False):
    """Start the RLM-Session server."""
    mode = get_mode_settings(native)
    
    if is_server_running(native):
        print(f"Server is already running on port {mode['port']}!")
        try:
            response = httpx.get(f"http://localhost:{mode['port']}/health", timeout=2)
            data = response.json()
            print(f"URL: http://localhost:{mode['port']}")
            print(f"Version: {data.get('version')}, Mode: {data.get('mode')}")
        except:
            pass
        return 0
    
    print(f"Starting RLM-Session server ({'Native' if native else 'Proxy'} mode)...")
    
    log_file = f"/tmp/rlm-{'native' if native else 'session'}.log"
    cmd = f"nohup rlm-opencode serve{' --native' if native else ''} > {log_file} 2>&1 &"
    os.system(cmd)
    
    for i in range(10):
        time.sleep(1)
        if is_server_running(native):
            print(f"Server started on http://localhost:{mode['port']}")
            return 0
    
    print(f"Failed to start server. Check {log_file}")
    return 1


def main():
    # Parse --native flag
    native = "--native" in sys.argv or "-n" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--native", "-n")]
    
    if not args:
        print(__doc__)
        return 0
    
    command = args[0].lower()
    
    if command in ("install", "add", "enable"):
        return install(native)
    elif command in ("uninstall", "remove", "disable"):
        return uninstall(native)
    elif command in ("status", "check"):
        return status()
    elif command in ("serve", "start"):
        return serve(native)
    else:
        print(f"Unknown command: {command}")
        print("Use: install, uninstall, status, or serve")
        print("Add --native for native mode")
        return 1


if __name__ == "__main__":
    sys.exit(main())
