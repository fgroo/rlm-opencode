# RLM-OpenCode

> **True RLM for AI Coding Assistants**

A Recursive Language Model implementation that gives AI coding assistants unlimited context through tool-based access.

## What is this?

RLM-OpenCode is based on the [Recursive Language Models paper (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601). It enables AI coding assistants to handle **100M+ character contexts** by treating context as an external resource accessed via tools, rather than stuffing everything into the prompt.

```
Traditional:  Context (60M chars) → Model → 💀 FAILS
RLM-OpenCode: Context metadata → Model → Tools → Context chunks → SUCCESS
```

## Quick Start

### Install

```bash
pip install -e .
```

### Add to OpenCode

```bash
rlm-opencode-setup install
```

### Use

```bash
opencode -m rlm-opencode/rlm-internal.rlm-core-v1
```

## How It Works

### The Problem

```
┌─────────────────────────────────────────┐
│  Your Codebase: 60M chars               │
│  Model Context Window: 200K tokens      │
│                                         │
│  60M / 800K = 73x too much!             │
└─────────────────────────────────────────┘
```

### The RLM Solution

```
┌─────────────────────────────────────────┐
│  1. Model receives METADATA only:       │
│     "Context: 60M chars, 5000 lines"    │
│                                         │
│  2. Model gets TOOLS:                   │
│     - rlm_get_context(offset, length)   │
│     - rlm_search(pattern)               │
│     - rlm_find(text)                    │
│                                         │
│  3. Model calls tools on-demand         │
│     to peek/search context              │
│                                         │
│  Result: Unlimited context access!      │
└─────────────────────────────────────────┘
```

## Context Tools

The model has access to these tools:

| Tool | Description |
|------|-------------|
| `rlm_get_context(offset, length)` | Get a chunk of context |
| `rlm_search(pattern, max_results)` | Search with regex |
| `rlm_find(text, max_results)` | Find exact text |
| `rlm_stats()` | Get context statistics |
| `rlm_get_entries(type)` | List context entries |

## Architecture

```
OpenCode (Client)                    RLM-OpenCode Server
┌─────────────────┐                  ┌──────────────────────┐
│                 │   API Request    │                      │
│  opencode run   │ ───────────────▶ │  FastAPI Server      │
│  -m rlm-opencode│                  │  (port 8769)         │
│                 │                  │                      │
└─────────────────┘                  └──────────┬───────────┘
                                                │
        ┌───────────────────────────────────────┘
        ▼
┌─────────────────┐                  ┌──────────────────────┐
│                 │   Tool Calls     │                      │
│  Model (GLM-5)  │ ◀─────────────── │  Context Store       │
│                 │   Context Data   │  (Session Files)     │
│                 │ ───────────────▶ │  100M+ chars         │
└─────────────────┘                  └──────────────────────┘
```

## Comparison

| Feature | rlm-server | rlm-opencode |
|---------|------------|--------------|
| Context Access | Code execution | Tool calls |
| Integration | Single-shot API | Full OpenCode |
| Natural Feel | Script-like | Conversational |
| Permissions | Sandbox | Handled by OpenCode |
| Best For | Batch processing | Agentic workflows |

## Files

```
rlm-opencode/
├── src/rlm_opencode/
│   ├── server.py          # True RLM server
│   ├── context_tools.py   # Tool definitions
│   ├── session.py         # Context storage
│   ├── providers/         # Model API clients
│   └── cli.py             # CLI commands
├── ARCHITECTURE.md        # Detailed architecture
├── README.md              # This file
└── pyproject.toml
```

## Session Data

Sessions are stored in `~/.local/share/rlm-opencode/`:

- `sessions/` - Context files and metadata
- `mappings/` - Directory → Session mapping

Context accumulates across OpenCode calls, persisting between sessions.

## License

MIT
