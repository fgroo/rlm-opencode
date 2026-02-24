# RLM-OpenCode

> **True RLM for AI Coding Assistants**

A Recursive Language Model implementation that gives AI coding assistants unlimited context through tool-based access.

## What is this?

RLM-OpenCode is based on the [Recursive Language Models paper (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601). It enables AI coding assistants to handle **242M+ character contexts** by treating context as an external resource accessed via tools, rather than stuffing everything into the prompt.

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
| `rlm_summarize(query, max_results)` | Semantically summarize parts of memory |
| `rlm_forget(pattern)` | Delete context entries matching regex |

## Architecture

```
OpenCode (Client)                    RLM-OpenCode Server
┌─────────────────┐                  ┌──────────────────────┐
│                 │   API Request    │                      │
│  opencode run   │ ───────────────▶│  FastAPI Server      │
│  -m rlm-opencode│                  │  (port 8769)         │
│                 │                  │                      │
└─────────────────┘                  └──────────┬───────────┘
                                                │
        ┌───────────────────────────────────────┘
        ▼
┌─────────────────┐                  ┌──────────────────────┐
│                 │   Tool Calls     │                      │
│  Model (LLM)    │ ◀───────────────│ Context Store        │
│                 │   Context Data   │  (Session Files)     │
│                 │ ───────────────▶│  242M+ chars         │
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

## Setup Details

`rlm-opencode-setup install` modifies your `~/.config/opencode/opencode.json`:

- Adds an **`rlm-opencode`** provider pointing to `http://localhost:8769/v1`
- Creates model entries for **every** model in your existing OpenCode config under the `rlm-opencode/` prefix
- Sets context limit to **67M tokens** (~300M chars) — tells OpenCode to send the full conversation history
- The server truncates this to fit the upstream model's real context window (default 128K tokens)

After setup, all your models are available as `rlm-opencode/<provider>.<model>`:

```bash
opencode -m rlm-opencode/your_provider.your_model
```

## Configuration

All thresholds are dynamically configurable via the built-in CLI config tool (which saves to a persistent JSON store) or via environment variables (which act as fallbacks).

```bash
rlm-opencode config rlm_max_payload_chars 75000
```

| Config Key / Env Var | Default | Description |
|----------------------|---------|-------------|
| `strict_mode_level` | 0 | [0-4] Forces LLM to rely on tools (4 = Maximum Amnesia) |
| `rlm_upstream_max_tokens` | 128,000 | The raw token limit of the underlying LLM's architecture |
| `rlm_token_reserve` | 16,000 | Budget reserved for the model's generation output and tool responses |
| `rlm_max_payload_chars` | 250,000 | Absolute size limit of the immediate workspace injected natively |
| `rlm_capture_min_chars` | 500 | Min chars for a tool result to be saved into context (0 = all) |
| `rlm_capture_max_chars` | 50,000 | Max chars per single entry in the context lake before truncation |
| `rlm_user_min_chars` | 0 | Min chars for a user message to be captured (0 = all) |
| `rlm_assistant_min_chars` | 50 | Min chars for an assistant response to be captured |
| `rlm_summarize_model` | None | Specific model override for rlm_summarize tool calls |

## CLI Commands

```bash
rlm-opencode serve          # Start server (foreground)
rlm-opencode serve --bg     # Start in background
rlm-opencode stop           # Stop the server
rlm-opencode restart        # Stop → reinstall → restart
rlm-opencode config         # View or set persistent settings
rlm-opencode branch         # Branch, backup, restore, or transfer session memory
rlm-opencode log [-f]       # View visually rendered session history tree (or follow logs)
rlm-opencode sessions       # List sessions with context stats
rlm-opencode status         # Server status + active configuration
rlm-opencode clear --all    # Clear all sessions and logs
```

## Session Data

Sessions are stored in `~/.local/share/rlm-opencode/`:

- `sessions.db` — Session metadata (SQLite)
- `contexts/` — Context files (one per session, append-only)

Context accumulates across OpenCode calls, persisting between sessions. Each OpenCode chat gets its own isolated session via request fingerprinting.

## License

MIT
