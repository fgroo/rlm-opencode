# RLM-OpenCode Architecture

> **True RLM for AI Coding Assistants**

A Recursive Language Model implementation that transforms how AI coding assistants handle massive contexts - making 100M+ character contexts feel natural within OpenCode.

---

## The Problem

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE CONTEXT WALL                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Model Context Window: ~200K tokens (~800K chars)              │
│                                                                  │
│   Your Codebase:                                                 │
│   ├── src/               2.3M chars                             │
│   ├── tests/             1.1M chars                             │
│   ├── docs/              500K chars                             │
│   ├── node_modules/     45M chars  ← 💀                         │
│   └── logs/             10M chars                               │
│                         ─────────                                │
│   Total:               ~59M chars                               │
│                                                                  │
│   59M / 800K = 73x the model's limit                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Traditional approaches fail:
- **Stuff everything** → Model hallucinates, forgets, times out
- **RAG/Search** → Loses context, can't aggregate across chunks
- **Summarization** → Loses critical details, irreversible

---

## The RLM Solution

Based on the paper [Recursive Language Models (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601)

### Core Insight

```
┌─────────────────────────────────────────────────────────────────┐
│              TRADITIONAL vs RLM APPROACH                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   TRADITIONAL:                                                   │
│   ┌─────────┐      ┌─────────────────────────────────┐          │
│   │  User   │ ───▶ │ Context (60M chars) + Prompt   │ ───▶ 💀  │
│   │ Prompt  │      └─────────────────────────────────┘          │
│   └─────────┘                                                    │
│                                                                  │
│   RLM:                                                           │
│   ┌─────────┐      ┌──────────┐      ┌─────────────────────┐    │
│   │  User   │ ───▶ │ Metadata │ ───▶ │ Model sees ONLY     │    │
│   │ Prompt  │      │ (1KB)    │      │ - "Context: 60M"    │    │
│   └─────────┘      └──────────┘      │ - Tools to access   │    │
│                                      │ - Your question     │    │
│                                      └─────────────────────┘    │
│                                             │                    │
│                                             ▼                    │
│                                      ┌─────────────┐            │
│                                      │ Model calls │            │
│                                      │ TOOLS to    │            │
│                                      │ peek/search │            │
│                                      └─────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLM REQUEST FLOW                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. REQUEST                                                      │
│     ┌──────────────────────────────────────────────────┐        │
│     │ User: "Find all API endpoints in my codebase"    │        │
│     │ Context: 60M characters (stored externally)       │        │
│     └──────────────────────────────────────────────────┘        │
│                          │                                       │
│                          ▼                                       │
│  2. MODEL RECEIVES (not the full context!)                      │
│     ┌──────────────────────────────────────────────────┐        │
│     │ System: You have access to 60M chars of context  │        │
│     │ Tools: rlm_get_context(), rlm_search()            │        │
│     │ User: Find all API endpoints...                   │        │
│     └──────────────────────────────────────────────────┘        │
│                          │                                       │
│                          ▼                                       │
│  3. MODEL WRITES CODE (or calls tools)                          │
│     ┌──────────────────────────────────────────────────┐        │
│     │ rlm_search("@app\.(get|post|put|delete)")        │        │
│     └──────────────────────────────────────────────────┘        │
│                          │                                       │
│                          ▼                                       │
│  4. SERVER EXECUTES                                              │
│     ┌──────────────────────────────────────────────────┐        │
│     │ Searching 60M chars... Found 47 matches!         │        │
│     │ Returning: ["/api/users", "/api/posts", ...]     │        │
│     └──────────────────────────────────────────────────┘        │
│                          │                                       │
│                          ▼                                       │
│  5. MODEL RESPONDS                                               │
│     ┌──────────────────────────────────────────────────┐        │
│     │ "I found 47 API endpoints in your codebase..."   │        │
│     └──────────────────────────────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RLM-OpenCode SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐                         ┌─────────────────────────┐   │
│  │             │                         │                         │   │
│  │  OpenCode   │ ◀─────── API ─────────▶ │    RLM-OpenCode         │   │
│  │  (Client)   │      localhost:8768     │    Server               │   │
│  │             │                         │                         │   │
│  └─────────────┘                         └───────────┬─────────────┘   │
│        │                                             │                  │
│        │                                             │                  │
│        ▼                                             ▼                  │
│  ┌─────────────┐                         ┌─────────────────────────┐   │
│  │             │                         │                         │   │
│  │   Model     │ ◀─────── Tools ───────▶ │    Context Store        │   │
│  │  (GLM-5)    │   rlm_get_context()     │    (Session Files)      │   │
│  │             │   rlm_search()          │    Up to 100M+ chars    │   │
│  │             │   rlm_find()            │                         │   │
│  └─────────────┘                         └─────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Components

```
rlm-opencode/
├── src/rlm_opencode/
│   ├── __init__.py           # Package init
│   ├── server.py             # FastAPI server (proxy mode)
│   ├── native_server.py      # Direct API server (main)
│   ├── cli.py                # CLI commands
│   ├── setup.py              # Install/uninstall to opencode
│   ├── session.py            # Context storage management
│   ├── detector.py           # OpenCode session detection
│   ├── context_tools.py      # ← NEW: RLM context tools
│   └── providers/
│       ├── __init__.py
│       ├── base.py           # Provider interface
│       ├── openai_compatible.py  # OpenAI-compatible streaming
│       └── registry.py       # Model discovery
├── pyproject.toml
├── README.md
├── ARCHITECTURE.md           # This file
└── paper.pdf                 # RLM paper (add manually)
```

---

## Context Tools (The RLM Magic)

### Tool Definitions

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "rlm_get_context",
        "description": "Get accumulated session context",
        "parameters": {
          "type": "object",
          "properties": {
            "offset": {"type": "integer", "default": 0},
            "length": {"type": "integer", "default": 10000}
          }
        }
      }
    },
    {
      "type": "function", 
      "function": {
        "name": "rlm_search",
        "description": "Search context with regex pattern",
        "parameters": {
          "type": "object",
          "properties": {
            "pattern": {"type": "string"},
            "max_results": {"type": "integer", "default": 50}
          },
          "required": ["pattern"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "rlm_find",
        "description": "Find exact text occurrences",
        "parameters": {
          "type": "object",
          "properties": {
            "text": {"type": "string"},
            "max_results": {"type": "integer", "default": 100}
          },
          "required": ["text"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "rlm_stats",
        "description": "Get context statistics",
        "parameters": {"type": "object", "properties": {}}
      }
    }
  ]
}
```

### Tool Behavior

```
┌─────────────────────────────────────────────────────────────────┐
│                    TOOL EXECUTION FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Model calls: rlm_search("def\\s+\\w+\\(.*\\):")                │
│                          │                                       │
│                          ▼                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ RLM-OpenCode Server                                    │     │
│  │                                                        │     │
│  │  1. Identify session (by directory)                   │     │
│  │  2. Load context (up to 100M+ chars)                  │     │
│  │  3. Execute regex search                               │     │
│  │  4. Return results                                     │     │
│  │                                                        │     │
│  └────────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼                                       │
│  Tool Result:                                                    │
│  {                                                               │
│    "matches": [                                                  │
│      {"line": 45, "text": "def process_data(input):"},         │
│      {"line": 128, "text": "def validate_user(user):"},        │
│      ...                                                         │
│    ],                                                            │
│    "total": 47,                                                  │
│    "truncated": false                                            │
│  }                                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Comparison: rlm-server vs rlm-opencode

| Feature | rlm-server (Port 8765) | rlm-opencode (Port 8768) |
|---------|------------------------|--------------------------|
| **Context Access** | Code execution (Python) | Tool calls |
| **Model Sees** | Metadata + code template | Metadata + tool definitions |
| **Integration** | Single-shot API calls | Full OpenCode integration |
| **Tools** | `load_context()`, `llm_query()` | `rlm_get_context()`, `rlm_search()` |
| **Recursion** | `llm_query()` for sub-calls | Via OpenCode tool loop |
| **Permissions** | Sandbox (no issues) | Handled by outer OpenCode |
| **Natural Feel** | Script-like | Conversational |
| **Best For** | Batch processing | Agentic workflows |

---

## Session Management

```
┌─────────────────────────────────────────────────────────────────┐
│                    SESSION LIFECYCLE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ~/.local/share/rlm-opencode/                                   │
│  ├── sessions/                                                   │
│  │   ├── sess_abc123_context.txt    # Accumulated context       │
│  │   ├── sess_abc123.json           # Session metadata          │
│  │   └── ...                                                     │
│  └── mappings/                                                   │
│      └── directory_to_rlm.json      # Path → Session mapping    │
│                                                                  │
│  Session Mapping:                                                │
│  ┌────────────────────────┬─────────────────┐                   │
│  │ Directory              │ Session ID      │                   │
│  ├────────────────────────┼─────────────────┤                   │
│  │ /home/user/project-a   │ sess_abc123     │                   │
│  │ /home/user/project-b   │ sess_def456     │                   │
│  └────────────────────────┴─────────────────┘                   │
│                                                                  │
│  Context Accumulation:                                           │
│  - Tool results (file reads, grep output, etc.)                 │
│  - Large outputs are stored, not in message history              │
│  - Persists across multiple OpenCode calls                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Usage

### Setup

```bash
# Install
pip install -e .

# Add to OpenCode
rlm-opencode-setup install

# Verify
opencode models | grep rlm-opencode
```

### In OpenCode

```bash
# Use RLM-OpenCode model
opencode -m rlm-opencode/rlm-internal.rlm-core-v1

# The model now has access to tools:
# - rlm_get_context(offset, length)
# - rlm_search(pattern, max_results)  
# - rlm_find(text, max_results)
# - rlm_stats()
```

### Example Interaction

```
User: What API patterns are used in my codebase?

Model: Let me search your accumulated context...
       [calls rlm_search("@(get|post|put|delete)\\(")]

Model: I found 47 API endpoints using these patterns:
       - Flask-style: @app.get(), @app.post()
       - FastAPI-style: @router.get(), @router.post()
       ...
```

---

## Performance

### Benchmarked Results (NIAH - Needle in a Haystack)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTEXT SCALING                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Context Size     rlm-server    rlm-opencode    Vanilla        │
│  ─────────────    ──────────    ────────────    ─────────      │
│  1M chars         ✓ 41s         ✓ ~30s          ✓ 15s          │
│  10M chars        ✓ 52s         ✓ ~60s          ✗ TIMEOUT      │
│  40M chars        ✓ 109s        ✓ ~120s         ✗ FAIL         │
│  100M chars       ✓ 65s         🔄 (goal)       ✗ IMPOSSIBLE   │
│                                                                  │
│  Key: ✓ = Success, ✗ = Failure, 🔄 = In Progress                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Goals

### Phase 1: Foundation ✅
- [x] Basic server with OpenAI-compatible API
- [x] Direct provider API calls
- [x] Session management
- [x] Context accumulation
- [x] Streaming responses

### Phase 2: True RLM (Current)
- [ ] Context tools: `rlm_get_context()`, `rlm_search()`
- [ ] Tool-based context access (no injection)
- [ ] Metadata-only prompts
- [ ] 100M+ char support

### Phase 3: Optimization
- [ ] Chunked context loading
- [ ] Lazy context retrieval
- [ ] Parallel tool execution
- [ ] Context compression hints

### Phase 4: Advanced
- [ ] Recursive sub-calls via tools
- [ ] Multi-session context sharing
- [ ] Context versioning
- [ ] Collaborative sessions

---

## Contributing

This project implements the RLM paper's Algorithm 1 with adaptations for AI coding assistants.

Key files to understand:
- `native_server.py` - Main server logic
- `context_tools.py` - Tool definitions and execution
- `session.py` - Context storage and retrieval

---

## References

- [Recursive Language Models Paper](https://arxiv.org/abs/2512.24601)
- [OpenCode](https://github.com/opencode-ai/opencode)
- [RLM Paper GitHub](https://github.com/alexzhang13/rlm)
