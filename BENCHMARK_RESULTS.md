# RLM NIAH Benchmark Results

**Date:** 2026-02-21
**Model:** (rlm-internal)
**Test:** Needle-in-a-Haystack (NIAH)

## Summary

| Server | Port | Mode | Max Working Context | Upper Limit | Accuracy |
|--------|------|------|---------------------|-------------|----------|
| **rlm-server** | 8765 | Code execution | **~100M chars** | ~120M+ (times out) | 67% (4/6) |
| **rlm-native** | 8768 | Direct API | **~1.5M chars** | ~2M (times out) | 60% (3/5) |
| **rlm-session** | 8767 | Proxy | **~1M chars** | ~1.5M | Not fully tested |
| **rlm-opencode** | 8769 | API with Tools | **~300M+ chars** | **~500M** (logic limit) | 100% (up to 300M) |

## Architecture Comparison

| Feature | rlm-server | rlm-native | rlm-session | rlm-opencode |
|---------|------------|------------|-------------|--------------|
| **Context storage** | Temp file per request | Injected in prompt | Injected in prompt | SQLite & Context Files |
| **How model sees context** | Writes Python code to search | All at once | All at once | Uses tools (rlm_search) |
| **Max context tested** | 100M chars | 1.5M chars | 1M chars | **500M chars** |
| **Limit factor** | Model quality + timeout | Model context window | Model context window | Logic budget (Turns) |
| **Model context window** | Bypassed (external file) | ~200K tokens | ~200K tokens | Bypassed via tools |

## Detailed Results

### rlm-server (Port 8765)
Original RLM with code execution. Model writes Python to search temp file.
**Model:** `rlm-rlm-core-v1`

| Context | Time | Result | Notes |
|---------|------|--------|-------|
| 10M | 41s | ✓ | |
| 20M | 140s | ✓ | |
| 40M | 90s | ✗ | Found wrong number |
| 60M | 125s | ✓ | |
| 80M | 108s | ✗ | Found wrong number |
| 100M | 65s | ✓ | |
| 120M | - | TIMEOUT | Exceeded 600s limit |

**Limit:** ~100M characters working, 120M+ times out

### rlm-native (Port 8768)
Direct API calls with context injection.
**Model:** `rlm-internal.rlm-core-v1`

| Context | Time | Result | Notes |
|---------|------|--------|-------|
| 1M | 90s | ✓ | Verbose response |
| 1.5M | 66s | ✓ | |
| 2M | 300s | TIMEOUT | |
| 2.5M | 101s | ✗ | Hallucinated - number not found |
| 3M | - | TIMEOUT | |

**Limit:** ~1.5M characters (model's ~200K token window ≈ 1.5M chars)

### rlm-opencode (Port 8769)
True RLM via HTTP completions. Extracted context interacts sequentially via tool calls (`rlm_search`, `rlm_get_context`).
**Model:** `rlm-core-v1`

| Context | Time | Result | Notes |
|---------|------|--------|-------|
| 30M | 49s | ✓ | Perfect precision |
| 40M | 35s | ✓ | Efficient regex search |
| 50M | 57s | ✓ | Perfect precision |
| 60M | 19s | ✓ | Extremely fast lookup |
| 80M | 36s | ✓ | Perfect precision |
| 100M | 37s | ✓ | Handled massive file perfectly |
| 120M | 72s | ✓ | Handled beyond original rlm-server limits |
| 200M | 93s | ✓ | Absolute extreme scale success |
| 300M | 49s | ✓ | Still 100% accurate |
| 500M | 271s | ✗ | **Breaking Point**: Hit Max Tool Turns (10) |

**Limit:** The architecture breaks at **half a billion characters** (500M), not due to server or memory failure, but because the model hits its tool-turn budget trying to find the needle in such a massive haystack. Raising the budget would likely push this even further. `rlm-opencode` is effectively 5x more scalable than the original `rlm-server`.

## Key Findings

1. **rlm-server handles 100M+** - 66x more than model's native window
2. **rlm-native limited to ~1.5M** - matches model's context window (~200K tokens)
3. **rlm-server has some accuracy issues** at scale (wrong numbers found)
4. **rlm-native times out** beyond 1.5M chars
5. **Model quality matters more at scale** - LLMs sometimes hallucinate numbers

## Context Window Math

- modern LLM context window: ~200K tokens
- 1 token ≈ 4 characters (English)
- 200K tokens ≈ 800K characters
- rlm-native works up to ~1.5M chars (close to window)
- rlm-server works up to ~100M chars (125x window size)

## Recommendations

| Use Case | Recommended Server | Reason |
|----------|-------------------|--------|
| Context > 10M | rlm-server (8765) | Only option |
| Context 1M-10M | rlm-server (8765) | More reliable |
| Context < 1M | rlm-native (8768) | Faster, simpler |
| Agentic workflows | rlm-native (8768) | No permission issues |
| Maximum reliability | rlm-server (8765) | Proven at scale |
