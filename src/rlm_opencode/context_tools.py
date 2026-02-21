"""Context tools for RLM-OpenCode.

These tools allow the model to access context on-demand instead of
having it injected into the prompt. This is the key to True RLM.

Tool Strategy:
1. Model receives metadata about context size
2. Model calls tools to peek/search context
3. Server intercepts tool calls and returns context chunks
4. Model never sees full context at once
"""
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ContextToolResult:
    """Result from a context tool execution."""
    tool_name: str
    success: bool
    result: dict[str, Any]
    error: str | None = None


def get_context_tools_definition() -> list[dict]:
    """Get OpenAI-compatible tool definitions for context access.
    
    Returns tool definitions that can be passed to the model.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "rlm_get_context",
                "description": "Get a chunk of the accumulated session context. Use this to peek at the context when you need to read specific parts. Default chunk size is 10K chars.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "offset": {
                            "type": "integer",
                            "description": "Start position in context (default: 0)",
                            "default": 0
                        },
                        "length": {
                            "type": "integer", 
                            "description": "Number of characters to retrieve (default: 10000, max: 50000)",
                            "default": 10000
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rlm_search",
                "description": "Search the accumulated context with a regex pattern. Returns matching lines with their positions. Use this to find specific patterns like function definitions, API endpoints, error messages, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for (e.g., 'def\\\\s+\\\\w+', '@app\\\\.(get|post)', 'ERROR.*')"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 50, max: 200)",
                            "default": 50
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": "Lines of context around each match (default: 2)",
                            "default": 2
                        }
                    },
                    "required": ["pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rlm_find",
                "description": "Find exact text occurrences in the context. Simpler than rlm_search but faster for literal strings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Exact text to find"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results (default: 100)",
                            "default": 100
                        }
                    },
                    "required": ["text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rlm_stats",
                "description": "Get statistics about the accumulated context: total size, entry counts, last update time.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rlm_get_entries",
                "description": "Get information about context entries (tool results, file reads, etc.). Use this to understand what's in the context before searching.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entry_type": {
                            "type": "string",
                            "description": "Filter by entry type: 'tool_result', 'thinking', or 'all' (default: 'all')",
                            "default": "all"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max entries to return (default: 20)",
                            "default": 20
                        }
                    }
                }
            }
        }
    ]


def execute_context_tool(
    tool_name: str,
    arguments: dict[str, Any],
    context: str,
    session_stats: dict[str, Any] | None = None,
    session_entries: list[dict] | None = None,
) -> ContextToolResult:
    """Execute a context tool and return the result.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments
        context: Full session context string
        session_stats: Statistics about the session
        session_entries: List of context entries
        
    Returns:
        ContextToolResult with success/failure and data
    """
    try:
        if tool_name == "rlm_get_context":
            return _tool_get_context(context, arguments)
        
        elif tool_name == "rlm_search":
            return _tool_search(context, arguments)
        
        elif tool_name == "rlm_find":
            return _tool_find(context, arguments)
        
        elif tool_name == "rlm_stats":
            return _tool_stats(context, session_stats)
        
        elif tool_name == "rlm_get_entries":
            return _tool_get_entries(session_entries or [], arguments)
        
        else:
            return ContextToolResult(
                tool_name=tool_name,
                success=False,
                result={},
                error=f"Unknown tool: {tool_name}"
            )
            
    except Exception as e:
        return ContextToolResult(
            tool_name=tool_name,
            success=False,
            result={},
            error=str(e)
        )


def _tool_get_context(context: str, args: dict) -> ContextToolResult:
    """Get a chunk of context."""
    offset = max(0, args.get("offset", 0))
    length = min(50000, max(1, args.get("length", 10000)))
    
    if offset >= len(context):
        return ContextToolResult(
            tool_name="rlm_get_context",
            success=False,
            result={},
            error=f"Offset {offset} exceeds context length {len(context)}"
        )
    
    chunk = context[offset:offset + length]
    end_offset = min(offset + length, len(context))
    
    return ContextToolResult(
        tool_name="rlm_get_context",
        success=True,
        result={
            "content": chunk,
            "offset": offset,
            "length": len(chunk),
            "total_context": len(context),
            "has_more": end_offset < len(context),
        }
    )


def _tool_search(context: str, args: dict) -> ContextToolResult:
    """Search context with regex."""
    pattern = args.get("pattern", "")
    max_results = min(200, max(1, args.get("max_results", 50)))
    context_lines = max(0, args.get("context_lines", 2))
    
    if not pattern:
        return ContextToolResult(
            tool_name="rlm_search",
            success=False,
            result={},
            error="Pattern is required"
        )
    
    try:
        regex = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
    except re.error as e:
        return ContextToolResult(
            tool_name="rlm_search",
            success=False,
            result={},
            error=f"Invalid regex: {e}"
        )
    
    lines = context.split("\n")
    matches = []
    
    for i, line in enumerate(lines):
        if len(matches) >= max_results:
            break
        
        if regex.search(line):
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            
            context_block = "\n".join(
                f"{j+1}: {lines[j]}" for j in range(start, end)
            )
            
            matches.append({
                "line_number": i + 1,
                "line": line[:200] if len(line) > 200 else line,
                "context": context_block[:500] if context_lines > 0 else None,
            })
    
    return ContextToolResult(
        tool_name="rlm_search",
        success=True,
        result={
            "pattern": pattern,
            "matches": matches,
            "total_found": len(matches),
            "truncated": len(matches) >= max_results,
            "context_size": len(context),
        }
    )


def _tool_find(context: str, args: dict) -> ContextToolResult:
    """Find exact text occurrences."""
    text = args.get("text", "")
    max_results = min(500, max(1, args.get("max_results", 100)))
    
    if not text:
        return ContextToolResult(
            tool_name="rlm_find",
            success=False,
            result={},
            error="Text is required"
        )
    
    positions = []
    start = 0
    
    while len(positions) < max_results:
        pos = context.find(text, start)
        if pos == -1:
            break
        
        line_start = context.rfind("\n", 0, pos) + 1
        line_end = context.find("\n", pos)
        line_num = context[:line_start].count("\n") + 1
        
        positions.append({
            "position": pos,
            "line_number": line_num,
            "line": context[line_start:line_end][:200] if line_end != -1 else context[line_start:line_start+200],
        })
        
        start = pos + 1
    
    return ContextToolResult(
        tool_name="rlm_find",
        success=True,
        result={
            "text": text,
            "occurrences": positions,
            "total_found": len(positions),
            "truncated": len(positions) >= max_results,
        }
    )


def _tool_stats(context: str, stats: dict | None) -> ContextToolResult:
    """Get context statistics."""
    result = {
        "context_chars": len(context),
        "context_mb": round(len(context) / 1_000_000, 2),
        "context_lines": context.count("\n") + 1 if context else 0,
    }
    
    if stats:
        result.update({
            "files_read": stats.get("files_read", 0),
            "tool_outputs": stats.get("commands_run", 0),
            "thinking_blocks": stats.get("thinking_blocks", 0),
        })
    
    return ContextToolResult(
        tool_name="rlm_stats",
        success=True,
        result=result
    )


def _tool_get_entries(entries: list[dict], args: dict) -> ContextToolResult:
    """Get context entry information."""
    entry_type = args.get("entry_type", "all")
    limit = min(100, max(1, args.get("limit", 20)))
    
    filtered = entries
    if entry_type != "all":
        filtered = [e for e in entries if e.get("type") == entry_type]
    
    result_entries = []
    for entry in filtered[-limit:]:
        result_entries.append({
            "type": entry.get("type", "unknown"),
            "length": entry.get("length", 0),
            "timestamp": entry.get("timestamp"),
            "metadata": entry.get("metadata", {}),
        })
    
    return ContextToolResult(
        tool_name="rlm_get_entries",
        success=True,
        result={
            "entries": result_entries,
            "total": len(filtered),
            "showing": len(result_entries),
        }
    )


def format_tool_result_for_message(result: ContextToolResult) -> str:
    """Format a tool result for inclusion in assistant message.
    
    This creates a readable summary of the tool result.
    """
    if not result.success:
        return f"Error: {result.error}"
    
    data = result.result
    
    if result.tool_name == "rlm_get_context":
        return f"Retrieved {data['length']} chars from offset {data['offset']} (total context: {data['total_context']:,} chars)"
    
    elif result.tool_name == "rlm_search":
        truncated = " (truncated)" if data.get("truncated") else ""
        return f"Found {data['total_found']} matches for pattern '{data['pattern']}'{truncated}"
    
    elif result.tool_name == "rlm_find":
        truncated = " (truncated)" if data.get("truncated") else ""
        return f"Found {data['total_found']} occurrences of '{data['text']}'{truncated}"
    
    elif result.tool_name == "rlm_stats":
        return f"Context: {data['context_mb']} MB ({data['context_chars']:,} chars, {data['context_lines']:,} lines)"
    
    elif result.tool_name == "rlm_get_entries":
        return f"Showing {data['showing']}/{data['total']} entries"
    
    return json.dumps(data, indent=2)
