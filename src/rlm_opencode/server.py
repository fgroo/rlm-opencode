"""RLM-OpenCode Server - True RLM with tool-based context access.

This implements the RLM paper's approach:
1. Context stored externally (session files)
2. Model receives ONLY metadata about context
3. Model calls tools (rlm_get_context, rlm_search) to access context
4. Server intercepts tool calls, returns context chunks

Key difference from legacy context injection:
- Old approach: Injects full context into prompt (limited by model window)
- RLM-OpenCode: Model requests context via tools (unlimited context)
"""
import json
import time
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from rich.console import Console

from rlm_opencode import __version__
from rlm_opencode.session import session_manager
from rlm_opencode.providers.registry import get_registry
from rlm_opencode.context_tools import (
    get_context_tools_definition,
    execute_context_tool,
    format_tool_result_for_message,
    ContextToolResult,
)

console = Console()

app = FastAPI(
    title="RLM-OpenCode",
    description="True RLM with tool-based context access",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatMessage(BaseModel):
    role: str
    content: str | list[dict] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 1.0
    max_tokens: int | None = None
    stream: bool = False
    tools: list[dict] | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


# Configurable thresholds (env vars)
import os
RLM_CAPTURE_MIN_CHARS = int(os.environ.get("RLM_CAPTURE_MIN_CHARS", "500"))    # Min chars to capture (0 = all, 500 = skip small)
RLM_CAPTURE_MAX_CHARS = int(os.environ.get("RLM_CAPTURE_MAX_CHARS", "50000"))  # Max chars per entry
UPSTREAM_MAX_TOKENS = int(os.environ.get("RLM_UPSTREAM_MAX_TOKENS", "128000")) # Upstream model's real context window
TOKEN_RESERVE = int(os.environ.get("RLM_TOKEN_RESERVE", "16000"))              # Reserve for response + tools


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def estimate_message_tokens(msg: dict) -> int:
    """Estimate tokens for a single message."""
    total = 4  # message framing overhead
    content = msg.get("content", "")
    if content:
        total += estimate_tokens(content if isinstance(content, str) else json.dumps(content))
    if msg.get("tool_calls"):
        total += estimate_tokens(json.dumps(msg["tool_calls"]))
    if msg.get("name"):
        total += estimate_tokens(msg["name"])
    return total


def truncate_messages(
    messages: list[dict],
    max_tokens: int = UPSTREAM_MAX_TOKENS,
    reserve: int = TOKEN_RESERVE,
) -> list[dict]:
    """Truncate messages to fit the upstream model's context window.
    
    Per the RLM paper (Algorithm 1): only metadata + recent turns are sent
    to the LLM. The full context is accessible via tools (rlm_search, etc.).
    
    Strategy:
    1. Always keep: system messages (RLM instructions)
    2. Always keep: last user message (current task)
    3. Fill remaining budget backwards from most recent messages
    4. If messages were dropped, insert a truncation notice
    """
    budget = max_tokens - reserve
    
    # Separate system messages and conversation messages
    system_msgs = [m for m in messages if m.get("role") == "system"]
    conv_msgs = [m for m in messages if m.get("role") != "system"]
    
    if not conv_msgs:
        return messages
    
    # Calculate system prompt cost
    system_cost = sum(estimate_message_tokens(m) for m in system_msgs)
    remaining_budget = budget - system_cost
    
    if remaining_budget <= 0:
        return system_msgs + conv_msgs[-2:]  # At least keep last exchange
    
    # Fill backwards from most recent messages
    kept_msgs = []
    total_tokens = 0
    
    for msg in reversed(conv_msgs):
        msg_tokens = estimate_message_tokens(msg)
        if total_tokens + msg_tokens > remaining_budget:
            break
        kept_msgs.insert(0, msg)
        total_tokens += msg_tokens
    
    dropped_count = len(conv_msgs) - len(kept_msgs)
    
    if dropped_count > 0:
        # Insert truncation notice
        truncation_notice = {
            "role": "user",
            "content": (
                f"[RLM Context Notice] {dropped_count} earlier messages were truncated "
                f"to fit the model window. The full conversation history is stored in your "
                f"RLM context. Use rlm_search() to find earlier content, or "
                f"rlm_get_context(offset, length) to read specific sections."
            ),
        }
        result = system_msgs + [truncation_notice] + kept_msgs
        console.print(f"[yellow]  Truncated: dropped {dropped_count} messages, kept {len(kept_msgs)} (~{total_tokens:,} tokens)[/yellow]")
    else:
        result = system_msgs + kept_msgs
    
    return result


# Session detection
def get_or_create_session() -> str:
    """Get or create session bound to the current opencode chat.
    
    Each opencode chat gets its own isolated RLM session.
    Two agents in the same directory will NOT share context.
    """
    from rlm_opencode.detector import get_current_session
    
    detected = get_current_session()
    if detected and detected.get("id"):
        # Bind by opencode's unique chat session ID (per-chat isolation)
        session = session_manager.get_or_create_session_by_opencode_id(
            detected["id"],
            directory=detected.get("directory"),
        )
        console.print(f"[cyan]Session: {session.id} (opencode chat: {detected['id'][:12]}...)[/cyan]")
        return session.id
    
    session = session_manager.create_session()
    console.print(f"[green]Created standalone session: {session.id}[/green]")
    return session.id


def capture_content(messages: list[ChatMessage], session_id: str):
    """Capture conversation content into session context.
    
    Captures:
    - ALL tool results (file reads, command outputs, search results)
    - Assistant text responses (summaries, analysis, etc.)
    
    This builds the model's long-term memory across the chat.
    """
    # Build a tool_call_id → tool_name lookup from assistant messages
    tool_id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id", "")
                tc_name = tc.get("function", {}).get("name", "")
                if tc_id and tc_name:
                    tool_id_to_name[tc_id] = tc_name
    
    for msg in messages:
        if msg.role == "tool" and msg.content:
            tool_name = msg.name or tool_id_to_name.get(msg.tool_call_id or "", "") or "external_tool"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.strip() and len(content) >= RLM_CAPTURE_MIN_CHARS:
                session_manager.append(
                    session_id, 
                    "tool_result", 
                    f"[Tool: {tool_name}]\n{content[:RLM_CAPTURE_MAX_CHARS]}",
                    metadata={"tool": tool_name, "truncated": len(content) > RLM_CAPTURE_MAX_CHARS}
                )
        
        elif msg.role == "assistant" and msg.content:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if not msg.tool_calls and len(content) > 50:
                # Capture assistant text responses (final answers)
                session_manager.append(
                    session_id,
                    "assistant_response",
                    f"[Assistant]\n{content[:RLM_CAPTURE_MAX_CHARS]}",
                    metadata={"truncated": len(content) > RLM_CAPTURE_MAX_CHARS}
                )
        
        elif msg.role == "user" and msg.content:
            # Capture user messages (prompts, instructions)
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) >= RLM_CAPTURE_MIN_CHARS:
                session_manager.append(
                    session_id,
                    "user_message",
                    f"[User]\n{content[:RLM_CAPTURE_MAX_CHARS]}",
                    metadata={"truncated": len(content) > RLM_CAPTURE_MAX_CHARS}
                )


def build_rlm_system_prompt(session_id: str) -> str:
    """Build the RLM system prompt with context metadata (NOT the full context)."""
    session = session_manager.get_session(session_id)
    context = session_manager.get_context(session_id)
    stats = session.stats if session else None
    
    context_mb = len(context) / 1_000_000 if context else 0
    context_lines = context.count("\n") + 1 if context else 0
    
    prompt = f"""You are an AI assistant with RLM (Recursive Language Model) capabilities.

## Context Access

You have access to accumulated context from this session:
- **Total Context**: {len(context):,} characters ({context_mb:.2f} MB)
- **Lines**: {context_lines:,}
- **Tool Results Captured**: {stats.tool_outputs if stats else 0}

Instead of seeing the full context, you have TOOLS to access it on-demand:
- `rlm_get_context(offset, length)` - Get a chunk of context
- `rlm_search(pattern, max_results)` - Search with regex
- `rlm_find(text, max_results)` - Find exact text
- `rlm_stats()` - Get context statistics
- `rlm_get_entries(type)` - List context entries

## Strategy

1. For SMALL contexts (<100K chars): You can read directly with rlm_get_context(0, 100000)
2. For LARGE contexts: Use rlm_search() to find relevant sections, then rlm_get_context() to read them
3. Think about what you're looking for before searching - good patterns save tokens

Remember: You DON'T need to read the entire context. Search first, then read relevant sections.
"""
    return prompt


def inject_tools(request: ChatCompletionRequest, session_id: str) -> tuple[list[dict], list[dict]]:
    """Inject RLM context tools into the request."""
    messages = []
    
    rlm_system = build_rlm_system_prompt(session_id)
    has_system = any(msg.role == "system" for msg in request.messages)
    
    if not has_system:
        messages.append({"role": "system", "content": rlm_system})
    
    for msg in request.messages:
        msg_dict = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            msg_dict["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id
        if msg.name:
            msg_dict["name"] = msg.name
        messages.append(msg_dict)
    
    # TRUNCATE: Per RLM paper Algorithm 1, only recent turns + metadata
    # go to the LLM. Full context is accessible via tools.
    messages = truncate_messages(messages)
    
    rlm_tools = get_context_tools_definition()
    
    all_tools = list(rlm_tools)
    if request.tools:
        for tool in request.tools:
            if tool.get("function", {}).get("name", "").startswith("rlm_"):
                continue
            all_tools.append(tool)
    
    return messages, all_tools


@app.get("/")
async def root():
    return {"message": "RLM-OpenCode", "version": __version__, "mode": "true-rlm"}


@app.get("/health")
async def health():
    return {"status": "ok", "version": __version__, "mode": "true-rlm"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, x_rlm_session: str | None = Header(default=None)):
    """RLM chat completions with tool-based context access."""
    console.print(f"[bold cyan][API] Request: {request.model}[/bold cyan]")
    console.print(f"[dim]  Messages: {len(request.messages)}, Stream: {request.stream}[/dim]")
    
    if x_rlm_session and session_manager.get_session(x_rlm_session):
        session_id = x_rlm_session
        console.print(f"[cyan]Using provided session header: {session_id}[/cyan]")
    else:
        session_id = get_or_create_session()
        
    session = session_manager.get_session(session_id)
    
    capture_content(request.messages, session_id)
    
    registry = get_registry()
    resolved = registry.resolve_model(request.model)
    
    if not resolved:
        console.print(f"[red]Model not found: {request.model}[/red]")
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model}")
    
    provider, model_id = resolved
    console.print(f"[dim]  Provider: {provider.provider_id}, Model: {model_id}[/dim]")
    
    context = session_manager.get_context(session_id)
    console.print(f"[dim]  Context: {len(context):,} chars[/dim]")
    
    enhanced_messages, all_tools = inject_tools(request, session_id)
    
    if request.stream:
        return StreamingResponse(
            _stream_with_tools(
                provider, model_id, enhanced_messages, all_tools,
                request, session_id
            ),
            media_type="text/event-stream",
        )
    
    result = []
    finish_reason = None
    tool_calls_accumulated = []
    
    try:
        async for chunk in provider.stream(
            model_id,
            enhanced_messages,
            temperature=request.temperature or 1.0,
            max_tokens=request.max_tokens,
            tools=all_tools if all_tools else None,
        ):
            if chunk.content:
                result.append(chunk.content)
            if chunk.tool_calls:
                tool_calls_accumulated.extend(chunk.tool_calls)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise HTTPException(status_code=500, detail=str(e))
    
    response_text = "".join(result)
    console.print(f"[green]Response: {len(response_text)} chars[/green]")
    
    message_content = response_text if response_text else None
    message_tool_calls = tool_calls_accumulated if tool_calls_accumulated else None
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(
                    role="assistant",
                    content=message_content,
                    tool_calls=message_tool_calls,
                ),
                finish_reason=finish_reason or "stop",
            )
        ],
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "session_id": session_id,
            "context_chars": len(context),
        }
    )


async def _stream_with_tools(
    provider,
    model_id: str,
    messages: list[dict],
    tools: list[dict],
    request: ChatCompletionRequest,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """Stream response with RLM tool support."""
    chat_id = f"chatcmpl-{uuid4().hex[:8]}"
    created = int(time.time())
    
    console.print(f"[dim cyan][_stream] Starting for {model_id}...[/dim cyan]")
    
    collected_content = []
    collected_tool_calls = []
    collected_reasoning = []
    finish_reason = None
    
    try:
        async for chunk in provider.stream(
            model_id,
            messages,
            temperature=request.temperature or 1.0,
            max_tokens=request.max_tokens,
            tools=tools if tools else None,
        ):
            if chunk.content:
                collected_content.append(chunk.content)
            if chunk.tool_calls:
                collected_tool_calls.extend(chunk.tool_calls)
            if chunk.reasoning:
                collected_reasoning.append(chunk.reasoning)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            
            delta = {}
            if chunk.content:
                delta["content"] = chunk.content
            if chunk.tool_calls:
                delta["tool_calls"] = chunk.tool_calls
            if chunk.reasoning:
                delta["reasoning_content"] = chunk.reasoning
            
            data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": chunk.finish_reason,
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
        
        # Capture the model's response into RLM context
        full_content = "".join(collected_content)
        full_reasoning = "".join(collected_reasoning)
        
        if full_reasoning and len(full_reasoning) >= RLM_CAPTURE_MIN_CHARS:
            session_manager.append(
                session_id,
                "thinking",
                f"[Thinking]\n{full_reasoning[:RLM_CAPTURE_MAX_CHARS]}",
                metadata={"truncated": len(full_reasoning) > RLM_CAPTURE_MAX_CHARS}
            )
        
        if full_content and not collected_tool_calls and len(full_content) > 50:
            session_manager.append(
                session_id,
                "assistant_response",
                f"[Assistant]\n{full_content[:RLM_CAPTURE_MAX_CHARS]}",
                metadata={"truncated": len(full_content) > RLM_CAPTURE_MAX_CHARS}
            )
        
        console.print(f"[green][_stream] Complete: {len(full_content)} chars, {len(collected_tool_calls)} tool calls[/green]")
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        console.print(f"[red][_stream] Error: {e}[/red]")
        error_data = {
            "id": chat_id,
            "object": "error",
            "error": {"message": str(e), "type": "stream_error"}
        }
        yield f"data: {json.dumps(error_data)}\n\n"


def run_server():
    """Entry point for running the server."""
    import uvicorn
    console.print("[bold blue]Starting RLM-OpenCode Server[/bold blue]")
    console.print("[dim]True RLM with tool-based context access[/dim]")
    uvicorn.run(app, host="0.0.0.0", port=8769)
