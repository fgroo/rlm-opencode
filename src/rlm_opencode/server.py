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

from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
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
# Default configurations
RLM_DEFAULT_SETTINGS = {
    "rlm_capture_min_chars": 500,
    "rlm_capture_max_chars": 50000,
    "rlm_user_min_chars": 0,
    "rlm_assistant_min_chars": 50,
    "rlm_upstream_max_tokens": 128000,
    "rlm_token_reserve": 16000,
    "rlm_max_payload_chars": 250000,
    "rlm_summarize_model": None,
}

RLM_SETTING_DESCRIPTIONS = {
    "strict_mode_level": "[0-4] Forces LLM to rely on tools (4 = Maximum Amnesia)",
    "rlm_capture_min_chars": "Minimum characters for a tool result to be saved into the context lake (0 = all)",
    "rlm_capture_max_chars": "Maximum characters allowed per single entry in the context lake before truncation",
    "rlm_user_min_chars": "Minimum characters for a user message to be captured into the context lake (0 = all)",
    "rlm_assistant_min_chars": "Minimum characters for an assistant response to be captured into the context lake",
    "rlm_upstream_max_tokens": "The raw token limit of the underlying LLM's architecture",
    "rlm_token_reserve": "Budget reserved for the model's generation output and tool responses",
    "rlm_max_payload_chars": "Absolute size limit of the immediate workspace injected natively into the LLM prompt",
    "rlm_summarize_model": "Optional specific model override used exclusively for rlm_summarize tool calls",
}

def get_setting(key: str) -> any:
    """Get configuration with priority: Persistent JSON -> ENV var -> Default."""
    # 1. Try persistent config
    config = session_manager.get_config()
    if key in config:
        return config[key]
    
    # 2. Try ENV var
    env_key = key.upper()
    if env_key in os.environ:
        val = os.environ[env_key]
        if key == "rlm_summarize_model":
            return val
        return int(val)
        
    # 3. Default
    return RLM_DEFAULT_SETTINGS.get(key)

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
    max_tokens: int | None = None,
    reserve: int | None = None,
    max_chars: int | None = None,
) -> list[dict]:
    """Truncate messages to fit the upstream model's context window.
    
    Per the RLM paper (Algorithm 1): only metadata + recent turns are sent
    to the LLM. The full context is accessible via tools (rlm_search, etc.).
    
    Strategy:
    1. Always keep: system messages (RLM instructions)
    2. Always keep: last user message (current task)
    3. Fill remaining budget backwards from most recent messages
    4. Enforce an absolute character limit to avoid hidden API gateway 500 errors
    """
    if max_tokens is None:
        max_tokens = get_setting("rlm_upstream_max_tokens")
    if reserve is None:
        reserve = get_setting("rlm_token_reserve")
    if max_chars is None:
        max_chars = get_setting("rlm_max_payload_chars")
        
    budget = max_tokens - reserve
    
    # Separate system messages and conversation messages
    system_msgs = [m for m in messages if m.get("role") == "system"]
    conv_msgs = [m for m in messages if m.get("role") != "system"]
    
    if not conv_msgs:
        return messages
    
    # Calculate system prompt cost
    system_cost = sum(estimate_message_tokens(m) for m in system_msgs)
    system_chars = sum(len(str(m.get("content", ""))) for m in system_msgs)
    
    remaining_budget = budget - system_cost
    remaining_chars = max_chars - system_chars
    
    if remaining_budget <= 0 or remaining_chars <= 0:
        return system_msgs + conv_msgs[-2:]  # At least keep last exchange
    
    # Keep adding recent messages until token/char budget is full
    kept_msgs = []
    total_tokens = 0
    total_chars = 0
    
    for msg in reversed(conv_msgs):
        msg_tokens = estimate_message_tokens(msg)
        msg_chars = len(str(msg.get("content", "")))
        
        if total_tokens + msg_tokens > remaining_budget or total_chars + msg_chars > remaining_chars:
            break
            
        kept_msgs.insert(0, msg)
        total_tokens += msg_tokens
        total_chars += msg_chars
    
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


def _extract_text(content) -> str:
    """Extract plain text from message content.
    
    Handles both formats:
    - String: "hello world" → "hello world"
    - Multimodal: [{"type": "text", "text": "hello"}, ...] → "hello"
    
    Strips system-reminder blocks which opencode injects.
    """
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # Multimodal content — extract text parts only
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                t = part.get("text", "")
                # Skip system-reminder blocks (opencode injects these)
                if "<system-reminder>" in t:
                    continue
                parts.append(t)
        text = "\n".join(parts) if parts else str(content)
    else:
        text = str(content)
    
    return text.strip()


# Title generation requests from opencode — skip these
_TITLE_PREFIXES = ("generate a title", "suggest a title", "create a title")


# Session detection
def _fingerprint_messages(messages: list) -> str:
    """Create a stable fingerprint from the first REAL user message.
    
    Handles multimodal content, strips system-reminders, and
    ignores opencode's internal title-generation requests.
    """
    import hashlib
    for msg in messages:
        role = msg.role if hasattr(msg, 'role') else msg.get('role', '')
        content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
        if role == "user" and content:
            text = _extract_text(content)
            
            # Skip title-generation requests (opencode internal)
            if text and text.lower().startswith(_TITLE_PREFIXES):
                continue
                
            # If there's no text (e.g., pure image prompt), hash the raw content structure
            hash_target = text if text else str(content)
            
            if hash_target:
                return hashlib.sha256(hash_target.encode()).hexdigest()[:16]
                
    return ""


def get_or_create_session(messages: list = None) -> str:
    """Get or create session bound to the current opencode chat.
    
    Uses request fingerprinting: hashes the first user message text
    to create a stable session identity. Handles multimodal content
    and ignores opencode's internal requests (title generation).
    """
    fingerprint = _fingerprint_messages(messages) if messages else ""
    
    if fingerprint:
        session = session_manager.get_or_create_session_by_opencode_id(
            fingerprint,
            directory=None,
        )
        console.print(f"[cyan]Session: {session.id} (fingerprint: {fingerprint})[/cyan]")
        return session.id
    
    session = session_manager.create_session()
    console.print(f"[green]Created standalone session: {session.id}[/green]")
    return session.id

# Track how many messages we've already captured per session
# to avoid duplicating content since OpenCode resends full history each request
_session_msg_count: dict[str, int] = {}


def capture_content(messages: list[ChatMessage], session_id: str):
    """Capture NEW conversation content into session context.
    
    OpenCode sends the FULL message history with every request.
    We track how many messages we've already processed and only
    capture new ones to avoid duplication.
    """
    global _session_msg_count
    
    prev_count = _session_msg_count.get(session_id, 0)
    current_count = len(messages)
    
    if current_count <= prev_count:
        return  # No new messages
    
    # Only process new messages (from prev_count onwards)
    new_messages = messages[prev_count:]
    _session_msg_count[session_id] = current_count
    
    # Build tool_call_id → tool_name lookup from ALL messages (need full history for resolution)
    tool_id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id", "")
                tc_name = tc.get("function", {}).get("name", "")
                if tc_id and tc_name:
                    tool_id_to_name[tc_id] = tc_name
    
    captured = 0
    for msg in new_messages:
        if msg.role == "tool" and msg.content:
            tool_name = msg.name or tool_id_to_name.get(msg.tool_call_id or "", "") or "external_tool"
            content = _extract_text(msg.content)
            # Truncate content if too huge to keep DB small
            cap_min = get_setting("rlm_capture_min_chars")
            cap_max = get_setting("rlm_capture_max_chars")
            if content.strip() and len(content) >= cap_min:
                session_manager.append(
                    session_id,
                    "tool_output",
                    f"[Tool: {tool_name}]\n{content[:cap_max]}",
                    metadata={"tool": tool_name, "truncated": len(content) > cap_max}
                )
                captured += 1
        
        elif msg.role == "assistant" and msg.content:
            # Skip — assistant responses are captured immediately in _stream_with_tools
            pass
        
        elif msg.role == "user" and msg.content:
            content = _extract_text(msg.content)
            # Skip title-generation requests
            if content.lower().startswith(_TITLE_PREFIXES):
                continue
            usr_min = get_setting("rlm_user_min_chars")
            cap_max = get_setting("rlm_capture_max_chars")
            if content.strip() and len(content) >= usr_min:
                session_manager.append(
                    session_id,
                    "user_message",
                    f"[User]\n{content[:cap_max]}",
                    metadata={"truncated": len(content) > cap_max}
                )
                captured += 1
    
    if captured > 0:
        console.print(f"[dim]  Captured {captured} new entries from {len(new_messages)} new messages[/dim]")


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
- `rlm_summarize(offset, length, focus)` - Get a dense summary of a huge chunk
- `rlm_get_context(offset, length)` - Get an exact extract of context
- `rlm_search(pattern, max_results)` - Search with regex
- `rlm_find(text, max_results)` - Find exact text
- `rlm_stats()` - Get context statistics
- `rlm_get_entries(type)` - List context entries

## Strategy

1. For SMALL contexts (<100K chars): You can read directly with rlm_get_context(0, 100000)
2. For LARGE contexts: 
   - If you don't know the exact keywords to search for, use `rlm_summarize` to quickly skim huge sections.
   - Once you find the right section, use `rlm_search` or `rlm_get_context` to read the exact details.
3. Think about what you're looking for before searching - good patterns save tokens

Remember: You DON'T need to read the entire context. Summarize large chunks or search first, then read relevant sections.
"""

    strict_level = session_manager.get_config().get("strict_mode_level", 0)
    
    if strict_level >= 1:
        prompt += "\n## Strict Mode Guidance\n"
        
        if strict_level == 1:
            prompt += "Note: Your visible conversation history has been truncated. Use your tools if you need older info.\n"
        elif strict_level == 2:
            prompt += "Do not guess past information. Use your tools to search the context lake.\n"
        elif strict_level == 3:
            prompt += """🚨 STRICT MODE 🚨
Your visible conversation history has been heavily truncated. Older messages were deliberately deleted.
If the user references ANY past code, decision, or file that is not in your immediate visible messages, you MUST use `rlm_summarize`, `rlm_search`, or `rlm_get_context` to find it.
DO NOT GUESS. DO NOT SAY YOU DON'T KNOW. USE YOUR TOOLS FIRST.
"""
        elif strict_level >= 4:
            prompt += """🚨 MAXIMUM STRICTNESS 🚨 
When you are over 128k tokens, YOU SUFFER FROM MASSIVE AMNESIA. 
Your visible messages are ONLY the last few turns. The vast majority of the project is hidden from you.
You absolutely MUST use `rlm_summarize` and `rlm_search` repeatedly to understand the history, verify past tasks, and rebuild connections. 
DO NOT TRUST YOUR PRE-TRAINING OR MEMORY. USE THE TOOLS.
"""

    return prompt


def inject_tools(request: ChatCompletionRequest, session_id: str) -> tuple[list[dict], list[dict]]:
    """Inject RLM context tools into the request."""
    messages = []
    
    rlm_system = build_rlm_system_prompt(session_id)
    system_injected = False
    
    for msg in request.messages:
        msg_dict = {"role": msg.role, "content": msg.content}
        
        if msg.role == "system" and not system_injected:
            msg_dict["content"] = msg_dict["content"] + "\n\n" + rlm_system
            system_injected = True
            
        if msg.tool_calls:
            msg_dict["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id
        if msg.name:
            msg_dict["name"] = msg.name
        messages.append(msg_dict)
        
    if not system_injected:
        messages.insert(0, {"role": "system", "content": rlm_system})
    
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


class DashboardManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, event_type: str, data: dict):
        if not self.active_connections:
            return
        message = {"type": event_type, "data": data}
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except Exception:
                pass

dashboard_mgr = DashboardManager()


@app.get("/")
async def root():
    dashboard_path = Path(__file__).parent / "dashboard.html"
    if dashboard_path.exists():
        content = dashboard_path.read_text()
        strict_level = session_manager.get_config().get("strict_mode_level", 0)
        content = content.replace("___STRICT_MODE_TEMPLATE___", str(strict_level))
        return HTMLResponse(content=content, status_code=200)
    return HTMLResponse(content="Dashboard not found", status_code=404)


@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    await dashboard_mgr.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        dashboard_mgr.disconnect(websocket)


@app.get("/health")
async def health():
    strict_level = session_manager.get_config().get("strict_mode_level", 0)
    return {"status": "ok", "version": __version__, "mode": "true-rlm", "strict_level": strict_level}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, x_rlm_session: str | None = Header(default=None)):
    """RLM chat completions with tool-based context access."""
    console.print(f"[bold cyan][API] Request: {request.model}[/bold cyan]")
    console.print(f"[dim]  Messages: {len(request.messages)}, Stream: {request.stream}[/dim]")
    
    if x_rlm_session and session_manager.get_session(x_rlm_session):
        session_id = x_rlm_session
        console.print(f"[cyan]Using provided session header: {session_id}[/cyan]")
    else:
        session_id = get_or_create_session(request.messages)
        
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
    """Stream response with internal RLM tool execution.
    
    Key architecture: rlm_ tool calls are intercepted and executed
    server-side. Only non-rlm responses are streamed to opencode.
    
    Loop:
    1. Query upstream model
    2. If model calls rlm_ tools → execute locally, feed results back, goto 1
    3. If model produces text or non-rlm tool calls → stream to opencode
    """
    chat_id = f"chatcmpl-{uuid4().hex[:8]}"
    created = int(time.time())
    
    console.print(f"[dim cyan][_stream] Starting for {model_id}...[/dim cyan]")
    
    # Get context for rlm tool execution
    context = session_manager.get_context(session_id)
    session = session_manager.get_session(session_id)
    session_entries = [e.__dict__ if hasattr(e, '__dict__') else e for e in (session.entries if session else [])]
    session_stats = session.stats.__dict__ if session and session.stats else None
    
    await dashboard_mgr.broadcast("stream_start", {
        "session_id": session_id,
        "model": model_id,
        "context_chars": len(context) if context else 0,
    })
    
    current_messages = list(messages)
    max_iterations = 10  # Prevent infinite loops
    
    for iteration in range(max_iterations):
        # Collect full response from upstream model
        collected_content = []
        collected_tool_calls: list[dict] = []
        collected_reasoning = []
        finish_reason = None
        
        try:
            async for chunk in provider.stream(
                model_id,
                current_messages,
                temperature=request.temperature or 1.0,
                max_tokens=request.max_tokens,
                tools=tools if tools else None,
            ):
                if chunk.content:
                    collected_content.append(chunk.content)
                if chunk.reasoning:
                    collected_reasoning.append(chunk.reasoning)
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                
                # Accumulate tool calls from streaming deltas
                if chunk.tool_calls:
                    for tc_delta in chunk.tool_calls:
                        idx = tc_delta.get("index", 0)
                        while len(collected_tool_calls) <= idx:
                            collected_tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        
                        tc = collected_tool_calls[idx]
                        if "id" in tc_delta:
                            tc["id"] = tc_delta["id"]
                        if "function" in tc_delta:
                            fn = tc_delta["function"]
                            if "name" in fn:
                                tc["function"]["name"] += fn["name"]
                            if "arguments" in fn:
                                tc["function"]["arguments"] += fn["arguments"]
        except Exception as e:
            console.print(f"[red][_stream] Error: {e}[/red]")
            error_data = {
                "id": chat_id,
                "object": "error",
                "error": {"message": str(e), "type": "stream_error"}
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        full_content = "".join(collected_content)
        full_reasoning = "".join(collected_reasoning)
        
        # Check if any tool calls are rlm_ tools
        rlm_tool_calls = [tc for tc in collected_tool_calls if tc["function"]["name"].startswith("rlm_")]
        non_rlm_tool_calls = [tc for tc in collected_tool_calls if not tc["function"]["name"].startswith("rlm_")]
        
        if not rlm_tool_calls:
            # No rlm_ tool calls — stream everything to opencode
            # Stream reasoning FIRST (model thinks before responding)
            if full_reasoning:
                for i in range(0, len(full_reasoning), 50):
                    chunk_text = full_reasoning[i:i+50]
                    data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"reasoning_content": chunk_text}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            
            # Then stream tool calls (actions execute before text response)
            if non_rlm_tool_calls:
                for i, tc in enumerate(non_rlm_tool_calls):
                    tc_with_index = {**tc, "index": i}
                    data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"tool_calls": [tc_with_index]}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            
            # Finally stream content (the visible response)
            if full_content:
                for i in range(0, len(full_content), 50):
                    chunk_text = full_content[i:i+50]
                    data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            
            # Send finish
            data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason or "stop"}]
            }
            yield f"data: {json.dumps(data)}\n\n"
            
            # Capture thinking into RLM context (opencode strips these from message history)
            if full_reasoning and len(full_reasoning) >= get_setting("rlm_capture_min_chars"):
                session_manager.append(session_id, "thinking", f"[Thinking]\n{full_reasoning[:get_setting('rlm_capture_max_chars')]}")
            # Capture assistant response immediately (not on next request)
            if full_content and len(full_content) > get_setting("rlm_assistant_min_chars"):
                session_manager.append(session_id, "assistant_response", f"[Assistant]\n{full_content[:get_setting('rlm_capture_max_chars')]}")
            
            console.print(f"[green][_stream] Complete (iter {iteration+1}): {len(full_content)} chars, {len(non_rlm_tool_calls)} tool calls[/green]")
            await dashboard_mgr.broadcast("stream_end", {"iteration": iteration + 1})
            yield "data: [DONE]\n\n"
            return
        
        # rlm_ tool calls found — execute locally and loop back
        console.print(f"[cyan][_stream] Iteration {iteration+1}: executing {len(rlm_tool_calls)} rlm tool(s) locally[/cyan]")
        
        # Refresh context (may have been updated)
        context = session_manager.get_context(session_id)
        
        # Build assistant message with tool calls
        assistant_msg = {"role": "assistant", "content": full_content or None, "tool_calls": collected_tool_calls}
        current_messages.append(assistant_msg)
        
        # Execute each rlm_ tool and add results
        for tc in rlm_tool_calls:
            tool_name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            
            summarizer_model = get_setting("rlm_summarize_model")
            summarizer_model = summarizer_model or model_id
            
            result = await execute_context_tool(
                tool_name, args, context,
                session_stats=session_stats,
                session_entries=session_entries,
                provider=provider,
                summarize_model_id=summarizer_model,
                session_id=session_id,
            )
            
            result_text = json.dumps(result.result, indent=2) if result.success else f"Error: {result.error}"
            console.print(f"[dim]  {tool_name}({args}) → {len(result_text)} chars[/dim]")
            
            await dashboard_mgr.broadcast("tool_call", {"tool_name": tool_name})
            
            # Capture the rlm tool invocation into context
            summary = format_tool_result_for_message(result)
            result_text_for_capture = summary if tool_name == "rlm_summarize" else result.result.get("text", "")
            result_formatted = (
                f"[{tool_name}]\nQuery: {json.dumps(args)}\nResult: {summary}\n{result_text_for_capture[:get_setting('rlm_capture_max_chars')]}"
            )
            session_manager.append(session_id, "tool_output", result_formatted, metadata={"tool": tool_name})
            
            current_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result_text,
            })
        
        # If there are non-rlm tool calls mixed with rlm ones,
        # we CANNOT stream them to opencode right now because we are looping back internally.
        # We must tell the LLM that they failed and force it to reissue them.
        for tc in non_rlm_tool_calls:
            current_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": "ERROR: You mixed rlm_ tools and external tools. The external tools were NOT executed. Please re-issue your external tool calls in your next turn.",
            })
    
    # Max iterations reached
    console.print(f"[yellow][_stream] Max iterations ({max_iterations}) reached[/yellow]")
    data = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {"content": "[RLM: Maximum tool iterations reached]"}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(data)}\n\n"
    await dashboard_mgr.broadcast("stream_end", {"iteration": max_iterations})
    yield "data: [DONE]\n\n"


def run_server():
    """Entry point for running the server."""
    import uvicorn
    console.print("[bold blue]Starting RLM-OpenCode Server[/bold blue]")
    console.print("[dim]True RLM with tool-based context access[/dim]")
    uvicorn.run(app, host="0.0.0.0", port=8769)
