"""OpenAI-compatible provider for RLM-Native.

Handles providers that use OpenAI's API format:
- OpenAI
- OpenRouter
- Most custom providers
"""

import json
from typing import AsyncGenerator

import httpx

from rlm_opencode.providers.base import BaseProvider, StreamChunk


class OpenAICompatibleProvider(BaseProvider):
    """Provider for OpenAI-compatible APIs."""
    
    async def stream(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream completion using OpenAI-compatible API."""
        
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError(f"No API key found for provider {self.provider_id}")
        
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        # Add custom headers from config
        custom_headers = self.config.get("options", {}).get("headers", {})
        headers.update(custom_headers)
        
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        if tools:
            payload["tools"] = tools
        
        # Add any extra options
        payload.update(kwargs)
        
        import sys
        import asyncio
        
        max_retries = 10
        base_backoff = 2.0
        
        print(f"[OpenAICompat] POST {url} model={model_id}", file=sys.stderr, flush=True)
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    async with client.stream(
                        "POST",
                        url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        print(f"[OpenAICompat] Response status: {response.status_code}", file=sys.stderr, flush=True)
                        
                        if response.status_code >= 500 or response.status_code == 429:
                            error_text = await response.aread()
                            error_msg = f"API error {response.status_code}: {error_text.decode()}"
                            if attempt < max_retries - 1:
                                wait_time = base_backoff * (2 ** attempt)
                                print(f"[OpenAICompat] Transient error ({response.status_code}). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})", file=sys.stderr, flush=True)
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                print(f"[OpenAICompat] Error: {error_text.decode()}", file=sys.stderr, flush=True)
                                raise Exception(error_msg)
                        elif response.status_code != 200:
                            # 4xx errors are usually fatal (bad request, unauthorized) so we don't retry
                            error_text = await response.aread()
                            print(f"[OpenAICompat] Fatal Error {response.status_code}: {error_text.decode()}", file=sys.stderr, flush=True)
                            raise Exception(f"API error {response.status_code}: {error_text.decode()}")
                        
                        line_count = 0
                        chunk_count = 0
                        async for line in response.aiter_lines():
                            line_count += 1
                            if not line:
                                continue
                            
                            if line.startswith("data: "):
                                data_str = line[6:]
                                
                                if data_str == "[DONE]":
                                    print(f"[OpenAICompat] Stream done: {line_count} lines, {chunk_count} chunks", file=sys.stderr, flush=True)
                                    return
                                
                                try:
                                    data = json.loads(data_str)
                                except json.JSONDecodeError:
                                    continue
                                
                                choices = data.get("choices", [])
                                if not choices:
                                    continue
                                
                                delta = choices[0].get("delta", {})
                                finish_reason = choices[0].get("finish_reason")
                                
                                content = delta.get("content", "")
                                tool_calls = delta.get("tool_calls")
                                reasoning = delta.get("reasoning_content", "") or delta.get("reasoning", "")
                                
                                if content or tool_calls or finish_reason or reasoning:
                                    chunk_count += 1
                                    yield StreamChunk(
                                        content=content,
                                        finish_reason=finish_reason,
                                        tool_calls=tool_calls,
                                        reasoning=reasoning,
                                    )
                        
                        print(f"[OpenAICompat] Stream ended: {line_count} lines, {chunk_count} chunks", file=sys.stderr, flush=True)
                        return # Subroutine complete safely without exception.
            except httpx.RequestError as e:
                # Catch network level issues: ReadTimeout, ConnectTimeout, NetworkError
                if attempt < max_retries - 1:
                    wait_time = base_backoff * (2 ** attempt)
                    print(f"[OpenAICompat] Network request error: {e}. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})", file=sys.stderr, flush=True)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"[OpenAICompat] Fatal Network Error after {max_retries} attempts: {e}", file=sys.stderr, flush=True)
                    raise Exception(f"Network error: {e}")
