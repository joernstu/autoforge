"""
LiteLLM Proxy Router
=====================

Anthropic-to-OpenAI translation proxy for providers (like IONOS) that only
support the OpenAI /v1/chat/completions format.

The Claude Code CLI sends requests in Anthropic format (/v1/messages).
This router accepts those requests, translates them to OpenAI format via
litellm, and returns Anthropic-format responses (including SSE streaming).

Flow:
  Claude CLI -> POST /proxy/v1/messages (Anthropic format)
             -> litellm.acompletion (OpenAI format to IONOS)
             <- OpenAI response translated back to Anthropic format
"""

import json
import logging
import os
import sys
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Ensure root is on sys.path for registry import
from ..services.chat_constants import ROOT_DIR

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from registry import API_PROVIDERS, get_all_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/proxy", tags=["proxy"])

# Compatibility preamble injected before Claude Code's system prompt.
# Claude Code's system prompt is optimised for Claude and may confuse other
# models (e.g. Llama) into returning "More information needed" instead of
# using the available tools.  This preamble overrides that tendency.
_COMPAT_PREAMBLE = (
    "You are a capable AI coding assistant. "
    "You have access to tools such as Bash (for running shell commands), "
    "Read/Write/Edit (for file operations), and various MCP tools. "
    "IMPORTANT: Always complete the requested task by calling the available tools "
    "directly. Do NOT ask for clarification or additional information – proceed "
    "immediately using the tools provided. "
    "If a task requires reading a file, call the Read tool. "
    "If a task requires running a command, call the Bash tool. "
    "If a task requires creating features, call the MCP feature tools. "
    "Think step-by-step and use tools to accomplish every goal.\n\n"
)

# Suppress HuggingFace tokenizer download warnings on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Prevent litellm from downloading tokenizers for token counting
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# Suppress noisy litellm debug output
try:
    import litellm
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    # Use a simple token counter to avoid HuggingFace downloads
    litellm.token_counter = None  # type: ignore[assignment]
except ImportError:
    litellm = None  # type: ignore[assignment]
    logger.warning("litellm not installed - proxy endpoints will return 501")


def _count_tokens_estimate(body: dict[str, Any]) -> int:
    """Rough token count estimate: total characters / 4."""
    total_chars = 0
    system = body.get("system", "")
    if isinstance(system, str):
        total_chars += len(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                total_chars += len(block.get("text", ""))
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(block.get("text", "")) + len(json.dumps(block.get("input", "")))
    return max(1, total_chars // 4)


# =============================================================================
# Anthropic -> OpenAI Translation
# =============================================================================

def _translate_system_prompt(system: Any) -> str:
    """Convert Anthropic system field to a plain text string.

    The system field can be a string or a list of content blocks
    (e.g. [{"type": "text", "text": "..."}]).
    """
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(system)


def _translate_messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic message format to OpenAI message format.

    Handles:
    - String content pass-through
    - Text content blocks flattened to string
    - tool_use blocks -> assistant tool_calls
    - tool_result blocks -> tool role messages
    """
    oai_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        # Simple string content - pass through
        if isinstance(content, str):
            oai_messages.append({"role": role, "content": content})
            continue

        # Content is a list of blocks
        if isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []

            for block in content:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type")

                if block_type == "text":
                    text_parts.append(block.get("text", ""))

                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })

                elif block_type == "tool_result":
                    # Extract text content from tool_result
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_parts = []
                        for rc in result_content:
                            if isinstance(rc, dict) and rc.get("type") == "text":
                                result_parts.append(rc.get("text", ""))
                            elif isinstance(rc, str):
                                result_parts.append(rc)
                        result_content = "\n".join(result_parts)
                    elif not isinstance(result_content, str):
                        result_content = str(result_content)

                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": result_content,
                    })

            # Build the OpenAI message for this Anthropic message
            if role == "assistant" and tool_calls:
                # Assistant message with tool calls
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                }
                if text_parts:
                    assistant_msg["content"] = "\n".join(text_parts)
                else:
                    assistant_msg["content"] = None
                oai_messages.append(assistant_msg)
            elif tool_results:
                # User message containing tool_result blocks becomes separate tool messages.
                # If there is also text content, emit the text as a user message first.
                if text_parts:
                    oai_messages.append({"role": role, "content": "\n".join(text_parts)})
                for tr in tool_results:
                    oai_messages.append(tr)
            elif text_parts:
                oai_messages.append({"role": role, "content": "\n".join(text_parts)})
            else:
                # Fallback: empty content
                oai_messages.append({"role": role, "content": ""})
        else:
            # Unexpected content type - pass through as-is
            oai_messages.append({"role": role, "content": str(content) if content else ""})

    return oai_messages


def _translate_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert Anthropic tool definitions to OpenAI function tool format.

    Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI:    {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    if not tools:
        return None

    oai_tools = []
    for tool in tools:
        oai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return oai_tools


def _translate_tool_choice(tool_choice: Any) -> Any:
    """Convert Anthropic tool_choice to OpenAI format.

    Anthropic: {"type": "auto"} | {"type": "any"} | {"type": "tool", "name": "..."}
    OpenAI:    "auto" | "required" | {"type": "function", "function": {"name": "..."}}
    """
    if tool_choice is None:
        return None

    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "tool":
            return {
                "type": "function",
                "function": {"name": tool_choice.get("name", "")},
            }

    return None


# =============================================================================
# OpenAI -> Anthropic Translation
# =============================================================================

def _translate_finish_reason(reason: str | None) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(reason or "", "end_turn")


def _build_anthropic_response(oai_response: Any, model: str) -> dict[str, Any]:
    """Convert a non-streaming OpenAI response to Anthropic message format."""
    choice = oai_response.choices[0] if oai_response.choices else None
    message = choice.message if choice else None

    content_blocks: list[dict[str, Any]] = []

    if message:
        # Text content
        if message.content:
            content_blocks.append({"type": "text", "text": message.content})

        # Tool calls
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    input_data = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    input_data = {}

                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id or f"toolu_{uuid.uuid4().hex[:12]}",
                    "name": tc.function.name,
                    "input": input_data,
                })

    # Usage
    usage = oai_response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    stop_reason = _translate_finish_reason(
        choice.finish_reason if choice else None
    )

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


# =============================================================================
# Streaming: OpenAI SSE -> Anthropic SSE
# =============================================================================

async def _stream_anthropic_sse(oai_stream: Any, model: str):
    """Consume an OpenAI async streaming response and yield Anthropic SSE events.

    Emits the full Anthropic streaming protocol:
      message_start -> content_block_start -> ping -> content_block_delta* ->
      content_block_stop -> message_delta -> message_stop
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    output_tokens = 0
    block_index = 0
    # Track whether we have started a content block
    current_block_type: str | None = None
    # Accumulate tool call data per index from the OpenAI stream
    tool_call_accumulators: dict[int, dict[str, Any]] = {}
    # For diagnostic logging: capture first 300 chars of text + tool names
    _log_text_preview: list[str] = []
    _log_tool_names: list[str] = []

    def sse(event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    # Emit message_start
    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    yield sse("ping", {"type": "ping"})

    try:
        async for chunk in oai_stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            # Handle text content deltas
            if delta and getattr(delta, "content", None):
                if sum(len(s) for s in _log_text_preview) < 300:
                    _log_text_preview.append(delta.content)
                if current_block_type != "text":
                    # Close previous block if any
                    if current_block_type is not None:
                        yield sse("content_block_stop", {
                            "type": "content_block_stop",
                            "index": block_index,
                        })
                        block_index += 1

                    # Start new text block
                    current_block_type = "text"
                    yield sse("content_block_start", {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "text", "text": ""},
                    })

                yield sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": delta.content},
                })
                output_tokens += 1  # Rough approximation

            # Handle tool call deltas
            if delta and getattr(delta, "tool_calls", None):
                for tc_delta in delta.tool_calls:
                    tc_idx = tc_delta.index if hasattr(tc_delta, "index") else 0

                    if tc_idx not in tool_call_accumulators:
                        # Close previous block if needed
                        if current_block_type is not None:
                            yield sse("content_block_stop", {
                                "type": "content_block_stop",
                                "index": block_index,
                            })
                            block_index += 1

                        # Initialize accumulator for this tool call
                        tool_id = (
                            getattr(tc_delta, "id", None)
                            or f"toolu_{uuid.uuid4().hex[:12]}"
                        )
                        tool_name = ""
                        if hasattr(tc_delta, "function") and tc_delta.function:
                            tool_name = getattr(tc_delta.function, "name", "") or ""

                        tool_call_accumulators[tc_idx] = {
                            "id": tool_id,
                            "name": tool_name,
                            "arguments": "",
                            "block_index": block_index,
                        }

                        current_block_type = "tool_use"
                        _log_tool_names.append(tool_name)
                        yield sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": tool_name,
                                "input": {},
                            },
                        })

                    # Accumulate arguments
                    acc = tool_call_accumulators[tc_idx]
                    if hasattr(tc_delta, "function") and tc_delta.function:
                        fn = tc_delta.function
                        # Update name if provided in this delta
                        if getattr(fn, "name", None):
                            acc["name"] = fn.name
                        # Append argument fragment
                        arg_fragment = getattr(fn, "arguments", None) or ""
                        if arg_fragment:
                            acc["arguments"] += arg_fragment
                            yield sse("content_block_delta", {
                                "type": "content_block_delta",
                                "index": acc["block_index"],
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": arg_fragment,
                                },
                            })

            # Handle finish
            if finish_reason:
                # Close current block
                if current_block_type is not None:
                    yield sse("content_block_stop", {
                        "type": "content_block_stop",
                        "index": block_index,
                    })

                stop_reason = _translate_finish_reason(finish_reason)
                logger.info(
                    "Proxy ← %s | stream done | stop=%s tool_calls=%s text_preview=%r",
                    model, stop_reason, _log_tool_names or False,
                    "".join(_log_text_preview)[:300],
                )

                yield sse("message_delta", {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": output_tokens},
                })

                yield sse("message_stop", {"type": "message_stop"})
                return

    except Exception as e:
        logger.error("Error during proxy streaming: %s", e, exc_info=True)
        # Ensure we close cleanly even on error
        if current_block_type is not None:
            yield sse("content_block_stop", {
                "type": "content_block_stop",
                "index": block_index,
            })
        yield sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        })
        yield sse("message_stop", {"type": "message_stop"})

    # If stream ended without a finish_reason, close gracefully
    if current_block_type is not None:
        yield sse("content_block_stop", {
            "type": "content_block_stop",
            "index": block_index,
        })
    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield sse("message_stop", {"type": "message_stop"})


# =============================================================================
# Route Handler
# =============================================================================

@router.post("/v1/messages")
async def proxy_messages(request: Request):
    """Accept an Anthropic /v1/messages request and proxy it via litellm to an OpenAI-compatible provider.

    This endpoint enables Claude Code CLI to work with providers like IONOS
    that only support the OpenAI chat completions format.
    """
    if litellm is None:
        return JSONResponse(
            status_code=501,
            content={"error": {"type": "not_implemented", "message": "litellm is not installed. Run: pip install litellm"}},
        )

    # Parse the Anthropic request body
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": {"type": "invalid_request", "message": "Invalid JSON body"}},
        )

    # Read provider settings from the registry
    all_settings = get_all_settings()
    provider_id = all_settings.get("api_provider", "claude")
    provider = API_PROVIDERS.get(provider_id)

    if not provider:
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "configuration_error", "message": f"Unknown provider: {provider_id}"}},
        )

    api_base = all_settings.get("api_base_url") or provider.get("base_url", "")
    auth_token = all_settings.get("api_auth_token", "")
    model = body.get("model") or all_settings.get("api_model") or provider.get("default_model", "")
    stream = body.get("stream", False)

    if not api_base:
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "configuration_error", "message": "No API base URL configured for provider"}},
        )

    # Build OpenAI messages from Anthropic format
    oai_messages: list[dict[str, Any]] = []

    # System prompt – prepend compatibility preamble so non-Claude models
    # (e.g. Llama) understand they must use tools instead of asking for info.
    system = body.get("system")
    system_text = _translate_system_prompt(system) if system else ""
    oai_messages.append({"role": "system", "content": _COMPAT_PREAMBLE + system_text})

    # User/assistant messages
    anthropic_messages = body.get("messages", [])
    oai_messages.extend(_translate_messages_to_openai(anthropic_messages))

    # Tools
    oai_tools = _translate_tools_to_openai(body.get("tools"))
    tool_choice = _translate_tool_choice(body.get("tool_choice"))

    # Parameters
    max_tokens = body.get("max_tokens", 4096)
    temperature = body.get("temperature", 1.0)
    # For tool-calling requests lower temperature yields more reliable tool use.
    # Claude defaults to 1.0 but non-Claude models work better with ≤ 0.6.
    if oai_tools and temperature > 0.6:
        temperature = 0.6
    top_p = body.get("top_p")

    # Build litellm call kwargs
    call_kwargs: dict[str, Any] = {
        "model": f"openai/{model}",
        "messages": oai_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "api_base": api_base,
        "api_key": auth_token,
        "stream": stream,
    }

    if top_p is not None:
        call_kwargs["top_p"] = top_p

    if oai_tools:
        call_kwargs["tools"] = oai_tools
    if tool_choice is not None and oai_tools:
        call_kwargs["tool_choice"] = tool_choice
    elif oai_tools and tool_choice is None:
        # Explicitly request tool use when tools are provided but no choice was set.
        call_kwargs["tool_choice"] = "auto"

    logger.info(
        "Proxy → %s | model=%s stream=%s msgs=%d tools=%d sys_len=%d temp=%.2f",
        provider_id, model, stream,
        len(oai_messages), len(oai_tools or []),
        len(system_text), temperature,
    )

    try:
        response = await litellm.acompletion(**call_kwargs)
    except Exception as e:
        logger.error("LiteLLM proxy error: %s", e, exc_info=True)
        error_msg = str(e)
        # Try to extract a more useful error message
        if "AuthenticationError" in error_msg or "401" in error_msg:
            error_msg = f"Authentication failed with {provider_id}. Check your API key."
        return JSONResponse(
            status_code=502,
            content={"error": {"type": "proxy_error", "message": error_msg}},
        )

    if stream:
        logger.info("Proxy ← %s | streaming response started", model)
        return StreamingResponse(
            _stream_anthropic_sse(response, model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        choice = response.choices[0] if response.choices else None
        if choice and choice.message:
            resp_preview = (choice.message.content or "")[:300]
            has_tools = bool(getattr(choice.message, "tool_calls", None))
            logger.info(
                "Proxy ← %s | stop=%s tool_calls=%s preview=%r",
                model, choice.finish_reason, has_tools, resp_preview,
            )
        anthropic_response = _build_anthropic_response(response, model)
        return JSONResponse(content=anthropic_response)


@router.post("/v1/messages/count_tokens")
async def proxy_count_tokens(request: Request):
    """Return a token count estimate for an Anthropic-format message batch.

    Called by Claude Code CLI before/after API calls (with ?beta=true).
    Returns a rough estimate (characters / 4) to avoid downloading tokenizers.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"type": "invalid_request", "message": "Invalid JSON body"}})

    return JSONResponse(content={"input_tokens": _count_tokens_estimate(body)})
