"""Anthropic → OpenAI request conversion."""

import json
import logging

log = logging.getLogger(__name__)


def _make_data_url(data: str, media_type: str) -> str:
    """Convert base64 data to a data: URL, handling pre-prefixed data."""
    if "base64" in data:
        data = data.split("base64")[-1]
        if data.startswith(","):
            data = data[1:]
    return f"data:{media_type};base64,{data}"


def _convert_image_source(source: dict) -> dict:
    """Convert an Anthropic image source to OpenAI image_url format."""
    if source.get("type") == "base64":
        url = _make_data_url(source["data"], source["media_type"])
    else:
        url = source.get("url", "")
    return {"type": "image_url", "image_url": {"url": url}}


def _convert_system(system) -> list[dict]:
    """Convert Anthropic system prompt to OpenAI messages."""
    if not system:
        return []
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    # Array form (with cache_control items)
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, str):
                parts.append({"type": "text", "text": block})
            elif isinstance(block, dict) and block.get("type") == "text":
                part = {"type": "text", "text": block["text"]}
                if "cache_control" in block:
                    part["cache_control"] = block["cache_control"]
                parts.append(part)
        return [{"role": "system", "content": parts}]
    return []


def _convert_user_message(msg: dict) -> list[dict]:
    """Convert an Anthropic user message to one or more OpenAI messages.

    Returns a list because tool_result blocks become separate tool messages,
    and images in tool_results become separate user messages.
    """
    content = msg.get("content")

    # Simple string content
    if isinstance(content, str):
        return [{"role": "user", "content": content}]

    if not isinstance(content, list):
        return [{"role": "user", "content": str(content)}]

    result_messages = []
    user_parts = []

    for block in content:
        btype = block.get("type")

        if btype == "tool_result":
            # Extract tool result — separate images from text
            tool_content = block.get("content")
            tool_images = []

            if isinstance(tool_content, str):
                text = tool_content
            elif isinstance(tool_content, list):
                text_parts = []
                for item in tool_content:
                    if isinstance(item, dict) and item.get("type") == "image" and item.get("source"):
                        tool_images.append(item)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    else:
                        text_parts.append(json.dumps(item) if isinstance(item, dict) else str(item))
                text = "\n".join(text_parts) if text_parts else (
                    "[image returned]" if tool_images else json.dumps(tool_content)
                )
            else:
                text = str(tool_content) if tool_content is not None else ""

            tool_msg = {
                "role": "tool",
                "content": text,
                "tool_call_id": block.get("tool_use_id", ""),
            }
            if "cache_control" in block:
                tool_msg["cache_control"] = block["cache_control"]
            result_messages.append(tool_msg)

            # Images from tool_result → separate user message
            if tool_images:
                result_messages.append({
                    "role": "user",
                    "content": [
                        _convert_image_source(img["source"]) for img in tool_images
                    ],
                })

        elif btype == "text":
            user_parts.append({"type": "text", "text": block.get("text", "")})

        elif btype == "image":
            source = block.get("source", {})
            user_parts.append(_convert_image_source(source))

    if user_parts:
        result_messages.append({"role": "user", "content": user_parts})

    return result_messages


def _convert_assistant_message(msg: dict) -> dict:
    """Convert an Anthropic assistant message to OpenAI format."""
    content = msg.get("content")
    result: dict = {"role": "assistant"}

    if isinstance(content, str):
        result["content"] = content
        return result

    if not isinstance(content, list):
        result["content"] = str(content) if content else ""
        return result

    text_parts = []
    tool_calls = []
    thinking = None

    for block in content:
        btype = block.get("type")

        if btype == "text":
            text_parts.append(block.get("text", ""))

        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

        elif btype == "thinking":
            thinking = {
                "content": block.get("thinking", ""),
                "signature": block.get("signature", ""),
            }

    if text_parts:
        result["content"] = "\n".join(text_parts)
    if tool_calls:
        result["tool_calls"] = tool_calls
    if thinking:
        result["thinking"] = thinking

    return result


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI format."""
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return result


def _convert_thinking(thinking: dict) -> dict:
    """Convert Anthropic extended thinking config to OpenAI reasoning params."""
    budget = thinking.get("budget_tokens", 10000)

    # Map budget to effort level
    if budget <= 5000:
        effort = "low"
    elif budget <= 20000:
        effort = "medium"
    else:
        effort = "high"

    return {
        "reasoning": {"effort": effort, "enabled": True, "max_tokens": budget},
        "enable_thinking": True,
        "chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False},
    }


def convert_request(body: dict) -> dict:
    """Convert a full Anthropic /v1/messages request body to OpenAI /v1/chat/completions format."""
    messages = []

    # System prompt
    messages.extend(_convert_system(body.get("system")))

    # Messages
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role == "user":
            messages.extend(_convert_user_message(msg))
        elif role == "assistant":
            messages.append(_convert_assistant_message(msg))

    openai_body: dict = {
        "messages": messages,
        "model": body.get("model", "auto"),
        "stream": True,
        "max_completion_tokens": body.get("max_tokens", 4096),
    }

    # Tools
    tools = body.get("tools")
    if tools:
        openai_body["tools"] = _convert_tools(tools)

    # Temperature
    if "temperature" in body:
        openai_body["temperature"] = body["temperature"]

    # Top-p
    if "top_p" in body:
        openai_body["top_p"] = body["top_p"]

    # Stop sequences
    if "stop_sequences" in body:
        openai_body["stop"] = body["stop_sequences"]

    # Extended thinking
    thinking = body.get("thinking")
    if thinking and thinking.get("type") == "enabled":
        thinking_params = _convert_thinking(thinking)
        openai_body.update(thinking_params)

    return openai_body
