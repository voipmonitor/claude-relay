"""Tests for Anthropic -> OpenAI request conversion."""

from claude_relay.convert_request import convert_request


def test_system_role_in_messages_merged_into_system():
    """Newer Claude Code sends role: system in messages array. Should merge with top-level system."""
    body = {
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hello"},
        ],
    }

    converted = convert_request(body)

    assert converted["messages"] == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hello"},
    ]


def test_system_role_in_messages_merges_after_existing_system():
    """String system + string system msg get collapsed to single string for OpenAI compat."""
    body = {
        "system": "Keep this. ",
        "messages": [
            {"role": "system", "content": "Add this too."},
            {"role": "user", "content": "Hello"},
        ],
    }

    converted = convert_request(body)

    assert converted["messages"] == [
        {"role": "system", "content": "Keep this. Add this too."},
        {"role": "user", "content": "Hello"},
    ]


def test_system_role_in_messages_preserves_user_assistant_order():
    body = {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Tell me more"},
        ],
    }

    converted = convert_request(body)
    roles = [m["role"] for m in converted["messages"]]

    assert roles == ["system", "user", "assistant", "user"]


def test_no_system_role_in_messages_unchanged():
    body = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
    }

    converted = convert_request(body)

    assert len(converted["messages"]) == 2
    assert converted["messages"][0]["role"] == "user"


def test_system_role_in_messages_content_blocks_collapsed_to_string():
    """Pure text blocks from both sources get collapsed to single string."""
    body = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Block system."}]},
            {"role": "user", "content": "Hello"},
        ],
        "system": [{"type": "text", "text": "Existing system. "}],
    }

    converted = convert_request(body)

    assert converted["messages"][0]["role"] == "system"
    assert converted["messages"][0]["content"] == "Existing system. Block system."


def test_multiple_system_messages_with_list_content():
    """Multiple role:system with list content should flatten, not nest."""
    body = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "First block."}]},
            {"role": "system", "content": [{"type": "text", "text": "Second block."}]},
            {"role": "user", "content": "Hi"},
        ],
    }

    converted = convert_request(body)

    assert converted["messages"][0]["role"] == "system"
    assert converted["messages"][0]["content"] == "First block.Second block."


def test_system_message_cache_control_preserves_blocks():
    """cache_control on system blocks prevents string collapse."""
    body = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Long base prompt.", "cache_control": {"type": "ephemeral"}},
                ],
            },
            {"role": "user", "content": "Hello"},
        ],
    }

    converted = convert_request(body)

    assert converted["messages"][0]["role"] == "system"
    assert converted["messages"][0]["content"] == [
        {"type": "text", "text": "Long base prompt.", "cache_control": {"type": "ephemeral"}},
    ]


def test_tool_choice_tool_converts_to_openai_function_choice():
    body = {
        "messages": [{"role": "user", "content": "Use the tool."}],
        "tools": [{
            "name": "get_weather",
            "description": "Get weather.",
            "input_schema": {"type": "object"},
        }],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }

    converted = convert_request(body)

    assert converted["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


def test_output_config_high_maps_to_reasoning_effort():
    body = {
        "messages": [{"role": "user", "content": "Hello"}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }

    converted = convert_request(body)

    assert converted["chat_template_kwargs"] == {
        "reasoning_effort": "high",
        "enable_thinking": True,
    }


def test_output_config_max_maps_to_reasoning_effort_max():
    body = {
        "messages": [{"role": "user", "content": "Hello"}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "max"},
    }

    converted = convert_request(body)

    assert converted["chat_template_kwargs"] == {
        "reasoning_effort": "max",
        "enable_thinking": True,
    }


def test_output_config_low_maps_to_reasoning_effort():
    body = {
        "messages": [{"role": "user", "content": "Hello"}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"},
    }

    converted = convert_request(body)

    assert converted["chat_template_kwargs"] == {
        "reasoning_effort": "low",
        "enable_thinking": True,
    }


def test_output_config_ignored_when_thinking_enabled():
    """When thinking.type == 'enabled', output_config is ignored (budget path wins)."""
    body = {
        "messages": [{"role": "user", "content": "Hello"}],
        "thinking": {"type": "enabled", "budget_tokens": 10000},
        "output_config": {"effort": "low"},
    }

    converted = convert_request(body)

    # Should use budget-derived medium effort, not low from output_config
    assert converted["chat_template_kwargs"]["reasoning_effort"] == "medium"
    assert converted["chat_template_kwargs"]["enable_thinking"] is True


def test_output_config_ignored_when_thinking_not_active():
    """No thinking.active or adaptive → output_config does NOT inject reasoning_effort."""
    body = {
        "messages": [{"role": "user", "content": "Hello"}],
        "output_config": {"effort": "high"},
    }

    converted = convert_request(body)

    assert "chat_template_kwargs" not in converted


def test_output_config_ignored_without_effort_field():
    body = {
        "messages": [{"role": "user", "content": "Hello"}],
        "thinking": {"type": "adaptive"},
        "output_config": {"other": "stuff"},
    }

    converted = convert_request(body)

    assert "chat_template_kwargs" not in converted


def test_no_output_config_no_chat_template_kwargs():
    body = {"messages": [{"role": "user", "content": "Hello"}]}

    converted = convert_request(body)

    assert "chat_template_kwargs" not in converted


def test_tool_choice_any_converts_to_required():
    body = {
        "messages": [{"role": "user", "content": "Use a tool."}],
        "tools": [{
            "name": "get_weather",
            "description": "Get weather.",
            "input_schema": {"type": "object"},
        }],
        "tool_choice": {"type": "any"},
    }

    converted = convert_request(body)

    assert converted["tool_choice"] == "required"
