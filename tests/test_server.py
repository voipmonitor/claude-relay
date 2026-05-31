"""Tests for server-side request handling helpers."""

import pytest

from claude_relay.config import ProxyConfig
from claude_relay.server import (
    ContextWindowError,
    _cap_max_completion_tokens,
    _completion_token_margin,
    _context_overflow_retry_from_error,
    _delta_has_visible_output,
    _peek_with_keepalive,
)
from claude_relay.sse import SSEEvent


def test_completion_token_margin_is_fixed_reserve():
    assert _completion_token_margin(local_input_tokens=1) == 128
    assert _completion_token_margin(local_input_tokens=139000) == 128


def test_cap_max_completion_tokens_uses_fixed_margin():
    capped, margin = _cap_max_completion_tokens(
        64000,
        ctx_limit=65536,
        input_tokens=1537,
    )

    assert margin == 128
    assert capped == 63871


def test_cap_max_completion_tokens_keeps_lower_request():
    capped, margin = _cap_max_completion_tokens(
        32000,
        ctx_limit=65536,
        input_tokens=1537,
    )

    assert margin == 128
    assert capped == 32000


def test_cap_max_completion_tokens_caps_to_available_context():
    capped, margin = _cap_max_completion_tokens(
        64000,
        ctx_limit=65536,
        input_tokens=53479,
    )

    assert margin == 128
    assert capped == 11929


def test_cap_max_completion_tokens_rejects_when_input_exceeds_context():
    with pytest.raises(ContextWindowError):
        _cap_max_completion_tokens(
            64000,
            ctx_limit=65536,
            input_tokens=65409,
        )


def test_non_200_retry_detects_completion_token_limit():
    config = ProxyConfig(min_completion_tokens=4096)
    error_body = "max_completion_tokens=250000 cannot be greater than max_model_len=202,752"

    retry = _context_overflow_retry_from_error(error_body, config)

    assert retry is not None
    assert retry.reason == "completion_limit"
    assert retry.ctx_limit == 202752
    assert retry.input_tokens is None
    assert retry.max_completion_tokens == 202652


def test_non_200_retry_uses_available_context_for_completion_token_limit():
    config = ProxyConfig(min_completion_tokens=4096)
    error_body = "max_completion_tokens=250000 cannot be greater than max_model_len=202,752"

    retry = _context_overflow_retry_from_error(error_body, config, estimated_input_tokens=139000)

    assert retry is not None
    assert retry.reason == "completion_limit"
    assert retry.ctx_limit == 202752
    assert retry.input_tokens == 139000
    assert retry.max_completion_tokens == 63652


def test_non_200_retry_detects_vllm_context_limit():
    config = ProxyConfig(min_completion_tokens=4096)
    error_body = (
        "This model's maximum context length is 202752 tokens. "
        "However, you requested 64000 output tokens and your prompt contains 139000 input tokens."
    )

    retry = _context_overflow_retry_from_error(error_body, config)

    assert retry is not None
    assert retry.reason == "context_limit"
    assert retry.ctx_limit == 202752
    assert retry.input_tokens == 139000
    assert not retry.input_tokens_is_lower_bound
    assert retry.max_completion_tokens == 63652


def test_non_200_retry_detects_vllm_at_least_context_limit():
    config = ProxyConfig(min_completion_tokens=4096)
    error_body = (
        "This model's maximum context length is 65536 tokens. "
        "However, you requested 64000 output tokens and your prompt contains at least 1537 input tokens, "
        "for a total of at least 65537 tokens. Please reduce the length of the input prompt or the number "
        "of requested output tokens. (parameter=input_tokens, value=1537)"
    )

    retry = _context_overflow_retry_from_error(error_body, config)

    assert retry is not None
    assert retry.reason == "context_limit"
    assert retry.ctx_limit == 65536
    assert retry.input_tokens == 1537
    assert retry.input_tokens_is_lower_bound
    assert retry.max_completion_tokens == 62975


def test_non_200_retry_uses_larger_margin_for_vllm_at_least_boundary_error():
    config = ProxyConfig(min_completion_tokens=4096)
    error_body = (
        "This model's maximum context length is 262144 tokens. "
        "However, you requested 63798 output tokens and your prompt contains at least 198347 input tokens, "
        "for a total of at least 262145 tokens. Please reduce the length of the input prompt or the number "
        "of requested output tokens. (parameter=input_tokens, value=198347)"
    )

    retry = _context_overflow_retry_from_error(error_body, config)

    assert retry is not None
    assert retry.reason == "context_limit"
    assert retry.ctx_limit == 262144
    assert retry.input_tokens == 198347
    assert retry.input_tokens_is_lower_bound
    assert retry.max_completion_tokens == 62773


def test_non_200_retry_detects_openai_compatible_context_limit():
    config = ProxyConfig(min_completion_tokens=4096)
    error_body = (
        "This model's maximum context length is 202752 tokens. "
        "However, you requested 203000 tokens (139000 in the messages, 64000 in the completion)."
    )

    retry = _context_overflow_retry_from_error(error_body, config)

    assert retry is not None
    assert retry.reason == "context_limit"
    assert retry.ctx_limit == 202752
    assert retry.input_tokens == 139000
    assert retry.max_completion_tokens == 63652


def test_non_200_retry_detects_requested_token_count_context_limit():
    config = ProxyConfig(min_completion_tokens=4096)
    error_body = (
        "Requested token count exceeds the model's maximum context length of 202752 tokens. "
        "You requested a total of 206272 tokens: 142272 tokens from the input messages "
        "and 64000 tokens for the completion. Please reduce the number of tokens in the "
        "input messages or the completion to fit within the limit."
    )

    retry = _context_overflow_retry_from_error(error_body, config)

    assert retry is not None
    assert retry.reason == "context_limit"
    assert retry.ctx_limit == 202752
    assert retry.input_tokens == 142272
    assert retry.max_completion_tokens == 60380


def test_non_200_retry_respects_min_completion_token_floor():
    config = ProxyConfig(min_completion_tokens=4096)
    error_body = (
        "This model's maximum context length is 202752 tokens. "
        "However, you requested 64000 output tokens and your prompt contains 202000 input tokens."
    )

    retry = _context_overflow_retry_from_error(error_body, config)

    assert retry is not None
    assert retry.max_completion_tokens == 4096


def test_non_200_retry_ignores_unrelated_errors():
    config = ProxyConfig(min_completion_tokens=4096)

    retry = _context_overflow_retry_from_error("backend is unavailable", config)

    assert retry is None


def test_reasoning_only_delta_is_not_visible_output():
    assert not _delta_has_visible_output({"reasoning": "I should continue"})


def test_reasoning_content_delta_is_not_visible_output():
    assert not _delta_has_visible_output({"reasoning_content": "I should continue"})


def test_whitespace_content_is_not_visible_output():
    assert not _delta_has_visible_output({"content": "\n\n"})


def test_text_content_is_visible_output():
    assert _delta_has_visible_output({"content": "hello"})


def test_tool_call_delta_is_visible_output():
    assert _delta_has_visible_output({"tool_calls": [{"index": 0}]})


class _DummyResponse:
    async def write(self, chunk: bytes) -> None:
        pass


async def _events(data_chunks: list[str]):
    for chunk in data_chunks:
        yield SSEEvent(data=chunk)


@pytest.mark.asyncio
async def test_peek_buffers_reasoning_only_stream_until_done():
    buffered, has_content = await _peek_with_keepalive(
        _events([
            '{"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}',
            '{"choices":[{"delta":{"reasoning":"hidden"},"finish_reason":null}]}',
            '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
            '[DONE]',
        ]),
        _DummyResponse(),
    )

    assert not has_content
    assert [event.data for event in buffered][-1] == "[DONE]"


@pytest.mark.asyncio
async def test_peek_stops_at_first_tool_call():
    buffered, has_content = await _peek_with_keepalive(
        _events([
            '{"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}',
            '{"choices":[{"delta":{"reasoning":"hidden"},"finish_reason":null}]}',
            '{"choices":[{"delta":{"tool_calls":[{"index":0}]},"finish_reason":null}]}',
            '[DONE]',
        ]),
        _DummyResponse(),
    )

    assert has_content
    assert len(buffered) == 3
