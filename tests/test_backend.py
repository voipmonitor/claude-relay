"""Tests for backend request routing."""

from types import SimpleNamespace

import pytest

from claude_relay import backend as backend_module
from claude_relay.backend import send_to_backend
from claude_relay.config import ModelRoute, ProxyConfig


class FakeSession:
    def __init__(self):
        self.posts = []

    async def post(self, url, **kwargs):
        self.posts.append({"url": url, **kwargs})
        return SimpleNamespace(status=200, content_type="text/event-stream")


@pytest.mark.asyncio
async def test_send_to_backend_uses_configured_model_route(monkeypatch):
    calls = []

    async def fake_detect_backend(session, backend_url):
        calls.append(backend_url)
        return SimpleNamespace(model="detected-model", backend_type="vllm")

    monkeypatch.setattr(backend_module, "detect_backend", fake_detect_backend)

    session = FakeSession()
    config = ProxyConfig(
        backend_url="http://default:8000",
        model_routes={
            "claude-opus": ModelRoute(
                backend_url="http://opus:8000",
                upstream_model="Kimi-K2.6",
            ),
        },
    )
    body = {"model": "claude-opus", "messages": [], "stream": True}

    await send_to_backend(session, config, body)

    assert calls == ["http://opus:8000"]
    assert session.posts[0]["url"] == "http://opus:8000/v1/chat/completions"
    assert session.posts[0]["json"]["model"] == "Kimi-K2.6"
    assert session.posts[0]["json"]["stream_options"] == {"include_usage": True}
    assert body == {"model": "claude-opus", "messages": [], "stream": True}


@pytest.mark.asyncio
async def test_send_to_backend_default_route_uses_detected_model(monkeypatch):
    async def fake_detect_backend(session, backend_url):
        return SimpleNamespace(model="detected-default", backend_type="vllm")

    monkeypatch.setattr(backend_module, "detect_backend", fake_detect_backend)

    session = FakeSession()
    config = ProxyConfig(backend_url="http://default:8000")

    await send_to_backend(session, config, {"model": "claude-sonnet", "messages": []})

    assert session.posts[0]["url"] == "http://default:8000/v1/chat/completions"
    assert session.posts[0]["json"]["model"] == "detected-default"


@pytest.mark.asyncio
async def test_send_to_backend_injects_sampling_from_detected_model(monkeypatch):
    async def fake_detect_backend(session, backend_url):
        return SimpleNamespace(model="deepseek-v4-flash", backend_type="vllm")

    monkeypatch.setattr(backend_module, "detect_backend", fake_detect_backend)

    session = FakeSession()
    config = ProxyConfig(
        backend_url="http://default:8000",
        model_sampling={
            "deepseek-v4-flash": {"temperature": 1.0, "top_p": 1.0},
        },
    )
    body = {"model": "claude-opus", "messages": [], "stream": True}

    await send_to_backend(session, config, body)

    assert session.posts[0]["json"]["temperature"] == 1.0
    assert session.posts[0]["json"]["top_p"] == 1.0
    # Params not in model_sampling should not appear
    assert "top_k" not in session.posts[0]["json"]
    assert "presence_penalty" not in session.posts[0]["json"]
    assert body == {"model": "claude-opus", "messages": [], "stream": True}


@pytest.mark.asyncio
async def test_send_to_backend_client_params_beat_sampling_defaults(monkeypatch):
    async def fake_detect_backend(session, backend_url):
        return SimpleNamespace(model="deepseek-v4-flash", backend_type="vllm")

    monkeypatch.setattr(backend_module, "detect_backend", fake_detect_backend)

    session = FakeSession()
    config = ProxyConfig(
        backend_url="http://default:8000",
        model_sampling={
            "deepseek-v4-flash": {"temperature": 1.0, "top_p": 1.0},
        },
    )
    # Client already set temperature — should not be overridden
    body = {"model": "claude-opus", "messages": [], "temperature": 0.7, "stream": True}

    await send_to_backend(session, config, body)

    assert session.posts[0]["json"]["temperature"] == 0.7
    assert session.posts[0]["json"]["top_p"] == 1.0
