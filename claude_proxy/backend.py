import json
import time
import logging

import aiohttp

from .config import ProxyConfig

log = logging.getLogger(__name__)


DEFAULT_CONTEXT_LIMIT = 131072  # 128k fallback if backend doesn't report


class BackendState:
    def __init__(self, ttl: int = 30):
        self.model: str | None = None
        self.backend_type: str | None = None  # "sglang" or "vllm"
        self.context_limit: int = DEFAULT_CONTEXT_LIMIT
        self.last_check: float = 0
        self.ttl = ttl

    @property
    def stale(self) -> bool:
        return self.model is None or (time.time() - self.last_check) >= self.ttl

    def info(self) -> dict:
        return {
            "model": self.model,
            "backend_type": self.backend_type,
            "context_limit": self.context_limit,
            "last_check_ago": f"{time.time() - self.last_check:.0f}s"
            if self.last_check
            else "never",
        }


_state = BackendState()


async def detect_backend(session: aiohttp.ClientSession, backend_url: str) -> BackendState:
    global _state
    if not _state.stale:
        return _state

    try:
        async with session.get(
            f"{backend_url}/v1/models",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            data = await resp.json()
            model_info = data["data"][0]
            _state.model = model_info["id"]
            owned_by = model_info.get("owned_by", "").lower()

            # Detect backend type and context limit from model info
            max_model_len = model_info.get("max_model_len")  # vLLM includes this

            if "sglang" in owned_by:
                _state.backend_type = "sglang"
            elif "vllm" in owned_by:
                _state.backend_type = "vllm"
            else:
                # Fallback: try SGLang-specific endpoint
                try:
                    async with session.get(
                        f"{backend_url}/get_model_info",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as r2:
                        _state.backend_type = "sglang" if r2.status == 200 else "vllm"
                except Exception:
                    _state.backend_type = "vllm"

            # Query context limit from backend-specific endpoints
            if max_model_len:
                _state.context_limit = int(max_model_len)
            elif _state.backend_type == "sglang":
                try:
                    async with session.get(
                        f"{backend_url}/get_model_info",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as r2:
                        if r2.status == 200:
                            info = await r2.json()
                            ctx = info.get("max_total_num_tokens") or info.get("context_length")
                            if ctx:
                                _state.context_limit = int(ctx)
                except Exception:
                    pass  # keep default

            _state.last_check = time.time()
            log.info(
                "backend=%s model=%s context_limit=%d (cached %ds)",
                _state.backend_type,
                _state.model,
                _state.context_limit,
                _state.ttl,
            )
    except Exception as e:
        log.warning("backend probe failed: %s, using cached values", e)
        if not _state.model:
            _state.model = "default"
            _state.backend_type = "vllm"

    return _state


def init_state(ttl: int):
    global _state
    _state = BackendState(ttl=ttl)


def get_state() -> BackendState:
    return _state


async def send_to_backend(
    session: aiohttp.ClientSession,
    config: ProxyConfig,
    openai_body: dict,
    req_id: str = "",
) -> aiohttp.ClientResponse:
    """Detect backend, remap model, inject sglang kwargs, POST to backend.

    Returns the raw aiohttp response (caller must consume it).
    """
    _r = f"[{req_id}] " if req_id else ""
    state = await detect_backend(session, config.backend_url)

    openai_body["model"] = state.model

    if state.backend_type == "sglang":
        kwargs = openai_body.setdefault("chat_template_kwargs", {})
        kwargs["enable_thinking"] = True
        kwargs["clear_thinking"] = False

    # Request usage stats in streaming mode so we can report token counts to Claude Code
    if openai_body.get("stream"):
        openai_body["stream_options"] = {"include_usage": True}

    url = f"{config.backend_url}/v1/chat/completions"
    num_msgs = len(openai_body.get("messages", []))
    num_tools = len(openai_body.get("tools", []))
    log.info("%sbackend: POST %s model=%s msgs=%d tools=%d max_tokens=%s stream=%s",
             _r, url, openai_body.get("model"), num_msgs, num_tools,
             openai_body.get("max_completion_tokens"), openai_body.get("stream"))

    resp = await session.post(
        url,
        json=openai_body,
        timeout=aiohttp.ClientTimeout(total=config.request_timeout),
    )
    log.info("%sbackend: response status=%d content_type=%s", _r, resp.status, resp.content_type)
    return resp
