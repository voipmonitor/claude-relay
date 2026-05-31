import json
import time
import logging
from copy import deepcopy

import aiohttp

from .config import BackendTarget, ProxyConfig

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


_states: dict[str, BackendState] = {}
_ttl = 30


async def detect_backend(session: aiohttp.ClientSession, backend_url: str) -> BackendState:
    state = _states.get(backend_url)
    if state is None:
        state = BackendState(ttl=_ttl)
        _states[backend_url] = state

    if not state.stale:
        return state

    try:
        async with session.get(
            f"{backend_url}/v1/models",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            data = await resp.json()
            model_info = data["data"][0]
            state.model = model_info["id"]
            owned_by = model_info.get("owned_by", "").lower()

            # Detect backend type and context limit from model info
            max_model_len = model_info.get("max_model_len")  # vLLM includes this

            if "sglang" in owned_by:
                state.backend_type = "sglang"
            elif "vllm" in owned_by:
                state.backend_type = "vllm"
            else:
                # Fallback: try SGLang-specific endpoint
                try:
                    async with session.get(
                        f"{backend_url}/get_model_info",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as r2:
                        state.backend_type = "sglang" if r2.status == 200 else "vllm"
                except Exception:
                    state.backend_type = "vllm"

            # Query context limit from backend-specific endpoints
            if max_model_len:
                state.context_limit = int(max_model_len)
            elif state.backend_type == "sglang":
                try:
                    async with session.get(
                        f"{backend_url}/get_model_info",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as r2:
                        if r2.status == 200:
                            info = await r2.json()
                            ctx = info.get("max_total_num_tokens") or info.get("context_length")
                            if ctx:
                                state.context_limit = int(ctx)
                except Exception:
                    pass  # keep default

            state.last_check = time.time()
            log.info(
                "backend_url=%s backend=%s model=%s context_limit=%d (cached %ds)",
                backend_url,
                state.backend_type,
                state.model,
                state.context_limit,
                state.ttl,
            )
    except Exception as e:
        log.warning("backend probe failed for %s: %s, using cached values", backend_url, e)
        if not state.model:
            state.model = "default"
            state.backend_type = "vllm"

    return state


def init_state(ttl: int):
    global _states, _ttl
    _ttl = ttl
    _states = {}


def get_state(backend_url: str | None = None) -> BackendState:
    if backend_url is not None:
        return _states.setdefault(backend_url, BackendState(ttl=_ttl))
    if _states:
        return next(iter(_states.values()))
    return BackendState(ttl=_ttl)


def get_states_info() -> dict[str, dict]:
    return {backend_url: state.info() for backend_url, state in _states.items()}


async def send_to_backend(
    session: aiohttp.ClientSession,
    config: ProxyConfig,
    openai_body: dict,
    req_id: str = "",
    backend_target: BackendTarget | None = None,
) -> aiohttp.ClientResponse:
    """Detect backend, remap model, inject sglang kwargs, POST to backend.

    Returns the raw aiohttp response (caller must consume it).
    """
    _r = f"[{req_id}] " if req_id else ""
    request_model = str(openai_body.get("model") or "auto")
    target = backend_target or config.resolve_backend(request_model)
    state = await detect_backend(session, target.backend_url)

    backend_body = deepcopy(openai_body)
    backend_body["model"] = target.upstream_model or state.model

    # Apply per-model sampling defaults from detected upstream model name
    # Only fills params not already set by client (setdefault).
    sampling = config.model_sampling.get(state.model)
    if sampling:
        for k, v in sampling.items():
            backend_body.setdefault(k, v)
        log.info("%ssampling: %s (detected model=%s)", _r, sampling, state.model)

    # When thinking is active: use route/effort default. When inactive: send "none"
    # so vLLM skips reasoning compute entirely. This changes the prompt slightly
    # (costing a KV cache refill on toggle) but avoids burning compute on hidden reasoning.
    thinking_active = backend_body.pop("_thinking_active", False)
    re = target.reasoning_effort
    kwargs = backend_body.setdefault("chat_template_kwargs", {})
    if thinking_active:
        kwargs.setdefault("reasoning_effort", re or "max")
        kwargs.setdefault("enable_thinking", True)
    else:
        kwargs.setdefault("reasoning_effort", "none")
    log.info("%sreasoning_effort: %s (route=%s, active=%s)", _r, kwargs["reasoning_effort"], target.route_name, thinking_active)

    # Request usage stats in streaming mode so we can report token counts to Claude Code
    if backend_body.get("stream"):
        backend_body["stream_options"] = {"include_usage": True}

    url = f"{target.backend_url}/v1/chat/completions"
    num_msgs = len(backend_body.get("messages", []))
    num_tools = len(backend_body.get("tools", []))
    log.info(
        "%sbackend: POST %s route=%s request_model=%s model=%s msgs=%d tools=%d max_tokens=%s stream=%s",
        _r, url, target.route_name, target.request_model, backend_body.get("model"),
        num_msgs, num_tools, backend_body.get("max_completion_tokens"), backend_body.get("stream"),
    )

    resp = await session.post(
        url,
        json=backend_body,
        headers={"Connection": "close"},
        timeout=aiohttp.ClientTimeout(
            total=config.request_timeout,
            sock_read=config.sock_read_timeout,
        ),
    )
    log.info("%sbackend: response status=%d content_type=%s", _r, resp.status, resp.content_type)
    return resp
