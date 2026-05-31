from dataclasses import dataclass, field, fields
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


@dataclass(frozen=True)
class ModelRoute:
    backend_url: str
    upstream_model: str | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class BackendTarget:
    request_model: str
    backend_url: str
    upstream_model: str | None = None
    route_name: str = "default"
    reasoning_effort: str | None = None


@dataclass
class ProxyConfig:
    host: str = "0.0.0.0"
    port: int = 5021
    backend_url: str = "http://localhost:30000"
    backend_detect_ttl: int = 30
    vision_url: str = ""
    vision_model: str = ""
    image_agent_enabled: bool = True
    force_vision: bool = False           # force analyzeImage call on every image via tool_choice
    image_cache_max_size: int = 100
    image_cache_ttl: int = 300
    min_completion_tokens: int = 4096  # minimum completion tokens to allow
    client_max_size: int = 200 * 1024 * 1024  # 200MB
    max_image_b64_chars: int = 10_000_000  # 10MB per image (base64 chars)
    max_tools: int = 500  # Maximum tools array length
    request_timeout: int = 1800  # 30 minutes
    sock_read_timeout: int = 60  # Max silence between chunks
    log_level: str = "INFO"
    sort_tools: bool = True              # sort tools alphabetically for stable KV prefix
    strip_billing_nonce: bool = True     # remove x-anthropic-billing-header from system
    strip_cache_control: bool = True     # remove cache_control fields (unused by sglang/vLLM)
    strip_date_injection: bool = True    # remove "Today's date is YYYY-MM-DD." from user msgs
    dump_requests: bool = False          # dump request bodies to debug/ directory
    dump_responses: bool = False         # dump Anthropic SSE response bodies to debug/ directory
    tool_debug: bool = False             # dump backend SSE streams for requests with tools
    debug_max_age_hours: int | None = 24  # auto-delete debug files older than this (None disables)
    model_routes: dict[str, ModelRoute] = field(default_factory=dict)
    model_sampling: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def resolve_backend(self, request_model: str | None) -> BackendTarget:
        model_raw = (request_model or "auto").strip() or "auto"
        if model_raw in self.model_routes:
            route = self.model_routes[model_raw]
            return BackendTarget(model_raw, route.backend_url, route.upstream_model, model_raw, route.reasoning_effort)

        model_lower = model_raw.lower()
        for pattern, route in self.model_routes.items():
            if _is_glob_pattern(pattern) and fnmatchcase(model_lower, pattern.lower()):
                return BackendTarget(model_raw, route.backend_url, route.upstream_model, pattern, route.reasoning_effort)

        return BackendTarget(model_raw, self.backend_url)

    def backend_urls(self) -> list[str]:
        urls = [self.backend_url]
        for route in self.model_routes.values():
            if route.backend_url not in urls:
                urls.append(route.backend_url)
        return urls


def _is_glob_pattern(value: str) -> bool:
    return any(char in value for char in "*?[")


def _coerce_model_route(name: str, value: Any) -> ModelRoute:
    if isinstance(value, str):
        backend_url = value.strip()
        upstream_model = None
    elif isinstance(value, dict):
        backend_url = str(value.get("backend_url") or value.get("url") or "").strip()
        upstream = value.get("upstream_model", value.get("model"))
        upstream_model = str(upstream).strip() if upstream is not None else None
        upstream_model = upstream_model or None
    else:
        raise ValueError(f"model route {name!r} must be a string or table")

    if not backend_url:
        raise ValueError(f"model route {name!r} is missing backend_url")

    reasoning_effort = None
    if isinstance(value, dict):
        re = value.get("reasoning_effort")
        if isinstance(re, str) and re.strip():
            reasoning_effort = re.strip()

    return ModelRoute(backend_url=backend_url, upstream_model=upstream_model, reasoning_effort=reasoning_effort)


def parse_model_route(spec: str) -> tuple[str, ModelRoute]:
    """Parse MODEL=BACKEND_URL[,UPSTREAM_MODEL] CLI route specs."""
    if "=" not in spec:
        raise ValueError("model route must use MODEL=BACKEND_URL[,UPSTREAM_MODEL]")
    model, value = spec.split("=", 1)
    model = model.strip()
    if not model:
        raise ValueError("model route is missing MODEL")
    backend_url, _, upstream_model = value.partition(",")
    route = _coerce_model_route(
        model,
        {
            "backend_url": backend_url.strip(),
            "upstream_model": upstream_model.strip() or None,
        },
    )
    return model, route


def load_proxy_config(path: str | Path, base: ProxyConfig | None = None) -> tuple[ProxyConfig, set[str]]:
    """Load a TOML config file and return (config, keys_set_in_file)."""
    config = base or ProxyConfig()
    path = Path(path)
    with path.open("rb") as file:
        data = tomllib.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a TOML table")

    route_tables = {}
    for route_key in ("model_routes", "model_profiles"):
        value = data.pop(route_key, None)
        if value:
            if not isinstance(value, dict):
                raise ValueError(f"{route_key} must be a TOML table")
            route_tables.update(value)

    config_field_names = {field.name for field in fields(ProxyConfig)}
    unknown = sorted(set(data) - config_field_names)
    if unknown:
        raise ValueError(f"unknown config option(s): {', '.join(unknown)}")

    values = {field.name: getattr(config, field.name) for field in fields(ProxyConfig)}
    keys_set = set(data)
    for key, value in data.items():
        if key == "model_routes":
            continue
        values[key] = value

    routes = dict(config.model_routes)
    for name, route_value in route_tables.items():
        routes[str(name)] = _coerce_model_route(str(name), route_value)
    if route_tables:
        keys_set.add("model_routes")
    values["model_routes"] = routes

    return ProxyConfig(**values), keys_set
