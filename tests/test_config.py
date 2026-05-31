"""Tests for proxy configuration and model routing."""

from claude_relay.config import ModelRoute, ProxyConfig, load_proxy_config, parse_model_route


def test_resolve_backend_uses_exact_route_before_default():
    config = ProxyConfig(
        backend_url="http://default:8000",
        model_routes={
            "claude-3-5-sonnet": ModelRoute(
                backend_url="http://sonnet:8000",
                upstream_model="Qwen3.5",
            ),
        },
    )

    target = config.resolve_backend("claude-3-5-sonnet")

    assert target.request_model == "claude-3-5-sonnet"
    assert target.backend_url == "http://sonnet:8000"
    assert target.upstream_model == "Qwen3.5"
    assert target.route_name == "claude-3-5-sonnet"


def test_resolve_backend_supports_glob_routes():
    config = ProxyConfig(
        backend_url="http://default:8000",
        model_routes={
            "claude-opus-*": ModelRoute(
                backend_url="http://opus:8000",
                upstream_model="Kimi-K2.6",
            ),
        },
    )

    target = config.resolve_backend("claude-opus-4-5-20251101")

    assert target.backend_url == "http://opus:8000"
    assert target.upstream_model == "Kimi-K2.6"
    assert target.route_name == "claude-opus-*"


def test_resolve_backend_falls_back_to_default_backend():
    config = ProxyConfig(backend_url="http://default:8000")

    target = config.resolve_backend("unknown-model")

    assert target.request_model == "unknown-model"
    assert target.backend_url == "http://default:8000"
    assert target.upstream_model is None
    assert target.route_name == "default"


def test_parse_model_route_cli_spec():
    model, route = parse_model_route("claude-haiku=http://haiku:8000,Qwen2.5")

    assert model == "claude-haiku"
    assert route == ModelRoute("http://haiku:8000", "Qwen2.5")


def test_load_proxy_config_reads_toml_routes(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
backend_url = "http://default:8000"
request_timeout = 120
dump_responses = true

[model_routes."claude-3-5-sonnet"]
backend_url = "http://sonnet:8000"
upstream_model = "Qwen3.5"

[model_profiles."claude-opus-*"]
backend_url = "http://opus:8000"
model = "Kimi-K2.6"
""",
        encoding="utf-8",
    )

    config, keys = load_proxy_config(path)

    assert config.backend_url == "http://default:8000"
    assert config.request_timeout == 120
    assert config.dump_responses is True
    assert keys >= {"backend_url", "request_timeout", "dump_responses", "model_routes"}
    assert config.resolve_backend("claude-3-5-sonnet").backend_url == "http://sonnet:8000"
    opus = config.resolve_backend("claude-opus-latest")
    assert opus.backend_url == "http://opus:8000"
    assert opus.upstream_model == "Kimi-K2.6"
