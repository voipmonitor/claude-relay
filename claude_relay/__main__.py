"""Entry point: python -m claude_relay"""

import argparse
import logging
import sys

from aiohttp import web

from .config import ProxyConfig, load_proxy_config, parse_model_route
from .backend import init_state
from .server import create_app


def main():
    parser = argparse.ArgumentParser(description="Claude Relay")
    parser.add_argument("--config", help="TOML config file")
    parser.add_argument("--host", default=None, help="Listen address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Listen port (default: 5021)")
    parser.add_argument("--backend", default=None, help="Default sglang/vLLM backend URL")
    parser.add_argument("--model-route", action="append", default=None, metavar="MODEL=URL[,UPSTREAM_MODEL]",
                        help="Route a request model to a backend URL; may be repeated")
    parser.add_argument("--vision-url", default=None, help="Vision model endpoint (enables image agent)")
    parser.add_argument("--vision-model", default=None, help="Vision model name (required with --vision-url)")
    parser.add_argument("--ttl", type=int, default=None, help="Backend detect cache TTL")
    parser.add_argument("--no-image-agent", action="store_true", default=None, help="Disable image agent")
    parser.add_argument("--force-vision", action="store_true", default=None, help="Force analyzeImage on every image")
    parser.add_argument("--no-sort-tools", action="store_true", default=None, help="Disable tool sorting for KV cache")
    parser.add_argument("--no-strip-billing", action="store_true", default=None, help="Keep billing nonce in system prompt")
    parser.add_argument("--no-strip-cache-control", action="store_true", default=None, help="Keep cache_control fields")
    parser.add_argument("--no-strip-date", action="store_true", default=None, help="Keep date injection in user messages")
    parser.add_argument("--dump-requests", action="store_true", default=None, help="Dump request bodies to debug/ directory")
    parser.add_argument("--dump-responses", action="store_true", default=None, help="Dump Anthropic SSE response bodies to debug/ directory")
    parser.add_argument("--tool-debug", action="store_true", default=None, help="Dump raw backend SSE for tool requests and log tool-call diagnostics")
    parser.add_argument("--max-image-b64-chars", type=int, default=None, help="Maximum base64 chars per image (default: 10MB)")
    parser.add_argument("--max-tools", type=int, default=None, help="Maximum tools array length (default: 500)")
    parser.add_argument("--request-timeout", type=int, default=None, help="Total request timeout in seconds (default 1800)")
    parser.add_argument("--sock-read-timeout", type=int, default=None, help="Max silence between chunks in seconds (default 60)")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    config = ProxyConfig()
    config_keys: set[str] = set()
    if args.config:
        try:
            config, config_keys = load_proxy_config(args.config, base=config)
        except Exception as exc:
            print(f"Failed to load config {args.config}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    if args.host is not None:
        config.host = args.host
    if args.port is not None:
        config.port = args.port
    if args.backend is not None:
        config.backend_url = args.backend
    if args.ttl is not None:
        config.backend_detect_ttl = args.ttl
    if args.vision_url is not None:
        config.vision_url = args.vision_url
    if args.vision_model is not None:
        config.vision_model = args.vision_model
    if args.force_vision:
        config.force_vision = True
    if args.no_sort_tools:
        config.sort_tools = False
    if args.no_strip_billing:
        config.strip_billing_nonce = False
    if args.no_strip_cache_control:
        config.strip_cache_control = False
    if args.no_strip_date:
        config.strip_date_injection = False
    if args.dump_requests:
        config.dump_requests = True
    if args.dump_responses:
        config.dump_responses = True
    if args.tool_debug:
        config.tool_debug = True
    if args.max_image_b64_chars is not None:
        config.max_image_b64_chars = args.max_image_b64_chars
    if args.max_tools is not None:
        config.max_tools = args.max_tools
    if args.request_timeout is not None:
        config.request_timeout = args.request_timeout
    if args.sock_read_timeout is not None:
        config.sock_read_timeout = args.sock_read_timeout
    if args.log_level is not None:
        config.log_level = args.log_level
    if args.model_route:
        routes = dict(config.model_routes)
        try:
            for spec in args.model_route:
                name, route = parse_model_route(spec)
                routes[name] = route
        except ValueError as exc:
            print(f"Invalid --model-route: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        config.model_routes = routes

    if args.no_image_agent:
        config.image_agent_enabled = False
    elif args.vision_url is not None:
        config.image_agent_enabled = bool(config.vision_url)
    elif "image_agent_enabled" not in config_keys:
        config.image_agent_enabled = bool(config.vision_url)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    init_state(config.backend_detect_ttl)

    print(f"Claude relay: {config.host}:{config.port} -> {config.backend_url}")
    if config.model_routes:
        print("Model routes:")
        for name, route in config.model_routes.items():
            upstream = f" model={route.upstream_model}" if route.upstream_model else ""
            print(f"  {name} -> {route.backend_url}{upstream}")
    print(f"Vision: {config.vision_url} ({config.vision_model})")
    print(f"Image agent: {'enabled' if config.image_agent_enabled else 'disabled'}{' (force-vision)' if config.force_vision else ''}")
    print(f"Backend detect TTL: {config.backend_detect_ttl}s")
    norm_flags = [
        f"sort_tools={config.sort_tools}",
        f"strip_billing={config.strip_billing_nonce}",
        f"strip_cache_control={config.strip_cache_control}",
        f"strip_date={config.strip_date_injection}",
    ]
    print(f"KV normalization: {', '.join(norm_flags)}")
    print(f"Response dumps: {'enabled' if config.dump_responses else 'disabled'}")
    print(f"Tool debug: {'enabled' if config.tool_debug else 'disabled'}")

    app = create_app(config)
    web.run_app(app, host=config.host, port=config.port, print=lambda _: None)


if __name__ == "__main__":
    main()
