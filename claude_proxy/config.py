from dataclasses import dataclass


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
    request_timeout: int = 1800  # 30 minutes
    log_level: str = "INFO"
    sort_tools: bool = True              # sort tools alphabetically for stable KV prefix
    strip_billing_nonce: bool = True     # remove x-anthropic-billing-header from system
    strip_cache_control: bool = True     # remove cache_control fields (unused by sglang/vLLM)
    strip_date_injection: bool = True    # remove "Today's date is YYYY-MM-DD." from user msgs
    dump_requests: bool = False          # dump request bodies to debug/ directory
