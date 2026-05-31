"""KV cache normalization: stabilize request prefixes for better cache hits."""

import re

from .config import ProxyConfig

_BILLING_RE = re.compile(r"x-anthropic-billing-header:[^\n]*\n?")
_DATE_RE = re.compile(r"Today's date is \d{4}-\d{2}-\d{2}\.\n?")


def normalize_request(body: dict, config: ProxyConfig) -> tuple[dict, list[str]]:
    """Apply KV-cache-friendly normalizations to an Anthropic request body.

    Returns (modified_body, list_of_change_descriptions).
    """
    changes: list[str] = []

    if config.sort_tools:
        changes += _sort_tools(body)

    if config.strip_billing_nonce:
        changes += _strip_billing(body)

    if config.strip_cache_control:
        changes += _strip_cache_control(body)

    if config.strip_date_injection:
        changes += _strip_date(body)

    return body, changes


# ---------------------------------------------------------------------------
# 1. Tool sorting
# ---------------------------------------------------------------------------

def _sort_tools(body: dict) -> list[str]:
    tools = body.get("tools")
    if not tools or not isinstance(tools, list):
        return []
    names_before = [t.get("name", "") for t in tools]
    body["tools"] = sorted(tools, key=lambda t: t.get("name", ""))
    names_after = [t.get("name", "") for t in body["tools"]]
    if names_before != names_after:
        return [f"sorted {len(tools)} tools"]
    return []


# ---------------------------------------------------------------------------
# 2. Billing nonce stripping
# ---------------------------------------------------------------------------

def _strip_billing(body: dict) -> list[str]:
    system = body.get("system")
    if system is None:
        return []

    count = 0
    if isinstance(system, str):
        cleaned, n = _BILLING_RE.subn("", system)
        if n:
            body["system"] = cleaned
            count += n
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                cleaned, n = _BILLING_RE.subn("", text)
                if n:
                    block["text"] = cleaned
                    count += n

    return [f"stripped {count} billing nonce(s)"] if count else []


# ---------------------------------------------------------------------------
# 3. Cache control removal
# ---------------------------------------------------------------------------

def _strip_cache_control(body: dict) -> list[str]:
    count = 0

    # System blocks
    system = body.get("system")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and "cache_control" in block:
                del block["cache_control"]
                count += 1
    elif isinstance(system, dict) and "cache_control" in system:
        del system["cache_control"]
        count += 1

    # Messages
    for msg in body.get("messages", []):
        # Message-level cache_control
        if "cache_control" in msg:
            del msg["cache_control"]
            count += 1

        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    del block["cache_control"]
                    count += 1

    return [f"removed {count} cache_control field(s)"] if count else []


# ---------------------------------------------------------------------------
# 4. Date injection stripping
# ---------------------------------------------------------------------------

def _strip_date(body: dict) -> list[str]:
    count = 0

    for msg in body.get("messages", []):
        if msg.get("role") != "user":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            cleaned, n = _DATE_RE.subn("", content)
            if n:
                msg["content"] = cleaned
                count += n
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    cleaned, n = _DATE_RE.subn("", text)
                    if n:
                        block["text"] = cleaned
                        count += n

    return [f"stripped {count} date injection(s)"] if count else []
