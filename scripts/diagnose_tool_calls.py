#!/usr/bin/env python3
"""Smoke-test OpenAI and Anthropic/proxy tool-call streaming.

The script sends the same forced get_weather-style tool call to:
1. The OpenAI-compatible backend, usually http://localhost:8000
2. The claude-relay Anthropic endpoint, usually http://localhost:5021

It writes raw SSE streams plus a JSON report under claude_relay/debug/ by default.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator


def _base_url(url: str) -> str:
    return url.rstrip("/")


def _first_backend_model(base_url: str, timeout: float) -> str:
    url = f"{_base_url(base_url)}/v1/models"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["data"][0]["id"]


def _iter_sse_lines(resp) -> Iterator[tuple[str | None, str]]:
    event_type = None
    data_lines: list[str] = []
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if line == "":
            if data_lines:
                yield event_type, "\n".join(data_lines)
            event_type = None
            data_lines = []
        elif line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if data_lines:
        yield event_type, "\n".join(data_lines)


def _post_sse(url: str, payload: dict[str, Any], timeout: float, raw_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", "accept": "text/event-stream"},
        method="POST",
    )
    events: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, raw_path.open("w", encoding="utf-8") as raw_file:
            meta = {
                "ok": 200 <= resp.status < 300,
                "status": resp.status,
                "content_type": resp.headers.get("content-type", ""),
                "raw_stream_path": str(raw_path),
            }
            for seq, (event_type, data) in enumerate(_iter_sse_lines(resp), start=1):
                record = {"seq": seq, "event": event_type, "data": data}
                events.append(record)
                raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            return meta, events
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raw_path.write_text(body, encoding="utf-8")
        return {
            "ok": False,
            "status": e.code,
            "error": body[:2000],
            "raw_stream_path": str(raw_path),
        }, events
    except urllib.error.URLError as e:
        return {"ok": False, "error": str(e), "raw_stream_path": str(raw_path)}, events
    except TimeoutError as e:
        return {"ok": False, "error": f"timeout: {e}", "raw_stream_path": str(raw_path)}, events


def _summarize_tool_states(tool_states: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index, state in sorted(tool_states.items()):
        args = state.get("arguments", "")
        item = {
            "index": index,
            "id": state.get("id"),
            "name": state.get("name"),
            "chunks": state.get("chunks", 0),
            "arguments": args,
            "arguments_chars": len(args),
        }
        if args:
            try:
                item["arguments_json"] = json.loads(args)
                item["arguments_json_valid"] = True
            except json.JSONDecodeError as e:
                item["arguments_json_valid"] = False
                item["arguments_json_error"] = str(e)
        result.append(item)
    return result


def _analyze_openai_events(events: list[dict[str, Any]], expected_tool: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "events": len(events),
        "done": 0,
        "bad_json": 0,
        "finish_reasons": {},
        "delta_keys": {},
        "reasoning_chars": 0,
        "content_chars": 0,
        "tool_call_deltas": 0,
        "tool_like_content": [],
        "usage": None,
    }
    tool_states: dict[str, dict[str, Any]] = {}

    for event in events:
        data = event["data"]
        if data == "[DONE]":
            summary["done"] += 1
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            summary["bad_json"] += 1
            continue

        if payload.get("usage"):
            summary["usage"] = payload["usage"]

        choices = payload.get("choices") or []
        if not choices:
            continue
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            summary["finish_reasons"][finish_reason] = summary["finish_reasons"].get(finish_reason, 0) + 1
        for key in delta:
            summary["delta_keys"][key] = summary["delta_keys"].get(key, 0) + 1

        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
        if not reasoning and isinstance(delta.get("thinking"), dict):
            reasoning = delta["thinking"].get("content")
        if reasoning:
            summary["reasoning_chars"] += len(str(reasoning))

        content = delta.get("content")
        if content:
            content_text = str(content)
            summary["content_chars"] += len(content_text)
            lower = content_text.lower()
            if (
                "tool_call" in lower
                or "function_call" in lower
                or "<tool" in lower
                or "<function" in lower
                or f"{expected_tool}(" in content_text
            ):
                summary["tool_like_content"].append({"seq": event["seq"], "preview": content_text[:300]})

        for tool_call in delta.get("tool_calls") or []:
            summary["tool_call_deltas"] += 1
            index = str(tool_call.get("index", 0))
            state = tool_states.setdefault(index, {"id": None, "name": None, "arguments": "", "chunks": 0})
            func = tool_call.get("function") or {}
            if tool_call.get("id"):
                state["id"] = tool_call.get("id")
            if func.get("name"):
                state["name"] = func.get("name")
            if isinstance(func.get("arguments"), str):
                state["arguments"] += func.get("arguments")
            state["chunks"] += 1

    summary["tool_calls"] = _summarize_tool_states(tool_states)
    return summary


def _analyze_anthropic_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "events": len(events),
        "event_types": {},
        "stop_reason": None,
        "tool_use_blocks": 0,
        "text_blocks": 0,
        "thinking_blocks": 0,
        "bad_json": 0,
        "errors": [],
    }
    tool_blocks: dict[str, dict[str, Any]] = {}

    for event in events:
        event_type = event["event"]
        if event_type:
            summary["event_types"][event_type] = summary["event_types"].get(event_type, 0) + 1
        try:
            payload = json.loads(event["data"])
        except json.JSONDecodeError:
            summary["bad_json"] += 1
            continue

        payload_type = payload.get("type") or event_type
        if payload_type == "content_block_start":
            index = str(payload.get("index"))
            block = payload.get("content_block") or {}
            block_type = block.get("type")
            if block_type == "tool_use":
                summary["tool_use_blocks"] += 1
                tool_blocks[index] = {
                    "index": index,
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": "",
                }
            elif block_type == "text":
                summary["text_blocks"] += 1
            elif block_type == "thinking":
                summary["thinking_blocks"] += 1
        elif payload_type == "content_block_delta":
            index = str(payload.get("index"))
            delta = payload.get("delta") or {}
            if delta.get("type") == "input_json_delta" and index in tool_blocks:
                tool_blocks[index]["arguments"] += delta.get("partial_json", "")
        elif payload_type == "message_delta":
            delta = payload.get("delta") or {}
            summary["stop_reason"] = delta.get("stop_reason")
        elif payload_type == "error":
            summary["errors"].append(payload.get("error", payload))

    summary["tool_calls"] = _summarize_tool_states(tool_blocks)
    return summary


def _openai_tool(tool_name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }


def _anthropic_tool(tool_name: str) -> dict[str, Any]:
    tool = _openai_tool(tool_name)["function"]
    return {
        "name": tool["name"],
        "description": tool["description"],
        "input_schema": tool["parameters"],
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_out_dir = repo_root / "claude_relay" / "debug"
    parser = argparse.ArgumentParser(description="Diagnose GLM/tool-call streaming through claude-relay.")
    parser.add_argument("--backend", default="http://localhost:8000", help="OpenAI-compatible backend base URL")
    parser.add_argument("--proxy", default="http://localhost:5021", help="claude-relay base URL")
    parser.add_argument("--model", default="", help="Backend model for direct OpenAI request; defaults to /v1/models[0]")
    parser.add_argument("--tool-name", default="get_weather")
    parser.add_argument("--city", default="Boston")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--skip-backend", action="store_true")
    parser.add_argument("--skip-proxy", action="store_true")
    args = parser.parse_args()

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prompt = (
        f"Use the {args.tool_name} tool to look up the weather in {args.city}. "
        "Do not answer in plain text."
    )
    report: dict[str, Any] = {"run_id": run_id, "prompt": prompt}

    if not args.skip_backend:
        backend_model = args.model
        try:
            backend_model = backend_model or _first_backend_model(args.backend, args.timeout)
            openai_payload = {
                "model": backend_model,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [_openai_tool(args.tool_name)],
                "tool_choice": {"type": "function", "function": {"name": args.tool_name}},
                "stream": True,
                "max_completion_tokens": args.max_tokens,
            }
            meta, events = _post_sse(
                f"{_base_url(args.backend)}/v1/chat/completions",
                openai_payload,
                args.timeout,
                args.out_dir / f"{run_id}_direct_backend_stream.ndjson",
            )
            report["backend"] = {
                **meta,
                "model": backend_model,
                "request": openai_payload,
                "analysis": _analyze_openai_events(events, args.tool_name),
            }
        except Exception as e:
            report["backend"] = {"ok": False, "error": str(e)}

    if not args.skip_proxy:
        anthropic_payload = {
            "model": "claude-tool-diagnostic",
            "messages": [{"role": "user", "content": prompt}],
            "tools": [_anthropic_tool(args.tool_name)],
            "tool_choice": {"type": "tool", "name": args.tool_name},
            "stream": True,
            "max_tokens": args.max_tokens,
        }
        meta, events = _post_sse(
            f"{_base_url(args.proxy)}/v1/messages",
            anthropic_payload,
            args.timeout,
            args.out_dir / f"{run_id}_proxy_stream.ndjson",
        )
        report["proxy"] = {
            **meta,
            "request": anthropic_payload,
            "analysis": _analyze_anthropic_events(events),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / f"{run_id}_tool_diagnosis.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    selected = [value for key, value in report.items() if key in {"backend", "proxy"}]
    return 1 if any(not item.get("ok") for item in selected) else 0


if __name__ == "__main__":
    sys.exit(main())
