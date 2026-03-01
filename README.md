# claude-relay

Proxy server that lets [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) (and any Anthropic API client) talk to self-hosted **sglang** / **vLLM** backends. Converts Anthropic API format to OpenAI-compatible format on the fly, streams responses back in Anthropic SSE format, and applies KV cache normalizations that push prefix-cache hit rates from ~22% to 95%+.

## Features

- **Anthropic <-> OpenAI API conversion** -- full bidirectional streaming translation including extended thinking, tool use, multi-turn conversations
- **KV cache normalization** -- stabilizes token prefixes so sglang/vLLM prefix caching actually works (see [KV Cache Normalization](#kv-cache-normalization))
- **Vision routing** -- optional image agent that intercepts images, routes them to a dedicated vision model, and injects descriptions back into the text conversation (see [Vision Routing](#vision-routing))
- **Auto context overflow recovery** -- detects context-too-long errors from the backend and automatically retries with reduced `max_completion_tokens`
- **Backend auto-detection** -- probes `/v1/models` to detect sglang vs vLLM and sets backend-specific parameters automatically
- **Token counting** -- `/v1/messages/count_tokens` endpoint for Claude Code's `/context` command

## Quick start

```bash
git clone https://github.com/voipmonitor/claude-relay.git
cd claude-relay
pip install -r requirements.txt

# Minimal -- text-only, no vision
python -m claude_proxy --backend http://localhost:30000

# With vision routing
python -m claude_proxy \
  --backend http://localhost:30000 \
  --vision-url http://localhost:8000/v1/chat/completions \
  --vision-model Qwen2.5-VL-72B
```

Point Claude Code at the proxy:

```bash
export ANTHROPIC_BASE_URL=http://localhost:5021
claude
```

## Command-line options

```
Usage: python -m claude_proxy [OPTIONS]

Server:
  --host HOST               Listen address (default: 0.0.0.0)
  --port PORT               Listen port (default: 5021)
  --backend URL             sglang/vLLM backend URL (default: http://localhost:30000)
  --ttl SECONDS             Backend auto-detection cache TTL (default: 30)
  --log-level LEVEL         Logging level (default: INFO)

Vision (image agent):
  --vision-url URL          Vision model endpoint -- enables image agent
  --vision-model NAME       Vision model name (required with --vision-url)
  --no-image-agent          Force-disable image agent even if vision URL is set

KV cache normalization (all enabled by default):
  --no-sort-tools           Don't sort tools alphabetically
  --no-strip-billing        Keep x-anthropic-billing-header in system prompt
  --no-strip-cache-control  Keep cache_control fields (unused by sglang/vLLM)
  --no-strip-date           Keep "Today's date is YYYY-MM-DD." in user messages
```

## How it works

```
Claude Code (Anthropic API)
        |
        v
  +--------------+
  | claude-relay  |
  |              |
  |  1. Normalize (KV cache)
  |  2. Strip images -> placeholders (optional)
  |  3. Convert Anthropic -> OpenAI
  |  4. Route to backend
  |  5. Convert OpenAI SSE -> Anthropic SSE
  |  6. Intercept vision tool calls (optional)
  +--------------+
        |
        v
  sglang / vLLM (OpenAI API)
        |
        v
  (optional) Vision model
```

### Request flow

1. Claude Code sends a standard Anthropic `/v1/messages` request
2. **Normalization** -- request body is cleaned up for KV cache stability (tool sorting, nonce stripping, etc.)
3. **Image stripping** (if vision enabled) -- images in the last user message are replaced with `[Image #N]` placeholders and cached in memory
4. **Conversion** -- Anthropic request format is converted to OpenAI `/v1/chat/completions` format (system prompts, messages, tools, thinking config)
5. **Backend routing** -- request is sent to the sglang/vLLM backend with auto-detected model name and backend-specific parameters
6. **Stream conversion** -- OpenAI SSE chunks are converted back to Anthropic SSE format and streamed to the client
7. **Vision interception** (if images were present) -- if the model calls the injected `analyzeImage` tool, the proxy intercepts it, calls the vision model, and continues the conversation with the description

### Conversion details

| Anthropic | OpenAI |
|-----------|--------|
| `system` (string or text blocks) | `messages[0].role = "system"` |
| `messages[].content` (text/tool_use/tool_result/thinking blocks) | Equivalent OpenAI message structure |
| `tools[].input_schema` | `tools[].function.parameters` |
| `thinking.budget_tokens` | `reasoning.effort` (low/medium/high) + `reasoning.max_tokens` |
| `max_tokens` | `max_completion_tokens` |
| `stop_sequences` | `stop` |
| Anthropic SSE (`message_start`, `content_block_delta`, ...) | Converted from OpenAI SSE (`choices[].delta`) |

## KV cache normalization

Self-hosted sglang and vLLM use **prefix KV caching** -- if any token in the prefix changes, the entire cached computation for that prefix is invalidated. Claude Code injects several elements that change between requests, destroying cache hits.

Without normalization, typical cache hit rates are around 22%. With all normalizations enabled, hit rates reach 95%+.

Based on the analysis from [buster-ripper](https://github.com/AlexGS74/buster-ripper).

### What gets normalized

| Normalization | Flag | Problem | Fix |
|---|---|---|---|
| **Tool sorting** | `--no-sort-tools` | MCP servers reconnect in arbitrary order, shuffling tool definitions at position 0 of the prompt prefix. Any reorder = entire KV cache miss. | Sort `tools[]` alphabetically by `name`. Tools are a declarative set, order has no effect on model behavior. |
| **Billing nonce** | `--no-strip-billing` | Claude Code injects a unique `x-anthropic-billing-header:...` string into system prompt blocks on every request. | Regex-strip from system blocks. |
| **Cache control** | `--no-strip-cache-control` | `cache_control` markers migrate between messages each turn, changing content hashes. sglang/vLLM don't use Anthropic's explicit cache_control anyway. | Remove `cache_control` fields from system blocks, message content blocks, and message-level. |
| **Date injection** | `--no-strip-date` | Claude Code injects `Today's date is YYYY-MM-DD.` into user messages. Changes daily at midnight = cache bust. | Regex-strip from user messages only. |

All normalizations are enabled by default. Disable individually with `--no-*` flags if needed.

## Vision routing

The image agent solves a fundamental problem: text-only LLMs (most large models served via sglang/vLLM) can't process images, but Claude Code sends screenshots, diagrams, and other images as part of the conversation.

The image agent is **disabled by default** and activates only when you provide `--vision-url` and `--vision-model`. You can force-disable it with `--no-image-agent` even if a vision URL is configured.

### How vision routing works

When the image agent is enabled and the proxy detects images in the last user message, it performs a two-phase streaming interception:

**Phase 1 -- Image stripping and tool injection**

1. All image blocks in the last user message (including images inside `tool_result` blocks) are extracted and stored in a per-session in-memory LRU cache (keyed by session ID from `metadata.user_id`)
2. Each image is replaced with a text placeholder: `[Image #N](call analyzeImage with imageId "N" to view this image)`
3. An `analyzeImage` tool is injected into the request's tool list with parameters:
   - `imageId` -- array of image IDs from `[Image #N]` placeholders
   - `task` -- what to analyze (the model formulates this based on the user's question)
   - `context` -- conversation context for the vision model
4. A system prompt is injected instructing the model to always call `analyzeImage` before responding about images
5. The modified (image-free) request is sent to the text backend

**Phase 2 -- Vision model call and follow-up**

6. The proxy streams the backend's response while watching for `analyzeImage` tool calls
7. When detected, the proxy:
   - Retrieves the original images from cache
   - Converts them to data URLs
   - Sends them to the vision model (`--vision-url`) with the task and context from the tool call
   - Gets back a detailed text description
8. The proxy builds a follow-up request appending:
   - The assistant's `analyzeImage` tool call
   - A tool result containing the vision model's description
9. This follow-up is sent to the text backend, and the response is streamed back to the client
10. The two response streams are stitched together with proper block index numbering so the client sees a single coherent response

**Image cache:**
- Per-session LRU cache, max 100 images (configurable via `image_cache_max_size`)
- TTL of 300 seconds (configurable via `image_cache_ttl`)
- Session is identified by `metadata.user_id` sent by Claude Code

**Supported image formats:** Any format supported by the vision model. Images are passed as base64 data URLs or direct URLs, matching the Anthropic API's `image` content block format.

### Why this approach?

- The text model (which handles reasoning, code generation, tool use) can be a large, powerful model that doesn't support vision
- The vision model can be a smaller, specialized multimodal model
- This decouples the two concerns and lets you independently scale/upgrade each model
- The text model "sees" images through detailed text descriptions, which is sufficient for most Claude Code use cases (screenshots of errors, UI mockups, diagrams)

## Installation as a Linux service

### 1. Install dependencies

```bash
# Clone the repository
git clone https://github.com/voipmonitor/claude-relay.git /opt/claude-relay
cd /opt/claude-relay

# Create a virtual environment
python3 -m venv /opt/claude-relay/.venv
source /opt/claude-relay/.venv/bin/activate
pip install -r requirements.txt
```

### 2. Create a systemd service

```bash
cat > /etc/systemd/system/claude-relay.service << 'EOF'
[Unit]
Description=Claude Relay - Anthropic API to sglang/vLLM proxy
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/claude-relay
ExecStart=/opt/claude-relay/.venv/bin/python -m claude_proxy \
    --backend http://localhost:30000 \
    --port 5021
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Optional: uncomment and adjust for vision routing
# ExecStart=/opt/claude-relay/.venv/bin/python -m claude_proxy \
#     --backend http://localhost:30000 \
#     --vision-url http://localhost:8000/v1/chat/completions \
#     --vision-model Qwen2.5-VL-72B \
#     --port 5021

[Install]
WantedBy=multi-user.target
EOF
```

### 3. Enable and start

```bash
systemctl daemon-reload
systemctl enable claude-relay
systemctl start claude-relay

# Check status
systemctl status claude-relay

# View logs
journalctl -u claude-relay -f
```

### 4. Configure Claude Code

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export ANTHROPIC_BASE_URL=http://localhost:5021
```

Or for a remote server:

```bash
export ANTHROPIC_BASE_URL=http://your-server:5021
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/messages` | Main Anthropic-compatible messages endpoint |
| POST | `/v1/messages/count_tokens` | Token counting (for Claude Code `/context`) |
| GET | `/health` | Health check with backend status |

## Debugging

Request bodies are dumped to `claude_proxy/debug/` as JSON files:
- `{req_id}_anthropic.json` -- original Anthropic request
- `{req_id}_openai.json` -- converted OpenAI request

Set `--log-level DEBUG` for verbose logging of all conversions and streaming events.

## License

MIT
