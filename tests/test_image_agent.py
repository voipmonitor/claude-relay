"""Tests for image_agent.py session-scoped caching and LRU eviction."""
import pytest
from claude_relay.image_agent import (
    ImageCache,
    has_images,
    strip_and_cache_images,
    IMAGE_AGENT_SYSTEM_PROMPT,
)


class TestImageCacheSessionScoping:
    """Test that ImageCache properly scopes keys by session_id."""

    def test_store_and_retrieve_same_session(self):
        """Cache stores and retrieves images within same session."""
        cache = ImageCache(max_size=10, ttl=300)
        session_id = "session_abc"
        image_key = "session_abc_Image#1"
        source = {"data": "base64data", "media_type": "image/png"}

        cache.store(session_id, image_key, source)
        result = cache.get(session_id, image_key)

        assert result == source

    def test_different_sessions_isolated(self):
        """Different sessions don't collide - cache is properly scoped."""
        cache = ImageCache(max_size=10, ttl=300)
        session_a = "session_a"
        session_b = "session_b"
        image_key = "Image#1"
        source_a = {"data": "data_a", "media_type": "image/png"}
        source_b = {"data": "data_b", "media_type": "image/jpeg"}

        cache.store(session_a, image_key, source_a)
        cache.store(session_b, image_key, source_b)

        # Each session should get its own data
        result_a = cache.get(session_a, image_key)
        result_b = cache.get(session_b, image_key)

        assert result_a == source_a
        assert result_b == source_b
        assert result_a != result_b

    def test_nonexistent_session_returns_none(self):
        """Getting from nonexistent session returns None."""
        cache = ImageCache(max_size=10, ttl=300)
        result = cache.get("nonexistent", "Image#1")
        assert result is None

    def test_nonexistent_key_returns_none(self):
        """Getting nonexistent key from existing session returns None."""
        cache = ImageCache(max_size=10, ttl=300)
        cache.store("session_x", "Image#1", {"data": "test"})
        result = cache.get("session_x", "Image#999")
        assert result is None

    def test_cache_key_collision_prevention(self):
        """Verify keys from different sessions with same image number don't collide.

        This is the critical regression test - previously, keys like "Image#1"
        were shared across sessions, causing data corruption.
        """
        cache = ImageCache(max_size=10, ttl=300)

        # Store same logical image in two sessions
        cache.store("sess_1", "sess_1_Image#1", {"data": "session1_data"})
        cache.store("sess_2", "sess_2_Image#1", {"data": "session2_data"})

        # Verify no collision
        r1 = cache.get("sess_1", "sess_1_Image#1")
        r2 = cache.get("sess_2", "sess_2_Image#1")

        assert r1["data"] == "session1_data"
        assert r2["data"] == "session2_data"


class TestImageCacheLRUEviction:
    """Test LRU eviction behavior in ImageCache."""

    def test_lru_eviction_max_size(self):
        """Oldest entries are evicted when max_size exceeded."""
        cache = ImageCache(max_size=3, ttl=300)
        session = "test_session"

        cache.store(session, "Image#1", {"data": "1"})
        cache.store(session, "Image#2", {"data": "2"})
        cache.store(session, "Image#3", {"data": "3"})
        cache.store(session, "Image#4", {"data": "4"})  # Should evict Image#1

        assert cache.get(session, "Image#1") is None
        assert cache.get(session, "Image#2") == {"data": "2"}
        assert cache.get(session, "Image#3") == {"data": "3"}
        assert cache.get(session, "Image#4") == {"data": "4"}

    def test_lru_access_prevents_eviction(self):
        """Accessing an entry moves it to end, preventing eviction."""
        cache = ImageCache(max_size=3, ttl=300)
        session = "test_session"

        cache.store(session, "Image#1", {"data": "1"})
        cache.store(session, "Image#2", {"data": "2"})
        cache.store(session, "Image#3", {"data": "3"})

        # Access Image#1 to move it to end
        cache.get(session, "Image#1")

        # Add new entry - should evict Image#2 (now oldest)
        cache.store(session, "Image#4", {"data": "4"})

        assert cache.get(session, "Image#1") == {"data": "1"}  # Still there
        assert cache.get(session, "Image#2") is None  # Evicted
        assert cache.get(session, "Image#3") == {"data": "3"}
        assert cache.get(session, "Image#4") == {"data": "4"}

    def test_per_session_eviction(self):
        """Eviction is per-session, not global."""
        cache = ImageCache(max_size=2, ttl=300)

        # Fill session_a
        cache.store("session_a", "Image#1", {"data": "a1"})
        cache.store("session_a", "Image#2", {"data": "a2"})

        # Fill session_b (should not affect session_a)
        cache.store("session_b", "Image#1", {"data": "b1"})
        cache.store("session_b", "Image#2", {"data": "b2"})

        # Add to session_a - should evict within session_a only
        cache.store("session_a", "Image#3", {"data": "a3"})

        assert cache.get("session_a", "Image#1") is None
        assert cache.get("session_a", "Image#2") == {"data": "a2"}
        assert cache.get("session_b", "Image#1") == {"data": "b1"}
        assert cache.get("session_b", "Image#2") == {"data": "b2"}


class TestImageCacheTTL:
    """Test TTL expiration in ImageCache."""

    def test_expired_entry_returns_none(self):
        """Entries expire after TTL seconds."""
        cache = ImageCache(max_size=10, ttl=0)  # Immediate expiration

        cache.store("session", "Image#1", {"data": "test"})
        result = cache.get("session", "Image#1")

        assert result is None

    def test_cleanup_expired_removes_empty_sessions(self):
        """cleanup_expired removes sessions with no remaining entries."""
        cache = ImageCache(max_size=10, ttl=0)

        cache.store("session_temp", "Image#1", {"data": "test"})
        cache.cleanup_expired()

        assert "session_temp" not in cache._sessions


class TestHasImages:
    """Test has_images() helper function."""

    def test_detects_image_in_last_user_message(self):
        """Detects image blocks in the last user message."""
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": [
                    {"type": "text", "text": "Check this"},
                    {"type": "image", "source": {"data": "base64", "media_type": "image/png"}}
                ]},
            ]
        }
        assert has_images(body) is True

    def test_no_images_returns_false(self):
        """Returns False when no images present."""
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        assert has_images(body) is False

    def test_only_old_images_returns_false(self):
        """Images in old messages (not last user message) return False."""
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"data": "old", "media_type": "image/png"}}
                ]},
                {"role": "assistant", "content": "I see the image"},
                {"role": "user", "content": [{"type": "text", "text": "Thanks"}]},
            ]
        }
        assert has_images(body) is False

    def test_detects_image_in_tool_result(self):
        """Detects images inside tool_result content arrays."""
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "content": [
                        {"type": "image", "source": {"data": "tool_img", "media_type": "image/png"}}
                    ]}
                ]},
            ]
        }
        assert has_images(body) is True


class TestStripAndCacheImages:
    """Test strip_and_cache_images() function."""

    def test_replaces_images_with_placeholders(self):
        """Images are replaced with [Image #N] placeholders."""
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Look at"},
                    {"type": "image", "source": {"data": "img1", "media_type": "image/png"}},
                    {"type": "text", "text": "and"},
                    {"type": "image", "source": {"data": "img2", "media_type": "image/jpeg"}},
                ]},
            ]
        }
        cache = ImageCache()
        result = strip_and_cache_images(body, "test_session", cache)

        content = result["messages"][0]["content"]
        assert content[0]["text"] == "Look at"
        assert "[Image #1]" in content[1]["text"]
        assert "analyzeImage(imageId=[\"1\"])" in content[1]["text"]
        assert "[Image #2]" in content[3]["text"]
        assert "analyzeImage(imageId=[\"2\"])" in content[3]["text"]

    def test_caches_original_images(self):
        """Original image sources are cached with session-scoped keys."""
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"data": "img_data", "media_type": "image/png"}},
                ]},
            ]
        }
        cache = ImageCache()
        strip_and_cache_images(body, "session_xyz", cache)

        # Verify cache has the image with session-scoped key
        cached = cache.get("session_xyz", "session_xyz_Image#1")
        assert cached == {"data": "img_data", "media_type": "image/png"}

    def test_injects_system_prompt(self):
        """IMAGE_AGENT_SYSTEM_PROMPT is injected at start of system."""
        body = {"messages": []}
        cache = ImageCache()
        result = strip_and_cache_images(body, "session", cache)

        assert result["system"].startswith(IMAGE_AGENT_SYSTEM_PROMPT)

    def test_injects_analyze_image_tool(self):
        """analyzeImage tool is injected into tools list."""
        body = {"messages": []}
        cache = ImageCache()
        result = strip_and_cache_images(body, "session", cache)

        tools = result.get("tools", [])
        analyze_tool = next((t for t in tools if t.get("name") == "analyzeImage"), None)
        assert analyze_tool is not None
        assert "input_schema" in analyze_tool

    def test_doesnt_duplicate_tool_if_exists(self):
        """analyzeImage tool is not duplicated if already present."""
        body = {
            "messages": [],
            "tools": [{"name": "analyzeImage", "description": "existing"}],
        }
        cache = ImageCache()
        result = strip_and_cache_images(body, "session", cache)

        analyze_tools = [t for t in result["tools"] if t.get("name") == "analyzeImage"]
        assert len(analyze_tools) == 1

    def test_preserves_non_user_messages(self):
        """Assistant and system messages are not modified."""
        body = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "assistant", "content": "I see it"},
                {"role": "user", "content": [{"type": "image", "source": {"data": "x"}}]},
            ]
        }
        cache = ImageCache()
        result = strip_and_cache_images(body, "session", cache)

        assert result["messages"][0]["content"] == "You are helpful"
        assert result["messages"][1]["content"] == "I see it"

    def test_handles_empty_content(self):
        """Handles empty or non-list content gracefully."""
        body = {
            "messages": [
                {"role": "user", "content": []},
                {"role": "user", "content": "string not list"},
            ]
        }
        cache = ImageCache()
        result = strip_and_cache_images(body, "session", cache)

        assert result["messages"][0]["content"] == []
        # String content is preserved as-is (not iterated)

    def test_mixed_tool_result_content_preserves_order(self):
        """Mixed text and image content in tool_result preserves exact order.

        Tests that strip_and_cache_images() handles tool_result messages with
        interleaved text and images without reordering or losing content.
        """
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "content": [
                        {"type": "text", "text": "First text block"},
                        {"type": "image", "source": {"data": "img1_data", "media_type": "image/png"}},
                        {"type": "text", "text": "Second text block"},
                        {"type": "image", "source": {"data": "img2_data", "media_type": "image/jpeg"}},
                    ]},
                ]},
            ]
        }
        cache = ImageCache()
        result = strip_and_cache_images(body, "test_session", cache)

        tool_result = result["messages"][0]["content"][0]
        assert tool_result["type"] == "tool_result"

        content = tool_result["content"]
        assert len(content) == 4

        # Verify order is preserved: text, image placeholder, text, image placeholder
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "First text block"

        assert content[1]["type"] == "text"
        assert "[Image #1]" in content[1]["text"]
        assert "analyzeImage(imageId=[\"1\"])" in content[1]["text"]

        assert content[2]["type"] == "text"
        assert content[2]["text"] == "Second text block"

        assert content[3]["type"] == "text"
        assert "[Image #2]" in content[3]["text"]
        assert "analyzeImage(imageId=[\"2\"])" in content[3]["text"]

        # Verify images are cached in correct order
        cached_img1 = cache.get("test_session", "test_session_Image#1")
        assert cached_img1 == {"data": "img1_data", "media_type": "image/png"}

        cached_img2 = cache.get("test_session", "test_session_Image#2")
        assert cached_img2 == {"data": "img2_data", "media_type": "image/jpeg"}


class TestEdgeCasesMalformedInput:
    """Test edge cases and malformed input handling."""

    def test_image_block_missing_source(self):
        """Image block with type='image' but no source field is preserved as-is."""
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Here:"},
                    {"type": "image"},  # Missing source
                    {"type": "text", "text": "Done"},
                ]},
            ]
        }
        cache = ImageCache()
        result = strip_and_cache_images(body, "session", cache)

        content = result["messages"][0]["content"]
        # Image block without source should be preserved (not replaced)
        assert content[0]["text"] == "Here:"
        assert content[1]["type"] == "image"
        assert "source" not in content[1]
        assert content[2]["text"] == "Done"

    def test_image_source_missing_data(self):
        """Image with source but missing data field is converted to placeholder (doesn't crash)."""
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"media_type": "image/png"}},  # Missing data
                ]},
            ]
        }
        cache = ImageCache()
        # Should not crash even with incomplete source
        result = strip_and_cache_images(body, "session", cache)

        content = result["messages"][0]["content"]
        # Image gets converted to text placeholder (source dict is cached as-is)
        assert content[0]["type"] == "text"
        assert "[Image #1]" in content[0]["text"]

    def test_image_source_missing_media_type(self):
        """Image with source but missing media_type is converted to placeholder (doesn't crash)."""
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"data": "base64data"}},  # Missing media_type
                ]},
            ]
        }
        cache = ImageCache()
        # Should not crash even with incomplete source
        result = strip_and_cache_images(body, "session", cache)

        content = result["messages"][0]["content"]
        # Image gets converted to text placeholder
        assert content[0]["type"] == "text"
        assert "[Image #1]" in content[0]["text"]

    def test_system_prompt_none(self):
        """Message with system=None (null) doesn't crash."""
        body = {
            "messages": [],
            "system": None,
        }
        cache = ImageCache()
        # Should not crash on None system prompt
        result = strip_and_cache_images(body, "session", cache)

        # System prompt injection handles None gracefully (preserves None or sets prompt)
        # The key test is that it didn't crash
        assert "system" in result  # Either None or the injected prompt

    def test_empty_messages_list(self):
        """Empty messages list doesn't crash."""
        body = {"messages": []}
        cache = ImageCache()
        result = strip_and_cache_images(body, "session", cache)

        assert result["messages"] == []
        assert result["system"].startswith(IMAGE_AGENT_SYSTEM_PROMPT)

    def test_message_missing_role(self):
        """Message without role field is handled gracefully."""
        body = {
            "messages": [
                {"content": [{"type": "text", "text": "no role"}]},
                {"role": "user", "content": [{"type": "text", "text": "has role"}]},
            ]
        }
        cache = ImageCache()
        result = strip_and_cache_images(body, "session", cache)

        # Should not crash, should process valid messages
        assert len(result["messages"]) == 2

    def test_tool_result_missing_content(self):
        """tool_result without content field is handled gracefully."""
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "123"},  # Missing content
                    {"type": "text", "text": "after"},
                ]},
            ]
        }
        cache = ImageCache()
        result = strip_and_cache_images(body, "session", cache)

        content = result["messages"][0]["content"]
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "123"
        assert content[1]["text"] == "after"

    def test_tool_result_with_string_content(self):
        """tool_result with string (not list) content is handled gracefully."""
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "content": "string result"},
                ]},
            ]
        }
        cache = ImageCache()
        result = strip_and_cache_images(body, "session", cache)

        content = result["messages"][0]["content"]
        assert content[0]["content"] == "string result"

    def test_empty_body(self):
        """Empty body dict doesn't crash."""
        body = {}
        cache = ImageCache()
        result = strip_and_cache_images(body, "session", cache)

        # Function adds system prompt and tools even to empty body
        assert "system" in result
        assert "tools" in result
        assert result["system"].startswith(IMAGE_AGENT_SYSTEM_PROMPT)

    def test_has_images_with_empty_last_user_content(self):
        """has_images returns False for last user message with empty content."""
        body = {
            "messages": [
                {"role": "user", "content": []},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": []},  # Last user message is empty
            ]
        }
        assert has_images(body) is False

    def test_has_images_with_none_content(self):
        """has_images returns False when content is None."""
        body = {
            "messages": [
                {"role": "user", "content": None},
            ]
        }
        assert has_images(body) is False

    def test_has_images_no_messages(self):
        """has_images returns False for empty messages list."""
        body = {"messages": []}
        assert has_images(body) is False

    def test_has_images_no_user_messages(self):
        """has_images returns False when only assistant messages exist."""
        body = {
            "messages": [
                {"role": "assistant", "content": [{"type": "image", "source": {"data": "x"}}]},
            ]
        }
        assert has_images(body) is False


class TestMultiTurnHistory:
    """Test multi-turn conversation with image cache clearing.

    The proxy processes full conversation history on each turn (stateless).
    Session cache is cleared before repopulating to ensure fresh images each turn.
    Placeholder numbering is sequential within each request.
    """

    def test_two_turn_multi_image_history(self):
        """Verifies stateless request processing with cache clearing.

        Turn 1: 2 images -> cached as #1, #2
        Turn 2: Full history resent (with placeholders) + 1 new image
                -> cache cleared, then new image cached as #1

        This is the correct behavior for stateless request processing where
        the client sends full history each turn and expects fresh image caching.
        """
        session_id = "multi_turn_session"
        cache = ImageCache(max_size=100, ttl=300)

        # === TURN 1: User sends message with 2 images ===
        turn1_body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Check these images:"},
                    {"type": "image", "source": {"data": "turn1_img1_data", "media_type": "image/png"}},
                    {"type": "text", "text": "and"},
                    {"type": "image", "source": {"data": "turn1_img2_data", "media_type": "image/jpeg"}},
                ]},
            ]
        }

        turn1_result = strip_and_cache_images(turn1_body, session_id, cache)

        # Verify Turn 1 placeholders
        turn1_content = turn1_result["messages"][0]["content"]
        assert turn1_content[0]["text"] == "Check these images:"
        assert "[Image #1]" in turn1_content[1]["text"]
        assert 'analyzeImage(imageId=["1"])' in turn1_content[1]["text"]
        assert "[Image #2]" in turn1_content[3]["text"]
        assert 'analyzeImage(imageId=["2"])' in turn1_content[3]["text"]

        # Verify Turn 1 images cached
        cached_turn1_img1 = cache.get(session_id, f"{session_id}_Image#1")
        cached_turn1_img2 = cache.get(session_id, f"{session_id}_Image#2")
        assert cached_turn1_img1 == {"data": "turn1_img1_data", "media_type": "image/png"}
        assert cached_turn1_img2 == {"data": "turn1_img2_data", "media_type": "image/jpeg"}

        # === TURN 2: Full history resent with placeholders + new image ===
        # Client sends full conversation history (turn 1 already has placeholders)
        turn2_body = {
            "messages": [
                # Turn 1 history (placeholders already in place)
                {"role": "user", "content": [
                    {"type": "text", "text": "Check these images:"},
                    {"type": "text", "text": '[Image #1] — YOU CANNOT SEE THIS IMAGE. Call analyzeImage(imageId=["1"]) to view it.'},
                    {"type": "text", "text": "and"},
                    {"type": "text", "text": '[Image #2] — YOU CANNOT SEE THIS IMAGE. Call analyzeImage(imageId=["2"]) to view it.'},
                ]},
                # Turn 1 assistant response
                {"role": "assistant", "content": "I'll analyze those images for you."},
                # Turn 2: NEW user message with 1 NEW image
                {"role": "user", "content": [
                    {"type": "text", "text": "Also check this new one:"},
                    {"type": "image", "source": {"data": "turn2_new_img_data", "media_type": "image/webp"}},
                ]},
            ]
        }

        # Cache is cleared at start of turn2 processing, so new image becomes #1
        turn2_result = strip_and_cache_images(turn2_body, session_id, cache)

        # Turn 1 messages unchanged (placeholders remain)
        turn1_user_content = turn2_result["messages"][0]["content"]
        assert "[Image #1]" in turn1_user_content[1]["text"]
        assert "[Image #2]" in turn1_user_content[3]["text"]

        # Turn 2 new message gets placeholder #1 (cache was cleared)
        turn2_user_content = turn2_result["messages"][2]["content"]
        assert turn2_user_content[0]["text"] == "Also check this new one:"
        assert "[Image #1]" in turn2_user_content[1]["text"]
        assert 'analyzeImage(imageId=["1"])' in turn2_user_content[1]["text"]

        # === CACHE VERIFICATION ===
        # Only the new image is in cache (old ones were cleared)
        cached_turn2_img = cache.get(session_id, f"{session_id}_Image#1")
        assert cached_turn2_img == {"data": "turn2_new_img_data", "media_type": "image/webp"}

        # Old images are gone (cache was cleared before repopulating)
        assert cache.get(session_id, f"{session_id}_Image#2") is None

        # Total images in cache: 1 (only the new one)
        session_cache = cache._sessions.get(session_id, {})
        assert len(session_cache) == 1
