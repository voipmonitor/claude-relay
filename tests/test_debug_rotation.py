"""Tests for debug file rotation/cleanup functionality."""
import os
import time
import pytest


class TestCleanupDebugFiles:
    """Test _cleanup_debug_files() function behavior."""

    def test_deletes_old_files(self, tmp_path, monkeypatch):
        """Files older than max_age_hours are deleted."""
        old_file = tmp_path / "old.json"
        old_file.write_text("{}")
        old_time = time.time() - (3 * 3600)
        os.utime(old_file, (old_time, old_time))

        import claude_relay.server as server_mod
        monkeypatch.setattr(server_mod, "DEBUG_DIR", str(tmp_path))

        deleted = server_mod._cleanup_debug_files(max_age_hours=2)

        assert deleted == 1
        assert not old_file.exists()

    def test_keeps_recent_files(self, tmp_path, monkeypatch):
        """Files newer than max_age_hours are preserved."""
        recent_file = tmp_path / "recent.json"
        recent_file.write_text("{}")

        import claude_relay.server as server_mod
        monkeypatch.setattr(server_mod, "DEBUG_DIR", str(tmp_path))

        deleted = server_mod._cleanup_debug_files(max_age_hours=2)

        assert deleted == 0
        assert recent_file.exists()

    def test_ignores_non_json_files(self, tmp_path, monkeypatch):
        """Non-debug-dump files are ignored."""
        old_file = tmp_path / "old.txt"
        old_file.write_text("test")
        old_time = time.time() - (3 * 3600)
        os.utime(old_file, (old_time, old_time))

        import claude_relay.server as server_mod
        monkeypatch.setattr(server_mod, "DEBUG_DIR", str(tmp_path))

        deleted = server_mod._cleanup_debug_files(max_age_hours=2)

        assert deleted == 0
        assert old_file.exists()

    def test_deletes_old_ndjson_files(self, tmp_path, monkeypatch):
        """Raw stream dump files are deleted with other debug dumps."""
        old_file = tmp_path / "old.ndjson"
        old_file.write_text("{}\n")
        old_time = time.time() - (3 * 3600)
        os.utime(old_file, (old_time, old_time))

        import claude_relay.server as server_mod
        monkeypatch.setattr(server_mod, "DEBUG_DIR", str(tmp_path))

        deleted = server_mod._cleanup_debug_files(max_age_hours=2)

        assert deleted == 1
        assert not old_file.exists()

    def test_handles_missing_directory(self, monkeypatch):
        """Missing directory returns 0 without error."""
        import claude_relay.server as server_mod
        monkeypatch.setattr(server_mod, "DEBUG_DIR", "/nonexistent/path")

        deleted = server_mod._cleanup_debug_files(max_age_hours=2)
        assert deleted == 0

    def test_handles_subdirectories(self, tmp_path, monkeypatch):
        """Subdirectories are ignored (only files deleted)."""
        old_dir = tmp_path / "old_dir"
        old_dir.mkdir()
        old_time = time.time() - (3 * 3600)
        os.utime(old_dir, (old_time, old_time))

        import claude_relay.server as server_mod
        monkeypatch.setattr(server_mod, "DEBUG_DIR", str(tmp_path))

        deleted = server_mod._cleanup_debug_files(max_age_hours=2)

        assert deleted == 0
        assert old_dir.exists()

    def test_mixed_old_and_recent_files(self, tmp_path, monkeypatch):
        """Only old files are deleted, recent ones preserved."""
        old_file = tmp_path / "old.json"
        old_file.write_text("{}")
        old_time = time.time() - (3 * 3600)
        os.utime(old_file, (old_time, old_time))

        recent_file = tmp_path / "recent.json"
        recent_file.write_text("{}")

        import claude_relay.server as server_mod
        monkeypatch.setattr(server_mod, "DEBUG_DIR", str(tmp_path))

        deleted = server_mod._cleanup_debug_files(max_age_hours=2)

        assert deleted == 1
        assert not old_file.exists()
        assert recent_file.exists()

    def test_concurrent_deletion_race(self, tmp_path, monkeypatch):
        """Handles files deleted concurrently (OSError during removal)."""
        test_file = tmp_path / "test.json"
        test_file.write_text("{}")
        old_time = time.time() - (3 * 3600)
        os.utime(test_file, (old_time, old_time))

        def raise_os_error(*args):
            raise OSError("File already deleted")

        monkeypatch.setattr(os, "remove", raise_os_error)

        import claude_relay.server as server_mod
        monkeypatch.setattr(server_mod, "DEBUG_DIR", str(tmp_path))

        deleted = server_mod._cleanup_debug_files(max_age_hours=2)
        assert deleted == 0
