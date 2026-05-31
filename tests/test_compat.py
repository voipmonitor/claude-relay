"""Compatibility tests for the old claude_proxy module name."""

import importlib


def test_old_package_imports_alias_new_modules():
    relay_server = importlib.import_module("claude_relay.server")
    proxy_server = importlib.import_module("claude_proxy.server")

    assert proxy_server is relay_server


def test_old_entrypoint_imports_new_main():
    relay_main = importlib.import_module("claude_relay.__main__")
    proxy_main = importlib.import_module("claude_proxy.__main__")

    assert proxy_main.main is relay_main.main
