"""Compatibility wrapper for ``claude_proxy.sse``."""

import sys

from claude_relay import sse as _module

sys.modules[__name__] = _module
