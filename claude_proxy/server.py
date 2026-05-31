"""Compatibility wrapper for ``claude_proxy.server``."""

import sys

from claude_relay import server as _module

sys.modules[__name__] = _module
