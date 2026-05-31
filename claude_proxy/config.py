"""Compatibility wrapper for ``claude_proxy.config``."""

import sys

from claude_relay import config as _module

sys.modules[__name__] = _module
