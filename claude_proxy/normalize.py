"""Compatibility wrapper for ``claude_proxy.normalize``."""

import sys

from claude_relay import normalize as _module

sys.modules[__name__] = _module
