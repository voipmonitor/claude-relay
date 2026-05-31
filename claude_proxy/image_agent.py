"""Compatibility wrapper for ``claude_proxy.image_agent``."""

import sys

from claude_relay import image_agent as _module

sys.modules[__name__] = _module
