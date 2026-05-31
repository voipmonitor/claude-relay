"""Compatibility wrapper for ``claude_proxy.convert_request``."""

import sys

from claude_relay import convert_request as _module

sys.modules[__name__] = _module
