"""Compatibility wrapper for ``claude_proxy.convert_stream``."""

import sys

from claude_relay import convert_stream as _module

sys.modules[__name__] = _module
