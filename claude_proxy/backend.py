"""Compatibility wrapper for ``claude_proxy.backend``."""

import sys

from claude_relay import backend as _module

sys.modules[__name__] = _module
