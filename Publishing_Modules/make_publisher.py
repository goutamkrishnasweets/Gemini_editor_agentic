"""
Publishing_Modules / make_publisher.py — DEPRECATED COMPATIBILITY SHIM
================================================================================
Make.com webhook publishing has been deprecated and replaced by the native
Buffer API (buffer_publisher.py) for direct multi-channel syndication.

This shim exists solely to prevent import errors in legacy scripts.
"""

from typing import Optional, Dict, Any
from Publishing_Modules.buffer_publisher import PublishQueue, resolve_public_media_url


def is_make_configured(user_id: Optional[str] = None) -> bool:
    """Make.com is deprecated in favor of native Buffer API."""
    return False


def publish_to_make(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Deprecated: Make.com publishing is replaced by Buffer."""
    return {
        "status": "skipped",
        "message": "Make.com publisher is deprecated. Multi-channel syndication is handled by buffer_publisher."
    }


def build_make_payload(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Deprecated dummy payload builder."""
    return {}


__all__ = [
    "is_make_configured",
    "publish_to_make",
    "build_make_payload",
    "resolve_public_media_url",
    "PublishQueue",
]
