"""
tests/test_model_router_programmatic.py
======================================
Tests verifying programmatic model routing, governor-led rotation,
exclusion of unstable preview models (3.x/thinking), and priority of production Flash.
"""

import os
import sys
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Gemini_Modules.gemini_router_module.list_models import (
    _is_valid_generative_model,
    get_models_by_capability,
    get_active_models_and_ratings,
)
from Gemini_Modules.gemini_router_module.gemini_governor import gemini_router


def test_preview_models_excluded_from_discovery():
    """Verify non-generative, image, and unstable preview models are strictly excluded while 3.x Flash is valid."""
    assert not _is_valid_generative_model("gemini-3.1-pro-preview")
    assert not _is_valid_generative_model("gemini-3-flash-preview")
    assert not _is_valid_generative_model("gemini-2.5-flash-thinking")
    assert not _is_valid_generative_model("gemini-2.5-flash-image")
    assert not _is_valid_generative_model("gemini-2.5-flash-native-audio-preview-09-2025")
    assert not _is_valid_generative_model("gemini-2.0-flash-realtime-exp")
    assert not _is_valid_generative_model("text-embedding-004")
    assert _is_valid_generative_model("gemini-3.8-flash")
    assert _is_valid_generative_model("gemini-3.5-flash")
    assert _is_valid_generative_model("gemini-2.5-flash")
    assert _is_valid_generative_model("gemini-2.0-flash")


def test_reasoning_capability_prioritizes_production_flash():
    """Verify get_models_by_capability puts top production Flash models before Lite and Pro."""
    ranked = get_models_by_capability("reasoning")
    assert len(ranked) >= 2
    assert "flash" in ranked[0].lower()
    assert "flash" in ranked[1].lower()


def test_programmatic_governor_rotation():
    """Verify governor selects top available Flash model and cleanly rotates on 429."""
    # Reset governor model states to ACTIVE for clean test run
    with gemini_router.state_lock:
        for state in gemini_router.model_states.values():
            state["status"] = "ACTIVE"
            state["ban_remaining_seconds"] = 0

    m1 = gemini_router.get_available_model(task_type="reasoning")
    assert m1 is not None and "flash" in m1.lower()

    # Simulate 429 quota exhaustion (RPM/TPM: 45s uniform quarantine)
    gemini_router.mark_model_banned(m1, error_type="429", error_message="ResourceExhausted: 429 Quota exceeded")
    assert gemini_router.model_states[m1]["ban_remaining_seconds"] == 45
    m2 = gemini_router.get_available_model(task_type="reasoning")
    assert m2 != m1

    # Simulate 403 Permission Denied (24h ban)
    gemini_router.mark_model_banned(m2, error_type="403", error_message="PermissionDenied: 403 Your project has been denied access")
    assert gemini_router.model_states[m2]["ban_remaining_seconds"] == 86400

    # Clean up test bans
    with gemini_router.state_lock:
        for state in gemini_router.model_states.values():
            state["status"] = "ACTIVE"
            state["ban_remaining_seconds"] = 0
