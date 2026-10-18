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

    # Simulate 429 RPD daily quota exhaustion (real-time difference until 12:00 AM midnight)
    gemini_router.mark_model_banned(m2, error_type="429", error_message="ResourceExhausted: 429 GenerateRequestsPerDay quota exceeded")
    assert gemini_router.model_states[m2]["ban_remaining_seconds"] > 60

    # Simulate 403 Permission Denied (24h ban)
    gemini_router.mark_model_banned(m2, error_type="403", error_message="PermissionDenied: 403 Your project has been denied access")
    assert gemini_router.model_states[m2]["ban_remaining_seconds"] == 86400

    # Clean up test bans
    with gemini_router.state_lock:
        for state in gemini_router.model_states.values():
            state["status"] = "ACTIVE"
            state["ban_remaining_seconds"] = 0


def test_rpd_exact_time_difference_to_midnight():
    """Verify RPD quarantine calculates the exact seconds from any time (e.g. 3pm) to 12:00 AM midnight."""
    from datetime import datetime
    from unittest.mock import patch
    from Gemini_Modules.gemini_router_module.gemini_governor import get_seconds_until_midnight

    # Simulate 3:00 PM (15:00:00) -> 9 hours to 12:00 AM midnight
    fake_3pm = datetime(2026, 10, 19, 15, 0, 0)
    with patch("Gemini_Modules.gemini_router_module.gemini_governor.datetime") as mock_dt:
        mock_dt.now.return_value = fake_3pm
        diff = get_seconds_until_midnight()
        assert diff == 9 * 3600  # Exactly 32,400 seconds (9.0 hours)

    # Simulate 10:30 PM (22:30:00) -> 1.5 hours to 12:00 AM midnight
    fake_1030pm = datetime(2026, 10, 19, 22, 30, 0)
    with patch("Gemini_Modules.gemini_router_module.gemini_governor.datetime") as mock_dt:
        mock_dt.now.return_value = fake_1030pm
        diff = get_seconds_until_midnight()
        assert diff == 5400  # Exactly 5,400 seconds (1.5 hours)


def test_creative_task_fallback_returns_none_avoiding_json_decode_error():
    """Verify governor absolute fallback returns None for 'creative' to avoid JSON parsing errors in synthesis."""
    # When all models and Ollama are unavailable, 'creative' should return None, but 'caption' returns clean string
    with gemini_router.state_lock:
        for s in gemini_router.model_states.values():
            s["status"] = "BANNED"
            s["ban_remaining_seconds"] = 86400

    from unittest.mock import patch
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-test"}), patch.object(gemini_router, "_call_ollama", return_value=None):
        creative_res = gemini_router.generate(
            prompt="test prompt",
            task_type="creative",
            module_name="test_synthesis"
        )
        assert creative_res is None, "Expected None for 'creative' fallback so callers trigger structured defaults"

        caption_res = gemini_router.generate(
            prompt="test prompt",
            task_type="caption",
            module_name="test_caption"
        )
        assert isinstance(caption_res, str) and len(caption_res) > 0, "Expected string for 'caption' text fallback"

    # Reset states
    with gemini_router.state_lock:
        for s in gemini_router.model_states.values():
            s["status"] = "ACTIVE"
            s["ban_remaining_seconds"] = 0

