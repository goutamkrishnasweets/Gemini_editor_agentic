import os
import sys
import importlib
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Gemini_Modules.gemini_ffmpeg_synthesis import GeminiFFmpegEngine, enforce_feature_flags
from Watermark_and_Inpainting.static_patch_engine import StaticPatchReuseEngine

# Dynamic import for module starting with digits
rhythm_mod = importlib.import_module("Phase_2.05_rhythm_timeline")
_calculate_optimal_audio_window = rhythm_mod._calculate_optimal_audio_window


def test_schema_allows_gentle_and_zoom_modes():
    """Ensure GEMINI_FFMPEG_SCHEMA accepts gentle, zoom_in, zoom_out in mode."""
    engine = GeminiFFmpegEngine()
    for mode in ["gentle", "zoom_in", "zoom_out"]:
        plan = {
            "editing_intent": f"Test {mode} mode",
            "operations": [
                {
                    "operation_type": "cinematic_zoom",
                    "mode": mode,
                    "zoom_rate": 0.08
                }
            ]
        }
        assert engine.validate_schema(plan), f"Schema rejected valid zoom mode: {mode}"


def test_enforce_feature_flags_zoom_passes_schema():
    """Ensure enforce_feature_flags auto-added gentle zoom passes validate_schema."""
    engine = GeminiFFmpegEngine()
    plan = {
        "editing_intent": "Test auto gentle zoom",
        "operations": []
    }
    flags = {"enable_cinematic_zoom": True}
    new_plan, changes = enforce_feature_flags(plan, flags=flags)
    assert engine.validate_schema(new_plan), "Auto-added gentle zoom failed schema validation"


def test_optimal_audio_window_handles_dict_tension_and_drops():
    """Ensure _calculate_optimal_audio_window doesn't crash on dict elements in tension_arc or drops."""
    lyric_intel = {
        "sections": [
            {"type": "verse", "start": {"time": 0.0}},
            {"type": "chorus", "start": 10.0}
        ],
        "drops": [
            {"time": 4.0, "intensity": 0.9},
            5.0
        ],
        "tension_arc": [
            {"time": {"start": 0.0}, "tension": {"value": 0.3}},
            {"time": 2.0, "tension": 0.7},
            {"time": 4.0, "tension": 0.9}
        ]
    }
    offset = _calculate_optimal_audio_window(lyric_intel, bgm_duration=15.0, target_duration=5.0)
    assert isinstance(offset, float)
    assert offset >= 0.0


def test_static_patch_reuse_engine_interface():
    """Ensure StaticPatchReuseEngine implements analyze_stability and apply_patch."""
    assert hasattr(StaticPatchReuseEngine, "analyze_stability")
    assert hasattr(StaticPatchReuseEngine, "apply_patch")
    mode = StaticPatchReuseEngine.analyze_stability("dummy_nonexistent.mp4")
    assert mode in ("dynamic", "static", "rigid_motion")
    assert StaticPatchReuseEngine.apply_patch("dummy.mp4", [], "out.mp4") is False
