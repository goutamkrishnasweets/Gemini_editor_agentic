import os
import sys
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Gemini_Modules.gemini_ffmpeg_synthesis import enforce_feature_flags


def test_feature_flags_disabled():
    fake_plan = {
        "editing_intent": "Test flags disabling",
        "voiceover": "/tmp/fake_vo.mp3",
        "operations": [
            {"operation_type": "speed_ramp", "speed_factor": 1.5},
            {"operation_type": "cinematic_zoom", "mode": "gentle"},
            {
                "operation_type": "multi_trim",
                "trims": [
                    {"start_time": 0.0, "end_time": 1.2},
                    {"start_time": 2.0, "end_time": 5.0}
                ]
            },
            {"operation_type": "audio_ducking_mix", "voiceover_input": "/tmp/fake_vo.mp3"}
        ]
    }
    flags = {
        "enable_speed_ramps": False,
        "enable_cinematic_zoom": False,
        "enable_fast_pacing": False,
        "enable_voiceover": False
    }
    extra_inputs = {"voiceover": "/tmp/fake_vo.mp3"}

    new_plan, changes = enforce_feature_flags(fake_plan, flags, extra_inputs=extra_inputs)

    # 1. enable_speed_ramps == False -> remove all speed_ramp operations
    op_types = [op["operation_type"] for op in new_plan["operations"]]
    assert "speed_ramp" not in op_types

    # 2. enable_cinematic_zoom == False -> remove zoom/zoompan operations
    assert "cinematic_zoom" not in op_types
    assert not any("zoom" in t.lower() for t in op_types)

    # 3. enable_fast_pacing == False -> floor multi_trim shot length at 2.0s
    multi_trim_op = next(op for op in new_plan["operations"] if op["operation_type"] == "multi_trim")
    shot1 = multi_trim_op["trims"][0]
    assert (shot1["end_time"] - shot1["start_time"]) >= 2.0

    # 4. enable_voiceover == False -> remove the TTS/voiceover audio input
    assert "voiceover" not in new_plan
    assert "voiceover" not in extra_inputs
    assert "audio_ducking_mix" not in op_types
    assert len(changes) >= 4


def test_feature_flags_enabled():
    fake_plan = {
        "editing_intent": "Test flags enabling",
        "voiceover": "/tmp/fake_vo.mp3",
        "operations": [
            {"operation_type": "speed_ramp", "speed_factor": 1.5},
            {
                "operation_type": "multi_trim",
                "trims": [
                    {"start_time": 0.0, "end_time": 4.5},
                    {"start_time": 5.0, "end_time": 6.0}
                ]
            }
        ]
    }
    flags = {
        "enable_speed_ramps": True,
        "enable_cinematic_zoom": True,
        "enable_fast_pacing": True,
        "enable_voiceover": True
    }
    extra_inputs = {"voiceover": "/tmp/fake_vo.mp3"}

    new_plan, changes = enforce_feature_flags(fake_plan, flags, extra_inputs=extra_inputs)

    op_types = [op["operation_type"] for op in new_plan["operations"]]

    # 1. enable_speed_ramps == True -> speed_ramp kept
    assert "speed_ramp" in op_types

    # 2. enable_cinematic_zoom == True and plan has none -> add one gentle zoom op
    assert any("zoom" in t.lower() for t in op_types)

    # 3. enable_fast_pacing == True -> cap multi_trim shot length at 2.5s max
    multi_trim_op = next(op for op in new_plan["operations"] if op["operation_type"] == "multi_trim")
    shot0 = multi_trim_op["trims"][0]
    assert (shot0["end_time"] - shot0["start_time"]) <= 2.5

    # 4. enable_voiceover == True -> voiceover kept
    assert new_plan.get("voiceover") == "/tmp/fake_vo.mp3"
    assert extra_inputs.get("voiceover") == "/tmp/fake_vo.mp3"


def test_feature_flags_missing_leaves_plan_unchanged():
    fake_plan = {
        "editing_intent": "Test missing flags",
        "voiceover": "/tmp/fake_vo.mp3",
        "operations": [
            {"operation_type": "speed_ramp", "speed_factor": 1.5},
            {"operation_type": "cinematic_zoom", "mode": "gentle"}
        ]
    }
    new_plan, changes = enforce_feature_flags(fake_plan, flags={})
    assert len(changes) == 0
    op_types = [op["operation_type"] for op in new_plan["operations"]]
    assert "speed_ramp" in op_types
    assert "cinematic_zoom" in op_types
    assert new_plan.get("voiceover") == "/tmp/fake_vo.mp3"


def test_forensic_context_nested_flags():
    fake_plan = {
        "editing_intent": "Test nested context",
        "operations": [
            {"operation_type": "speed_ramp", "speed_factor": 1.5}
        ]
    }
    forensic_context = {
        "feature_flags": {
            "enable_speed_ramps": False
        }
    }
    new_plan, changes = enforce_feature_flags(fake_plan, flags=forensic_context)
    op_types = [op["operation_type"] for op in new_plan["operations"]]
    assert "speed_ramp" not in op_types
    assert len(changes) == 1
