import os
import sys
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Gemini_Modules.gemini_ffmpeg_synthesis import enforce_route_params, GeminiFFmpegEngine


def test_route_params_overrides_plan():
    """
    Fake plan with wrong speed, wrong transition, and wrong duck level:
    assert each is overridden by route_params.
    """
    fake_plan = {
        "editing_intent": "Test route enforcement overrides",
        "operations": [
            {"operation_type": "speed_ramp", "speed_factor": 1.0},
            {"operation_type": "transition", "transition_type": "fade"},
            {"operation_type": "audio_ducking_mix", "ducking_threshold": 0.5}
        ]
    }
    route = {
        "speed_factor": 1.5,
        "transition_type": "wipeleft",
        "bgm_ducking_db": -20.0
    }
    flags = {"enable_speed_ramps": True}

    new_plan, changes = enforce_route_params(fake_plan, route=route, flags=flags)

    ops = new_plan["operations"]

    # 1. Speed factor overridden
    speed_op = next(op for op in ops if op["operation_type"] == "speed_ramp")
    assert speed_op["speed_factor"] == 1.5
    assert any("speed_factor 1.0 -> 1.5" in c for c in changes)

    # 2. Transition type overridden with valid executor transition
    trans_op = next(op for op in ops if op["operation_type"] == "transition")
    assert trans_op["transition_type"] == "wipeleft"
    assert any("transition_type fade -> wipeleft" in c for c in changes)

    # 3. Ducking threshold overridden
    duck_op = next(op for op in ops if op["operation_type"] == "audio_ducking_mix")
    assert duck_op["ducking_threshold"] == -20.0
    assert any("ducking_threshold 0.5 -> -20.0" in c for c in changes)


def test_route_speed_untouched_when_ramps_disabled():
    """
    Precedence: when enable_speed_ramps is False, speed must stay untouched.
    """
    fake_plan = {
        "editing_intent": "Test disabled speed ramps keeps speed untouched",
        "operations": [
            {"operation_type": "speed_change", "speed_factor": 1.25},
            {"operation_type": "transition", "transition_type": "fade"},
            {"operation_type": "audio_ducking_mix", "ducking_threshold": 0.4}
        ]
    }
    route = {
        "speed_factor": 1.8,
        "transition_type": "slideleft",
        "bgm_ducking_db": -12.0
    }
    flags = {"enable_speed_ramps": False}

    new_plan, changes = enforce_route_params(fake_plan, route=route, flags=flags)

    ops = new_plan["operations"]

    # Speed factor remains untouched
    speed_op = next(op for op in ops if op["operation_type"] == "speed_change")
    assert speed_op["speed_factor"] == 1.25
    assert not any("speed_factor" in c for c in changes)

    # Transition and ducking are still enforced
    trans_op = next(op for op in ops if op["operation_type"] == "transition")
    assert trans_op["transition_type"] == "slideleft"

    duck_op = next(op for op in ops if op["operation_type"] == "audio_ducking_mix")
    assert duck_op["ducking_threshold"] == -12.0


def test_route_speed_clamping():
    """
    Assert route speed_factor is clamped to [0.5, 2.0].
    """
    # Over 2.0 -> clamped to 2.0
    plan1 = {"operations": [{"operation_type": "speed_change", "speed_factor": 1.0}]}
    res1, changes1 = enforce_route_params(plan1, route={"speed_factor": 3.5})
    assert res1["operations"][0]["speed_factor"] == 2.0
    assert any("speed_factor 1.0 -> 2.0" in c for c in changes1)

    # Under 0.5 -> clamped to 0.5
    plan2 = {"operations": [{"operation_type": "speed_change", "speed_factor": 1.0}]}
    res2, changes2 = enforce_route_params(plan2, route={"speed_factor": 0.1})
    assert res2["operations"][0]["speed_factor"] == 0.5
    assert any("speed_factor 1.0 -> 0.5" in c for c in changes2)


def test_route_speed_added_when_missing():
    """
    When plan has no speed ops and route speed_factor != 1.0, add speed_change op.
    """
    plan = {
        "operations": [
            {"operation_type": "trim", "start_time": 0.0, "end_time": 4.0}
        ]
    }
    res, changes = enforce_route_params(plan, route={"speed_factor": 1.4})
    speed_op = next((op for op in res["operations"] if op["operation_type"] == "speed_change"), None)
    assert speed_op is not None
    assert speed_op["speed_factor"] == 1.4
    assert any("speed_factor None -> 1.4" in c for c in changes)


def test_route_unsupported_transition_ignored():
    """
    Unsupported transition enum value (e.g. whip_pan) must NOT override plan.
    """
    plan = {
        "operations": [
            {"operation_type": "transition", "transition_type": "dissolve"}
        ]
    }
    res, changes = enforce_route_params(plan, route={"transition_type": "whip_pan"})
    assert res["operations"][0]["transition_type"] == "dissolve"
    assert len(changes) == 0


def test_engine_enforce_route_params_method():
    """
    Verify GeminiFFmpegEngine instance method delegates to enforce_route_params.
    """
    engine = GeminiFFmpegEngine()
    plan = {
        "operations": [
            {"operation_type": "speed_ramp", "speed_factor": 1.0}
        ]
    }
    res, changes = engine.enforce_route_params(plan, route={"speed_factor": 1.6})
    assert res["operations"][0]["speed_factor"] == 1.6
    assert len(changes) == 1
