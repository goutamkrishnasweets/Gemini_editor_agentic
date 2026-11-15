import os
import sys
import subprocess
import pytest
import numpy as np
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Gemini_Modules.gemini_ffmpeg_synthesis import enforce_feature_flags, GeminiFFmpegEngine


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


def test_real_cinematic_zoom_render(tmp_path):
    in_clip = str(tmp_path / "synthetic_3s.mp4")
    out_clip = str(tmp_path / "zoomed_out.mp4")

    # 1. Render 3s synthetic clip with ffmpeg testsrc
    p_in = subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "testsrc=duration=3:size=1080x1920:rate=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        in_clip
    ], capture_output=True, text=True)
    assert p_in.returncode == 0, f"ffmpeg testsrc failed: {p_in.stderr}"

    # 2. Execute through real executor with a zoom op
    engine = GeminiFFmpegEngine()
    plan = {
        "editing_intent": "Test cinematic zoom real render",
        "operations": [
            {
                "operation_type": "cinematic_zoom",
                "duration": 3.0,
                "zoom_rate": 0.08,
                "target_width": 1080,
                "target_height": 1920,
                "mode": "zoom_in"
            }
        ]
    }
    res = engine.execute_pipeline(
        gemini_json=plan,
        input_path=in_clip,
        output_path=out_clip,
        extra_inputs={"force_multipass": True, "source_duration_s": 3.0}
    )

    # Assert ffmpeg execution succeeded (exit code 0 / status SUCCESS)
    assert os.path.exists(out_clip), f"Output video was not generated: {res}"
    assert os.path.getsize(out_clip) > 1024, "Output video file is empty"
    assert res.get("status") in ("SUCCESS", None), f"Pipeline execution failed: {res}"

    # 3. Extract first frame (t=0.1s) and last frame (t=2.8s)
    f0_path = str(tmp_path / "frame_first.png")
    f1_path = str(tmp_path / "frame_last.png")

    p0 = subprocess.run([
        "ffmpeg", "-y", "-ss", "0.1", "-i", out_clip,
        "-frames:v", "1", f0_path
    ], capture_output=True)
    assert p0.returncode == 0, f"Extract first frame failed: {p0.stderr}"

    p1 = subprocess.run([
        "ffmpeg", "-y", "-ss", "2.8", "-i", out_clip,
        "-frames:v", "1", f1_path
    ], capture_output=True)
    assert p1.returncode == 0, f"Extract last frame failed: {p1.stderr}"

    # 4. Compare first vs last frame to verify real zoom
    img0 = np.array(Image.open(f0_path)).astype(np.float32)
    img1 = np.array(Image.open(f1_path)).astype(np.float32)

    diff = np.abs(img0 - img1)
    # Assert frames differ consistent with 8% zoom over 3s
    assert diff.mean() > 0.0, "First and last frames are identical: zoom did not affect render"
