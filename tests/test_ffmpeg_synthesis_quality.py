import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from Gemini_Modules.gemini_ffmpeg_synthesis import FFmpegCommandGenerator, GeminiFFmpegEngine

def test_encoder_flags_lossless_intermediate():
    gen = FFmpegCommandGenerator()
    
    # 1. Intermediate mode must be lossless (CRF 0, ultrafast)
    flags_inter = gen._get_encoder_flags(is_intermediate=True)
    assert "-crf" in flags_inter
    crf_idx = flags_inter.index("-crf")
    assert flags_inter[crf_idx + 1] == "0"
    assert "-preset" in flags_inter
    preset_idx = flags_inter.index("-preset")
    assert flags_inter[preset_idx + 1] == "ultrafast"
    # Lossless mode must NOT specify profile:v main (which breaks x264)
    assert "main" not in flags_inter

    # 2. Master delivery mode must be high quality (default CRF 16, preset medium)
    flags_master = gen._get_encoder_flags(is_intermediate=False)
    assert "-crf" in flags_master
    m_crf_idx = flags_master.index("-crf")
    assert flags_master[m_crf_idx + 1] == "16"
    assert "-profile:v" in flags_master
    prof_idx = flags_master.index("-profile:v")
    assert flags_master[prof_idx + 1] == "high"


def test_single_pass_clean_bgm_no_dangling_audio_pad():
    gen = FFmpegCommandGenerator()
    
    # Simulate the exact conditions from the runner log:
    # 6 trims, clean BGM replacement (video_volume=0.00, music_volume=0.90)
    micro_shots = [
        {"start": 0.0, "end": 2.0},
        {"start": 3.94, "end": 6.0},
        {"start": 8.0, "end": 12.0},
    ]
    gemini_ops = [
        {"operation_type": "trim", "start_time": 0.0, "end_time": 2.0},
        {"operation_type": "trim", "start_time": 3.94, "end_time": 6.0},
        {"operation_type": "trim", "start_time": 8.0, "end_time": 12.0},
        {"operation_type": "scale_aspect", "mode": "blur_pad"},
        {"operation_type": "bgm_mix", "music_volume": 0.90, "video_volume": 0.00, "audio_start_time": 4.0}
    ]
    
    step = gen.build_single_pass_filtergraph(
        input_path="dummy_input.mp4",
        output_path="dummy_output.mp4",
        micro_shots=micro_shots,
        bgm_path="dummy_bgm.wav",
        gemini_operations=gemini_ops,
        extra_inputs={"preserve_original_audio": False}
    )
    
    cmd_str = step["terminal_command"]
    # With video_volume=0.0 and external BGM, [ac] should NEVER be produced by concat
    # and [bgm_chain] connects to [aout]
    assert "[aout]" in cmd_str
    # If concat is used, it must NOT emit [ac] when input audio is not wanted
    assert "concat=n=3:v=1:a=0[vc]" in cmd_str or "[ac]" not in cmd_str


def test_synthesize_from_gemini_json_intermediate_vs_master_encoding():
    engine = GeminiFFmpegEngine()
    gemini_json = {
        "editing_intent": "Rhythm short with multi trim and BGM",
        "operations": [
            {"operation_type": "trim", "start_time": 0.0, "end_time": 2.0},
            {"operation_type": "trim", "start_time": 4.0, "end_time": 6.0},
            {"operation_type": "scale_aspect", "mode": "blur_pad"},
            {"operation_type": "brand_watermark", "text": "TEST_BRAND"}
        ]
    }
    
    res = engine.synthesize_from_gemini_json(
        gemini_json,
        input_path="test_in.mp4",
        output_path="test_out.mp4"
    )
    
    steps = res["command_steps"]
    assert len(steps) >= 3
    
    # Sub-trims and intermediate steps must use lossless intermediate encoding (crf 0, ultrafast)
    trim_steps = [s for s in steps if s.get("operation") == "trim"]
    for ts in trim_steps:
        cmd_str = ts["terminal_command"]
        assert "-crf 0" in cmd_str
        assert "-preset ultrafast" in cmd_str
    
    # The final step targeting output_path must use master delivery encoding (crf 16, preset medium)
    final_step = steps[-1]
    assert final_step["output"] == "test_out.mp4"
    assert "-crf 16" in final_step["terminal_command"]
    assert "-preset medium" in final_step["terminal_command"]


def test_execute_pipeline_mode_selection():
    engine = GeminiFFmpegEngine()
    gemini_json = {
        "editing_intent": "Multi-pass validation test",
        "operations": [
            {"operation_type": "scale_aspect", "mode": "blur_pad"}
        ]
    }
    
    # 1. When force_multipass is True, execute_pipeline returns mode="MULTI_PASS"
    res = engine.execute_pipeline(
        gemini_json=gemini_json,
        input_path="dummy_input.mp4",
        output_path="dummy_output.mp4",
        extra_inputs={"force_multipass": True},
        dry_run=True
    )
    assert res.get("status") == "SUCCESS"
    assert res.get("mode") == "MULTI_PASS"

    # 2. When synthesis_mode="multi_pass", execute_pipeline returns mode="MULTI_PASS"
    res2 = engine.execute_pipeline(
        gemini_json=gemini_json,
        input_path="dummy_input.mp4",
        output_path="dummy_output.mp4",
        extra_inputs={"synthesis_mode": "multi_pass"},
        dry_run=True
    )
    assert res2.get("status") == "SUCCESS"
    assert res2.get("mode") == "MULTI_PASS"

