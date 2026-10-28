"""
Import_Modules / phase2_imports.py
==================================
Central Hub exporting all Phase 2 capabilities (Vision Perception, BGM Selection,
Rhythm Speed Ramps, FFmpeg Master Director Synthesis, and Rendering).

Importing from this file ensures all Phase 2 module pointers are managed in ONE SINGLE location.

Usage:
  from Import_Modules.phase2_imports import run_phase2_orchestration, edit_video_master
"""

import os
import sys
import importlib

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SAMPLE_UPDATE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for candidate in (_REPO_ROOT, _SAMPLE_UPDATE_ROOT):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

# Lazy wrappers for indexed step modules to allow lightweight imports without numpy/torch
def step01_scan_clip_targets(*args, **kwargs):
    m = importlib.import_module("Phase_2.01_folder_scanner")
    return m.scan_clip_targets(*args, **kwargs)

def step02_run_forensic_perception(*args, **kwargs):
    m = importlib.import_module("Phase_2.02_forensic_perception")
    return m.run_forensic_perception(*args, **kwargs)

def step03_extract_targeted_frames(*args, **kwargs):
    m = importlib.import_module("Phase_2.03_vector_frame_extractor")
    return m.extract_targeted_frames(*args, **kwargs)

def step04_select_clip_bgm(*args, **kwargs):
    m = importlib.import_module("Phase_2.04_bgm_selector")
    return m.select_clip_bgm(*args, **kwargs)

def step05_build_rhythm_timeline(*args, **kwargs):
    m = importlib.import_module("Phase_2.05_rhythm_timeline")
    return m.build_rhythm_timeline(*args, **kwargs)

def step06_synthesize_editing_plan(*args, **kwargs):
    m = importlib.import_module("Phase_2.06_ffmpeg_synthesis")
    return m.synthesize_editing_plan(*args, **kwargs)

def step07_verify_master_render(*args, **kwargs):
    m = importlib.import_module("Phase_2.07_master_render")
    return m.verify_master_render(*args, **kwargs)

def run_phase2_orchestration(*args, **kwargs):
    from Phase_2.phase2_orchestrator import run_phase2_pipeline
    return run_phase2_pipeline(*args, **kwargs)

def edit_video_master(*args, **kwargs):
    from Main_Modules.master_ai_editor import edit_video_master as _evm
    return _evm(*args, **kwargs)

__all__ = [
    "step01_scan_clip_targets",
    "step02_run_forensic_perception",
    "step03_extract_targeted_frames",
    "step04_select_clip_bgm",
    "step05_build_rhythm_timeline",
    "step06_synthesize_editing_plan",
    "step07_verify_master_render",
    "run_phase2_orchestration",
    "edit_video_master",
]
