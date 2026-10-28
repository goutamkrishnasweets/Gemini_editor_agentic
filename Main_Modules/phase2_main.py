"""
phase2_main.py — Phase 2 Orchestrator (Master AI Editing & Synthesis)
================================================────────────────======
Phase 2 Orchestrator CLI for AMTCE.

Workflow:
1. Scans `downloads/` for clip subfolders containing `video.mp4` & `metadata.json`.
2. Runs OpenCV Res10 300x300 Caffe SSD face detection & updates RAG Creator Face Store in `cache/face_cache/`.
3. Pre-scans candidate BGM audio track metadata (BPM, Energy, Beat Drops) in `Audio_Modules/Original_audio/`.
4. Sends keyframes + creator hint + audio candidate table to Gemini 2.5 Flash Vision.
5. Gemini selects matching BGM track and generates structured `creative_possibilities` ClipPlan.
6. Calls `master_ai_editor.py` to synthesize FFmpeg filtergraph (cuts, rhythm speed ramps, audio ducking) and render final master reel into `master_edits/`.

Usage:
    python phase2_main.py
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Callable, Union, Set

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase2_main")

# Ensure the canonical workspace root is on sys.path before the sample tree.
# This lets imports resolve to the real AMTCE modules instead of the nested
# shadow copies under `simpler update/`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SAMPLE_UPDATE_ROOT = os.path.dirname(os.path.abspath(__file__))

# Remove any direct sample-tree path entries that would otherwise shadow the
# real package roots. We then place the canonical repo root first and keep the
# sample tree available only as a fallback.
for shadow_path in [p for p in sys.path if os.path.abspath(p) == os.path.abspath(_SAMPLE_UPDATE_ROOT)]:
    sys.path.remove(shadow_path)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SAMPLE_UPDATE_ROOT not in sys.path:
    sys.path.append(_SAMPLE_UPDATE_ROOT)

from Phase_2.phase2_orchestrator import run_phase2_pipeline

def run_phase2_orchestration(
    downloads_dir: Optional[str] = None,
    master_edits_dir: Optional[str] = None,
    limit: Optional[int] = None,
    input_path: Optional[str] = None,
    target_dirs: Optional[List[str]] = None,
    skip_existing: bool = False,
    on_rendered_callback: Optional[Callable[[str], None]] = None,
    user_edit_directive: Optional[str] = None,
    excluded_audio: Optional[Union[str, List[str], Set[str]]] = None,
    requestor_chat_id: Optional[Union[int, str]] = None,
) -> Dict[str, Any]:
    """
    Delegates Phase 2 Master AI Editing Pipeline to Phase_2.phase2_orchestrator.
    """
    return run_phase2_pipeline(
        input_path=input_path,
        downloads_dir=downloads_dir,
        master_edits_dir=master_edits_dir,
        limit=limit,
        target_dirs=target_dirs,
        skip_existing=skip_existing,
        on_rendered_callback=on_rendered_callback,
        user_edit_directive=user_edit_directive,
        excluded_audio=excluded_audio,
        requestor_chat_id=requestor_chat_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMTCE Phase 2 Orchestrator (Master AI Editing)")
    parser.add_argument("-i", "--input", type=str, default=None, help="Target single input clip subfolder or video file")
    parser.add_argument("--downloads", type=str, default=None, help="Input downloads directory (for batch mode)")
    parser.add_argument("--output", type=str, default=None, help="Output master edits directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clips to edit")
    parser.add_argument("--skip-existing", action="store_true", help="Skip clips that already have rendered master edits in output directory (default: False, always render fresh creative possibilities)")
    args = parser.parse_args()

    run_phase2_orchestration(
        input_path=args.input,
        downloads_dir=args.downloads,
        master_edits_dir=args.output,
        limit=args.limit,
        skip_existing=args.skip_existing
    )
