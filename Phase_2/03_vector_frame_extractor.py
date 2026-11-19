"""
Phase_2 / 03_vector_frame_extractor.py
======================================
Step 3: OpenCV Vector-Guided Frame Extraction.
Reads visual_vectors.targeted_timestamps_sec (from Gemini Call 1) and extracts exact targeted frames.
Includes OpenCV Optical Flow verification as a defensive guard.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("Phase2.Step03")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Core_Modules.scene_intel import extract_frames_from_vectors, extract_strategic_frame_files


def extract_targeted_frames(
    video_path: str,
    tmp_dir: str,
    visual_vectors: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Extracts targeted frames using Gemini visual_vectors.
    Returns list of absolute JPEG file paths.
    """
    logger.info(f"👁️ [STEP 03] Extracting vector-guided frames for: {os.path.basename(video_path)}")

    if visual_vectors and visual_vectors.get("targeted_timestamps_sec"):
        frame_paths = extract_frames_from_vectors(video_path, visual_vectors, tmp_dir)
    else:
        frame_paths = extract_strategic_frame_files(video_path, tmp_dir, include_micro_crops=False)

    logger.info(f"✓ [STEP 03 SUCCESS] Extracted {len(frame_paths)} vector-guided keyframe(s).")
    return frame_paths
