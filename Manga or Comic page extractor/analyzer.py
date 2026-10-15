"""
Phase_1/analyzer.py — Deterministic Panel Grouping & 3-Way Hybrid Pacing Calculator
====================================================================================
Combines:
  1. OpenCV Color Histogram Visual Scene Cut Detection (_is_visually_similar)
  2. OpenCV Panel Area Ratio Calculation (Physical pixel scaling factor)
  3. 3-Way Hybrid Pacing Math Engine:
     Duration = max(TTS_Audio_Length + 0.5s, 2.5s * Gemini_Importance * OpenCV_Area_Ratio)
"""

import os

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

from typing import List, Dict, Any, Optional, Tuple


def _clamp(val: float, min_v: float, max_v: float) -> float:
    """Clamps float value between min_v and max_v."""
    return max(min_v, min(float(val), max_v))


class SceneGrouper:
    def __init__(self, base_panel_duration_sec: float = 2.5):
        self.base_duration = base_panel_duration_sec

    def group_panels(self, panel_items: List[Any]) -> List[List[Any]]:
        """
        Groups panels into scenes using deterministic signal heuristics.
        Signals: Visual similarity (color histogram), dramatic weight boundaries, geometry.
        Accepts either a list of file path strings or a list of panel dictionaries.
        """
        if not panel_items:
            return []

        def _get_path(item: Any) -> str:
            if isinstance(item, dict):
                return item.get("panel_path", "")
            return str(item)

        def _get_weight(item: Any) -> float:
            if isinstance(item, dict):
                return float(item.get("dramatic_weight", 1.0))
            return 1.0

        scenes = [[panel_items[0]]]

        for i in range(1, len(panel_items)):
            prev_item = panel_items[i - 1]
            curr_item = panel_items[i]
            prev_path = _get_path(prev_item)
            curr_path = _get_path(curr_item)
            curr_weight = _get_weight(curr_item)

            # Major dramatic climax panels (weight >= 1.8) stand alone as epic scene cuts
            if curr_weight >= 1.8:
                scenes.append([curr_item])
                continue

            # Visual similarity check: Color Histogram
            if self._is_visually_similar(prev_path, curr_path):
                scenes[-1].append(curr_item)
            else:
                scenes.append([curr_item])

        # Limit scene size to avoid overly long clips (deterministic cap of 4 panels per scene)
        final_scenes = []
        for scene in scenes:
            while len(scene) > 4:
                final_scenes.append(scene[:4])
                scene = scene[4:]
            if scene:
                final_scenes.append(scene)

        return final_scenes


    def _is_visually_similar(self, path1: str, path2: str) -> bool:
        """Determines if two images are visually similar using OpenCV color histograms."""
        if cv2 is None or not os.path.exists(path1) or not os.path.exists(path2):
            return False

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        if img1 is None or img2 is None:
            return False

        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return float(similarity) > 0.7  # Deterministic threshold

    def compute_panel_area_ratio(
        self,
        panel_path: str,
        page_width: int = 2480,
        page_height: int = 3508
    ) -> float:
        """
        Computes OpenCV physical panel area ratio relative to full page.
        Returns scaling factor clamped between 0.8 (small corner panel) and 1.5 (splash page).
        """
        if cv2 is None or not os.path.exists(panel_path):
            return 1.0

        img = cv2.imread(panel_path)
        if img is None:
            return 1.0

        pw = int(page_width) if page_width else 2480
        ph = int(page_height) if page_height else 3508

        h, w = img.shape[:2]
        panel_area = w * h
        page_area = pw * ph

        raw_ratio = panel_area / max(1.0, float(page_area))
        # Scale: 0.1 ratio -> 0.85 multiplier, 0.8 ratio -> 1.45 multiplier
        multiplier = 0.8 + (raw_ratio * 0.8)
        return _clamp(multiplier, 0.8, 1.5)

    def compute_panel_pacing(
        self,
        panel_path: str,
        tts_duration_sec: float = 0.0,
        gemini_importance: float = 1.0,
        page_dimensions: Tuple[int, int] = (2480, 3508)
    ) -> Dict[str, Any]:
        """
        3-Way Hybrid Pacing Math Engine:
          Final Duration = max(TTS_Audio_Length + 0.5s, 2.5s * Gemini_Importance * OpenCV_Area_Ratio)

        Args:
            panel_path: File path to panel image.
            tts_duration_sec: Spoken voiceover length in seconds (0.0 if silent action panel).
            gemini_importance: Gemini narrative dramatic weight (0.5 to 1.8).
            page_dimensions: Tuple (page_width, page_height).

        Returns:
            Dictionary containing final duration seconds, opencv area multiplier, and pacing breakdown.
        """
        page_w = page_dimensions[0] if (page_dimensions and len(page_dimensions) > 0 and page_dimensions[0]) else 2480
        page_h = page_dimensions[1] if (page_dimensions and len(page_dimensions) > 1 and page_dimensions[1]) else 3508
        opencv_area_mult = self.compute_panel_area_ratio(panel_path, page_width=page_w, page_height=page_h)

        # Clamp Gemini importance weight safety limits
        gemini_weight = _clamp(gemini_importance, 0.5, 2.0)

        # Calculate visual weight duration
        visual_duration = self.base_duration * gemini_weight * opencv_area_mult

        # Audio hard minimum padding (audio length + 0.5s pause)
        audio_duration = tts_duration_sec + 0.5 if tts_duration_sec > 0 else 0.0

        # Final duration is maximum of audio requirement vs visual pacing requirement
        final_duration = max(audio_duration, visual_duration)

        return {
            "panel_path": panel_path,
            "final_duration_sec": round(final_duration, 2),
            "tts_duration_sec": round(tts_duration_sec, 2),
            "gemini_importance_weight": round(gemini_weight, 2),
            "opencv_area_multiplier": round(opencv_area_mult, 2),
            "pacing_mode": "audio_driven" if audio_duration > visual_duration else "visual_driven"
        }

    def compute_panel_pacing_from_meta(
        self,
        panel_meta: Dict[str, Any],
        tts_duration_sec: float = 0.0,
        page_dimensions: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Calculates pacing directly from a semantic panel dictionary produced by scanner.py.
        Extracts dramatic_weight, camera_shot, action_summary, and actual page dimensions automatically.
        """
        panel_path = panel_meta.get("panel_path", "")
        gemini_importance = float(panel_meta.get("dramatic_weight", 1.0))

        # Dynamic Canvas Dimension Resolution:
        # 1. Use page_width / page_height attached directly to panel_meta
        # 2. Use page_dimensions tuple passed by caller if provided
        # 3. Fallback to 2480x3508 only if completely unavailable
        if page_dimensions is not None:
            actual_dims = page_dimensions
        elif panel_meta.get("page_width") and panel_meta.get("page_height"):
            actual_dims = (int(panel_meta["page_width"]), int(panel_meta["page_height"]))
        else:
            actual_dims = (2480, 3508)

        pacing = self.compute_panel_pacing(
            panel_path=panel_path,
            tts_duration_sec=tts_duration_sec,
            gemini_importance=gemini_importance,
            page_dimensions=actual_dims
        )
        pacing["camera_shot"] = panel_meta.get("camera_shot", "static")
        pacing["action_summary"] = panel_meta.get("action_summary", "")
        pacing["dialogue"] = panel_meta.get("dialogue", "")
        return pacing

