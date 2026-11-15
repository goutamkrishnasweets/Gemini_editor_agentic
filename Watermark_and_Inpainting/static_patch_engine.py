"""
Watermark_and_Inpainting / static_patch_engine.py
=================================================
Static Patch Motion Detection Engine.
Checks whether the background behind a watermark bounding box is stationary or moving.
"""

import os
import logging
from typing import Dict, Any, Optional

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

logger = logging.getLogger("static_patch_engine")


class StaticPatchReuseEngine:
    @staticmethod
    def check_pixel_motion(video_path: str, wm_box: Optional[Dict[str, Any]] = None, threshold: float = 12.0) -> bool:
        """
        Returns True if pixel motion is detected in the watermark region between frames.
        Returns False if the region is static/stationary.
        """
        if not _HAS_CV2 or not video_path or not os.path.exists(video_path):
            return True

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return True

        ret1, frame1 = cap.read()
        for _ in range(15):
            cap.read()
        ret2, frame2 = cap.read()
        cap.release()

        if not ret1 or not ret2 or frame1 is None or frame2 is None:
            return True

        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            if wm_box and isinstance(wm_box, dict) and "w" in wm_box and "h" in wm_box:
                x = max(0, int(wm_box.get("x", 0)))
                y = max(0, int(wm_box.get("y", 0)))
                w = int(wm_box.get("w", 50))
                h = int(wm_box.get("h", 50))
                crop1 = gray1[y:y+h, x:x+w]
                crop2 = gray2[y:y+h, x:x+w]
                if crop1.size > 0 and crop2.size > 0:
                    diff = cv2.absdiff(crop1, crop2)
                    mean_diff = float(diff.mean())
                    return mean_diff > threshold

            diff = cv2.absdiff(gray1, gray2)
            return float(diff.mean()) > threshold
        except Exception as e:
            logger.debug(f"[StaticPatchReuseEngine] check_pixel_motion fallback: {e}")
            return True


__all__ = ["StaticPatchReuseEngine"]

