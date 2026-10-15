try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

import logging
import os
from typing import List, Tuple, Any, Dict

logger = logging.getLogger(__name__)

import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from config import PANEL_CONFIDENCE_THRESHOLD, TEMP_DIR
except ImportError:
    PANEL_CONFIDENCE_THRESHOLD = 0.5
    TEMP_DIR = os.path.join(_REPO_ROOT, "temp")


class PanelDetector:
    _ocr_engine = None  # Class-level shared singleton across all pages and panels

    def __init__(self):
        pass

    def detect_panels(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect panels using contour detection.
        Returns list of (x, y, w, h) bounding boxes.
        """
        if cv2 is None or not os.path.exists(image_path):
            return []

        img = cv2.imread(image_path)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding to handle various page backgrounds
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations to close panel borders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours representing panels
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = img.shape[:2]
        min_panel_area = (w * h) * 0.02  # Minimum 2% of page area
        
        panels = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            
            # Filter noise and entire-page boundaries
            if area > min_panel_area and area < (w * h * 0.95):
                panels.append((x, y, bw, bh))

        # Fallback if detection fails
        if not panels:
            panels = self._apply_fallbacks(img)

        # Sort in standard reading order (top-to-bottom, right-to-left)
        # Using a row-based sorting to properly handle manga layout
        panels.sort(key=lambda p: (p[1] // (h // 5), -p[0]))

        return panels

    def _apply_fallbacks(self, img) -> List[Tuple[int, int, int, int]]:
        """Applies fallback splits if no panels detected."""
        h, w = img.shape[:2]
        # Fallback 1: Wide spread -> vertical split
        if w > h * 1.5:
            return [(0, 0, w//2, h), (w//2, 0, w//2, h)]
        # Fallback 2: Tall page -> horizontal split
        if h > w * 1.5:
            return [(0, 0, w, h//2), (0, h//2, w, h//2)]
        # Fallback 3: Single panel
        return [(0, 0, w, h)]

    def extract_and_save_panels(self, image_path: str, output_prefix: str) -> List[str]:
        """Extracts panels and saves them as individual files."""
        img = cv2.imread(image_path) if cv2 else None
        panels = self.detect_panels(image_path)
        paths = []

        for i, (x, y, w, h) in enumerate(panels):
            path = os.path.join(TEMP_DIR, f"{output_prefix}_panel_{i}.png")
            if img is not None:
                panel_img = img[y:y+h, x:x+w]
                cv2.imwrite(path, panel_img)
            paths.append(path)
            
        return paths

    def _parse_box_to_pixels(self, box: List[Any], w: int, h: int) -> Tuple[int, int, int, int]:
        """
        Robust independent coordinate parser.
        Converts Gemini box (which may be mixed normalized floats 0.0-1.0 or 0-1000 per-mille)
        into validated pixel coordinates (x1, y1, x2, y2).
        """
        raw_vals = [float(v) for v in box[:4]]

        # Normalize each coordinate independently to 0.0 - 1.0 range
        norm = []
        for v in raw_vals:
            if 0.0 <= v <= 1.0:
                norm.append(v)
            elif 1.0 < v <= 1000.0:
                # Gemini per-mille coordinate (e.g. 349 -> 0.349)
                norm.append(v / 1000.0)
            else:
                # Value > 1000 accidentally mixed in: clamp to 1.0
                norm.append(1.0)

        ny0, nx0, ny1, nx1 = norm[0], norm[1], norm[2], norm[3]
        # Gemini standard is [ymin, xmin, ymax, xmax]
        if ny1 >= ny0 and nx1 >= nx0:
            y1, x1, y2, x2 = int(ny0 * h), int(nx0 * w), int(ny1 * h), int(nx1 * w)
        elif nx1 >= ny0 and ny1 >= nx0:  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = int(ny0 * w), int(nx0 * h), int(ny1 * w), int(nx1 * h)
        else:
            # Fallback: sort pairs
            min_y, max_y = min(ny0, ny1), max(ny0, ny1)
            min_x, max_x = min(nx0, nx1), max(nx0, nx1)
            x1, y1, x2, y2 = int(min_x * w), int(min_y * h), int(max_x * w), int(max_y * h)

        # Ensure min <= max
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Clamp strictly to image boundaries
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        return x1, y1, x2, y2

    def crop_panels_from_normalized_boxes(
        self,
        image_path: str,
        normalized_boxes: List[Any],
        output_prefix: str
    ) -> List[dict]:
        """
        Gemini-Guided OpenCV Panel Cutter:
        Takes normalized bounding boxes [ymin, xmin, ymax, xmax] or semantic panel dicts
        from Gemini Vision, converts them to exact pixel dimensions, and uses OpenCV to crop
        crystal clear panel images in < 1 millisecond.
        """
        if not os.path.exists(image_path):
            return []

        img = cv2.imread(image_path) if cv2 else None
        if img is None:
            try:
                from PIL import Image
                img_pil = Image.open(image_path)
                w, h = img_pil.size
            except Exception:
                return []
        else:
            h, w = img.shape[:2]

        # Extract box coordinates and associate with metadata if dicts are passed
        box_entries = []
        if normalized_boxes:
            for item in normalized_boxes:
                if isinstance(item, dict):
                    raw_box = item.get("box_2d") or item.get("bbox_norm") or item.get("bbox") or []
                    box_entries.append({"box": raw_box, "meta": item})
                elif isinstance(item, (list, tuple)):
                    box_entries.append({"box": list(item), "meta": {}})
        
        # Fallback to contour detection if Gemini provided no boxes
        if not box_entries:
            boxes_px = self.detect_panels(image_path)
            for (x, y, bw, bh) in boxes_px:
                box_entries.append({
                    "box": [y/h, x/w, (y+bh)/h, (x+bw)/w],
                    "meta": {}
                })

        # ── Sort boxes in strict Manga Reading Order (Top-to-Bottom, Right-to-Left) ──
        def _manga_sort_key(entry):
            b = entry.get("box", [])
            if not b or len(b) < 4:
                return (999, 999)
            x1, y1, x2, y2 = self._parse_box_to_pixels(b, w, h)
            row = int(y1 / max(1, int(h * 0.18)))
            return (row, -x2)

        box_entries = sorted(box_entries, key=_manga_sort_key)

        results = []
        os.makedirs(TEMP_DIR, exist_ok=True)

        contour_boxes_cached = None
        used_contour_indices = set()

        for idx, entry in enumerate(box_entries):
            box = entry.get("box", [])
            meta = entry.get("meta", {})
            if not box or len(box) < 4:
                continue

            x1, y1, x2, y2 = self._parse_box_to_pixels(box, w, h)
            crop_w = x2 - x1
            crop_h = y2 - y1
            panel_source = "contour_fallback" if entry.get("is_contour_fallback") else "gemini_box"

            # ── Post-Crop Sanity Check: Per-Panel Contour Recovery ──
            if crop_w < 15 or crop_h < 15:
                if contour_boxes_cached is None:
                    contour_boxes_cached = self.detect_panels(image_path)
                
                recovered = False
                # Try finding an overlapping contour candidate
                for c_i, (cx, cy, cw, ch) in enumerate(contour_boxes_cached):
                    if c_i in used_contour_indices or cw < 15 or ch < 15:
                        continue
                    x_overlap = max(0, min(x2, cx + cw) - max(x1, cx))
                    y_overlap = max(0, min(y2, cy + ch) - max(y1, cy))
                    if x_overlap > 0 or y_overlap > 0:
                        x1, y1, x2, y2 = cx, cy, cx + cw, cy + ch
                        crop_w, crop_h = cw, ch
                        used_contour_indices.add(c_i)
                        recovered = True
                        panel_source = "contour_recovered"
                        logger.warning(
                            f"⚠️ [SCANNER] Recovered degenerate panel {idx} ({box}) via contour replacement -> {[x1, y1, x2, y2]} ({cw}x{ch}px)"
                        )
                        break

                if not recovered:
                    # Fallback to any unclaimed valid contour box
                    for c_i, (cx, cy, cw, ch) in enumerate(contour_boxes_cached):
                        if c_i not in used_contour_indices and cw >= 15 and ch >= 15:
                            x1, y1, x2, y2 = cx, cy, cx + cw, cy + ch
                            crop_w, crop_h = cw, ch
                            used_contour_indices.add(c_i)
                            recovered = True
                            panel_source = "contour_recovered"
                            logger.warning(
                                f"⚠️ [SCANNER] Recovered degenerate panel {idx} via unclaimed contour box -> {[x1, y1, x2, y2]} ({cw}x{ch}px)"
                            )
                            break

                if not recovered:
                    logger.error(
                        f"❌ [SCANNER] Dropped unrecoverable degenerate panel {idx} ({crop_w}x{crop_h}px) on {image_path}"
                    )
                    continue

            out_filename = f"{output_prefix}_panel_{idx}.png"
            out_path = os.path.join(TEMP_DIR, out_filename)

            if img is not None:
                panel_img = img[y1:y2, x1:x2]
                # ── Auto-Purge Scanlation Watermarks & Header Margins (4% Trim) ──
                ph, pw = panel_img.shape[:2]
                trim_top = int(ph * 0.04)
                trim_bot = int(ph * 0.04)
                if ph > 60 and (ph - trim_top - trim_bot) > 40:
                    panel_img = panel_img[trim_top:ph-trim_bot, 0:pw]

                # Fast credit text check via local EasyOCR singleton
                try:
                    if PanelDetector._ocr_engine is None:
                        from reader import OCREngine
                        PanelDetector._ocr_engine = OCREngine()
                    _t = (PanelDetector._ocr_engine.get_full_text_from_image(panel_img) or "").upper()
                    if any(k in _t for k in ["TRANSLATOR", "PROOFREADER", "TYPESETTER", "REDRAWER", "TCB SCANS", "TL QUALITY", "CLEANER"]):
                        logger.info(f"🚫 [SCANNER] Purging credit panel {out_filename} (contains scanlation text)")
                        continue
                except Exception:
                    pass

                cv2.imwrite(out_path, panel_img)
            else:
                panel_pil = img_pil.crop((x1, y1, x2, y2))
                pw, ph = panel_pil.size
                trim_top = int(ph * 0.04)
                trim_bot = int(ph * 0.04)
                if ph > 60 and (ph - trim_top - trim_bot) > 40:
                    panel_pil = panel_pil.crop((0, trim_top, pw, ph - trim_bot))
                panel_pil.save(out_path)

            results.append({
                "panel_index": idx,
                "panel_path": out_path,
                "bbox_pixel": [x1, y1, x2, y2],
                "bbox_norm": [round(y1/h, 4), round(x1/w, 4), round(y2/h, 4), round(x2/w, 4)],
                "width": crop_w,
                "height": crop_h,
                "page_width": w,
                "page_height": h,
                "extraction_source": panel_source,
                "dialogue": meta.get("dialogue", ""),
                "action_summary": meta.get("action_summary", ""),
                "dramatic_weight": float(meta.get("dramatic_weight", 1.0)),
                "camera_shot": meta.get("camera_shot", "static")
            })

        # ── Fallback to Contour Detection if all Gemini boxes were rejected ──
        if not results and (img is not None or 'img_pil' in locals()):
            logger.info(f"🔄 [SCANNER] Fallback to contour detection for {image_path} (all boxes rejected or missing)")
            contour_boxes = self.detect_panels(image_path)
            for c_idx, (cx, cy, cw, ch) in enumerate(contour_boxes):
                if cw < 15 or ch < 15:
                    continue
                out_filename = f"{output_prefix}_contour_panel_{c_idx}.png"
                out_path = os.path.join(TEMP_DIR, out_filename)
                if img is not None:
                    cv2.imwrite(out_path, img[cy:cy+ch, cx:cx+cw])
                else:
                    img_pil.crop((cx, cy, cx+cw, cy+ch)).save(out_path)
                results.append({
                    "panel_index": c_idx,
                    "panel_path": out_path,
                    "bbox_pixel": [cx, cy, cx+cw, cy+ch],
                    "bbox_norm": [round(cy/h, 4), round(cx/w, 4), round((cy+ch)/h, 4), round((cx+cw)/w, 4)],
                    "width": cw,
                    "height": ch,
                    "page_width": w,
                    "page_height": h,
                    "extraction_source": "contour_fallback",
                    "dialogue": "",
                    "action_summary": f"Contour Panel {c_idx+1}",
                    "dramatic_weight": 1.0,
                    "camera_shot": "static"
                })

        return results


