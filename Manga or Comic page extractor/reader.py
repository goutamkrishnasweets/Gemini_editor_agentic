"""
Phase_1/reader.py — 3-Tier Multi-Engine OCR System
=====================================================
Priority Chain:
  1. Tier 1 (PRIMARY): Gemini Vision Cloud OCR (0% local CPU/GPU, fast)
  2. Tier 2 (FALLBACK 1): Ollama Local Vision AI (llama3.2-vision / llava @ localhost:11434)
  3. Tier 3 (LAST RESORT): EasyOCR (Local PyTorch CPU/GPU Engine)
"""

import os
import json
import re
import base64
import logging
import urllib.request
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None

from typing import List, Dict, Optional, Any

import sys
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from config import OCR_LANGUAGES, USE_GPU
except ImportError:
    OCR_LANGUAGES = ["en"]
    USE_GPU = False


logger = logging.getLogger("ocr_engine")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None


import PIL.Image


try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None


try:
    from Tools.Gemini_Modules.gemini_router_module.gemini_governor import gemini_router
except ImportError:
    try:
        from Intelligence_Modules.gemini_governor import gemini_router
    except ImportError:
        try:
            from gemini_governor import gemini_router
        except ImportError:
            gemini_router = None


def _try_gemini_vision_ocr(image_path: str) -> Optional[str]:
    """
    Tier 1 Primary OCR: Calls gemini_router for Gemini Vision API.
    Returns extracted plain text string or None on failure/quota issue.
    """
    if not gemini_router:
        return None

    try:
        img = PIL.Image.open(image_path)
        prompt = "Extract all comic page text, speech bubbles, and dialogue verbatim in original reading order. Return plain text string only."
        text = gemini_router.generate(
            task_type="vision",
            prompt=[prompt, img],
            module_name="ocr_engine",
            gen_config={"temperature": 0.1}
        )
        if text and text.strip():
            logger.info(f"✨ [TIER 1 GEMINI OCR SUCCESS]: '{text.strip()[:50]}...'")
            return text.strip()
    except Exception as exc:
        logger.debug(f"[GEMINI OCR] Vision API error for {image_path}: {exc}")

    return None


def _parse_json_safe(raw: str) -> Optional[Any]:
    """Safely extracts and parses JSON from raw LLM output, stripping markdown fences."""
    if not raw or not raw.strip():
        return None
    cleaned = raw.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end+1])
        except Exception:
            pass
    start_arr = cleaned.find('[')
    end_arr = cleaned.rfind(']')
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        try:
            return json.loads(cleaned[start_arr:end_arr+1])
        except Exception:
            pass
    return None


def _try_gemini_semantic_analysis(image_path: str, page_number: int = 1) -> Optional[Dict[str, Any]]:
    """
    Tier 1 Semantic Vision: Extracts dialogue and 2D panel bounding boxes [ymin, xmin, ymax, xmax]
    in strict Manga Reading Order (Right-to-Left, Top-to-Bottom).
    """
    if not gemini_router or not os.path.exists(image_path):
        return None

    try:
        img = PIL.Image.open(image_path)
        prompt = (
            "You are an expert Manga Vision & Layout Director.\n"
            "Analyze this manga page image and return strict JSON with dialogue, panel 2D bounding boxes, and copyright safety checks:\n"
            "{\n"
            '  "page_text": "all verbatim speech bubble dialogue and narration in Japanese/English in reading order (Right-to-Left, Top-to-Bottom)",\n'
            '  "is_credits_page": false,\n'
            '  "is_high_risk_copyright": false,\n'
            '  "copyright_risk_reason": "none",\n'
            '  "panels": [\n'
            '    {\n'
            '      "panel_index": 1,\n'
            '      "box_2d": [0.05, 0.50, 0.45, 0.98],\n'
            '      "dialogue": "dialogue inside this panel only",\n'
            '      "action_summary": "characters visible and action description",\n'
            '      "dramatic_weight": 1.0,\n'
            '      "camera_shot": "zoom_in | pan_left | pan_right | wide_shot | close_up | static"\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "CRITICAL RULES:\n"
            "1. 'box_2d' MUST contain normalized coordinates [ymin, xmin, ymax, xmax] as floats between 0.0 and 1.0.\n"
            "2. Sort panels in strict Manga Reading Order (Right-to-Left, Top-to-Bottom).\n"
            "3. Set dramatic_weight from 0.5 (low/minor dialogue) to 2.0 (maximum climax/big reveal).\n"
            "4. COPYRIGHT & RECRUITMENT SAFETY GUARD:\n"
            "   Only classify a page as 'is_high_risk_copyright': true if the ENTIRE page is dedicated exclusively to:\n"
            "   - Full magazine cover or volume front cover with prominent publisher logos (Shonen Jump, Viz, Kodansha)\n"
            "   - Promotional color spreads, anime posters, or centerfold advertisements with NO story panels\n"
            "   - Dedicated standalone recruitment splash pages, Discord/Patreon donation pages with NO story dialogue\n"
            "   - Pure legal copyright warnings, editorial notice, Table of Contents, or next issue previews\n"
            "   If ANY of these FULL-PAGE splash scenarios apply, set 'is_high_risk_copyright': true, specify 'copyright_risk_reason' (e.g. 'publisher_cover', 'color_spread_poster', 'editorial_legal_notice', 'merchandise_ad', or 'scanlation_credits'), and set 'panels': [].\n"
            "   CRITICAL EXCEPTION FOR STORY PAGES: If the page is a regular comic/manga story page containing panels, character artwork, or narrative dialogue, set 'is_high_risk_copyright': false and 'is_credits_page': false! Do NOT flag story pages as copyright risks just because of small scanlation watermarks (e.g. TCB Scans) or translator notes in the outer margins — extract all story panels normally.\n"
            "5. Return raw JSON only with NO markdown fences or explanation."
        )
        res_text = gemini_router.generate(
            task_type="vision",
            prompt=[prompt, img],
            module_name="ocr_engine",
            gen_config={"temperature": 0.1}
        )
        parsed = _parse_json_safe(res_text)
        if parsed and isinstance(parsed, dict) and "panels" in parsed:
            parsed["page_number"] = page_number
            return parsed
    except Exception as exc:
        logger.debug(f"[GEMINI SEMANTIC OCR] Vision API error for {image_path}: {exc}")

    return None


def _try_gemini_batch_semantic_analysis(image_paths: List[str], start_page_num: int = 1) -> List[Dict[str, Any]]:
    """
    Batches up to 6 manga page images into a single Gemini Vision API call.
    Returns a list of per-page semantic analysis dictionaries with copyright protection.
    """
    if not gemini_router or not image_paths:
        return []

    try:
        pil_imgs = []
        for p in image_paths:
            if os.path.exists(p):
                pil_imgs.append(PIL.Image.open(p))

        if not pil_imgs:
            return []

        prompt = (
            f"You are an expert Manga Vision & Layout Director. Analyze these {len(pil_imgs)} consecutive manga pages.\n"
            "For EACH page in order, return its dialogue, panel 2D bounding boxes, and copyright safety checks in strict JSON:\n"
            "{\n"
            '  "pages": [\n'
            '    {\n'
            '      "page_text": "verbatim speech bubble dialogue and narration in reading order",\n'
            '      "is_credits_page": false,\n'
            '      "is_high_risk_copyright": false,\n'
            '      "copyright_risk_reason": "none",\n'
            '      "panels": [\n'
            '        {\n'
            '          "panel_index": 1,\n'
            '          "box_2d": [ymin, xmin, ymax, xmax],\n'
            '          "dialogue": "dialogue in this panel only",\n'
            '          "action_summary": "characters and action description",\n'
            '          "dramatic_weight": 1.0,\n'
            '          "camera_shot": "zoom_in | pan_left | pan_right | wide_shot | close_up | static"\n'
            '        }\n'
            '      ]\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "CRITICAL RULES:\n"
            "1. 'box_2d' must contain normalized coordinates [ymin, xmin, ymax, xmax] as floats (0.0 to 1.0).\n"
            "2. Sort panels in Manga Reading Order (Right-to-Left, Top-to-Bottom).\n"
            "3. COPYRIGHT & RECRUITMENT SAFETY GUARD:\n"
            "   Only classify a page as 'is_high_risk_copyright': true if the ENTIRE page is dedicated exclusively to:\n"
            "   - Full magazine cover or volume front cover with prominent publisher logos (Shonen Jump, Viz, Kodansha)\n"
            "   - Promotional color spreads, anime posters, or centerfold advertisements with NO story panels\n"
            "   - Dedicated standalone recruitment splash pages, Discord/Patreon donation pages with NO story dialogue\n"
            "   - Pure legal copyright warnings, editorial notice, Table of Contents, or next issue previews\n"
            "   If ANY of these FULL-PAGE splash scenarios apply, set 'is_high_risk_copyright': true, specify 'copyright_risk_reason' (e.g. 'publisher_cover', 'color_spread_poster', 'editorial_legal_notice', 'merchandise_ad', or 'scanlation_credits'), and set 'panels': [].\n"
            "   CRITICAL EXCEPTION FOR STORY PAGES: If the page is a regular comic/manga story page containing panels, character artwork, or narrative dialogue, set 'is_high_risk_copyright': false and 'is_credits_page': false! Do NOT flag story pages as copyright risks just because of small scanlation watermarks (e.g. TCB Scans) or translator notes in the outer margins — extract all story panels normally.\n"
            "4. Return raw JSON only with NO markdown fences."
        )

        res_text = gemini_router.generate(
            task_type="vision",
            prompt=[prompt] + pil_imgs,
            module_name="ocr_engine",
            gen_config={"temperature": 0.1}
        )
        parsed = _parse_json_safe(res_text)
        if parsed and isinstance(parsed, dict) and "pages" in parsed:
            pages = parsed.get("pages", [])
            for idx, pg in enumerate(pages):
                if isinstance(pg, dict):
                    pg["page_number"] = start_page_num + idx
            return pages
    except Exception as exc:
        logger.debug(f"[GEMINI BATCH SEMANTIC OCR] Batch error: {exc}")

    return []


def _encode_image_to_base64(image_path: str) -> Optional[str]:

    """Encodes image file to base64 string for Ollama Vision API."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.debug(f"[OLLAMA] Base64 encoding error for {image_path}: {e}")
        return None


def _try_ollama_vision_ocr(image_path: str) -> Optional[str]:
    """
    Tier 2 Fallback: Tries local Ollama Vision AI (localhost:11434).
    Supports models: llama3.2-vision, llava.
    """
    b64_img = _encode_image_to_base64(image_path)
    if not b64_img:
        return None

    ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"
    models_to_try = [os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision"), "llava"]

    for model_name in models_to_try:
        try:
            payload = json.dumps({
                "model": model_name,
                "prompt": "Extract ALL speech bubble text and dialogue from this comic image. Reply ONLY with the extracted plain text string.",
                "images": [b64_img],
                "stream": False
            }).encode("utf-8")

            req = urllib.request.Request(
                ollama_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            # Fast 5s timeout to prevent blocking pipeline if Ollama isn't running
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    text = (data.get("response") or "").strip()
                    if text:
                        logger.info(f"🦙 [TIER 2 OLLAMA SUCCESS] ({model_name}): '{text[:40]}...'")
                        return text
        except Exception:
            continue  # Silently fall through to next model or Tier 3

    return None


class OCREngine:
    """
    3-Tier OCR Engine:
      Tier 1: Gemini Vision (Primary)
      Tier 2: Ollama Vision (Fallback 1)
      Tier 3: EasyOCR (Last Resort Fallback)
    """

    _easyocr_warned = False

    def __init__(self):
        self.reader = None
        self._easyocr_initialized = False

    def _lazy_init_easyocr(self):
        """Initializes EasyOCR lazily ONLY when Tier 1 & Tier 2 both fail."""
        if self._easyocr_initialized:
            return

        self._easyocr_initialized = True
        if EASYOCR_AVAILABLE:
            try:
                use_gpu_flag = USE_GPU and TORCH_AVAILABLE and torch and torch.cuda.is_available()
                self.reader = easyocr.Reader(OCR_LANGUAGES, gpu=use_gpu_flag)
                logger.info(f"🤖 [TIER 3 EASYOCR INITIALIZED] (GPU CUDA: {use_gpu_flag})")
            except Exception as e:
                logger.warning(f"⚠️ EasyOCR initialization warning: {e}")
        else:
            if not OCREngine._easyocr_warned:
                logger.warning("⚠️ EasyOCR module not installed — Tier 3 fallback unavailable.")
                OCREngine._easyocr_warned = True

    def perform_ocr(
        self,
        image_path: str,
        gemini_text_hint: Optional[str] = None
    ) -> List[Dict]:
        """
        Performs OCR adhering strictly to the 3-Tier Priority Chain.

        Returns list of results: [{'text': str, 'box': list, 'confidence': float, 'tier': str}]
        """
        # ── TIER 1: Gemini Vision Cloud OCR (PRIMARY) ────────────────────────
        if gemini_text_hint and isinstance(gemini_text_hint, str) and gemini_text_hint.strip():
            logger.info(f"✨ [TIER 1 GEMINI OCR PRIMARY] Used hint text: '{gemini_text_hint[:40]}...'")
            return [{
                "text": gemini_text_hint.strip(),
                "box": [0, 0, 0, 0],
                "confidence": 0.95,
                "tier": "tier1_gemini"
            }]

        gemini_text = _try_gemini_vision_ocr(image_path)
        if gemini_text:
            return [{
                "text": gemini_text,
                "box": [0, 0, 0, 0],
                "confidence": 0.95,
                "tier": "tier1_gemini"
            }]

        # ── TIER 2: Ollama Local Vision AI (FALLBACK 1) ──────────────────────
        ollama_text = _try_ollama_vision_ocr(image_path)
        if ollama_text:
            return [{
                "text": ollama_text,
                "box": [0, 0, 0, 0],
                "confidence": 0.85,
                "tier": "tier2_ollama"
            }]

        # ── TIER 3: EasyOCR (LAST RESORT FALLBACK) ───────────────────────────
        logger.info(f"⚠️ [TIER 3 FALLBACK] Gemini & Ollama unavailable → initializing EasyOCR for {os.path.basename(image_path)}")
        self._lazy_init_easyocr()

        if not self.reader:
            return []

        try:
            results = self.reader.readtext(image_path)
            extracted = []
            for (bbox, text, prob) in results:
                extracted.append({
                    "text": text,
                    "box": [int(x) for point in bbox for x in point],
                    "confidence": float(prob),
                    "tier": "tier3_easyocr"
                })
            return extracted
        except Exception as e:
            logger.warning(f"⚠️ EasyOCR processing error on {image_path}: {e}")
            return []

    def get_full_text(self, image_path: str, gemini_text_hint: Optional[str] = None) -> str:
        """Returns all text found in the image as a single string."""
        results = self.perform_ocr(image_path, gemini_text_hint=gemini_text_hint)
        return " ".join([res["text"] for res in results if res.get("text")])

    def get_full_text_from_image(self, image_input: Any, local_only: bool = True) -> str:
        """
        Runs OCR directly on an in-memory image (numpy.ndarray or PIL.Image) or file path string.
        Enables in-memory watermark/credit panel purging in OpenCV cutter without disk I/O.
        Strictly uses local EasyOCR by default to prevent expensive remote Gemini calls on individual panels.
        """
        if image_input is None:
            return ""

        if isinstance(image_input, str) and not local_only:
            return self.get_full_text(image_input)

        try:
            self._lazy_init_easyocr()
            if self.reader is not None:
                # EasyOCR's reader.readtext accepts both file paths and numpy ndarrays
                results = self.reader.readtext(image_input)
                texts = [res[1] for res in results if len(res) > 1 and res[1]]
                return " ".join(texts)
        except Exception as e:
            logger.debug(f"[OCREngine] In-memory OCR error: {e}")

        return ""

    def read_page_semantic(self, image_path: str, page_number: int = 1) -> Dict[str, Any]:
        """
        Step 2 Primary Entry Point:
        Extracts dialogue text and 2D panel bounding boxes in a single Gemini Vision call.
        Falls back to local contour detection + OCR if Gemini is unavailable.
        """
        # Determine actual image dimensions
        h, w = (3508, 2480)
        if cv2 and os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]

        # Tier 1: Gemini Semantic Vision
        res = _try_gemini_semantic_analysis(image_path, page_number=page_number)
        if res and isinstance(res, dict) and res.get("panels"):
            res["page_width"] = w
            res["page_height"] = h
            res["extraction_method"] = "gemini_semantic"
            for p in res.get("panels", []):
                p.setdefault("page_width", w)
                p.setdefault("page_height", h)
            logger.info(f"✨ [SEMANTIC OCR SUCCESS] Page {page_number}: Found {len(res['panels'])} semantic panels ({w}x{h}px).")
            return res

        # Fallback: Local Contour Detection + OCR
        logger.info(f"⚠️ [SEMANTIC OCR FALLBACK] Page {page_number}: Falling back to local contour scanner + OCR.")
        ocr_text = self.get_full_text(image_path)
        panels = []
        try:
            from scanner import PanelDetector
            detector = PanelDetector()
            boxes_px = detector.detect_panels(image_path)

            for idx, (x, y, bw, bh) in enumerate(boxes_px):
                panels.append({
                    "panel_index": idx + 1,
                    "box_2d": [round(y/h, 4), round(x/w, 4), round((y+bh)/h, 4), round((x+bw)/w, 4)],
                    "page_width": w,
                    "page_height": h,
                    "dialogue": "",
                    "action_summary": f"Panel {idx + 1}",
                    "dramatic_weight": 1.0,
                    "camera_shot": "static"
                })
        except Exception as e:
            logger.warning(f"⚠️ [SEMANTIC OCR FALLBACK] Error detecting fallback panels: {e}")

        if not panels:
            panels = [{
                "panel_index": 1,
                "box_2d": [0.0, 0.0, 1.0, 1.0],
                "page_width": w,
                "page_height": h,
                "dialogue": ocr_text,
                "action_summary": "Full Page",
                "dramatic_weight": 1.0,
                "camera_shot": "static"
            }]

        return {
            "page_number": page_number,
            "page_text": ocr_text,
            "page_width": w,
            "page_height": h,
            "extraction_method": "contour_fallback",
            "is_credits_page": False,
            "panels": panels
        }

    def read_pages_batch_semantic(
        self,
        image_paths: List[str],
        start_page_num: int = 1,
        batch_size: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Batches multiple page images into Gemini Vision calls in chunks of `batch_size` (default 6).
        Falls back to page-by-page semantic reading if batch call fails.
        """
        all_results = []
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i:i + batch_size]
            chunk_start_num = start_page_num + i
            logger.info(f"🔍 [SEMANTIC BATCH OCR] Processing pages {chunk_start_num} to {chunk_start_num + len(chunk) - 1}...")

            batch_res = _try_gemini_batch_semantic_analysis(chunk, start_page_num=chunk_start_num)
            if batch_res and len(batch_res) == len(chunk):
                for idx, p_path in enumerate(chunk):
                    if cv2 and os.path.exists(p_path):
                        img = cv2.imread(p_path)
                        if img is not None:
                            h, w = img.shape[:2]
                            batch_res[idx]["page_width"] = w
                            batch_res[idx]["page_height"] = h
                            batch_res[idx]["extraction_method"] = "gemini_batch_semantic"
                            for p in batch_res[idx].get("panels", []):
                                p.setdefault("page_width", w)
                                p.setdefault("page_height", h)
                    else:
                        batch_res[idx]["extraction_method"] = "gemini_batch_semantic"
                all_results.extend(batch_res)
            else:
                # Fallback to single page semantic reading for this chunk
                for sub_idx, p in enumerate(chunk):
                    p_num = chunk_start_num + sub_idx
                    all_results.append(self.read_page_semantic(p, page_number=p_num))

        return all_results

