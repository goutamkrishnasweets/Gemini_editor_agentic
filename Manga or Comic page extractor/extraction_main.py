# extraction_main.py - Standalone Entry Point for Manga/Comic Extraction Phase
import os
import sys
import json
import argparse
import time

# Ensure directories are in sys.path for robust standalone imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from core.logger_setup import setup_unified_logging
    setup_unified_logging()
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from loader import process_manga_input
from scanner import PanelDetector
from analyzer import SceneGrouper
from reader import OCREngine

try:
    from config import TEMP_DIR, OUTPUT_DIR, PHASE1_JSON_DIR
except ImportError:
    TEMP_DIR = os.path.join(ROOT_DIR, "temp")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    PHASE1_JSON_DIR = OUTPUT_DIR

import PIL.Image


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


def batch_analyze_pages_gemini(page_chunk: list, start_idx: int) -> list:
    """
    Batches up to 6 manga page images into a SINGLE Gemini Vision call via OCREngine semantic vision.
    Extracts verbatim dialogue and 2D panel bounding boxes in one call.
    """
    ocr = OCREngine()
    raw_pages = ocr.read_pages_batch_semantic(page_chunk, start_page_num=start_idx + 1)
    results = []
    for idx, page_data in enumerate(raw_pages):
        page_num = page_data.get("page_number", start_idx + idx + 1)
        panels = page_data.get("panels", [])
        norm_boxes = [p.get("box_2d", []) for p in panels if p.get("box_2d")]
        results.append({
            "page_number": page_num,
            "ocr_text": page_data.get("page_text", ""),
            "is_credits_page": page_data.get("is_credits_page", False),
            "is_high_risk_copyright": page_data.get("is_high_risk_copyright", False),
            "copyright_risk_reason": page_data.get("copyright_risk_reason", "none"),
            "panel_boxes": panels if panels else norm_boxes,
            "scene_summary": f"Page {page_num} Action",
            "narration_script": page_data.get("page_text", ""),
            "camera_action": panels[0].get("camera_shot", "SLOW_ZOOM_IN") if panels else "SLOW_ZOOM_IN",
            "characters": [],
            "emotion": "DRAMATIC"
        })
    return results



class ExtractionPipeline:
    """
    Standalone Extraction Engine:
    Handles Input Loading -> PDF Extraction -> Gemini 6-Page Batch Forensic Vision Processing
    """
    def __init__(self, run_ocr: bool = True):
        self.detector = PanelDetector()
        self.grouper = SceneGrouper()
        self.run_ocr_flag = run_ocr
        self.ocr_engine = OCREngine() if run_ocr else None

    def run(self, input_path: str, save_json: str = None) -> dict:
        start_time = time.time()
        print(f"\n[Phase_1/extraction_main.py] 🚀 [EXTRACTION] Starting Standalone Page Extraction Pipeline")
        print(f"[Phase_1/extraction_main.py] 📥 Input: {input_path}")

        # 1. Process Input (PDF/URL/Image/Folder) to Page Images
        page_paths = process_manga_input(input_path)
        if not page_paths:
            print("[Phase_1/extraction_main.py] ❌ No pages extracted.")
            return {"error": "No pages found", "pages": [], "scenes": []}

        print(f"[Phase_1/extraction_main.py] 📄 Found {len(page_paths)} page(s) to process.")

        # 2. Process Pages in Batches of 6 via Gemini Forensic Analyzer
        pages_summary = []
        scenes_data = []

        batch_size = 6
        for batch_start in range(0, len(page_paths), batch_size):
            chunk = page_paths[batch_start:batch_start + batch_size]
            batch_results = batch_analyze_pages_gemini(chunk, batch_start) if self.run_ocr_flag else []

            # Map batch results or fallback page-by-page
            result_map = {res.get("page_number", batch_start + idx + 1): res for idx, res in enumerate(batch_results)}

            for idx, page_path in enumerate(chunk):
                page_idx = batch_start + idx
                page_num = page_idx + 1
                abs_page = os.path.abspath(page_path)

                res_info = result_map.get(page_num) or (batch_results[idx] if idx < len(batch_results) else {})
                ocr_text = res_info.get("ocr_text", "")
                text_upper = (ocr_text or "").upper()
                credit_keywords = ["TRANSLATOR", "PROOFREADER", "TYPESETTER", "REDRAWER", "TCB SCANS", "TL QUALITY", "CLEANER"]

                # ── Auto-Skip Scanlation Credits & High-Risk Copyright Pages ──
                is_credits = res_info.get("is_credits_page", False)
                is_copyright = res_info.get("is_high_risk_copyright", False)
                norm_boxes = res_info.get("panel_boxes", [])

                # Guard against false positives: A page is NEVER skipped if it contains substantial story dialogue or multiple story panels
                has_story_dialogue = len(ocr_text.strip()) > 50 and not all(kw in text_upper for kw in ["TRANSLATOR", "TYPESETTER"])
                has_story_panels = len(norm_boxes) >= 2

                if (is_credits or is_copyright) and not (has_story_dialogue or has_story_panels):
                    reason = res_info.get("copyright_risk_reason") or ("scanlation credits splash" if is_credits else "high copyright risk")
                    print(f"  [Phase_1/extraction_main.py] 🛡️ Page {page_num:02d}/{len(page_paths):02d} is Copyright/Credits Risk ({reason}) → Auto-Skipped.")
                    continue

                # Fallback to single page OCR if batch missed text
                if not ocr_text and self.run_ocr_flag and self.ocr_engine:
                    print(f"[Phase_1/extraction_main.py] 📖 Page {page_num}/{len(page_paths)} Fallback OCR ({os.path.basename(page_path)})...")
                    try:
                        ocr_text = self.ocr_engine.get_full_text(abs_page) or ""
                    except Exception as ocr_err:
                        print(f"   [Phase_1/extraction_main.py] ⚠️ OCR notice on page {page_num}: {ocr_err}")
                
                # ── Gemini-Guided OpenCV Panel Extraction ──
                printed_page_num = res_info.get("page_number", page_num)
                norm_boxes = res_info.get("panel_boxes", [])
                extracted_panels = self.detector.crop_panels_from_normalized_boxes(
                    abs_page, norm_boxes, f"page_{page_num:03d}"
                )
                for p in extracted_panels:
                    p["sequence_index"] = page_num
                    p["printed_page_number"] = printed_page_num
                    p["page_number"] = page_num

                num_p = len(extracted_panels)
                print(f"  ↳ [Phase_1/extraction_main.py] Page {page_num:02d}/{len(page_paths):02d} (Printed #{printed_page_num}) ({num_p} panels): Text: {ocr_text[:40]}..." if ocr_text else f"  ↳ [Phase_1/extraction_main.py] Page {page_num:02d}/{len(page_paths):02d} ({num_p} panels): [No text detected]")

                page_info = {
                    "sequence_index": page_num,
                    "printed_page_number": printed_page_num,
                    "page_index": page_idx,
                    "page_path": abs_page,
                    "ocr_text": ocr_text,
                    "scene_summary": res_info.get("scene_summary", ""),
                    "narration_script": res_info.get("narration_script", ""),
                    "camera_action": res_info.get("camera_action", "SLOW_ZOOM_IN"),
                    "characters": res_info.get("characters", []),
                    "emotion": res_info.get("emotion", "DRAMATIC"),
                    "panels": extracted_panels
                }
                pages_summary.append(page_info)
                scenes_data.append({
                    "sequence_index": page_num,
                    "printed_page_number": printed_page_num,
                    "scene_index": page_idx,
                    "page_path": abs_page,
                    "ocr_text": ocr_text,
                    "scene_summary": res_info.get("scene_summary", ""),
                    "narration_script": res_info.get("narration_script", ""),
                    "camera_action": res_info.get("camera_action", "SLOW_ZOOM_IN"),
                    "characters": res_info.get("characters", []),
                    "emotion": res_info.get("emotion", "DRAMATIC"),
                    "panels": extracted_panels
                })

        elapsed = round(time.time() - start_time, 2)
        all_panels = [p for page in pages_summary for p in page.get("panels", [])]
        gemini_pages_count = sum(1 for p in pages_summary if p.get("narration_script") or p.get("ocr_text"))
        fallback_pages_count = len(page_paths) - gemini_pages_count

        extraction_metrics = {
            "ocr_enabled": self.run_ocr_flag,
            "total_pages": len(page_paths),
            "gemini_success_pages": gemini_pages_count,
            "contour_fallback_pages": fallback_pages_count,
            "overall_source": "gemini_vision" if (gemini_pages_count == len(page_paths) and self.run_ocr_flag) else ("hybrid" if gemini_pages_count > 0 else "contour_fallback")
        }

        results = {
            "status": "success",
            "input_source": input_path,
            "execution_time_seconds": elapsed,
            "extraction_metrics": extraction_metrics,
            "total_pages": len(page_paths),
            "total_panels": len(all_panels),
            "pages": pages_summary,
            "panels": all_panels,
            "scenes": scenes_data
        }

        # 3. Save JSON Output
        os.makedirs(PHASE1_JSON_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        clean_name = "".join(c if c.isalnum() else "_" for c in base_name).strip("_")
        filename = f"phase_01_extraction_{clean_name}.json"

        if save_json:
            json_path = os.path.abspath(save_json)
            if os.path.isdir(json_path):
                json_path = os.path.join(json_path, filename)
        else:
            json_path = os.path.join(PHASE1_JSON_DIR, filename)

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        # Mirror only the exact named chapter file to OUTPUT_DIR
        output_dir_copy = os.path.join(OUTPUT_DIR, filename)
        if os.path.abspath(json_path) != os.path.abspath(output_dir_copy):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(output_dir_copy, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

        print(f"[Phase_1/extraction_main.py] 💾 Extraction results saved to: {json_path}")

        print(f"[Phase_1/extraction_main.py] ✨ [EXTRACTION COMPLETE] Finished {len(page_paths)} pages in {elapsed}s.\n")
        return results

def main():
    parser = argparse.ArgumentParser(description="Standalone Manga & Comic Extraction Tool")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input source (PDF path, image path, image folder, or image URL)")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR text extraction for faster processing")
    parser.add_argument("--output-json", "-j", type=str, default=None, help="Custom output JSON filepath")

    args = parser.parse_args()

    pipeline = ExtractionPipeline(run_ocr=not args.no_ocr)
    pipeline.run(args.input, save_json=args.output_json)

if __name__ == "__main__":
    main()
