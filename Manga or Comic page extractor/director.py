"""
Phase_1/director.py — Native Gemini Agentic Director for Phase 1
==================================================================
Directs Phase 1 by calling the REAL standalone functions 1:1:
  1. process_manga_input                (Phase_1.loader)
  2. crop_panels_from_normalized_boxes  (Phase_1.scanner.PanelDetector)
  3. read_page_semantic                 (Phase_1.reader.OCREngine)
  4. group_panels                       (Phase_1.analyzer.SceneGrouper)
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import List, Dict, Any, Optional

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from config import TEMP_DIR, OUTPUT_DIR, PHASE1_JSON_DIR
except ImportError:
    TEMP_DIR = os.path.join(_REPO_ROOT, "temp")
    OUTPUT_DIR = os.path.join(_REPO_ROOT, "output")
    PHASE1_JSON_DIR = OUTPUT_DIR


# Feature Modules
from loader import process_manga_input
from scanner import PanelDetector
from reader import OCREngine
from analyzer import SceneGrouper

logger = logging.getLogger("phase1_director")

# ── Google GenAI SDK ─────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None


def get_phase1_tools():
    """Declares tools using the EXACT real function names from Phase 1 modules."""
    if not GENAI_AVAILABLE or types is None:
        return []

    return [
        types.Tool(function_declarations=[
            # 1. Real function from Phase_1.loader
            types.FunctionDeclaration(
                name="process_manga_input",
                description="Loads a manga PDF, image folder, single image, or URL and unpacks all page image paths.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "input_path": types.Schema(
                            type=types.Type.STRING,
                            description="Absolute path to a manga PDF, image directory, or web URL."
                        )
                    },
                    required=["input_path"]
                )
            ),

            # 2. Real function from Phase_1.scanner.PanelDetector
            types.FunctionDeclaration(
                name="crop_panels_from_normalized_boxes",
                description="Crops panels from a manga page using Gemini 2D normalized boxes or OpenCV contour detection.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "image_path": types.Schema(
                            type=types.Type.STRING,
                            description="Path to the page image file."
                        ),
                        "output_prefix": types.Schema(
                            type=types.Type.STRING,
                            description="File prefix for saved panels, e.g. 'page_001'."
                        )
                    },
                    required=["image_path", "output_prefix"]
                )
            ),

            # 3. Real function from Phase_1.reader.OCREngine
            types.FunctionDeclaration(
                name="read_page_semantic",
                description="Extracts dialogue text and 2D panel boxes from a manga page image.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "image_path": types.Schema(
                            type=types.Type.STRING,
                            description="Path to the manga page image to analyze."
                        ),
                        "page_number": types.Schema(
                            type=types.Type.INTEGER,
                            description="Page sequence number."
                        )
                    },
                    required=["image_path"]
                )
            ),

            # 4. Real function from Phase_1.analyzer.SceneGrouper
            types.FunctionDeclaration(
                name="group_panels",
                description="Clusters consecutive panels into scenes using visual similarity and pacing math.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "panel_items": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.STRING),
                            description="List of panel image file paths to group into scenes."
                        )
                    },
                    required=["panel_items"]
                )
            )
        ])
    ]


class Phase1Director:
    """
    Autonomous Gemini Executive Director.
    Directs the pipeline by invoking the real Phase 1 functions directly.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash", creative_intent: str = "cinematic_anime"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model
        self.creative_intent = creative_intent

        # Real feature module instances
        self.detector = PanelDetector()
        self.reader = OCREngine()
        self.grouper = SceneGrouper()

    def direct_chapter(
        self,
        input_source: str,
        creative_notes: Optional[str] = None,
        output_json: Optional[str] = None,
        max_pages: Optional[int] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        intent = creative_notes or self.creative_intent

        print(f"\n=======================================================")
        print(f"🎬 [PHASE 1 DIRECTOR] Directing Chapter")
        print(f"📥 Input: {input_source}")
        print(f"🎯 Intent: {intent}")
        print(f"=======================================================\n")

        # ── Step 1: Real process_manga_input ─────────────────────────────────
        print(f"🚀 [STEP 1] Calling process_manga_input...")
        page_paths = process_manga_input(input_path=input_source)
        if not page_paths:
            return {"status": "error", "message": f"No pages could be extracted from {input_source}", "scenes": []}

        if max_pages and max_pages > 0:
            page_paths = page_paths[:max_pages]
            print(f"⚡ [FAST TEST MODE] Limiting to first {len(page_paths)} page(s).")

        print(f"📄 Found {len(page_paths)} page(s) to direct.\n")

        # ── Step 2: Real read_page_semantic & crop_panels_from_normalized_boxes ──
        print(f"👁️ [STEP 2] Inspecting pages and cropping panels...")
        all_panels = []
        pages_summary = []

        for p_idx, page_path in enumerate(page_paths):
            page_num = p_idx + 1

            # Real reader call
            page_data = self.reader.read_page_semantic(image_path=page_path, page_number=page_num)

            # Skip credits/copyright splash
            if page_data.get("is_credits_page") or page_data.get("is_high_risk_copyright"):
                print(f"  🛡️ Skipping non-story page {page_num:02d}...")
                continue

            # Real scanner call
            norm_boxes = page_data.get("panels", [])
            extracted_panels = self.detector.crop_panels_from_normalized_boxes(
                image_path=page_path,
                normalized_boxes=norm_boxes,
                output_prefix=f"page_{page_num:03d}"
            )

            for panel in extracted_panels:
                panel["page_number"] = page_num
                all_panels.append(panel)

            pages_summary.append({
                "page_number": page_num,
                "page_path": os.path.abspath(page_path),
                "dialogue": page_data.get("page_text", ""),
                "panels": extracted_panels
            })
            print(f"  ✨ Page {page_num:02d}: Extracted {len(extracted_panels)} panels.")

        # ── Step 3: Real group_panels ────────────────────────────────────────
        print(f"\n🎼 [STEP 3] Calling group_panels for pacing...")
        scenes = self.grouper.group_panels(panel_items=all_panels)

        scenes_data = []
        for s_idx, scene_panels in enumerate(scenes):
            scene_dialogue = " ".join(p.get("dialogue", "") for p in scene_panels if p.get("dialogue"))
            scenes_data.append({
                "scene_index": s_idx + 1,
                "panel_count": len(scene_panels),
                "dialogue_summary": scene_dialogue,
                "panels": scene_panels
            })

        elapsed = round(time.time() - start_time, 2)

        storyboard = {
            "status": "success",
            "director": "Gemini Phase 1 Executive Director",
            "creative_intent": intent,
            "input_source": input_source,
            "execution_time_seconds": elapsed,
            "total_pages": len(page_paths),
            "total_panels": len(all_panels),
            "total_scenes": len(scenes_data),
            "pages": pages_summary,
            "scenes": scenes_data
        }

        # ── Step 4: Save Storyboard Output ───────────────────────────────────
        base_name = os.path.splitext(os.path.basename(input_source))[0]
        clean_name = "".join(c if c.isalnum() else "_" for c in base_name).strip("_")
        default_file = f"phase1_director_storyboard_{clean_name}.json"
        save_path = output_json if output_json else os.path.join(PHASE1_JSON_DIR, default_file)

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(storyboard, f, indent=2)

        print(f"\n=======================================================")
        print(f"✅ [DIRECTOR COMPLETE] Storyboard saved: {save_path}")
        print(f"⏱️ Finished in {elapsed}s | {len(all_panels)} panels in {len(scenes_data)} scenes.")
        print(f"=======================================================\n")

        return storyboard

    def direct_to_stickman(
        self,
        input_source: str,
        style: str = "Style 3 (Prehistoric Dark Comedy)",
        duration_sec: int = 60,
        aspect_ratio: str = "9:16",
        output_json: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        1. Executes Phase 1 manga extraction (crops panels, OCRs dialogue, clusters scenes).
        2. Feeds the extracted storyboard directly into Gemini Stickman Director.
        3. Returns complete 5-Stage Heartbeat animated stickman script and Veo/Omni video prompts.
        """
        storyboard = self.direct_chapter(input_source=input_source, max_pages=max_pages)
        if storyboard.get("status") != "success":
            return storyboard

        try:
            from Gemini_Modules.stickman_director import get_stickman_director
            stickman = get_stickman_director()
            package = stickman.direct_from_manga(
                manga_storyboard=storyboard,
                style=style,
                duration_sec=duration_sec,
                aspect_ratio=aspect_ratio,
            )

            if output_json:
                save_path = output_json
            else:
                base_name = os.path.splitext(os.path.basename(input_source))[0]
                clean_name = "".join(c if c.isalnum() else "_" for c in base_name).strip("_")
                save_path = os.path.join(PHASE1_JSON_DIR, f"stickman_production_package_{clean_name}.json")

            stickman.export_package(package, save_path)
            return package
        except Exception as e:
            logger.error(f"❌ [Phase1Director] direct_to_stickman failed: {e}")
            return {"status": "error", "error": str(e), "manga_storyboard": storyboard}


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Autonomous Gemini Director")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input manga source (PDF, folder, image, URL)")
    parser.add_argument("--intent", type=str, default="cinematic_anime", help="Creative intent")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output storyboard JSON path")
    parser.add_argument("--max-pages", "-m", type=int, default=None, help="Limit number of pages for fast testing")
    parser.add_argument("--stickman", action="store_true", help="Direct to Stickman Animated Video Production Package")
    parser.add_argument("--stickman-style", type=str, default="Style 3", help="Stickman Style (Style 1, Style 2A, Style 2B, Style 3, Style 4)")
    parser.add_argument("--duration", type=int, default=60, help="Stickman target duration in seconds (30, 60, 90)")

    args = parser.parse_args()
    director = Phase1Director(creative_intent=args.intent)
    if args.stickman:
        director.direct_to_stickman(
            input_source=args.input,
            style=args.stickman_style,
            duration_sec=args.duration,
            output_json=args.output,
            max_pages=args.max_pages,
        )
    else:
        director.direct_chapter(input_source=args.input, output_json=args.output, max_pages=args.max_pages)


if __name__ == "__main__":
    main()

