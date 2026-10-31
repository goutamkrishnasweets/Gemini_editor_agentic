"""
Hybrid Watermark Manager (Gemini Authority)
-------------------------------------------
STRICT MODE:
- Gemini Vision is the ONLY detection source.
- No ML, No OpenCV Detection.
- If Gemini fails/quota -> Status: CLEAN (Skip).
- Geometric Hardening & Face Safety Enforced.

v2 UPDATE: detect_watermark() now returns (results, detected_niche) tuple.
           process_video() captures detected_niche and persists it to a
           video sidecar file (<video_path>.niche.json) for the uploader
           to consume at approval time.
"""

import os
import json
import logging
import cv2
import numpy as np
import time
import uuid
import shutil
try:
    from Watermark_and_Inpainting.import_gate import ImportGate
except ImportError:
    from import_gate import ImportGate

# Import FaceProtector for strict safety overrides
try:
    from Watermark_and_Inpainting.opencv_watermark import FaceProtector
except ImportError:
    try:
        from opencv_watermark import FaceProtector
    except ImportError:
        class FaceProtector:
            @staticmethod
            def is_safe_region(frame, box): return True, "Safe"
            @staticmethod
            def detect_faces(frame): return []

logger = logging.getLogger("hybrid_watermark")


# ── Sidecar helpers ────────────────────────────────────────────────────────────

def _niche_sidecar_path(video_path: str) -> str:
    """Returns the path for the niche sidecar file next to the video."""
    base, _ = os.path.splitext(video_path)
    return f"{base}.niche.json"


def save_detected_niche(video_path: str, niche: str) -> None:
    """Persist the detected niche to a sidecar JSON file beside the video."""
    try:
        sidecar = _niche_sidecar_path(video_path)
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump({"detected_niche": niche, "timestamp": time.time()}, f)
        
        abs_sidecar = os.path.abspath(sidecar)
        logger.info(f"💾 [NICHE] Sidecar saved: {niche} → {abs_sidecar}")
    except Exception as e:
        logger.warning(f"⚠️ [NICHE] Failed to save sidecar: {e}")


def load_detected_niche(video_path: str) -> str:
    """
    Read the detected niche from the sidecar file.
    Returns niche string, or 'General_Fallback' if not found.
    """
    try:
        sidecar = _niche_sidecar_path(video_path)
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
            niche = data.get("detected_niche", "General_Fallback")
            logger.info(f"📂 [NICHE] Sidecar loaded: {niche} (from {os.path.abspath(sidecar)})")
            return niche
    except Exception as e:
        logger.warning(f"⚠️ [NICHE] Failed to load sidecar: {e}")
    return "General_Fallback"


# ──────────────────────────────────────────────────────────────────────────────

class HybridWatermarkDetector:
    def __init__(self):
        self.session_blacklist = {}
        self.removability_cache = {}

    def _error_json(self, msg):
        return json.dumps({
            "watermarks": [],
            "count": 0,
            "status": "ERROR",
            "context": {"error": msg}
        })

    def confirm_learning(self, context: dict, is_positive: bool):
        """
        Logs user feedback (Reinforcement Learning Stub).
        In "Strict Mode", this just logs the failure to influence the next retry's prompt.
        """
        try:
            feedback = "POSITIVE" if is_positive else "NEGATIVE"
            logger.info(f"🧠 HybridWatermark Learning: Received {feedback} feedback.")
            if not is_positive:
                logger.info("   └─ Will trigger deeper scan on next retry.")
        except Exception as e:
            logger.warning(f"Learning feedback failed: {e}")

    def reset_quotas(self):
        """Resets per-video quotas (Gemini)."""
        try:
            gemini_enhance = ImportGate.get("gemini_enhance_for_watermark")
            if gemini_enhance and hasattr(gemini_enhance, 'quota_manager'):
                gemini_enhance.quota_manager.reset()
        except Exception as e:
            logger.warning(f"Failed to reset quotas: {e}")

    def process_video(self, video_path: str, aggressive: bool = False, keywords: str = None, retry_level: int = 0, title: str = None) -> str:
        """
        Main entry point. Gemini-Only Authority Mode.

        v2 CHANGE: detect_watermark() now returns a tuple (boxes, niche).
                   The niche is saved to a sidecar file for the uploader.
        """
        # 🔒 SYSTEM LOCK ASSERTION
        watermark_source = "gemini"

        logger.info(f"🎬 Processing (Gemini Authority): {video_path} (Retry Level: {retry_level})")

        # INJECT FEEDBACK PROMPT FOR RETRIES
        if retry_level > 0:
            feedback_prompt = " FEEDBACK:PREVIOUS_FAILED_SEEK_DEEPER_AND_SMALLER MODE:NUCLEAR_SENSITIVITY "
            if keywords: keywords += feedback_prompt
            else: keywords = feedback_prompt
            logger.info(f"🧠 Injecting Feedback Prompt: {feedback_prompt.strip()}")

        # Also inject studio logo focus on retries
        studio_hint = " | FOCUS: corner_studio_logo bottom_left bottom_right photographer_watermark"
        keywords = (keywords or "") + studio_hint

        if not os.path.exists(video_path):
            return self._error_json("Video file not found.")

        # 1. Select Representative Frame
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_count < 1:
            cap.release()
            return self._error_json("Empty video.")

        # 2. Select Representative Frames (VANGUARD 7-SHOT STRATEGY)
        frames_to_check = []
        scan_percentages = [0.05, 0.15, 0.33, 0.5, 0.66, 0.85, 0.95]

        for pct in scan_percentages:
            target_idx = int(frame_count * pct)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, scanned_frame = cap.read()
            if ret and scanned_frame is not None and np.mean(scanned_frame) > 3:  # Lowered: dark videos are valid
                frames_to_check.append(scanned_frame)

        # [DARK VIDEO GUARD] If dark-frame filter rejected everything, force-capture the first readable frame
        if not frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(min(frame_count, 30)):
                ret, scanned_frame = cap.read()
                if ret and scanned_frame is not None:
                    frames_to_check.append(scanned_frame)
                    break

        cap.release()

        if not frames_to_check:
            return self._error_json("Could not read any valid video frames.")

        # Use Middle Frame as reference for geometry and safety checks
        reference_frame = frames_to_check[len(frames_to_check) // 2]

        job_start_time = time.time()

        # ── TITLE OVERRIDE GUARD ────────────────────────────────────────────────
        # If the caller provides a title, it is the authoritative niche source.
        # We NEVER let Gemini's visual classification override an explicit title.
        title_niche_override = None
        if title and title.strip():
            _title_clean = title.strip()
            try:
                from Watermark_and_Inpainting.gemini_enhance_for_watermark import NICHE_LIST
            except ImportError:
                try:
                    from gemini_enhance_for_watermark import NICHE_LIST
                except ImportError:
                    NICHE_LIST = []

            # Exact match first
            if _title_clean in NICHE_LIST:
                title_niche_override = _title_clean
                logger.info(f"🏷️ [TITLE OVERRIDE] Title '{_title_clean}' matched NICHE_LIST exactly → niche locked.")
            else:
                # Case-insensitive fuzzy match
                _lower = _title_clean.lower()
                for n in NICHE_LIST:
                    if n.lower() == _lower:
                        title_niche_override = n
                        logger.info(f"🏷️ [TITLE OVERRIDE] Title '{_title_clean}' fuzzy-matched '{n}' → niche locked.")
                        break

            if title_niche_override is None:
                # Title was given but is NOT a recognized actress — route to General_Fallback
                title_niche_override = "General_Fallback"
                logger.info(
                    f"🏷️ [TITLE OVERRIDE] Title '{_title_clean}' is NOT in NICHE_LIST → "
                    f"niche forced to General_Fallback (Gemini visual ID suppressed)."
                )
        # ────────────────────────────────────────────────────────────────────────

        # 3. Multi-Frame Detection (returns tuple now)
        detected_boxes = []
        detected_niche = "General_Fallback"

        try:
            gemini_enhance = ImportGate.get("gemini_enhance_for_watermark")
            if gemini_enhance is None:
                # Fallback to direct import
                try:
                    from gemini_enhance_for_watermark import detect_watermark as gemini_detect
                    gemini_enhance = type('obj', (object,), {'detect_watermark': gemini_detect})()
                except ImportError:
                    logger.warning("⚠️ gemini_enhance_for_watermark module not available. Skipping Gemini detection.")
                    return self._error_json("gemini_enhance_for_watermark module not available")

            # Debug frame dump
            try:
                debug_dir = os.path.join(os.getcwd(), "debug_frames")
                os.makedirs(debug_dir, exist_ok=True)
                for i, frame in enumerate(frames_to_check):
                    cv2.imwrite(os.path.join(debug_dir, f"scan_{uuid.uuid4().hex[:4]}_{i}.jpg"), frame)
            except:
                pass

            logger.info(f"🔭 Initiating Gemini Detection (1 call) on {len(frames_to_check)} frames...")

            # STRICT CACHE POLICY: Inject Geometry Mode Salt (v11.0)
            salt = "FORENSIC_MODE:ULTRA | TARGET:STUDIO_LOGO_PHOTOGRAPHER_WATERMARK_CORNER | PRIORITY:CORNER_SWEEP_FIRST | BIAS:STRICT_ACCURACY_ONLY_NO_FALSE_POSITIVES"
            keywords = f"{keywords} | {salt}" if keywords else salt

            logger.info(f"    └─ 🔑 Injection: '{keywords[:60]}...'")

            # ── v2: unpack tuple (watermark boxes, niche) ──────────────────
            raw_result = None
            max_gemini_retries = 1
            for attempt in range(max_gemini_retries):
                # --- v10.0 Resolution Synchronization ---
                raw_result = gemini_enhance.detect_watermark(
                    frames_to_check, 
                    keywords=keywords,
                    force_width=w_img,
                    force_height=h_img
                )
                
                # Success checks
                if isinstance(raw_result, tuple) and raw_result[0] is not None:
                    break
                elif not isinstance(raw_result, tuple) and raw_result is not None:
                    break
                    
                if attempt < max_gemini_retries - 1:
                    logger.warning(f"⏳ Gemini Quota/Error (Attempt {attempt+1}/{max_gemini_retries}). Sleeping 45s for quota reset...")
                    time.sleep(45)

            # Backward-compatible unpack:
            # New gemini_enhance returns (list_or_none, niche_str_or_None)
            # Old gemini_enhance returns list_or_none directly
            if isinstance(raw_result, tuple) and len(raw_result) == 2:
                detected_boxes, gemini_niche = raw_result
                if not gemini_niche:
                    gemini_niche = "General_Fallback"
            else:
                # Legacy path: no niche returned
                detected_boxes = raw_result
                gemini_niche = "General_Fallback"
                logger.warning("⚠️ gemini_enhance.detect_watermark() returned a non-tuple result. "
                                "Niche routing will use General_Fallback. Please update gemini_enhance.")

            # Apply title override: if title was provided, IGNORE Gemini's niche result
            if title_niche_override is not None:
                if gemini_niche != title_niche_override:
                    logger.info(
                        f"🚫 [TITLE OVERRIDE] Gemini said '{gemini_niche}' — "
                        f"REJECTED. Title overrides to '{title_niche_override}'."
                    )
                detected_niche = title_niche_override
            else:
                detected_niche = gemini_niche

            # ── Persist niche to sidecar immediately (even if no watermarks) ──
            save_detected_niche(video_path, detected_niche)
            if title_niche_override is not None:
                # Re-write sidecar with source flag for auditing
                try:
                    sidecar = _niche_sidecar_path(video_path)
                    import time as _time
                    with open(sidecar, "w", encoding="utf-8") as _f:
                        import json as _json
                        _json.dump({"detected_niche": detected_niche, "source": "title_override",
                                    "title_provided": title, "timestamp": _time.time()}, _f)
                except Exception as _se:
                    logger.warning(f"⚠️ [TITLE OVERRIDE] Sidecar re-write failed: {_se}")
            logger.info(f"🎯 Niche for this video: {detected_niche}")

            # 3a. UNVERIFIED / QUOTA FAILURE: If Gemini returned None
            if detected_boxes is None:
                logger.warning("⚠️ Gemini returned None (Quota/Error/Budget). Marking as UNVERIFIED_EMPTY (NOT clean).")
                return json.dumps({
                    "watermarks": [],
                    "count": 0,
                    "status": "UNVERIFIED_EMPTY",
                    "context": {
                        "removal_success": False,
                        "reason": "Gemini Quota/Error (no detection)",
                        "is_clean": False,
                        "detection_failed": True,
                        "detected_niche": detected_niche
                    }
                }, indent=2)

            # 3b. CLEAN STATE — Trust Gemini Completely if it actually ran and verified 0
            if not detected_boxes:
                logger.info("✅ Gemini reported CLEAN. Video is watermark-free.")
                return json.dumps({
                    "watermarks": [],
                    "count": 0,
                    "status": "CLEAN",
                    "context": {"removal_success": True, "verified_by": "gemini_only", "is_clean": True, "detected_niche": detected_niche}
                }, indent=2)


        except Exception as e:
            logger.error(f"❌ Gemini Detection Exception: {e}")
            return json.dumps({
                "watermarks": [],
                "count": 0,
                "status": "UNVERIFIED_EMPTY",
                "context": {
                    "removal_success": False,
                    "reason": f"Exception: {e}",
                    "is_clean": False,
                    "detection_failed": True,
                    "detected_niche": detected_niche
                }
            }, indent=2)

        # 4. Process & Validate Detected Boxes
        # ── DEDUPLICATION: Merge same-watermark detections from multiple frames ──────
        # Gemini scans 7 frames. The same corner logo found in each frame becomes
        # 7 nearly-identical boxes. Use MEDIAN coordinates (not hull) to avoid
        # inflating the bounding box with frame-to-frame jitter.
        import statistics as _stats

        def _iou(a, b):
            ax1, ay1 = a['x'], a['y']
            ax2, ay2 = ax1 + a['w'], ay1 + a['h']
            bx1, by1 = b['x'], b['y']
            bx2, by2 = bx1 + b['w'], by1 + b['h']
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = a['w']*a['h'] + b['w']*b['h'] - inter
            return inter / union if union > 0 else 0.0

        merged_boxes = []
        used = [False] * len(detected_boxes)
        for i, box_i in enumerate(detected_boxes):
            if used[i]: continue
            cluster = [box_i]
            used[i] = True
            for j, box_j in enumerate(detected_boxes):
                if used[j]: continue
                if _iou(box_i, box_j) > 0.30:  # Strict overlap — only genuine duplicates
                    cluster.append(box_j)
                    used[j] = True
                    continue
                
                # If they share the exact same corner anchor from Gemini, they are the same mark
                anchor_a = box_i.get("semantic_anchor", "unknown")
                anchor_b = box_j.get("semantic_anchor", "unknown")
                if anchor_a == anchor_b and anchor_a not in ('unknown', 'floating', 'center'):
                    cluster.append(box_j)
                    used[j] = True
                    continue
                
                # NOTE: Distance-based merge REMOVED — it was grouping watermarks from
                # different parts of the frame (e.g. FILMYGYAN bottom-left + jacket element
                # upper-left) into one hull, producing wrong-location inpainting.

            # DENSEST-CLUSTER MEDIAN — replaces flat median.
            # Problem: Gemini scans 7 frames and may report the SAME static
            # watermark at wildly different y-coords across frames
            # (e.g. [547, 887, 1372, 1382] for @handle stuck at the bottom).
            # Flat median → 1129px, but the watermark is at 1377px → mask misses.
            # Fix: find the densest cluster (tightest group of values), then
            # compute median of only those values.  Outlier frames are ignored.
            CLUSTER_THRESH = max(120, int(h_img * 0.07))  # 7% of frame height ≈ 134px on 1920

            def _dense_cluster(vals):
                """Return the largest cluster of values within CLUSTER_THRESH of each other."""
                if len(vals) <= 2:
                    return vals
                best = []
                for v in vals:
                    grp = [x for x in vals if abs(x - v) <= CLUSTER_THRESH]
                    if len(grp) > len(best):
                        best = grp
                return best if best else vals

            # Use densest cluster for position, and build a MAX-HULL for size.
            # E.g. "Bollywindow" median width cuts off the "w". Hull covers the max extent.
            all_y = [b['y'] for b in cluster]
            dense_y = _dense_cluster(all_y)
            valid_boxes = [b for b in cluster if b['y'] in dense_y]

            if not valid_boxes:
                valid_boxes = cluster

            # Position: anchor to the top-leftmost of the valid cluster
            med_x = min(b['x'] for b in valid_boxes)
            med_y = int(_stats.median(b['y'] for b in valid_boxes))  # stable vertical anchor
            
            # Clip med_x to avoid going out of bounds via negative coords
            med_x = max(0, med_x - 5) # 5px left safety pad

            # Size: expand to the rightmost/bottommost bounds of ANY valid frame
            max_right = max(b['x'] + b['w'] for b in valid_boxes)
            max_bottom = max(b['y'] + b['h'] for b in valid_boxes)

            # Cap max_right/bottom just safely
            med_w = min(w_img - med_x, (max_right - med_x) + 15)  # 15px right safety pad
            med_h = min(h_img - med_y, (max_bottom - med_y) + 10) # 10px bottom safety pad

            if len(dense_y) < len(all_y):
                outliers = [y for y in all_y if y not in dense_y]
                logger.info(
                    f"🎯 [DEDUP] Outlier y-coords filtered: {outliers} → "
                    f"using dense cluster {dense_y} → y={med_y}"
                )

            representative = dict(cluster[0])  # copy first (preserves motion_hint, type, etc.)
            representative['x'] = med_x
            representative['y'] = med_y
            representative['w'] = med_w
            representative['h'] = med_h
            merged_boxes.append(representative)

            if len(cluster) > 1:
                logger.info(
                    f"🔗 [DEDUP] Merged {len(cluster)} near-identical boxes → "
                    f"1 cluster-median box at x={med_x},y={med_y},w={med_w},h={med_h}"
                )

        if len(merged_boxes) < len(detected_boxes):
            logger.info(f"✂️ [DEDUP] {len(detected_boxes)} raw detections → {len(merged_boxes)} unique watermarks after merge.")
        detected_boxes = merged_boxes

        # NOTE: Gap-Proximity Hull Merge REMOVED.
        # It was hull-merging FILMYGYAN (bottom-left) with jacket/clothing detections
        # (upper-left) into one giant box spanning the entire left column.
        # The auto-snap then picked different high-gradient regions within that
        # oversized ROI on each frame → shape-shifting patch on the wrong body part.
        # Each watermark is now kept as its own independent box — matches old module.


        final_watermarks = []


        # Define Center Region (Normalized 0-1)
        center_x_min = 0.3 * w_img
        center_x_max = 0.7 * w_img
        center_y_min = 0.3 * h_img
        center_y_max = 0.7 * h_img

        for i, box in enumerate(detected_boxes):
            x, y, w, h = box['x'], box['y'], box['w'], box['h']

            # --- VALIDATION 0: BRAND WATERMARK PROTECTOR ---
            # Protect intentional brand watermark ONLY when scanning an already-rendered master video
            # (_master.mp4 or video in Processed Shorts).
            # On RAW source videos (video.mp4 / upfront inpainting in downloads), ALL detected marks
            # are foreign creator watermarks and MUST be inpainted.
            v_name = os.path.basename(video_path).lower()
            is_master_video = "_master.mp4" in v_name or "processed shorts" in os.path.abspath(video_path).lower()

            is_brand_match = False
            if is_master_video:
                brand_text_env = (
                    os.getenv("BRAND_WATERMARK_TEXT", "").strip().lower()
                    or os.getenv("WATERMARK_TEXT", "").strip().lower()
                    or os.getenv("BRAND_NAME", "").strip().lower()
                )
                box_text = str(box.get("text", "")).strip().lower()
                box_reason = str(box.get("reason", "")).strip().lower()

                if brand_text_env:
                    if (brand_text_env in box_text or
                        (len(box_text) >= 3 and box_text in brand_text_env) or
                        brand_text_env in box_reason):
                        is_brand_match = True

                # Also check sidecar overlay box from single-pass render if present
                if not is_brand_match:
                    try:
                        video_dir = os.path.dirname(os.path.abspath(video_path))
                        sidecar_intel = os.path.join(video_dir, ".clip_intelligence.json")
                        if os.path.exists(sidecar_intel):
                            with open(sidecar_intel, "r", encoding="utf-8") as _sf:
                                _data = json.load(_sf)
                            _b_info = _data.get("output", {}).get("brand_overlay") or _data.get("editing_plan", {}).get("brand_overlay")
                            if _b_info and isinstance(_b_info, dict):
                                bx, by, bw, bh = _b_info.get("x", 0), _b_info.get("y", 0), _b_info.get("w", 100), _b_info.get("h", 50)
                                # Overlap or center-distance check
                                ov_x = max(0, min(x + w, bx + bw) - max(x, bx))
                                ov_y = max(0, min(y + h, by + bh) - max(y, by))
                                if (ov_x * ov_y) > 0 or abs(y - by) < 150:
                                    is_brand_match = True
                    except Exception:
                        pass

                # Failsafe for master: If box is located in bottom-center brand overlay zone (y >= 82% of frame height)
                if not is_brand_match and brand_text_env and y >= (h_img * 0.82):
                    logger.info(f"🛡️ [BRAND PROTECTOR] Master box {i} at y={y} (bottom 18% of frame height) matches intentional brand watermark zone → PROTECTED!")
                    is_brand_match = True

                if is_brand_match:
                    logger.info(f"🛡️ [BRAND PROTECTOR] Master candidate box {i} ('{box_text or brand_text_env}') matches intentional brand watermark → PROTECTED! (Skipping inpaint on master).")
                    continue


            # --- VALIDATION 1: GEOMETRY ---

            if h <= 0: continue
            ar = w / float(h)
            if ar < 0.05 or ar > 20.0:
                logger.warning("⚠️ [WM] Candidate rejected")
                logger.warning(f"    ├─ reason: aspect_ratio {ar:.2f}")
                logger.warning(f"    └─ box_id: {i}")
                continue

            # Area (Max 15%)
            area_pct = (w * h) / (w_img * h_img)
            if area_pct > 0.15:
                logger.warning("⚠️ [WM] Candidate rejected")
                logger.warning(f"    ├─ reason: area_too_large {area_pct:.1%}")
                logger.warning(f"    └─ box_id: {i}")
                continue
            if area_pct < 0.0001:
                if retry_level > 0:
                    logger.info(f"☢️ Nuclear Mode: Accepting tiny watermark ({area_pct:.5f})")
                else:
                    logger.info(f"Ignoring microscopic detection {i} (<0.01%)")
                    continue

            # --- VALIDATION 2: FACE SAFETY ---
            is_safe, reason = FaceProtector.is_safe_region(reference_frame, box)
            is_soft_warn = "SOFT_FACE_PROXIMITY" in reason

            if not is_safe:
                logger.warning(f"⛔ Safety Reject (Face): {reason}")
                continue

            if is_soft_warn:
                logger.info(f"⚠️ [WM] Allowed with Soft Face Proximity warning.")

            # --- CONFIDENCE CALCULATION ---
            conf = 0.95
            if area_pct > 0.10: conf -= 0.1
            if ar < 0.2 or ar > 5.0: conf -= 0.1
            conf = max(0.0, min(1.0, conf))

            min_conf = 0.55 if retry_level > 0 else 0.60  # Lowered: studio logos have odd shapes that reduce score
            if conf < min_conf:
                logger.warning("⚠️ [WM] Candidate rejected")
                logger.warning(f"    ├─ reason: low_confidence {conf:.2f} (Threshold: {min_conf})")
                logger.warning(f"    └─ box_id: {i}")
                continue

            if retry_level > 0:
                logger.info(f"☢️ Nuclear Mode: Accepted low confidence ({conf:.2f}) candidate.")

            # 🛡️ RULE 2: ABSOLUTE ACCURACY GEOMETRY (Aspect-Aware V2)
            ar = w / float(h)
            
            if ar > 3.0: 
                # Wide (Text line?) -> Expand Height slightly
                exp_w_pct = 0.10 # Reduced from 0.25 to prevent mask inflation
                exp_h_pct = 0.20 # Reduced from 0.45
            elif ar < (1.0 / 2.5): # h/w > 2.5
                # Tall (Vertical banner/Logo?) 
                exp_w_pct = 0.20 # Reduced from 0.40
                exp_h_pct = 0.10 # Reduced from 0.40
            else:
                # Standard (Square-ish Logo)
                exp_w_pct = 0.10 # Reduced from 0.30
                exp_h_pct = 0.10 # Reduced from 0.30
                
            exp_w = int(w * exp_w_pct)
            exp_h = int(h * exp_h_pct)
            
            nx = x - (exp_w // 2)
            ny = y - (exp_h // 2)
            nw = w + exp_w
            nh = h + exp_h
            
            nx = max(0, nx); ny = max(0, ny)
            nw = min(nw, w_img - nx); nh = min(nh, h_img - ny)

            # 🛡️ ITERATIVE EDGE EXPANSION (Auto-Grow)
            # Disabled: Caused masks to inflate up to 4x their size for small watermarks in textured areas
            # try:
            #     max_iter = 10
            #     expand_step = 8
            #     # Threshold: > 10% of border pixels are edges -> cut through text -> expand
            #     edge_border_thr = 0.10
            #
            #     for _ in range(max_iter):
            #         ...
            #         if not expanded_any:
            #             break
            #
            #         if (nw * nh) > (w_img * h_img * 0.25):
            #             logger.warning("⚠️ Auto-Grow halted: Max area (25%) reached.")
            #             break
            #
            # except Exception as e:
            #     logger.warning(f"Auto-Grow Validation failed: {e}")

            # Update Box (Final Lock)
            box['x'], box['y'], box['w'], box['h'] = nx, ny, nw, nh

            watermark_entry = {
                "id": i + 1,
                "coordinates": box,
                "confidence": conf,
                "safe_to_remove": True,
                "decision": "remove",
                "time_range": {"start": 0.0, "end": 0.0},
                "is_moving": box.get("motion_hint") == "dynamic",
                "motion_hint": box.get("motion_hint", "static"),
                "watermark_type": "GEMINI_EXACT",
                "face_proximity": is_soft_warn,
                "strategy": "inpaint_standard",
                "semantic_class": box.get("semantic_type", "unknown")
            }
            final_watermarks.append(watermark_entry)

        # 5. Final Packaging
        if final_watermarks:
            frame_path = os.path.join(os.path.dirname(video_path), f"frame_{uuid.uuid4().hex[:6]}.jpg")
            try:
                cv2.imwrite(frame_path, reference_frame)
            except:
                frame_path = "error_saving_frame.jpg"

            logger.info(f"🏁 Finalized {len(final_watermarks)} watermarks for removal.")

            return json.dumps({
                "watermarks": final_watermarks,
                "count": len(final_watermarks),
                "status": "DETECTED",
                "context": {
                    "frame_path": frame_path,
                    "removal_success": False,
                    "detected_niche": detected_niche
                },
                "processing_time": time.time() - job_start_time
            }, indent=2)
        else:
            return json.dumps({
                "watermarks": [],
                "count": 0,
                "status": "CLEAN",
                "context": {"removal_success": True, "detected_niche": detected_niche}
            }, indent=2)

    def generate_static_mask(self, video_path: str, box: dict, output_path: str, padding_ratio: float = 0.0, semantic_class: str = "text") -> bool:
        """
        Generates a STATIC mask video using Alpha-Safe Authority logic.
        """
        try:
            original_box = box.copy()

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return False

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cap.release()

            # --- TEMPORAL SMART REFINEMENT ---
            try:
                try:
                    from Watermark_and_Inpainting.opencv_watermark import TemporalSmartRefiner
                except ImportError:
                    from opencv_watermark import TemporalSmartRefiner

                refined_box = TemporalSmartRefiner.refine_box_temporal(video_path, box)
                box = refined_box  # Accept the tight fit (includes micro-pad)

                # Disable downstream padding because TemporalRefiner already micro-padded
                force_zero_padding = True

            except Exception as e:
                logger.warning(f"Temporal Refinement skipped: {e}")
                force_zero_padding = False

            x, y, w, h = box['x'], box['y'], box['w'], box['h']

            # --- ALPHA-SAFE MASK GENERATION ---
            core_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(core_mask, (x, y), (x + w, y + h), 255, -1)

            final_mask = core_mask

            try:
                cap_ref = cv2.VideoCapture(video_path)
                cap_ref.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret_ref, frame_ref = cap_ref.read()
                cap_ref.release()

                if ret_ref:
                    final_mask = core_mask

                    # --- GLYPH-SAFE EXPANSION (TEXT ONLY) ---
                    is_cpu_safe = os.getenv("COMPUTE_MODE") == "cpu"

                    if semantic_class == "text" and not is_cpu_safe:
                        logger.info(f"✍️ Applying Glyph-Safe Expansion (+2px V, +3px H) for {semantic_class}...")
                        kernel_glyph = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
                        expanded_mask = cv2.dilate(final_mask, kernel_glyph, iterations=1)

                        constraint_canvas = np.zeros_like(final_mask)
                        ox, oy, ow, oh = original_box['x'], original_box['y'], original_box['w'], original_box['h']
                        ox = max(0, ox); oy = max(0, oy)
                        ow = min(ow, width - ox); oh = min(oh, height - oy)

                        cv2.rectangle(constraint_canvas, (ox, oy), (ox + ow, oy + oh), 255, -1)
                        final_mask = cv2.bitwise_and(expanded_mask, constraint_canvas)

            except Exception as e:
                logger.warning(f"Smart Halo generation failed: {e}")
                final_mask = core_mask

            # --- FINAL SAFETY CLIP (Face Firewall) ---
            try:
                cap_safety = cv2.VideoCapture(video_path)
                cap_safety.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret_s, frame_s = cap_safety.read()
                cap_safety.release()

                if ret_s:
                    faces = FaceProtector.detect_faces(frame_s)
                    for (fx, fy, fw, fh) in faces:
                        protected_h = int(fh * 0.85)
                        cv2.rectangle(final_mask, (fx, fy), (fx + fw, fy + protected_h), 0, -1)
                        logger.info(f"🛡️ Face Firewall: Sanitized Face Region at ({fx},{fy}) - Top 85% Protected")

            except Exception as e:
                logger.warning(f"Face Firewall failed: {e}")

            mask_frame = final_mask

            base_path, _ = os.path.splitext(output_path)
            png_path = f"{base_path}.png"

            cv2.imwrite(png_path, mask_frame)
            logger.info(f"✅ Static Mask Generated: {os.path.basename(png_path)}")

            return True

        except Exception as e:
            logger.error(f"Mask Gen Error: {e}")
            return False

    def generate_tracked_mask(self, video_path: str, box: dict, output_path: str, padding_ratio: float = 0.0, semantic_class: str = "text") -> bool:
        """
        Smart Tracking Mask Generator (CPU-Safe).
        Uses Template Matching with Local Search Window to follow moving watermarks.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return False

            w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            ref_idx = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
            ret, ref_frame = cap.read()

            if not ret:
                cap.release()
                return False

            x, y, w, h = box['x'], box['y'], box['w'], box['h']

            x = max(0, x); y = max(0, y)
            w = min(w, w_vid - x); h = min(h, h_vid - y)

            if w < 10 or h < 10:
                cap.release()
                return False

            template = ref_frame[y:y + h, x:x + w]
            if template.size == 0:
                cap.release()
                return False

            curr_x, curr_y = x, y
            search_margin = max(50, int(max(w, h) * 0.5))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w_vid, h_vid), isColor=False)

            kernel_glyph = None
            if semantic_class == "text":
                kernel_glyph = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))

            # Detect faces ONCE on the reference frame and cache the exclusion zone.
            # Calling FaceProtector.detect_faces(frame) inside the per-frame loop
            # runs the DNN on every frame — ~100-200 ms × 900 frames = 90-180 s of
            # pure DNN overhead, which triggers the time-budget fast-blur fallback.
            cached_face_excl = np.zeros((h_vid, w_vid), dtype=np.uint8)
            try:
                ref_faces = FaceProtector.detect_faces(ref_frame)
                for (fx, fy, fw, fh) in ref_faces:
                    core_h = int(fh * 0.85)
                    cv2.rectangle(cached_face_excl, (fx, fy), (fx + fw, fy + core_h), 255, -1)
                if ref_faces:
                    logger.info(f"🛡️ Tracked Mask: face exclusion zone cached ({len(ref_faces)} face(s)).")
            except Exception as e:
                logger.warning(f"Face cache build failed: {e}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret: break

                sx = max(0, curr_x - search_margin)
                sy = max(0, curr_y - search_margin)
                ex = min(w_vid, curr_x + w + search_margin)
                ey = min(h_vid, curr_y + h + search_margin)

                search_region = frame[sy:ey, sx:ex]

                if search_region.shape[0] > h and search_region.shape[1] > w:
                    try:
                        res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)

                        if max_val > 0.65:
                            curr_x = sx + max_loc[0]
                            curr_y = sy + max_loc[1]
                    except:
                        pass

                mask_frame = np.zeros((h_vid, w_vid), dtype=np.uint8)

                px = int(w * padding_ratio)
                py = int(h * padding_ratio)

                draw_x = curr_x - px
                draw_y = curr_y - py
                draw_w = w + (px * 2)
                draw_h = h + (py * 2)

                cv2.rectangle(mask_frame, (draw_x, draw_y), (draw_x + draw_w, draw_y + draw_h), 255, -1)

                if kernel_glyph is not None:
                    mask_frame = cv2.dilate(mask_frame, kernel_glyph, iterations=1)

                # --- FACE SAFETY: use cached exclusion zone (no per-frame DNN) ---
                if cv2.countNonZero(cached_face_excl) > 0:
                    mask_frame = cv2.bitwise_and(mask_frame, cv2.bitwise_not(cached_face_excl))

                out.write(mask_frame)

            cap.release()
            out.release()
            logger.info(f"📍 Tracked Mask Generated: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Tracked Mask Gen Error: {e}")
            return False


# Singleton
hybrid_detector = HybridWatermarkDetector()