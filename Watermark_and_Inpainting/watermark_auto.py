"""
Automated Watermark Orchestrator
--------------------------------
Centralizes adaptive removal logic to unburden compiler.py.
Fixes: Mask Drift (Decoupled Motion), Advanced Strategies.
"""

import os
import shutil
import logging
from typing import List, Dict, Tuple, Optional

# Shared Modules
try:
    from Watermark_and_Inpainting import hybrid_watermark
    from Watermark_and_Inpainting.import_gate import ImportGate
    from Watermark_and_Inpainting.opencv_watermark import inpaint_video, check_watermark_residue, MaskVerifier, verify_visual_guarantee
    from Watermark_and_Inpainting.watermark_enhancers import MicroTextureBlender
except ImportError:
    import hybrid_watermark
    from import_gate import ImportGate
    from opencv_watermark import inpaint_video, check_watermark_residue, MaskVerifier, verify_visual_guarantee
    try:
        from watermark_enhancers import MicroTextureBlender
    except ImportError:
        MicroTextureBlender = None

try:
    from Watermark_and_Inpainting.static_patch_engine import StaticPatchReuseEngine
except ImportError:
    try:
        from static_patch_engine import StaticPatchReuseEngine
    except ImportError:
        StaticPatchReuseEngine = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("watermark_auto")

def run_adaptive_watermark_orchestration(
    input_video: str, 
    watermarks: List[Dict], 
    output_video: str, 
    job_dir: str, 
    original_height: int = 1080, 
    aggressive_mode: bool = True, 
    retry_level: int = 0
) -> Tuple[bool, str]:
    """
    Control System for Watermark Removal.
    Iteratively tunes mask padding and inpaint radius based on residue feedback.
    """
    logger.info(f"🧠 Starting Smart-Decay Watermark Orchestrator (Retry Level: {retry_level})...")
    
    # 1. Check Motion First (ALWAYS)
    # Support both 'coordinates' key and flat dict
    if watermarks and 'coordinates' in watermarks[0]:
        wm_box = watermarks[0]['coordinates']
    else:
        wm_box = watermarks[0] if watermarks else {'x':0,'y':0,'w':0,'h':0}
    
    # Get AI Hint for Watermark Motion
    wm_motion_hint = watermarks[0].get("motion_hint", "static")
    
    
    # Check Pixel Motion (Background checks)
    if StaticPatchReuseEngine:
        is_background_moving = StaticPatchReuseEngine.check_pixel_motion(input_video, wm_box)
    else:
        is_background_moving = True # Fallback safer

    
    motion_override = "static"
    radius_boost_level = 0
    
    # DRIFT FIX V2: "Innocent until proven Static"
    # If background moves, we must assume the watermark MIGHT move (e.g. on a shirt).
    # Only force Static if AI explicitly said "static".
    if is_background_moving:
         logger.info("🌊 Motion Detected (Background): Enabling Tracking.")
         motion_override = "dynamic"
         radius_boost_level = 2
         
         # Force Tracked Mask for safety on moving objects (Fixes Position Drift)
         # Even if it thinks it's static, if the background moves, we track.
         wm_motion_hint = "dynamic"
    else:
         motion_override = "static"
         radius_boost_level = 0

    if retry_level >= 2: radius_boost_level += 4
    elif retry_level == 1: radius_boost_level += 2

    # SMART-DECAY STRATEGY DEFINITIONS
    # Lite Model Compensation: If using Flash-Lite, start aggression higher.
    is_lite_model = "lite" in os.getenv("GEMINI_MODEL", "").lower()
    
    if is_lite_model:
        logger.info("⚠️ Lite Model Detected: Boosting inpaint aggression constraints.")
        strategies = [
            {'pad': 0.20, 'rad': 6, 'thresh': 0.25, 'name': 'Precision+ (Lite)'}, # Boosted
            {'pad': 0.30, 'rad': 8, 'thresh': 0.50, 'name': 'Balanced+ (Lite)'},
            {'pad': 0.40, 'rad': 10, 'thresh': 1.00, 'name': 'Nuclear (Force)'}
        ]
    else:
        strategies = [
            {'pad': 0.15, 'rad': 5, 'thresh': 0.15, 'name': 'Precision (Strict)'},
            {'pad': 0.25, 'rad': 7, 'thresh': 0.40, 'name': 'Balanced (Medium)'},
            {'pad': 0.35, 'rad': 9, 'thresh': 1.00, 'name': 'Nuclear (Force)'}
        ]
    
    radius_boost = (1 if original_height < 1080 else 0) + radius_boost_level
    
    for attempt, strat in enumerate(strategies, 1):
        pad_boost = retry_level * 0.10
        pad_ratio = strat['pad'] + pad_boost
        # FORCE PRIME RADIUS (Force Synchronized Strategy)
        # We use a 30% fill heuristic to estimate mask pixels from the bounding box
        wm_coords_ref = watermarks[0].get('coordinates', watermarks[0])
        est_mask_pixels = (wm_coords_ref.get('w', 0) * wm_coords_ref.get('h', 0)) * 0.30

        if est_mask_pixels < 300:
            radius = 7
        elif est_mask_pixels < 1200:
            radius = 11
        else:
            radius = 13
        threshold = strat['thresh']
        name = strat['name']
        
        logger.info(f"🛡️ Strategy {attempt}/{len(strategies)}: [{name}] (Inpaint: {motion_override} | Mask: {wm_motion_hint})")
        
        # 1. Generate Masks
        masks = []
        text_masks = []
        for i, watermrk in enumerate(watermarks):
            # Static Mask -> PNG (Sync Safe)
            # Tracked -> MP4
            current_wm_motion = watermrk.get("motion_hint", "static")
            
            if current_wm_motion == "dynamic":
                mpath = os.path.join(job_dir, f"mask_a{attempt}_{i}.mp4")
            else:
                mpath = os.path.join(job_dir, f"mask_a{attempt}_{i}.png")
            
            # MASK GENERATION ROUTING (The Drift Fix)
            # If AI says Watermark is STATIC, use Static Mask (Fixed Box).
            # ONLY use Tracked Mask if AI says the watermark ITSELF is moving.
            
            # Support both formats
            wm_coords = watermrk.get('coordinates', watermrk)
            
            if current_wm_motion == "dynamic":
                 # Tracked
                 gen_success = hybrid_watermark.hybrid_detector.generate_tracked_mask(
                    input_video, wm_coords, mpath, 
                    padding_ratio=pad_ratio, semantic_class=watermrk.get("semantic_class", "unknown")
                 )
            else:
                 # Static (Prevents Drift on fixed overlays)
                 gen_success = hybrid_watermark.hybrid_detector.generate_static_mask(
                    input_video, wm_coords, mpath, 
                    padding_ratio=pad_ratio, semantic_class=watermrk.get("semantic_class", "unknown")
                 )

            if gen_success:
                masks.append(mpath)
        
        if not masks: return False, "Mask Gen Failed"
        
        # 2. Inpaint
        out_candidate = os.path.join(job_dir, f"candidate_a{attempt}.mp4")
        
        # Extract semantic_vectors from first watermark box (all boxes share same surface context)
        _sem_vecs = watermarks[0].get("semantic_vectors") or watermarks[0].get("vectors") or {}

        success = inpaint_video(
            input_video, masks, out_candidate, 
            original_height=original_height, 
            radius_override=radius, 
            motion_hint_override=wm_motion_hint,
            semantic_vectors=_sem_vecs,
        )
        
        if not success:
            logger.error(f"Inpainting failed on strategy {strat['name']}")
            continue
        
        # 3. Residue Check & Exit
        residue = check_watermark_residue(input_video, out_candidate, masks, watermarks)
        score = residue.get("score", 1.0)
        logger.info(f"🔍 Residue Check for {strat['name']}: {score:.3f} ({residue.get('reason')})")
        
        if score < 0.25 or attempt == len(strategies) - 1:
            out_parent = os.path.dirname(os.path.abspath(output_video))
            if out_parent:
                os.makedirs(out_parent, exist_ok=True)
            shutil.move(out_candidate, output_video)
            status = "Completed_Clean" if score < 0.25 else "Completed_With_Residue"
            logger.info(f"✅ Strategy accepted ({status})")
            return True, status
            
        logger.warning(f"⚠️ Residue too high ({score:.3f}), escalating to next strategy...")

    return False, "Strategies Exhausted"

def process_video_with_watermark(input_path: str, output_path: str, retry_mode: bool = False, retry_level: int = 0, pre_detected_watermarks=None, **kwargs) -> Dict:
    logger.info(f"🛡️ Watermark Filter Process called for: {input_path}")
    job_dir = os.path.join("temp_watermark", f"job_{int(os.path.getmtime(input_path))}")
    os.makedirs(job_dir, exist_ok=True)
    
    # Extract title from kwargs so callers can enforce niche identity
    _title = kwargs.get("title", None)
    if _title:
        logger.info(f"🏷️ [WATERMARK_AUTO] Title provided: '{_title}' — will be used as niche authority.")

    try:
        import json
        is_error = False  # [FIX] Define early so it's always in scope
        watermarks = []
        if pre_detected_watermarks:
            logger.info("⚡ Using pre-detected watermarks (Gemini Master Analysis). Skipping detection call.")
            watermarks = pre_detected_watermarks
        else:
            # Fallback to detection if not provided
            json_res = hybrid_watermark.hybrid_detector.process_video(input_path, aggressive=retry_mode, retry_level=retry_level, title=_title)
            try:
                data = json.loads(json_res)
                watermarks = data.get("watermarks", [])
                ctx = data.get("context", {})
                is_error = ctx.get("removal_success") is False or "Quota/Error" in str(ctx.get("reason", ""))
            except:
                watermarks = []
                is_error = True
            
        if not watermarks:
            out_parent = os.path.dirname(os.path.abspath(output_path))
            if out_parent:
                os.makedirs(out_parent, exist_ok=True)
            shutil.copy(input_path, output_path)
            if is_error:
                return {"success": False, "watermark_detected": None, "context": {"reason": "API Quota/Error"}}
            return {"success": True, "watermark_detected": False, "context": None}
             
        success, reason = run_adaptive_watermark_orchestration(
            input_path, watermarks, output_path, job_dir, 
            aggressive_mode=retry_mode, retry_level=retry_level
        )
        
        wm_bbox = None
        if watermarks:
             # Find first coord dict (might be flat or nested)
             _c = watermarks[0].get("coordinates", watermarks[0])
             wm_bbox = [_c.get('x',0), _c.get('y',0), _c.get('w',0), _c.get('h',0)]
             
        return {"success": success, "watermark_detected": True, "bbox": wm_bbox, "context": {"reason": reason, "removal_success": success}}
    except Exception as e:
        logger.error(f"Watermark process failed: {e}")
        return {"success": False, "watermark_detected": None, "context": None}
    finally:
        if job_dir and os.path.exists(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)
            logger.debug(f"🧹 [WATERMARK_AUTO] Purged ephemeral job directory: {job_dir}")

def apply_text_watermark(*args, **kwargs): return False

def process(video_path, frames):
    """Legacy detection-only entry point for orchestrator."""
    import json
    json_res = hybrid_watermark.hybrid_detector.process_video(video_path)
    try:
        data = json.loads(json_res)
        return {"watermarks": data.get("watermarks", []), "count": data.get("count", 0)}
    except:
        return {"watermarks": [], "count": 0}
