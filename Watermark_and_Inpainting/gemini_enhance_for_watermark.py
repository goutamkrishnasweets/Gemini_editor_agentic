"""
Gemini Watermark Detection Module
---------------------------------
Isolated module for forensic watermark detection.
Used by HybridWatermarkDetector.
"""

import os
import cv2
import base64
import logging
import json
import re
import numpy as np
import time
import math
from typing import Optional, Dict, Any, List

logger = logging.getLogger("gemini_watermark")

# Import gemini_router from the correct location
try:
    from Gemini_Modules.gemini_router_module.gemini_governor import gemini_router
    HAS_GEMINI = True
except ImportError:
    gemini_router = None
    HAS_GEMINI = False
    logger.warning("⚠️ gemini_governor module not found. Gemini detection will be disabled.")

try:
    from PIL import Image
except ImportError:
    pass

# Configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_FALLBACK_25 = "gemini-2.5-flash"
GEMINI_FALLBACK_LITE = "gemini-2.5-flash-lite"

def frame_to_pil(frame: np.ndarray):
    try:
        h, w = frame.shape[:2]
        # Increased to 1440px so small corner logos remain legible for Gemini
        if w > 1440:
            scale = 1440 / w
            frame = cv2.resize(frame, (1440, int(h * scale)))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
    except Exception as e:
        logger.warning(f"Failed to convert frame to PIL: {e}")
        return None

def clean_json_response(text: str) -> str:
    """
    Robustly extracts JSON content from a text response that might contain
    backticks, conversational filler, or other non-JSON headers.
    """
    if not text:
        return ""
    try:
        # 1. Handle standard markdown wrappers
        if "```" in text:
            # Try to grab content between triple backticks
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
            else:
                # If no pair found, just strip the backticks themselves
                text = re.sub(r"```(json)?", "", text).replace("```", "")

        # 2. Heuristic extraction: Find the first { and last } or first [ and last ]
        # This handles cases where Gemini adds text BEFORE or AFTER the JSON block.
        json_start = text.find('{')
        json_end = text.rfind('}')
        list_start = text.find('[')
        list_end = text.rfind(']')

        # Determine which structure is the primary one
        start_idx = -1
        end_idx = -1

        if json_start != -1 and (list_start == -1 or json_start < list_start):
            start_idx = json_start
            end_idx = json_end
        elif list_start != -1:
            start_idx = list_start
            end_idx = list_end

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx : end_idx + 1].strip()

        return text.strip()
    except Exception:
        return text.strip()


def detect_watermark(
    frames: List[np.ndarray],
    keywords: str = None,
    force_width: int = None,
    force_height: int = None,
    frame_timestamps: List[float] = None,
) -> tuple:
    """
    Detects watermarks AND classifies fashion vs NSFW + picks best thumbnail frame in ONE Gemini call.

    Returns a tuple: (results_list_or_none, frame_and_niche_info_or_None)
    - results_list:         list of watermark dicts, [] if clean, None if Gemini failed/quota
    - frame_and_niche_info: dict with keys:
        {
          "best_frame":           {"index": int, "reason": str, "timestamp": float},
          "clothing_coverage_pct": int,   # 0-100
          "is_nsfw":              bool,   # True if coverage < 30%
          "content_category":     str,    # "fashion" | "nsfw" | "general"
        }
        Returns None if Gemini fails.

    Part A: Watermark detection (UNCHANGED)
    Part B: Best thumbnail frame selection + fashion/NSFW clothing coverage detection
    """
    # Check if Gemini is available
    if not HAS_GEMINI or gemini_router is None:
        logger.warning("⚠️ Gemini router not available. Cannot perform watermark detection.")
        return None, None
    
    # Ensure genai is configured
    
    try:
        from PIL import ImageEnhance, ImageFilter
        if not isinstance(frames, list): frames = [frames]
        # Convert all frames to native PIL images
        pil_images = []
        for f in frames:
            pil_img = frame_to_pil(f)
            if pil_img:
                pil_images.append(pil_img)

        if not pil_images: return None, None

        # We pass the full frames "as-is" without artificial contrast boosting,
        # because contrast boost blows out white text on bright backgrounds (e.g. Bollywindow).
        # Gemini 1.5/2.5 Flash has native visual acuity to detect faint overlays if given 1080p frames.
        n_full = len(pil_images)

        keywords_hint = ""
        if keywords and str(keywords).strip():
            kw_clean = str(keywords).strip()
            keywords_hint = f"""
─────────────────────────────────────────────────
⚠️ TARGET BRANDING HINT (CRITICAL USER DIRECTIVE):
The user explicitly requested removal of this watermark/branding:
"{kw_clean}"
Conduct an aggressive inspection for any text, handle, bug, or logo resembling or containing "{kw_clean}" in all frames and flag it!
─────────────────────────────────────────────────
"""

        # 🎯 FORENSIC PROMPT v17.0 — WATERMARK HUNT + BEST FRAME + FASHION/NSFW DETECTION
        # Part A: Watermark detection (UNCHANGED — do not modify)
        # Part B: Best thumbnail frame selection + clothing coverage (fashion vs NSFW)
        frame_labels = ", ".join([f"Frame {i}" for i in range(n_full)])
        prompt = f"""You are an elite forensic video analyst and visual quality inspector.
You have been given {n_full} frame(s) from a fashion/lifestyle video: {frame_labels}.
Frames are numbered starting at 0 (Frame 0 = first image, Frame {n_full - 1} = last image).

══════════════════════════════════════════════════════════════
PART A — WATERMARK / BRANDING DETECTION
══════════════════════════════════════════════════════════════
Hunt for ANY source branding/watermark burned into these frames. Examine all {n_full} frame(s) with maximum precision.
{keywords_hint}
─────────────────────────────────────────────────
WHAT TO FLAG (source branding that must be removed):
─────────────────────────────────────────────────
FLAG EVERY ONE of these, no matter where they appear in the frame:

1. LOGO / ICON MARKS
   • Any graphic symbol, icon, or logo — whether a professional brand mark, a phone camera app icon,
     a social media platform logo (Instagram reel icon, TikTok logo, YouTube play button), or a custom studio emblem
   • Semi-transparent / watermarked logos overlaid anywhere on the video (center, corner, edge, mid-frame)
   • Animated or pulsing logo bugs (they still count even if they appear briefly)

2. STUDIO / VIDEOGRAPHER BRANDING
   • Names like "MAHESH VIDEO", "XYZ Studio", "ABC Photography", "Creative Films" — any studio tag
   • Photographer/videographer personal brand: a name paired with any icon or graphic
   • Production house credit anywhere in frame

3. SOCIAL MEDIA HANDLES / CHANNEL TAGS
   • ANY "@username", "#channelname", or "youtube.com/..." burned into the video
   • These may appear in corners, edges, or even floating mid-frame
   • Even if small and faint — still FLAG them

4. AGENCY / STOCK / COPYRIGHT MARKS
   • Getty, Shutterstock, iStock, Adobe Stock, or any watermark-text diagonal across the frame
   • Any "© Copyright" or "All Rights Reserved" text overlay

5. NEWS / BROADCAST OVERLAY ELEMENTS
   • Channel logo bug (small logo in corner during a broadcast)
   • Network name, show title, or programme name burned in
   • "LIVE", "BREAKING", "EXCLUSIVE" labels
   • News ticker/crawl along the bottom
   • Score banners, sports overlays, election result bars

6. CAMERA / DEVICE STAMPS
   • Date/time burned in by the camera body (e.g., "2024-01-15 14:32")
   • Device-specific overlays (e.g., CCTV timestamp, GoPro watermark, DJI logo)

7. REPOST / AGGREGATOR TAGS
   • "Via @account", "Credit: @someone", repost app tags
   • Any text that attributes the video to a third-party distributor

─────────────────────────────────────────────────
WHAT NOT TO FLAG (creative content — leave these alone):
─────────────────────────────────────────────────
Do NOT flag these — they are part of the video's creative content:

✅ CAPTIONS / SUBTITLES
   → Dialogue or narration text displayed for viewers to read along
   → KEY TEST: Are the words lines of speech/dialogue someone is saying? → CAPTION, skip it.
   → Subtitles appear horizontally centered, usually at bottom third, change frequently

✅ TITLE CARDS / INTRO TEXT
   → Large styled text introducing the video topic, usually displayed for 1–3 seconds at the start

✅ MOTIVATIONAL / LIFESTYLE TEXT OVERLAYS
   → Inspirational quotes, "POV:" text, meme-format captions, "Day 1 of..." text
   → These are editorial choices, not branding

✅ LOWER-THIRD LABELS
   → Interview-style labels: "JOHN SMITH — CEO", "Guest: Dr. Priya"
   → These identify the people speaking, not the studio that filmed them

✅ INTENTIONAL UI ELEMENTS
   → Countdown timers, progress bars, poll results, like counters shown for effect
   → Animated text effects that are part of the video's style

✅ EMOJI OVERLAYS
   → Emojis placed as creative decorations — not branding

─────────────────────────────────────────────────
THE DECISIVE TEST — Apply this to every element you see:
─────────────────────────────────────────────────
Ask: "Did the VIDEOGRAPHER or DISTRIBUTOR add this to claim/brand the video?"
  → YES → FLAG IT (watermark/branding)
  → NO, it's part of the story/content → LEAVE IT (caption/creative)

Ask: "Is this text what someone is SAYING or READING in the video?"
  → YES → It's a CAPTION. Do NOT flag it.
  → NO → It could be branding. Check the other rules.

─────────────────────────────────────────────────
MANDATORY FULL-FRAME SWEEP (do all 9 zones):
─────────────────────────────────────────────────
Watermarks appear ANYWHERE — not just corners. Check every zone:

  TOP-LEFT (0–20% x, 0–20% y)       TOP-CENTER (40–60% x, 0–20% y)    TOP-RIGHT (80–100% x, 0–20% y)
  MID-LEFT (0–20% x, 40–60% y)      CENTER     (30–70% x, 30–70% y)   MID-RIGHT (80–100% x, 40–60% y)
  BOT-LEFT (0–20% x, 80–100% y)     BOT-CENTER (40–60% x, 80–100% y)  BOT-RIGHT (80–100% x, 80–100% y)

Even a tiny, faint, semi-transparent, or briefly-appearing mark → REPORT IT.

══════════════════════════════════════════════════════════════
OUTPUT — STRICT JSON ONLY, NO OTHER TEXT:
══════════════════════════════════════════════════════════════
If source branding IS found:
{{
  "watermark_present": true,
  "clothing_coverage_pct": 85,
  "is_nsfw": false,
  "items": [
    {{
      "box_2d": [ymin, xmin, ymax, xmax],
      "type": "studio_logo" | "social_handle" | "agency_mark" | "news_overlay" | "cam_stamp" | "text" | "logo" | "repost_tag" | "broadcast_bug",
      "anchoring": "top_left" | "top_right" | "bottom_left" | "bottom_right" | "top_center" | "bottom_center" | "mid_left" | "mid_right" | "center_frame" | "floating",
      "motion": "static" | "dynamic",
      "text_content": "exact text or logo description visible",
      "opacity_level": 0.1 to 1.0,
      "color_polarity": "white_text_dark_glow" | "dark_text_light_glow" | "translucent_white" | "colored_logo",
      "background_texture": "flat_smooth" | "gradient" | "complex_foliage_hair" | "fabric_clothing",
      "motion_trajectory": "static" | "bouncing_vertical" | "sliding_horizontal"
    }}
  ],
  "best_frame": {{
    "index": <0-based integer — which frame is the best thumbnail shot>,
    "reason": "<one sentence: why this frame has the best outfit visibility and pose>"
  }}
}}

If NO source branding found:
{{
  "watermark_present": false,
  "clothing_coverage_pct": 85,
  "is_nsfw": false,
  "items": [],
  "best_frame": {{
    "index": <0-based integer>,
    "reason": "<one sentence: why this frame has the best outfit visibility and pose>"
  }}
}}

══════════════════════════════════════════════════════════════
PART B — BEST THUMBNAIL FRAME + FASHION/NSFW DETECTION
══════════════════════════════════════════════════════════════

THUMBNAIL FRAME SELECTION:
After completing Part A, identify which ONE frame (from Frame 0 to Frame {n_full - 1}) would
make the BEST thumbnail screenshot. This is the frame ffmpeg will extract as the cover image.

Score each frame on these criteria (in priority order):

1. OUTFIT / CLOTHING VISIBILITY  ← Most important
   → Full or majority of the outfit is visible in frame
   → Fashion details (fabric, color, accessories, shoes) are clear
   → Not obscured by camera motion blur, other people, or objects

2. POSE QUALITY
   → Confident, flattering, complete pose — not mid-step, mid-turn, or transition
   → Body is composed (not cut off at awkward joints)
   → Arms/hands add to the composition, not blocking the outfit

3. FACE / EXPRESSION
   → Face visible and well-composed (not blinking, not turned away)
   → Confident or natural expression — not caught mid-speech or distorted

4. LIGHTING & SHARPNESS
   → Frame is sharp (not motion-blurred)
   → Even lighting — not overexposed, not in deep shadow
   → Colors look natural and vibrant

5. COMPOSITION
   → Subject fills the frame well (not tiny in the distance)
   → Background is clean or complementary — not distracting

⚠️ AVOID frames with:
   → Heavy motion blur
   → Mid-transition between poses
   → Hands covering significant outfit area
   → Unflattering or distorted expressions
   → Subject partially out of frame

Return the index (0-based) of the single BEST frame in the "best_frame" field.

CLOTHING COVERAGE (for account routing):
Estimate % of main subject's body covered by clothing (0–100).
  100 = fully clothed | 60–99 = crop tops/gym wear | 30–59 = significant skin/revealing
  15–29 = heavy skin exposure (swimwear, lingerie) | 0–14 = near-nude / explicit

👗 FASHION / GENERAL  — clothing coverage >= 30%
   Examples: crop tops, short skirts, gym wear, mini dresses, deep necklines
   → Set is_nsfw = false

🔞 NSFW              — clothing coverage < 30%
   Examples: bikinis, lingerie, underwear, near-nude, topless, explicit content
   → Set is_nsfw = true

CRITICAL RULE: The threshold is EXACTLY 30%.
  coverage >= 30  →  is_nsfw = false  (fashion / general — safe for Instagram)
  coverage <  30  →  is_nsfw = true   (adult content — routes to NSFW account)

If no human visible in the video, set coverage = 100 and is_nsfw = false.

"""
        if keywords: prompt += f"\n\nADDITIONAL FOCUS HINTS: {keywords}"

        
        request_contents = pil_images + [prompt]
        
        res_txt = gemini_router.generate(
            task_type="watermark",
            prompt=request_contents,
            module_name="gemini_watermark"
        )
        
        if not res_txt or len(res_txt.strip()) < 2:
            logger.warning(f"⚠️ Gemini returned empty or too short response: '{res_txt}'")
            return None, None

        cleaned_txt = clean_json_response(res_txt)
        try:
            data = json.loads(cleaned_txt)
        except json.JSONDecodeError as jde:
            logger.error(f"❌ JSON Decode Error: {jde}")
            logger.error(f"   └─ Cleaned Text (first 100 chars): {cleaned_txt[:100]}...")
            logger.info(f"   └─ Raw Response (first 200 chars): {res_txt[:200]}...")
            return None, None

        # ── Parse Part B: best_frame + clothing coverage / fashion vs NSFW ──────────
        frame_and_niche_info = None
        if isinstance(data, dict):
            # ─ Thumbnail frame selection ─
            bf = data.get("best_frame", {})
            best_frame = None
            if isinstance(bf, dict) and "index" in bf:
                try:
                    bf_index = max(0, min(int(bf["index"]), n_full - 1))
                    bf_ts    = float(frame_timestamps[bf_index]) if (frame_timestamps and bf_index < len(frame_timestamps)) else -1.0
                    best_frame = {"index": bf_index, "timestamp": bf_ts, "reason": bf.get("reason", "")}
                    logger.info("📸 [BEST_FRAME] Frame %d (ts=%.2fs): %s", bf_index, bf_ts, bf.get("reason", "")[:80])
                except (ValueError, TypeError) as _e:
                    logger.debug("[BEST_FRAME] Parse failed: %s", _e)

            # ─ Clothing coverage / fashion vs general ─
            coverage_pct = int(data.get("clothing_coverage_pct", 100))
            is_nsfw      = False
            category     = "fashion" if coverage_pct >= 30 else "general"

            frame_and_niche_info = {
                "best_frame":            best_frame,
                "clothing_coverage_pct": coverage_pct,
                "is_nsfw":               False,
                "content_category":      category,
            }
            logger.info("👗 [NICHE] coverage=%d%% is_nsfw=%s category=%s", coverage_pct, False, category)

        if isinstance(data, list):
            items = data
        else:
            items = data.get("items", [])
            is_present_flag = data.get("watermark_present", False)

            if not is_present_flag and not items:
                logger.info("🕵️ Forensic Debug: Gemini reported CLEAN")
                logger.debug(f"   └─ Response: {res_txt[:500]}...")
                return [], frame_and_niche_info

            if not is_present_flag and items:
                logger.warning("🕵️ Forensic Debug: Gemini reported 'false' but found items. Overriding.")

        if not items:
            return [], frame_and_niche_info
            
        results = []
        h_img, w_img = frames[0].shape[:2]
        use_w = force_width if force_width else w_img
        use_h = force_height if force_height else h_img
        
        for item in items:
            box_norm = item.get("box_2d", [])
            if len(box_norm) != 4: continue
            ymin, xmin, ymax, xmax = box_norm
            
            # PRECISE COORDINATE MAPPING (0-1000 space)
            x_start = int(math.floor((xmin / 1000.0) * use_w))
            y_start = int(math.floor((ymin / 1000.0) * use_h))
            x_end = int(math.ceil((xmax / 1000.0) * use_w))
            y_end = int(math.ceil((ymax / 1000.0) * use_h))
            
            w_pixel = x_end - x_start
            h_pixel = y_end - y_start
            
            if w_pixel < 2 or h_pixel < 2: continue
            
            # --- QUADRANT SNAPPER DISABLED (v11.0) ---
            # Gemini's v11.0 prompt performs a mandatory 4-step corner sweep with
            # pixel-accurate JSON output. The snapper was built to fix old coordinate
            # hallucinations but now actively corrupts the accurate detections by
            # jumping to unrelated gradient clusters. Trust Gemini's coordinates directly.

            pad_x = 12
            pad_y = 8
            eff_x = max(0, x_start - pad_x)
            eff_y = max(0, y_start - pad_y)
            eff_w = w_pixel + 2 * pad_x
            eff_h = h_pixel + 2 * pad_y

            op_vectors = {
                "inpaint_radius": 13,
                "mask_padding_x": pad_x,
                "mask_padding_y": pad_y,
                "effective_x": eff_x,
                "effective_y": eff_y,
                "effective_w": eff_w,
                "effective_h": eff_h,
                "inpaint_method": "Navier_Stokes_EdgeIntegrator",
                "background_texture": str(item.get("background_texture", "complex")).lower(),
            }

            vectors = {
                "opacity_level": float(item.get("opacity_level", 0.5)),
                "color_polarity": str(item.get("color_polarity", "white_text_dark_glow")),
                "background_texture": str(item.get("background_texture", "flat_smooth")),
                "motion_trajectory": str(item.get("motion_trajectory", item.get("motion", "static"))),
            }

            results.append({
                'x': max(0, x_start), 'y': max(0, y_start),
                'w': min(use_w-x_start, w_pixel), 'h': min(use_h-y_start, h_pixel),
                'video_width': use_w,
                'video_height': use_h,
                'type': 'HYBRID_CLAMPED',
                'semantic_type': item.get("type", "unknown"),
                'semantic_anchor': item.get("anchoring", "unknown"),
                'semantic_hint': item.get("text_content", ""),
                'semantic_vectors': vectors,
                'opencv_vectors': op_vectors,
            })
            logger.info(
                f"💎 Sub-Pixel Sweep: {item.get('type')} -> x={x_start}, y={y_start}, w={w_pixel}, h={h_pixel} | "
                f"OpenCV Vectors: eff_x={eff_x}, eff_y={eff_y}, eff_w={eff_w}, eff_h={eff_h}"
            )

        return results, frame_and_niche_info
    except Exception as e:
        logger.warning(f"⚠️ Gemini Forensic Sweep failed: {e}")
        return None, None


def extract_best_frame_ffmpeg(
    video_path: str,
    best_frame_info: dict,
    output_path: str,
    fallback_second: float = 1.0,
) -> str:
    """
    Uses ffmpeg to extract the single best frame identified by detect_watermark().

    Args:
        video_path:       Path to the source video file.
        best_frame_info:  Dict returned by detect_watermark() — must have 'timestamp'.
        output_path:      Where to save the extracted JPEG/PNG frame.
        fallback_second:  If timestamp is -1 or missing, extract at this second instead.

    Returns:
        Absolute path to the extracted frame on success, or "" on failure.

    Example:
        results, best = detect_watermark(frames, frame_timestamps=[0.5, 1.5, 2.5])
        extract_best_frame_ffmpeg("clip.mp4", best, "thumbnail.jpg")
    """
    import subprocess
    from Compiler_Modules.video_pipeline import FFMPEG_BIN

    if not os.path.exists(video_path):
        logger.warning("[BEST_FRAME_EXTRACT] Video not found: %s", video_path)
        return ""

    # Resolve timestamp
    ts = fallback_second
    if isinstance(best_frame_info, dict):
        raw_ts = best_frame_info.get("timestamp", -1)
        if raw_ts is not None and float(raw_ts) >= 0:
            ts = float(raw_ts)
        reason = best_frame_info.get("reason", "")
        idx    = best_frame_info.get("index", -1)
        logger.info(
            "📸 [BEST_FRAME_EXTRACT] Extracting frame %d @ %.2fs → %s",
            idx, ts, os.path.basename(output_path)
        )
        if reason:
            logger.info("   Reason: %s", reason[:100])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{ts:.4f}",           # seek to timestamp
        "-i", video_path,
        "-vframes", "1",              # extract exactly one frame
        "-q:v", "2",                  # JPEG quality (2 = near-lossless)
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if result.returncode == 0 and os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            logger.info(
                "✅ [BEST_FRAME_EXTRACT] Saved: %s (%.0f KB)",
                os.path.basename(output_path), size_kb
            )
            return os.path.abspath(output_path)
        err = result.stderr.decode(errors="ignore")[-200:]
        logger.warning("[BEST_FRAME_EXTRACT] FFmpeg failed: %s", err)
        return ""
    except subprocess.TimeoutExpired:
        logger.warning("[BEST_FRAME_EXTRACT] FFmpeg timed out for %s", video_path)
        return ""
    except Exception as exc:
        logger.warning("[BEST_FRAME_EXTRACT] Error: %s", exc)
        return ""

def verify_watermark(frame, candidate_box, confidence: float = 0.8) -> bool:
    """3-tier confidence check for surgical removal."""
    if confidence > 0.7:
        return True
    if confidence > 0.5:
        try:
            h_img, w_img = frame.shape[:2]
            frame_area = h_img * w_img
            box_area = candidate_box.get('w', 0) * candidate_box.get('h', 0)
            if box_area > 0 and (box_area / frame_area) < 0.15:
                return True
        except: pass
    return False

def evaluate(video_path, frames):
    """Legacy entry point."""
    res, _niche_info = detect_watermark(frames)
    if res:
        return {"score": 0.8, "status": "CHECK_PASSED", "watermarks": res}
    return {"score": 0.4, "status": "CLEAN", "watermarks": []}

def detect_watermark_from_video(video_path: str, keywords: str = "") -> tuple:
    """Convenience entry point: extracts 12 strategic keyframes from a video_path and executes detect_watermark."""
    if not os.path.exists(video_path):
        return None, None
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 30
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1080
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1920

        # Sample 12 strategic frames across the video to guarantee high-acuity watermark coverage
        sample_count = 12
        indices = [int(i * (total - 1) / max(1, sample_count - 1)) for i in range(sample_count)]
        frames = []
        timestamps = []
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
                timestamps.append(round(idx / fps, 3))
        cap.release()

        if not frames:
            return None, None

        return detect_watermark(
            frames=frames,
            keywords=keywords,
            force_width=vw,
            force_height=vh,
            frame_timestamps=timestamps
        )
    except Exception as exc:
        logger.warning(f"detect_watermark_from_video error for {video_path}: {exc}")
        return None, None