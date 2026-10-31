import os
import sys
import argparse
import logging
import json

# ─── PATH BOOTSTRAP ────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ─── ENV LOADING (watermark_env.env wins over global) ──────────────────────────
ENV_PATH_GLOBAL = os.path.join(PROJECT_ROOT, "Credentials", ".env")
if os.path.exists(ENV_PATH_GLOBAL):
    with open(ENV_PATH_GLOBAL, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

ENV_PATH_LOCAL = os.path.join(CURRENT_DIR, "watermark_env.env")
if os.path.exists(ENV_PATH_LOCAL):
    with open(ENV_PATH_LOCAL, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")
    print(f"📡 API CONNECTED: Using module-specific config from '{os.path.basename(ENV_PATH_LOCAL)}'")

# ─── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("watermark_main")

# ─── MODULE IMPORTS ────────────────────────────────────────────────────────────
try:
    from Watermark_and_Inpainting.watermark_auto import process_video_with_watermark
    from Watermark_and_Inpainting.hybrid_watermark import HybridWatermarkDetector
except ImportError:
    from watermark_auto import process_video_with_watermark
    from hybrid_watermark import HybridWatermarkDetector


# ═══════════════════════════════════════════════════════════════════════════════
# CORE PROCESSING FUNCTION (shared by CLI and Gradio)
# ═══════════════════════════════════════════════════════════════════════════════
def run_watermark_removal(
    input_path: str,
    output_path: str = None,
    keywords: str = "",
    retry_level: int = 0,
    predetected_watermarks: list = None,
    **kwargs
):
    """
    Full watermark detection + removal pipeline.
    Returns (output_path, status_log) or raises on fatal error.
    """
    import shutil
    log_lines = []

    def log(msg):
        logger.info(msg)
        log_lines.append(msg)

    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_no_watermark{ext}"
    output_path = os.path.abspath(output_path)

    log(f"🎬 Input:  {os.path.basename(input_path)}")
    log(f"📦 Output: {os.path.basename(output_path)}")

    # Step 1: AI Detection
    detection_verified_clean = False
    detection_failed = False

    if predetected_watermarks is not None:
        log(f"💎 Step 1: Reusing {len(predetected_watermarks)} pre-detected watermark bounding box(es) (duplicate detection skipped).")
        watermarks = predetected_watermarks
        if len(watermarks) == 0:
            detection_verified_clean = True
            detection_failed = False
    else:
        log("🔭 Step 1: Initiating AI Forensic Detection...")
        detector = HybridWatermarkDetector()
        detection_res_json = detector.process_video(input_path, keywords=keywords, retry_level=retry_level)
        try:
            detection_data = json.loads(detection_res_json)
        except Exception:
            detection_data = {}
        watermarks = detection_data.get("watermarks", [])
        log(f"✅ Detection finished. Found {len(watermarks)} watermarks.")

        status = detection_data.get("status", "")
        ctx = detection_data.get("context", {}) if isinstance(detection_data.get("context"), dict) else {}
        if status in ("ERROR", "UNVERIFIED_EMPTY") or ctx.get("detection_failed", False) or not ctx.get("is_clean", True):
            detection_verified_clean = False
            detection_failed = True
        else:
            detection_verified_clean = (len(watermarks) == 0)
            detection_failed = False

    if not watermarks:
        if detection_verified_clean:
            log("⏹️ No watermarks detected (verified clean). Copying source to destination...")
        else:
            log("⚠️ [ZERO-QUOTA / UNVERIFIED] Watermark detection could not verify clean status. Copying source as unverified fallback...")

        shutil.copy2(input_path, output_path)
        try:
            with open(f"{output_path}.coords.json", "w", encoding="utf-8") as _scf:
                json.dump({
                    "watermark_boxes": [],
                    "is_clean": detection_verified_clean,
                    "copied_from_raw": True,
                    "detection_failed": detection_failed,
                    "verified_by": "gemini_only" if detection_verified_clean else "none"
                }, _scf)
        except Exception:
            pass
        if detection_verified_clean:
            log("✅ PROCESS COMPLETE: Video was already clean.")
        else:
            log("⚠️ PROCESS COMPLETE: Video copied from raw (UNVERIFIED — not marked as clean).")
        return output_path, "\n".join(log_lines)

    # Probe raw input video dimensions for downstream aspect-ratio scaling & brand overlay alignment
    v_w, v_h = 1080, 1920
    try:
        import cv2
        cap = cv2.VideoCapture(input_path)
        if cap.isOpened():
            v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1080
            v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1920
            cap.release()
    except Exception as _ve:
        logger.debug("Could not probe video dimensions via OpenCV: %s", _ve)

    # Enrich watermarks with OpenCV inpaint mathematical vectors (effective padded ROI & inpaint math)
    enriched_watermarks = []
    for box in watermarks:
        b_copy = dict(box)
        b_copy["video_width"] = v_w
        b_copy["video_height"] = v_h

        # Extract coordinates from nested 'coordinates' dict (HybridWatermarkDetector format) or top-level
        coords = box.get("coordinates") if isinstance(box.get("coordinates"), dict) else box
        bx = int(coords.get("x") if coords.get("x") is not None else box.get("x", 0))
        by = int(coords.get("y") if coords.get("y") is not None else box.get("y", 0))
        bw = int(coords.get("w") if coords.get("w") is not None else box.get("w", 100))
        bh = int(coords.get("h") if coords.get("h") is not None else box.get("h", 50))

        # Flatten x, y, w, h at top-level of b_copy for all downstream consumers
        b_copy["x"] = bx
        b_copy["y"] = by
        b_copy["w"] = bw
        b_copy["h"] = bh

        pad_x = 12
        pad_y = 8
        eff_x = max(0, bx - pad_x)
        eff_y = max(0, by - pad_y)
        eff_w = bw + 2 * pad_x
        eff_h = bh + 2 * pad_y

        bg_texture = (
            str(coords.get("background_texture", ""))
            or str(b_copy.get("semantic_vectors", {}).get("background_texture", "complex"))
        ).lower()

        b_copy["opencv_vectors"] = {
            "inpaint_radius": 13,
            "mask_padding_x": pad_x,
            "mask_padding_y": pad_y,
            "effective_x": eff_x,
            "effective_y": eff_y,
            "effective_w": eff_w,
            "effective_h": eff_h,
            "inpaint_method": "Navier_Stokes_EdgeIntegrator",
            "background_texture": bg_texture
        }
        enriched_watermarks.append(b_copy)

    # Save detected watermark coordinates & OpenCV vectors to sidecar file beside output_path for FFmpeg brand overlay alignment
    try:
        coords_sidecar = output_path + ".coords.json"
        with open(coords_sidecar, "w", encoding="utf-8") as f:
            json.dump({
                "watermark_boxes": enriched_watermarks,
                "input_video": input_path,
                "video_width": v_w,
                "video_height": v_h
            }, f)
        log(f"💾 Saved {len(enriched_watermarks)} watermark box(es) with OpenCV math vectors ({v_w}x{v_h}) to sidecar: {os.path.basename(coords_sidecar)}")
    except Exception as _se:
        logger.warning(f"⚠️ Failed to save coords sidecar: {_se}")



    # Step 2: Adaptive Refinement
    log("🎨 Step 2: Running Adaptive Refinement Orchestration...")
    result = process_video_with_watermark(
        input_path,
        output_path,
        pre_detected_watermarks=watermarks,
        retry_level=retry_level
    )

    if result.get("success"):
        log(f"✅ PROCESS COMPLETE! Saved to: {output_path}")
    else:
        reason = result.get('context', {}).get('reason', 'Unknown error') if result.get('context') else 'Unknown error'
        log(f"❌ PROCESS FAILED: {reason}")

    return output_path, "\n".join(log_lines)


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════
def launch_gradio():
    import gradio as gr
    import tempfile
    import shutil

    ENABLE_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    PORT = int(os.getenv("GRADIO_PORT", 7860))
    # ── Auth: set GRADIO_AUTH_PASS in Credentials/.env to password-protect the UI
    GRADIO_USER = os.getenv("GRADIO_AUTH_USER", "admin")
    GRADIO_PASS = os.getenv("GRADIO_AUTH_PASS")  # None = no auth (local-only)
    _auth = (GRADIO_USER, GRADIO_PASS) if GRADIO_PASS else None
    if GRADIO_PASS:
        logger.info("🔒 [GRADIO] Password protection ENABLED.")
    elif ENABLE_SHARE:
        logger.warning("⚠️ [GRADIO] share=True but no GRADIO_AUTH_PASS set — UI is publicly accessible!")

    def process_fn(video_file, keywords, retry_level, progress=gr.Progress()):
        """Gradio handler: receives uploaded video path, returns cleaned video."""
        if video_file is None:
            return None, "❌ No file uploaded."

        # ── Pre-Flight Security Guard ─────────────────────────────
        try:
            from Compiler_Modules.video_pipeline import validate_video_input
            validate_video_input(video_file)
        except ImportError:
            pass  # Graceful: guard unavailable, proceed anyway
        except ValueError as _e:
            return None, f"🚫 Upload rejected: {_e}"

        progress(0.05, desc="📡 Connecting to AI...")
        try:
            # Gradio gives us the temp path directly
            input_path = video_file
            base = os.path.splitext(input_path)[0]
            output_path = f"{base}_no_watermark.mp4"

            progress(0.2, desc="🔭 Running AI Detection...")
            out_path, status = run_watermark_removal(
                input_path=input_path,
                output_path=output_path,
                keywords=keywords or "",
                retry_level=int(retry_level)
            )
            progress(1.0, desc="✅ Done!")
            return out_path, status

        except Exception as e:
            logger.error(f"Gradio process error: {e}")
            return None, f"💥 Fatal Error: {e}"

    # ── BUILD UI ──────────────────────────────────────────────────────────────
    with gr.Blocks(
        title="AMTCE Watermark Remover",
        theme=gr.themes.Glass(
            primary_hue="violet",
            secondary_hue="indigo",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container { max-width: 960px !important; margin: auto; }
        #title { text-align: center; margin-bottom: 8px; }
        #subtitle { text-align: center; color: #a78bfa; margin-bottom: 24px; font-size: 14px; }
        #status_box { font-family: monospace; font-size: 12px; }
        .upload-box { border: 2px dashed #6d28d9 !important; border-radius: 12px !important; }
        """
    ) as demo:

        gr.HTML("<h1 id='title'>🎬 AMTCE Watermark Remover</h1>")
        gr.HTML("<p id='subtitle'>Gemini AI · Pixel-Perfect Inpainting · Face Firewall Protected</p>")

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(
                    label="📤 Upload Video",
                    elem_classes=["upload-box"],
                    height=300,
                )
                keywords_in = gr.Textbox(
                    label="🔑 Detection Hints (optional)",
                    placeholder="e.g. BOLLYWOOD DAZZLE, logo in bottom right",
                    lines=1,
                )
                retry_in = gr.Slider(
                    label="🔁 Retry Level (0 = normal, 2 = nuclear scan)",
                    minimum=0, maximum=2, step=1, value=0
                )
                run_btn = gr.Button("🚀 Remove Watermark", variant="primary", size="lg")

            with gr.Column(scale=1):
                video_out = gr.Video(
                    label="📥 Cleaned Output",
                    height=300,
                )
                status_out = gr.Textbox(
                    label="📋 Processing Log",
                    lines=14,
                    max_lines=20,
                    elem_id="status_box",
                    interactive=False,
                )

        run_btn.click(
            fn=process_fn,
            inputs=[video_in, keywords_in, retry_in],
            outputs=[video_out, status_out],
        )

        gr.Markdown("""
        ---
        **Tips:**
        - Add detection hints if the AI misses a specific logo/text (e.g. `"PAPPS CENTRAL"`)
        - Use **Retry Level 2** for stubborn watermarks — it activates Nuclear Sensitivity mode
        - Face Firewall is always active: faces are never touched
        """)

    print("\n" + "="*55)
    print("🚀 AMTCE Watermark Remover — Gradio UI")
    print(f"   Opening at: http://localhost:{PORT}")
    print("   Press Ctrl+C to stop")
    print("="*55 + "\n")

    demo.launch(
        server_port=PORT,
        share=ENABLE_SHARE,
        auth=_auth,          # None = open (local), tuple = password-protected
        show_error=False,    # Never expose Python tracebacks in the browser UI
        inbrowser=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="AMTCE Individual Watermark Removal Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch Gradio UI (no args needed)
  python watermark_main.py

  # CLI with positional input
  python watermark_main.py "D:\\path\\to\\video.mp4"

  # CLI with --input flag (handles paths with special characters)
  python watermark_main.py --input "D:\\path\\to\\video.mp4"

  # With hints and output path
  python watermark_main.py --input "video.mp4" -k "LOGO TEXT" -o "clean.mp4"
        """
    )
    # Support BOTH positional and --input flag
    parser.add_argument("input", nargs="?", help="Path to input video file (positional)")
    parser.add_argument("--input", dest="input_flag", metavar="FILE", help="Path to input video file (named flag)")
    parser.add_argument("-o", "--output", help="Path to output video file (default: input_no_watermark.mp4)")
    parser.add_argument("-k", "--keywords", help="Optional detection hints", default="")
    parser.add_argument("--retry", type=int, default=0, help="Retry level (0-2)")
    parser.add_argument("--ui", action="store_true", help="Force launch Gradio UI")

    args = parser.parse_args()

    # Resolve input: --input flag wins over positional
    input_path = args.input_flag or args.input

    # If no input provided (or --ui flag), launch Gradio
    if not input_path or args.ui:
        launch_gradio()
        return

    # CLI mode
    print("\n" + "="*50)
    print("🚀 AMTCE INDIVIDUAL WATERMARK TOOL (CLI)")
    print("="*50 + "\n")

    try:
        out_path, status = run_watermark_removal(
            input_path=input_path,
            output_path=args.output,
            keywords=args.keywords,
            retry_level=args.retry,
        )
        print(f"\n📂 Output saved to:\n   {out_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")


if __name__ == "__main__":
    main()
