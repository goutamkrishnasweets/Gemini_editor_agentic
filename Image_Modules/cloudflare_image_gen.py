"""
Image_Modules/cloudflare_image_gen.py
=====================================
Cloudflare Workers AI Free Image Generation Engine.
Accesses 10,000 free Neurons/day on Cloudflare Workers AI (~200 free FLUX Schnell images/day).

Supported Free Models:
  1. @cf/black-forest-labs/flux-1-schnell  (Primary: 12B FLUX Diffusion, 4 steps, SOTA photorealism & detail)
  2. @cf/bytedance/stable-diffusion-xl-lightning (Sub-second SDXL 4-step)
  3. @cf/stabilityai/stable-diffusion-xl-base-1.0 (Full SDXL 1.0 1024x1024)

Architectural Flow:
  1. Gemini Call 1 analyzes video proxy & picks hero timestamp + micro-detailed prompt.
  2. extract_single_hero_frame() slices exact reference frame at timestamp_sec via FFmpeg/OpenCV.
  3. CloudflareImageGenerator dispatches micro_detail_image_prompt to Workers AI.
  4. Saves ultra-sharp 1024x1024 or 9:16 hero_cover_generated.png for Thumbnails & Video Hook Injection.
"""

import os
import sys
import json
import logging
import base64
import subprocess
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, Tuple, Union

logger = logging.getLogger("cloudflare_image_gen")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_CLOUDFLARE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
FAST_CLOUDFLARE_MODEL = "@cf/bytedance/stable-diffusion-xl-lightning"
SDXL_CLOUDFLARE_MODEL = "@cf/stabilityai/stable-diffusion-xl-base-1.0"


def extract_single_hero_frame(
    video_path: str,
    timestamp_sec: float,
    output_path: str,
) -> Optional[str]:
    """
    Extracts a single uncompressed reference frame at exact timestamp_sec.
    Primary: Fast FFmpeg seek with high-quality JPEG output.
    Fallback: OpenCV VideoCapture precise frame seek.
    """
    if not video_path or not os.path.isfile(video_path):
        logger.warning(f"Video file not found for hero extraction: {video_path}")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ts = max(0.0, float(timestamp_sec or 0.0))

    # 1. Primary: FFmpeg sub-second precision seek
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ts:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            output_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        if res.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 1024:
            logger.info(f"📸 [HERO SLICE] Sliced reference frame @ {ts:.2f}s via FFmpeg -> {output_path}")
            return output_path
    except Exception as fe:
        logger.debug(f"FFmpeg hero slice fallback notice: {fe}")

    # 2. Fallback: OpenCV VideoCapture
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_idx = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                logger.info(f"📸 [HERO SLICE] Sliced reference frame @ {ts:.2f}s via OpenCV -> {output_path}")
                return output_path
    except Exception as cve:
        logger.warning(f"OpenCV hero frame slice failed: {cve}")

    return None


class CloudflareImageGenerator:
    """
    Zero-Cost Cloudflare Workers AI Image Generation Client.
    Connects to Free Tier (10,000 neurons/day) without third-party dependencies.
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        api_token: Optional[str] = None,
    ):
        self.account_id = (account_id or "").strip()
        self.api_token = (api_token or "").strip()
        if not self.account_id or not self.api_token:
            self._hydrate_credentials()

    def _hydrate_credentials(self) -> None:
        """Resolves credentials from env vars, bot_config.json, or telegram_users.json."""
        if not self.account_id:
            self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        if not self.api_token:
            self.api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

        # Check data/bot_config.json
        if not self.account_id or not self.api_token:
            cfg_path = os.path.join(_REPO_ROOT, "data", "bot_config.json")
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        if not self.account_id:
                            self.account_id = str(cfg.get("cloudflare_account_id", "")).strip()
                        if not self.api_token:
                            self.api_token = str(cfg.get("cloudflare_api_token", "")).strip()
                except Exception:
                    pass

        # Check data/telegram_users.json
        if not self.account_id or not self.api_token:
            try:
                from Telegram_Storage_Modules.telegram_user_manager import load_all_users
                users = load_all_users()
                for u in users.values():
                    if isinstance(u, dict):
                        if not self.account_id and u.get("cloudflare_account_id"):
                            self.account_id = str(u["cloudflare_account_id"]).strip()
                        if not self.api_token and u.get("cloudflare_api_token"):
                            self.api_token = str(u["cloudflare_api_token"]).strip()
                        if self.account_id and self.api_token:
                            break
            except Exception:
                pass

        if self.account_id and not os.getenv("CLOUDFLARE_ACCOUNT_ID"):
            os.environ["CLOUDFLARE_ACCOUNT_ID"] = self.account_id
        if self.api_token and not os.getenv("CLOUDFLARE_API_TOKEN"):
            os.environ["CLOUDFLARE_API_TOKEN"] = self.api_token

    def is_configured(self) -> bool:
        """Returns True if valid Cloudflare Account ID and API Token are resolved."""
        return bool(self.account_id and self.api_token and len(self.account_id) >= 10 and len(self.api_token) >= 10)

    def generate_image(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: Optional[str] = None,
        model: str = DEFAULT_CLOUDFLARE_MODEL,
        steps: int = 4,
        width: int = 1024,
        height: int = 1024,
    ) -> Optional[str]:
        """
        Dispatches micro-detailed prompt to Cloudflare Workers AI.
        Returns absolute path to generated image file on success, or None on failure.
        """
        if not self.is_configured():
            logger.info("🎨 [CLOUDFLARE AI] Credentials unconfigured. Skipping AI image generation.")
            return None

        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            logger.warning("Empty prompt provided for Cloudflare image generation.")
            return None

        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{model}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "User-Agent": "GeminiEditor-CloudflareAI/1.0",
        }

        # Payload construction
        payload: Dict[str, Any] = {
            "prompt": clean_prompt,
            "num_steps": steps,
        }
        if "flux" in model.lower():
            # FLUX Schnell takes prompt & num_steps (4 is optimal for free tier)
            payload["steps"] = steps
        else:
            # SDXL models accept negative prompt and resolution dimensions
            if negative_prompt:
                payload["negative_prompt"] = negative_prompt
            payload["width"] = width
            payload["height"] = height

        logger.info(f"🎨 [CLOUDFLARE AI] Requesting '{model}' (steps={steps})...")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=45) as resp:
                content_type = resp.headers.get("Content-Type", "").lower()
                body = resp.read()

                # Case 1: Binary Image Stream (standard for FLUX / SDXL on Cloudflare)
                if any(t in content_type for t in ["image/", "application/octet-stream"]):
                    with open(output_path, "wb") as f:
                        f.write(body)
                    logger.info(f"✨ [CLOUDFLARE AI SUCCESS] Wrote {len(body)} bytes -> {output_path}")
                    return output_path

                # Case 2: JSON Response with base64 payload
                try:
                    resp_json = json.loads(body.decode("utf-8"))
                    img_b64 = None
                    if isinstance(resp_json, dict):
                        res_obj = resp_json.get("result")
                        if isinstance(res_obj, dict):
                            img_b64 = res_obj.get("image")
                        elif isinstance(res_obj, str) and len(res_obj) > 100:
                            img_b64 = res_obj

                    if img_b64:
                        img_bytes = base64.b64decode(img_b64)
                        with open(output_path, "wb") as f:
                            f.write(img_bytes)
                        logger.info(f"✨ [CLOUDFLARE AI SUCCESS] Decoded base64 image ({len(img_bytes)} bytes) -> {output_path}")
                        return output_path
                    else:
                        errs = resp_json.get("errors") or resp_json
                        logger.warning(f"⚠️ Cloudflare response did not contain image data: {errs}")
                except Exception as _parse_err:
                    logger.debug(f"JSON decode notice: {_parse_err}")

        except urllib.error.HTTPError as http_err:
            try:
                err_detail = http_err.read().decode("utf-8", errors="ignore")
            except Exception:
                err_detail = str(http_err)
            logger.error(f"❌ Cloudflare Workers AI HTTP {http_err.code} Error: {err_detail}")
        except Exception as e:
            logger.error(f"❌ Cloudflare Workers AI call failed: {e}")

        return None


def generate_hero_cover_for_clip(
    video_path: str,
    clip_folder: str,
    hero_visual: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-End Execution for Step 03b:
      1. Slices exact reference frame at hero_visual['timestamp_sec'].
      2. If Cloudflare AI configured: regenerates high-res image from micro_detail_image_prompt.
      3. Returns structured metadata with paths to raw frame & generated cover.
    """
    hv = hero_visual or {}
    ts = hv.get("timestamp_sec")
    if ts is None or float(ts) < 0:
        ts = 1.0  # Safe default to opening hook

    raw_frame_path = os.path.join(clip_folder, "hero_frame_raw.jpg")
    generated_cover_path = os.path.join(clip_folder, "hero_cover_generated.png")

    # 1. Exact reference slice
    sliced_path = extract_single_hero_frame(
        video_path=video_path,
        timestamp_sec=float(ts),
        output_path=raw_frame_path,
    )

    result_meta: Dict[str, Any] = {
        "timestamp_sec": float(ts),
        "selection_reason": hv.get("selection_reason", ""),
        "micro_detail_image_prompt": hv.get("micro_detail_image_prompt", ""),
        "negative_prompt": hv.get("negative_prompt", ""),
        "hero_frame_raw": sliced_path or raw_frame_path,
        "hero_cover_generated": None,
        "ai_generated": False,
        "status": "raw_frame_extracted",
    }

    prompt = hv.get("micro_detail_image_prompt")
    if not prompt:
        return result_meta

    # 2. Cloudflare AI Regeneration
    client = CloudflareImageGenerator()
    if client.is_configured():
        neg = hv.get("negative_prompt") or (
            "blurry, low quality, smooth plastic skin, cartoon, 3d render, distorted hands, deformed limbs, artifacts"
        )
        ai_img = client.generate_image(
            prompt=prompt,
            output_path=generated_cover_path,
            negative_prompt=neg,
            model=DEFAULT_CLOUDFLARE_MODEL,
            steps=4,
        )
        if ai_img and os.path.isfile(ai_img):
            result_meta["hero_cover_generated"] = ai_img
            result_meta["ai_generated"] = True
            result_meta["status"] = "ai_image_regenerated"
            logger.info("💎 [HERO COVER READY] Hyper-detailed AI hero cover generated for clip!")

    return result_meta
