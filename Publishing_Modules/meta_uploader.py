import os
import time
import logging
import asyncio
import json
import httpx
from typing import Dict, Optional, Tuple, Any

try:
    from Gemini_Modules.platform_seo_generator import strip_system_and_tracking_tokens
except ImportError:
    import re
    def strip_system_and_tracking_tokens(text):
        if not isinstance(text, str): return text
        text = re.sub(r'#?AQ[a-zA-Z0-9_-]{10,}', '', text)
        text = re.sub(r'#(?=[a-zA-Z0-9_-]{26,})(?=[a-zA-Z0-9_-]*\d)(?=[a-zA-Z0-9_-]*[a-zA-Z])[a-zA-Z0-9_-]+', '', text)
        text = re.sub(r'(?i)#?(?:creator_?unknown|niche_?unknown|unknown)\b', '', text)
        text = re.sub(r'#(?![a-zA-Z0-9_])', '', text)
        lines = [re.sub(r'[ \t]{2,}', ' ', l).strip() for l in text.split('\n')]
        return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()

# Initialize Logger
logger = logging.getLogger("meta_uploader")
logger.setLevel(logging.INFO)

# ── Dynamic API Versioning ──────────────────────────────────────────────────
DEFAULT_META_GRAPH_VERSION = "v21.0"

def get_meta_graph_version() -> str:
    """Returns the current Meta Graph API version from environment or default."""
    ver = os.getenv("META_GRAPH_VERSION", DEFAULT_META_GRAPH_VERSION).strip()
    if not ver:
        ver = DEFAULT_META_GRAPH_VERSION
    if not ver.startswith("v") and ver and ver[0].isdigit():
        ver = f"v{ver}"
    return ver

def set_meta_graph_version(version: str) -> str:
    """Sets the current Meta Graph API version dynamically in-memory."""
    ver = version.strip()
    if not ver.startswith("v") and ver and ver[0].isdigit():
        ver = f"v{ver}"
    os.environ["META_GRAPH_VERSION"] = ver
    logger.info("⚡ [META UPLOADER] Dynamic Meta Graph API version updated to: %s", ver)
    return ver

def get_graph_api_url() -> str:
    """Returns the dynamic base URL for Meta Graph API calls."""
    return f"https://graph.facebook.com/{get_meta_graph_version()}"

def get_rupload_video_url(video_id: str) -> str:
    """Returns the dynamic rupload endpoint for Facebook video uploads."""
    return f"https://rupload.facebook.com/video-upload/{get_meta_graph_version()}/{video_id}"

class _DynamicGraphApiUrl(str):
    """Dynamic string proxy that always resolves the current Meta Graph API base URL."""
    def __new__(cls):
        return super().__new__(cls, f"https://graph.facebook.com/{get_meta_graph_version()}")
    def __str__(self):
        return get_graph_api_url()
    def __repr__(self):
        return repr(get_graph_api_url())
    def __format__(self, format_spec):
        return get_graph_api_url().__format__(format_spec)

# Constants & Dynamic Endpoints
GRAPH_API_URL = _DynamicGraphApiUrl()
MAX_RETRIES = 2
RETRY_DELAY = 10

# ── Niche-toggle helpers ─────────────────────────────────────────────────────
_NICHE_ENV_MAP: dict = {
    "Adult Content & NSFW Content":         "NICHE_ADULT_CONTENT",
    "AI Tech & Futuristic Content":          "NICHE_AI_TECH",
    "Comedy & Relatable Meme":               "NICHE_COMEDY_MEME",
    "Educational & Quick Learning":          "NICHE_EDUCATIONAL",
    "face_refinement":                       "NICHE_FACE_REFINEMENT",
    "Fashion & Style":                       "NICHE_FASHION_STYLE",
    "Fitness & Body Transformation":         "NICHE_FITNESS",
    "Food & Cooking":                        "NICHE_FOOD_COOKING",
    "General_Fallback":                      "NICHE_GENERAL_FALLBACK",
    "Money & Online Income":                 "NICHE_MONEY_INCOME",
    "Motivation & Self-Improvement":         "NICHE_MOTIVATION",
    "Relationships & Dating Psychology":     "NICHE_RELATIONSHIPS",
    "Travel & Lifestyle":                    "NICHE_TRAVEL",
}

def _niche_env_key(niche: str) -> str:
    """Return the root-env prefix for a given niche folder name."""
    return _NICHE_ENV_MAP.get(niche.strip(), f"NICHE_{niche.upper().replace(' ', '_').replace('&', '').replace('__', '_')}")

def _is_niche_enabled(niche: str) -> bool:
    """
    Returns True when the niche has NICHE_<KEY>_ENABLED=yes in the root .env.
    Defaults to True for General_Fallback so the safety-net always works.
    Also returns True if a specific meta_config.json exists for this folder (e.g., actress folders).
    """
    if os.path.exists(os.path.join("Credentials", "social_media", niche, "meta_config.json")):
        return True
    if os.path.exists(os.path.join("Credentials", "social_media", niche.capitalize(), "meta_config.json")):
        return True
    if niche.strip() == "General_Fallback":
        key = "NICHE_GENERAL_FALLBACK_ENABLED"
        return os.getenv(key, "yes").strip().lower() in ("yes", "true", "on", "1")
    key = f"{_niche_env_key(niche)}_ENABLED"
    is_enabled = os.getenv(key, "no").strip().lower() in ("yes", "true", "on", "1")
    if is_enabled:
        return True
        
    # If the specific niche isn't explicitly enabled, allow it to fallback to General_Fallback
    fallback_key = "NICHE_GENERAL_FALLBACK_ENABLED"
    return os.getenv(fallback_key, "yes").strip().lower() in ("yes", "true", "on", "1")

def _niche_meta_enabled(niche: str) -> bool:
    """
    Returns True when 'meta' is listed in the NICHE_<KEY>_PLATFORMS variable
    for this niche in the root .env, OR if the specific meta_config.json exists.
    Prevents Instagram/Facebook upload attempts for YouTube-only niches.
    """
    if os.path.exists(os.path.join("Credentials", "social_media", niche, "meta_config.json")):
        return True
    if os.path.exists(os.path.join("Credentials", "social_media", niche.capitalize(), "meta_config.json")):
        return True
    key = f"{_niche_env_key(niche)}_PLATFORMS"
    platforms = os.getenv(key, "youtube").lower()
    if "meta" in platforms:
        return True
        
    # If the specific niche doesn't include meta, fallback to General_Fallback
    fallback_key = "NICHE_GENERAL_FALLBACK_PLATFORMS"
    fallback_platforms = os.getenv(fallback_key, "youtube,meta").lower()
    return "meta" in fallback_platforms

# ─────────────────────────────────────────────────────────────────────────────
class AsyncMetaUploader:

    @staticmethod
    def _resolve_meta_config(niche: str) -> Dict:
        """
        Resolves Meta credentials using a 3-tier priority chain:
          1. Credentials/social_media/{niche}/meta_config.json  (niche-specific)
          2. Credentials/social_media/General_Fallback/meta_config.json  (shared fallback)
          3. Root .env environment variables  (final fallback)

        Returns a dict with keys: IG_BUSINESS_ID, IG_BUSINESS_TOKEN,
        META_PAGE_ID, META_PAGE_TOKEN.  Missing keys are silently absent
        (callers already guard against empty strings).

        Gate: If the niche is disabled in root .env (NICHE_*_ENABLED != yes)
        or the niche's platform list does not include 'meta', an empty dict
        is returned immediately — no file I/O, no warnings.
        """
        REQUIRED_KEYS = ["IG_BUSINESS_ID", "IG_BUSINESS_TOKEN", "META_PAGE_ID", "META_PAGE_TOKEN"]

        # ── Niche-enable gate ─────────────────────────────────────────────────
        if not _is_niche_enabled(niche):
            logger.info(f"🔕 [META CONFIG] Niche '{niche}' is DISABLED in root .env — skipping credential load.")
            return {k: "" for k in REQUIRED_KEYS}

        if not _niche_meta_enabled(niche):
            logger.info(f"🔕 [META CONFIG] Niche '{niche}' platform list does not include 'meta' — skipping Meta credential load.")
            return {k: "" for k in REQUIRED_KEYS}

        # ── Tier 1 & 2: JSON config files ────────────────────────────────────
        search_dirs = [
            os.path.join("Credentials", "social_media", niche),
            os.path.join("Credentials", "social_media", niche.capitalize()),
            os.path.join("Credentials", "social_media", "General_Fallback"),
        ]

        for config_dir in search_dirs:
            config_path = os.path.join(config_dir, "meta_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                    
                    # --- DEMO DETECTION ---
                    # Check if the file contains placeholder strings like "your_id" or "DEMO_"
                    is_placeholder = any(
                        "DEMO_" in str(v) or "your_" in str(v) 
                        for v in data.values()
                    )
                    
                    if is_placeholder:
                        logger.warning(
                            f"⚠️ [META CONFIG] {config_path} contains placeholder values ('DEMO' or 'your_'). "
                            "Skipping to root .env fallback."
                        )
                        continue

                    # Validate: must contain at least one recognisable key
                    if any(k in data for k in REQUIRED_KEYS):
                        logger.info(
                            f"🗂️ [META CONFIG] Loaded niche credentials from: {config_path}"
                        )
                        return {k: str(data.get(k, "")).strip() for k in REQUIRED_KEYS}
                    else:
                        logger.warning(
                            f"⚠️ [META CONFIG] {config_path} exists but contains no recognised keys. Skipping."
                        )
                except Exception as e:
                    logger.warning(
                        f"⚠️ [META CONFIG] Failed to read {config_path}: {e}. Trying next tier."
                    )

        # ── Tier 3: Root .env (original behaviour) ───────────────────────────
        logger.info(
            f"🗂️ [META CONFIG] No meta_config.json found for niche='{niche}'. "
            "Falling back to root .env credentials."
        )
        return {k: os.getenv(k, "").strip() for k in REQUIRED_KEYS}

    @staticmethod
    async def upload_to_meta(
        video_path: str,
        caption: str,
        upload_type: str = "Reels",
        skip_facebook: bool = False,
        facebook_caption: str = None,
        thumbnail_path: str = None,
        niche: str = "General_Fallback",
        meta_config_override: Optional[Dict] = None,
    ) -> Dict:
        """
        Orchestrates uploads to enabled Meta platforms.

        The ``niche`` parameter drives credential resolution via
        _resolve_meta_config(), enabling per-niche Instagram/Facebook accounts.

        ``meta_config_override``: When provided, skips _resolve_meta_config() entirely
        and uses the supplied credentials directly. Used by the multi-tenant publisher
        to inject per-user credentials without touching root .env variables.
        """
        results = {
            "instagram": {"status": "skipped"},
            "facebook": {"status": "skipped"}
        }
        
        if not os.getenv("ENABLE_META_UPLOAD", "yes").lower() in ["yes", "true", "on"]:
            logger.info("Meta Upload Disabled in .env")
            return results

        # Resolve credentials once; pass the config dict to each sub-uploader.
        # If a per-user override is provided, use it directly without touching root .env.
        if meta_config_override is not None:
            meta_config = meta_config_override
            logger.info("[META AUTH] Using caller-supplied meta_config_override (user-isolated credentials).")
        else:
            meta_config = AsyncMetaUploader._resolve_meta_config(niche)

        caption_stripped = caption # logic if needed
        
        # 1. Instagram
        if os.getenv("SEND_TO_INSTAGRAM", "on").lower() in ["yes", "true", "on"]:
            vid_res = await AsyncMetaUploader._upload_to_instagram(video_path, caption, meta_config)
            results["instagram"]["video"] = vid_res
            results["instagram"]["status"] = vid_res.get("status", "skipped")
            if vid_res.get("id"):
                results["instagram"]["id"] = vid_res["id"]
            if vid_res.get("link"):
                results["instagram"]["link"] = vid_res["link"]
            
            # --- New: Upload Thumbnail as a separate Image Post ---
            if thumbnail_path and os.path.exists(thumbnail_path):
                logger.info("Valid thumbnail found. Attempting to post as Instagram Image...")
                photo_res = await AsyncMetaUploader._upload_photo_to_instagram(thumbnail_path, caption, meta_config)
                results["instagram"]["photo"] = photo_res
                
                # If video succeeded but photo failed, keep status as success but indicate photo failed
                if results["instagram"]["status"] == "success" and photo_res.get("status") != "success":
                    results["instagram"]["status"] = "success (photo failed)"
                elif results["instagram"]["status"] != "success" and photo_res.get("status") == "success":
                    results["instagram"]["status"] = "success (photo only)"
                    results["instagram"]["link"] = photo_res.get("link", "")
        else:
             logger.info("SEND_TO_INSTAGRAM is OFF.")
             results["instagram"] = {"status": "disabled"}
        
        # 2. Facebook (Independent)
        if not skip_facebook and os.getenv("SEND_TO_FACEBOOK", "on").lower() in ["yes", "true", "on"]:
            # Use specific FB caption if provided, else fall back to main caption
            final_fb_caption = facebook_caption if facebook_caption else caption
            results["facebook"] = await AsyncMetaUploader._upload_to_facebook(video_path, final_fb_caption, meta_config)
        else:
             if skip_facebook:
                  logger.info("Facebook Skipped (Restricted to Compilation Mode).")
             else:
                  logger.info("SEND_TO_FACEBOOK is OFF.")
             results["facebook"] = {"status": "disabled/skipped"}
        
        return results

    @staticmethod
    async def _upload_to_instagram(video_path: str, caption: str, meta_config: Dict) -> str:
        ig_id = meta_config.get("IG_BUSINESS_ID", "").strip()
        ig_token = meta_config.get("IG_BUSINESS_TOKEN", "").strip()
        upload_type_env = os.getenv("META_UPLOAD_TYPE", "Reels").strip().upper()
        
        if not ig_id or not ig_token:
            return {"status": "skipped_no_creds"}
            
        logger.info(f"📸 Starting Instagram Upload ({upload_type_env})...")
        
        url = f"{GRAPH_API_URL}/{ig_id}/media"
        
        caption = AsyncMetaUploader._clean_caption(caption)

        try:
            # Step 1: Init (Resumable)
            # NOTE: Use data= (form body) not params= (URL query) to prevent
            # caption with emojis/special chars from being malformed in the URL.
            init_data = {
                "upload_type": "resumable",
                "media_type": "REELS" if upload_type_env == "REELS" else "VIDEO",
                "caption": caption,
                "access_token": ig_token
            }

            # ── Trail Reels: inject trial_params when flagged ──────────────────
            # trial_params is applied only when the caller explicitly passes
            # meta_config["trail_reels"] = True (set by TrailReelsManager).
            # Graduation strategy is always MANUAL — user graduates on their phone.
            _is_trial = bool(meta_config.get("trail_reels", False)) if isinstance(meta_config, dict) else False
            if _is_trial:
                try:
                    from Publishing_Modules.instagram_features.trail_reels import TrailReelsManager
                    _tr_mgr = TrailReelsManager()
                    init_data = _tr_mgr.apply_trial_params(init_data, is_trial=True)
                except Exception as _tr_err:
                    logger.warning("[TrailReels] Could not inject trial_params: %s", _tr_err)
            # ──────────────────────────────────────────────────────────────────

            req_init = await AsyncMetaUploader._retry_request("POST", url, data=init_data)
            if "uri" not in req_init:
                # ── Auto-fallback: If Meta rejected init due to trial_params ineligibility ──
                if _is_trial and "trial" in json.dumps(req_init).lower():
                    logger.warning(
                        "⚠️ [TRIAL REELS] Meta rejected trial_params (account ineligible). "
                        "Falling back to standard Reel upload automatically."
                    )
                    try:
                        from Publishing_Modules.instagram_features.trail_reels import TrailReelsManager
                        init_data = TrailReelsManager().handle_trial_ineligible_fallback(init_data)
                    except Exception as _fb_err:
                        logger.warning("[TrailReels] Fallback handler error: %s", _fb_err)
                        init_data.pop("trial_params", None)
                    _is_trial = False
                    req_init = await AsyncMetaUploader._retry_request("POST", url, data=init_data)

                if "uri" not in req_init:
                    logger.error(f"IG Init Failed (400/error response): {req_init}")
                    return {"status": "failed_init", "error": str(req_init)}

            upload_url = req_init["uri"]
            container_id_from_init = req_init.get("id")
            logger.info(f"📤 [IG_UPLOAD] Got rupload URI. Container (init)={container_id_from_init}")

            # Step 2: Upload Binary
            # IMPORTANT: Do NOT set Content-Length manually — httpx sets it
            # automatically when content=bytes. A duplicate Content-Length
            # header causes rupload to return 400.
            file_size = os.path.getsize(video_path)
            with open(video_path, "rb") as f:
                video_bytes = f.read()

            upload_headers = {
                "Authorization": f"OAuth {ig_token}",
                "offset": "0",
                "file_offset": "0",           # required by some rupload versions
                "X-Entity-Length": str(file_size),
                "X-Entity-Type": "video/mp4",
                "Content-Type": "video/mp4",
            }
            logger.info(f"📤 [IG_UPLOAD] Uploading {file_size} bytes to rupload...")
            req_upload = await AsyncMetaUploader._retry_request(
                "POST", upload_url, content=video_bytes, headers=upload_headers
            )
            logger.info(f"📤 [IG_UPLOAD] rupload response: {req_upload}")

            # Extract Container ID
            container_id = None
            if isinstance(req_upload, dict) and "id" in req_upload:
                container_id = req_upload["id"]
            elif container_id_from_init:
                container_id = container_id_from_init

            if not container_id:
                logger.error(f"IG Upload: No Container ID found. Upload response: {req_upload}")
                return {"status": "failed_upload", "error": f"No Container ID. rupload said: {req_upload}"}

            logger.info(f"📤 [IG_UPLOAD] Container ID={container_id}. Waiting for IG processing...")

            # Wait for processing
            is_ready = await AsyncMetaUploader._wait_for_media_status(container_id, ig_token)
            if not is_ready:
                logger.error("IG Container processing failed or timed out.")
                return {"status": "failed_processing", "error": "Container status ERROR or timed out"}

            # Step 4: Publish
            pub_url = f"{GRAPH_API_URL}/{ig_id}/media_publish"
            pub_params = {
                "creation_id": container_id,
                "access_token": ig_token
            }

            pub_res = await AsyncMetaUploader._retry_request("POST", pub_url, params=pub_params)
            if "id" in pub_res:
                media_id = pub_res["id"]
                logger.info(f"✅ Instagram Upload Success: {media_id}")

                # Fetch Permalink
                link = ""
                try:
                    perm_url = f"{GRAPH_API_URL}/{media_id}"
                    perm_res = await AsyncMetaUploader._retry_request("GET", perm_url, params={"fields": "permalink,shortcode", "access_token": ig_token})
                    link = perm_res.get("permalink") or perm_res.get("shortcode") or ""
                except Exception as e:
                    logger.warning(f"Failed to fetch IG permalink: {e}")

                return {"status": "success", "id": media_id, "link": link}
            else:
                logger.error(f"IG Publish Failed: {pub_res}")
                return {"status": "failed_publish", "error": str(pub_res)}

        except Exception as e:
            logger.error(f"IG Upload Exception: {e}")
            return {"status": "failed", "error": str(e)}


    @staticmethod
    async def _host_temp_image(file_path: str) -> str:
        """
        Uploads the local image to a temporary public host so the Instagram
        Graph API can fetch it via URL.

        Tries multiple providers in sequence and returns the first working URL:
          1. imgBB  (reliable, no-sign-up key via env IMG_BB_API_KEY or anonymous)
          2. freeimage.host  (no API key required)
          3. tmpfiles.org  (fallback – has had 502 instability)
        """
        import base64

        logger.info("🌐 Temporarily hosting thumbnail to obtain public URL for Instagram API...")

        # ── Read file once ──────────────────────────────────────────────────
        with open(file_path, "rb") as fh:
            raw_bytes = fh.read()
        b64_image = base64.b64encode(raw_bytes).decode("utf-8")
        filename  = os.path.basename(file_path)

        async with httpx.AsyncClient(timeout=45.0) as client:

            # ── Provider 1: imgBB ───────────────────────────────────────────
            try:
                imgbb_key = os.getenv("IMG_BB_API_KEY", "")
                imgbb_url = "https://api.imgbb.com/1/upload"
                payload   = {"image": b64_image}
                if imgbb_key:
                    payload["key"] = imgbb_key
                resp = await client.post(imgbb_url, data=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    url  = data.get("data", {}).get("url", "")
                    if url:
                        logger.info(f"✅ [imgBB] Temporary Public URL obtained: {url}")
                        return url
                    logger.warning(f"⚠️ [imgBB] Unexpected response: {data}")
                else:
                    logger.warning(f"⚠️ [imgBB] Status {resp.status_code} — trying next provider.")
            except Exception as e:
                logger.warning(f"⚠️ [imgBB] Exception: {repr(e)} — trying next provider.")

            # ── Provider 2: freeimage.host ──────────────────────────────────
            try:
                fi_url  = "https://freeimage.host/api/1/upload"
                fi_key  = "6d207e02198a847aa98d0a2a901485a5"  # public demo key
                payload = {"key": fi_key, "source": b64_image, "format": "json"}
                resp    = await client.post(fi_url, data=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    url  = (data.get("image", {}) or {}).get("url", "")
                    if url:
                        logger.info(f"✅ [freeimage.host] Temporary Public URL obtained: {url}")
                        return url
                    logger.warning(f"⚠️ [freeimage.host] Unexpected response: {data}")
                else:
                    logger.warning(f"⚠️ [freeimage.host] Status {resp.status_code} — trying next provider.")
            except Exception as e:
                logger.warning(f"⚠️ [freeimage.host] Exception: {repr(e)} — trying next provider.")

            # ── Provider 3: tmpfiles.org (original, kept as last resort) ────
            try:
                files = {"file": (filename, raw_bytes, "image/jpeg")}
                resp  = await client.post("https://tmpfiles.org/api/v1/upload", files=files)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "success" and "url" in data.get("data", {}):
                        url = data["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")
                        logger.info(f"✅ [tmpfiles.org] Temporary Public URL obtained: {url}")
                        return url
                    logger.warning(f"⚠️ [tmpfiles.org] Unexpected response: {data}")
                else:
                    logger.error(f"❌ [tmpfiles.org] Status {resp.status_code} — all providers exhausted.")
            except Exception as e:
                logger.error(f"❌ [tmpfiles.org] Exception: {repr(e)} — all providers exhausted.")

        logger.error("❌ Failed to obtain a public image URL from any hosting provider.")
        return ""

    @staticmethod
    def _prepare_ig_photo(photo_path: str, caption: str) -> str:
        """
        Smart person-aware Instagram ratio formatter (4:5 = 1080x1350).

        Strategy: Detect the person's full body bounding box (head→feet),
        then crop ONLY background pixels to achieve the required 4:5 ratio.
        The subject (head, body, hands) is NEVER cut.

        If the person is taller than 4:5 allows (extreme close-ups):
          → Keep the head, trim from the bottom (feet > head priority).
        """
        try:
            from PIL import Image
            import re
            import glob
            import os

            TARGET_W = 1080
            TARGET_H = 1350  # 4:5 portrait (Instagram max portrait ratio)

            img = Image.open(photo_path)
            orig_w, orig_h = img.size

            # ── Step 1: Detect person bounding box via cv2 ─────────────────
            person_top    = 0
            person_bottom = orig_h
            person_left   = 0
            person_right  = orig_w

            try:
                import cv2
                import numpy as np

                frame = cv2.imread(photo_path)
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # -- Face: gives us the head-top anchor --
                    if hasattr(cv2, "CascadeClassifier"):
                        cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml" if hasattr(cv2, "data") else ""
                        if cascade_path and os.path.exists(cascade_path):
                            face_cascade = cv2.CascadeClassifier(cascade_path)
                            if not face_cascade.empty():
                                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
                                if len(faces) > 0:
                                    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                                    # head top = face top with a small headroom buffer
                                    person_top = max(0, fy - int(fh * 0.25))
                                    logger.info(f"📏 [THUMB_PREP] Face detected: top_y={person_top}")

                    # -- HOG body: gives us the feet-bottom anchor --
                    if hasattr(cv2, "HOGDescriptor"):
                        hog = cv2.HOGDescriptor()
                        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                        scale = min(1.0, 640 / max(orig_w, 1))
                        small = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))
                        rects, _ = hog.detectMultiScale(small, winStride=(8, 8), padding=(4, 4), scale=1.05)

                        if len(rects) > 0:
                            # Pick the most central detection
                            cx = (orig_w * scale) / 2
                            cy = (orig_h * scale) / 2

                            def _hog_score(r):
                                rx, ry, rw, rh = r
                                dist = ((rx + rw / 2 - cx) ** 2 + (ry + rh / 2 - cy) ** 2) ** 0.5
                                return -(dist / max(cx, 1)) + (rw * rh) / (orig_w * orig_h * scale * scale)

                            bx, by, bw, bh = max(rects, key=_hog_score)
                            body_bottom = int((by + bh) / scale)
                            person_bottom = min(orig_h, body_bottom + int(bh / scale * 0.05))
                            # Also refine left/right bounds from body box
                            person_left  = max(0, int(bx / scale))
                            person_right = min(orig_w, int((bx + bw) / scale))
                            logger.info(f"📏 [THUMB_PREP] HOG body: bottom_y={person_bottom}")

            except Exception as _det_e:
                logger.warning(f"⚠️ [THUMB_PREP] Person detection failed (falling back): {_det_e}")

            person_height = person_bottom - person_top
            person_width  = person_right - person_left

            # ── Step 2: Scale so the image fits TARGET_W at full quality ─────
            # Always scale width to 1080 first, then see if height needs trimming.
            scale_to_target_w = TARGET_W / orig_w
            scaled_w = TARGET_W
            scaled_h = int(orig_h * scale_to_target_w)

            img_scaled = img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

            # Scale person bounds to match
            s = scale_to_target_w
            pt = int(person_top    * s)
            pb = int(person_bottom * s)
            # pl = int(person_left   * s)  # not used for vertical crop
            subject_h = pb - pt  # person pixel height in scaled image

            # ── Step 3: Choose crop window (person-safe) ─────────────────────
            if scaled_h <= TARGET_H:
                # Image is already shorter (or equal) to 1350 — no cropping needed, just resize
                final_img = img_scaled.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS) \
                            if scaled_h != TARGET_H else img_scaled
                logger.info("📏 [THUMB_PREP] Image fits in 4:5 — no crop needed.")

            elif subject_h <= TARGET_H:
                # Person FITS in 1350px height — trim background from top and bottom
                # Center the crop window ON THE PERSON, then pull away any excess background.

                # Ideal: center person vertically in the crop window
                p_center   = (pt + pb) // 2
                crop_top   = p_center - TARGET_H // 2
                crop_bottom = crop_top + TARGET_H

                # Clamp: don't go above the image top
                if crop_top < 0:
                    crop_top    = 0
                    crop_bottom = TARGET_H

                # Clamp: don't go below the image bottom
                if crop_bottom > scaled_h:
                    crop_bottom = scaled_h
                    crop_top    = max(0, scaled_h - TARGET_H)

                # Safety: guarantee person top (head) is inside the crop window
                if pt < crop_top:
                    crop_top    = pt
                    crop_bottom = min(scaled_h, crop_top + TARGET_H)

                # Safety: guarantee person bottom (feet) is inside the crop window
                if pb > crop_bottom:
                    crop_bottom = pb
                    crop_top    = max(0, crop_bottom - TARGET_H)

                final_img = img_scaled.crop((0, crop_top, TARGET_W, crop_bottom))
                logger.info(
                    f"📏 [THUMB_PREP] Person-safe background crop: "
                    f"rows {crop_top}→{crop_bottom} (person at {pt}→{pb})"
                )

            else:
                # Person is TALLER than 1350 (extreme close-up / vertical overflow)
                # Priority: always keep the head visible; trim from the bottom.
                crop_top    = max(0, pt - 10)   # small headroom above face
                crop_bottom = crop_top + TARGET_H
                if crop_bottom > scaled_h:
                    crop_bottom = scaled_h
                    crop_top    = max(0, scaled_h - TARGET_H)
                final_img = img_scaled.crop((0, crop_top, TARGET_W, crop_bottom))
                logger.warning(
                    f"⚠️ [THUMB_PREP] Person taller than 4:5 frame — "
                    f"keeping head, trimming feet. Consider a wider shot."
                )

            # Ensure exact 1080x1350 output
            if final_img.size != (TARGET_W, TARGET_H):
                final_img = final_img.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)

            # ── Step 4: SEO Filename Formatting ──────────────────────────────
            clean_cap = re.sub(r'[^a-zA-Z0-9\s]', '', caption.split('\n')[0]).strip()
            words = clean_cap.split()[:5]
            base_name = "_".join(words).lower() if words else "ig_cover"
            if not base_name:
                base_name = "ig_cover"

            out_dir = os.path.dirname(photo_path) or "."

            existing_files = glob.glob(os.path.join(out_dir, f"{base_name}_*.jpg"))
            max_num = 0
            for f in existing_files:
                match = re.search(r'_(\d+)\.jpg$', f)
                if match:
                    max_num = max(max_num, int(match.group(1)))

            next_num = max_num + 1
            new_filename = f"{base_name}_{next_num:02d}.jpg"
            out_path = os.path.join(out_dir, new_filename)

            final_img = final_img.convert("RGB")
            final_img.save(out_path, format="JPEG", quality=95)
            logger.info(f"💾 Saved IG-ready thumbnail to: {out_path}")
            return out_path

        except ImportError:
            logger.warning("⚠️ Pillow not installed. Skipping IG formatting.")
            return photo_path
        except Exception as e:
            logger.error(f"⚠️ Failed to format IG photo: {e}")
            return photo_path


    @staticmethod
    async def _upload_photo_to_instagram(photo_path: str, caption: str, meta_config: Dict) -> Dict:
        """
        Uploads a standard Image Post to Instagram feed using the Graph API.
        Requires generating a temporary public URL first.
        """
        ig_id = meta_config.get("IG_BUSINESS_ID", "").strip()
        ig_token = meta_config.get("IG_BUSINESS_TOKEN", "").strip()
        
        if not ig_id or not ig_token:
            return {"status": "skipped_no_creds"}

        # 0. Format the image perfectly for Instagram (4:5) to stop cropping, rename for SEO
        padded_photo_path = AsyncMetaUploader._prepare_ig_photo(photo_path, caption)

        # 1. Get Temporary Public URL
        public_url = await AsyncMetaUploader._host_temp_image(padded_photo_path)
        if not public_url:
            return {"status": "failed", "error": "Could not host image publicly"}

        logger.info(f"📸 Starting Instagram PHOTO Post Initialization...")
        url = f"{GRAPH_API_URL}/{ig_id}/media"
        caption = AsyncMetaUploader._clean_caption(caption)

        try:
            # 2. Init Container
            init_params = {
                "image_url": public_url,
                "caption": caption,
                "access_token": ig_token
            }
            req_init = await AsyncMetaUploader._retry_request("POST", url, params=init_params)
            
            if "id" not in req_init:
                logger.error(f"IG Photo Init Failed: {req_init}")
                return {"status": "failed_init", "error": str(req_init)}
                
            container_id = req_init["id"]
            
            # Wait for IG to fetch and process the image from the temporary URL
            await AsyncMetaUploader._wait_for_media_status(container_id, ig_token)
            
            # 3. Publish
            pub_url = f"{GRAPH_API_URL}/{ig_id}/media_publish"
            pub_params = {
                "creation_id": container_id,
                "access_token": ig_token
            }
            
            pub_res = await AsyncMetaUploader._retry_request("POST", pub_url, params=pub_params)
            if "id" in pub_res:
                media_id = pub_res["id"]
                logger.info(f"✅ Instagram PHOTO Post Success: {media_id}")
                
                # Fetch Permalink
                link = ""
                try:
                    perm_url = f"{GRAPH_API_URL}/{media_id}"
                    perm_res = await AsyncMetaUploader._retry_request("GET", perm_url, params={"fields": "permalink,shortcode", "access_token": ig_token})
                    link = perm_res.get("permalink") or perm_res.get("shortcode") or ""
                except Exception as e:
                    logger.warning(f"Failed to fetch IG photo permalink: {e}")

                return {"status": "success", "id": media_id, "link": link}
            else:
                logger.error(f"IG Photo Publish Failed: {pub_res}")
                return {"status": "failed_publish", "error": str(pub_res)}

        except Exception as e:
            logger.error(f"IG Photo Upload Exception: {e}")
            return {"status": "failed", "error": str(e)}

    @staticmethod
    async def _upload_to_facebook(video_path: str, caption: str, meta_config: Dict) -> str:
        page_id = meta_config.get("META_PAGE_ID", "").strip()
        page_token = meta_config.get("META_PAGE_TOKEN", "").strip()
        # Allow specific override for Facebook, else fallback to global
        upload_type_env = os.getenv("META_UPLOAD_TYPE_FB", os.getenv("META_UPLOAD_TYPE", "Reels")).strip().upper()
        
        if not page_id or not page_token:
            return {"status": "skipped_no_creds"}

        # Validate that META_PAGE_ID is not accidentally set to Instagram Business ID (17841...)
        if page_id.startswith("17841"):
            logger.warning(
                f"⚠️ [FB UPLOAD] META_PAGE_ID ('{page_id}') is an Instagram Business ID, not a Facebook Page ID. "
                "Skipping Facebook Reels upload. Please set META_PAGE_ID to your Facebook Page ID."
            )
            return {"status": "skipped_invalid_page_id"}

        logger.info(f"📘 Starting Facebook Upload ({upload_type_env})...")

        caption = AsyncMetaUploader._clean_caption(caption)

        # Auto-resolve Page Access Token if User Access Token was provided (safely without tripping auth retry limit)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                tok_url = f"{GRAPH_API_URL}/{page_id}"
                tok_resp = await client.get(tok_url, params={"fields": "access_token", "access_token": page_token})
                if tok_resp.status_code == 200:
                    tok_data = tok_resp.json()
                    if isinstance(tok_data, dict) and tok_data.get("access_token"):
                        resolved_tok = tok_data["access_token"]
                        if resolved_tok != page_token:
                            logger.info(f"🔑 Auto-resolved Facebook Page Access Token for Page ID {page_id}")
                            page_token = resolved_tok
        except Exception as _tok_err:
            logger.debug(f"Notice auto-resolving FB Page token: {_tok_err}")

        try:
            endpoint = "video_reels" if upload_type_env == "REELS" else "videos"
            url = f"{GRAPH_API_URL}/{page_id}/{endpoint}"
            
            # 1. Init
            file_size = os.path.getsize(video_path)
            
            init_params = {
                "upload_phase": "start",
                "file_size": file_size,
                "access_token": page_token
            }
            
            req_init = await AsyncMetaUploader._retry_request("POST", url, params=init_params)
            
            if "video_id" not in req_init:
                logger.error(f"FB Init Failed: {req_init}")
                return {"status": "failed_init", "error": str(req_init)}
            
            logger.info(f"FB Init Success: {req_init}")
            # Use video_id for the upload URL (since rupload complains about invalid video id with session_id)
            video_id = req_init["video_id"]
            upload_session_id = req_init.get("upload_session_id", video_id)
            upload_url = get_rupload_video_url(video_id)
            
            # 2. Upload
            with open(video_path, "rb") as f:
                data = f.read()
            
            headers = {
                "Authorization": f"OAuth {page_token}",
                "offset": "0",
                "Content-Length": str(len(data)),
                "X-Entity-Length": str(len(data)),
                "Content-Type": "video/mp4"
            }
            req_upload = await AsyncMetaUploader._retry_request("POST", upload_url, content=data, headers=headers)
            
            if "success" not in str(req_upload) and "id" not in req_upload:
                 logger.error(f"FB Chunk Upload Failed: {req_upload}")
                 return {"status": "failed_upload", "error": str(req_upload)}
                 
            # 3. Finish / Publish
            finish_params = {
                "upload_phase": "finish",
                "upload_session_id": upload_session_id,
                "video_id": video_id,  # Required by Facebook API
                "description": caption,
                "access_token": page_token
            }
            
            if upload_type_env == "REELS":
                finish_params["video_state"] = "PUBLISHED"
                finish_params["title"] = caption[:50]
            else:
                # For standard /videos POST, explicitly set published=true
                finish_params["published"] = "true"
            
            req_finish = await AsyncMetaUploader._retry_request("POST", url, params=finish_params)
            
            if "success" in req_finish or "id" in req_finish or "video_id" in req_finish:
                # If finish response has ID, use it. Otherwise fallback to the ID from Init phase
                final_vid_id = req_finish.get('video_id') or req_finish.get('id') or video_id
                logger.info(f"✅ Facebook Upload Success: {final_vid_id}")
                # Construct Link based on type
                if upload_type_env == "REELS":
                     fb_link = f"https://www.facebook.com/reel/{final_vid_id}"
                else:
                     fb_link = f"https://www.facebook.com/watch/?v={final_vid_id}"
                     
                logger.info(f"✅ Facebook Upload Success: {final_vid_id} -> {fb_link}")
                return {"status": "success", "id": final_vid_id, "link": fb_link}
            else:
                logger.error(f"FB Finish Failed: {req_finish}")
                return {"status": "failed_publish", "error": str(req_finish)}
                
        except Exception as e:
            logger.error(f"FB Upload Exception: {e}")
            return {"status": "failed", "error": str(e)}
            
    @staticmethod
    def _clean_caption(caption: str) -> str:
        """
        Cleans captions of UTF-16 surrogates and applies regex sanitization to strip:
        - Instagram internal tracking tokens (#?AQ...)
        - Long mixed hash tags (>25 chars containing alphanumeric hashes)
        - System placeholders (#Creatorunknown, #Nicheunknown, #unknown)
        - Duplicate spaces and orphaned # symbols
        """
        if not caption:
            return ""
        try:
            # Re-encode with surrogatepass and decode to resolve any stray surrogates
            cleaned = caption.encode('utf-16', 'surrogatepass').decode('utf-16')
            return strip_system_and_tracking_tokens(cleaned)
        except Exception as e:
            logger.warning(f"⚠️ Caption cleaning failed: {e}")
            return strip_system_and_tracking_tokens(caption)

    @staticmethod
    async def _retry_request(method, url, timeout=1200.0, **kwargs) -> Dict:
        """
        Generic retry wrapper for HTTP requests using httpx.
        Default timeout increased to 1200s (20m) for large video uploads.
        """
        last_error = None
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    if method == "POST":
                        resp = await client.post(url, **kwargs)
                    else:
                        resp = await client.get(url, **kwargs)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        # Deep Auth Check
                        if isinstance(data, dict) and 'error' in data:
                            err = data['error']
                            if isinstance(err, dict):
                                e_type = err.get('type', '')
                                e_code = err.get('code')
                                e_sub = err.get('error_subcode')
                                
                                if e_type == 'OAuthException' or e_code == 190 or e_sub == 463:
                                    logger.error("❌ CRITICAL: META AUTH FAILURE. Token expired or invalid.")
                                    logger.info("⚠️ ACTION REQUIRED: Refresh 'IG_BUSINESS_TOKEN' and 'META_PAGE_TOKEN' in .env")
                        
                        return data
                    
                    if 400 <= resp.status_code < 600:
                        try: error_data = resp.json()
                        except: error_data = resp.text
                        
                        logger.warning(f"⚠️ API Error ({resp.status_code}): {error_data}")

                        # Critical Auth Check (Stop Retrying)
                        if isinstance(error_data, dict) and 'error' in error_data:
                            err = error_data['error']
                            if isinstance(err, dict):
                                e_type = err.get('type', '')
                                e_code = err.get('code')
                                e_sub = err.get('error_subcode')
                                # Code 200 = Permission error / User not capable
                                # Code 190 = Access Token Invalid
                                if e_type == 'OAuthException' or e_code in [190, 10, 200] or e_sub == 463:
                                    logger.error("❌ CRITICAL: META AUTH FAILURE. Stopping Retries.")
                                    logger.info("⚠️ ACTION REQUIRED: Check permissions or refresh 'IG_BUSINESS_TOKEN'/'META_PAGE_TOKEN'.")
                                    return error_data
                        
                        if attempt < MAX_RETRIES:
                            logger.info(f"🔄 Retrying in {RETRY_DELAY}s...")
                            await asyncio.sleep(RETRY_DELAY)
                            continue
                        else:
                            # Return error payload on failure
                            return error_data if isinstance(error_data, dict) else {"error": error_data}
                            
                    return resp.json()
                    
                except Exception as e:
                    last_error = e
                    # Enhanced Logging: Use repr() to see error type if str() is empty
                    logger.warning(f"⚠️ Network Exception: {repr(e)}")
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        break
        
        raise last_error

    @staticmethod
    async def _wait_for_media_status(container_id, token, timeout=300):
        """
        Polls Instagram container status until FINISHED or ERROR.
        Timeout default is 300s (5 min) — Reels processing can take 2-5 min.
        Images usually resolve in <10s.
        """
        start = time.time()
        url = f"{GRAPH_API_URL}/{container_id}"
        poll_interval = 8  # seconds between polls

        async with httpx.AsyncClient(timeout=15.0) as client:
            while time.time() - start < timeout:
                try:
                    resp = await client.get(url, params={"access_token": token, "fields": "status_code,status"})
                    res = resp.json()
                    status = res.get("status_code")
                    elapsed = int(time.time() - start)
                    logger.info(f"⏳ [IG_STATUS] container={container_id} status={status} elapsed={elapsed}s")
                    if status == "FINISHED":
                        return True
                    if status == "ERROR":
                        logger.error(f"❌ [IG_STATUS] Container processing ERROR: {res.get('status', res)}")
                        return False
                    await asyncio.sleep(poll_interval)
                except Exception as _poll_e:
                    logger.warning(f"⚠️ [IG_STATUS] Poll exception: {_poll_e}")
                    await asyncio.sleep(poll_interval)
        logger.error(f"❌ [IG_STATUS] Timed out after {timeout}s waiting for container {container_id}")
        return False


# ── 1-Click Meta OAuth & Account Auto-Discovery ─────────────────────────────

def build_meta_oauth_url(app_id: str, redirect_uri: str, scopes: Optional[str] = None) -> str:
    """Builds the 1-Click Meta OAuth authorization URL with all required publishing scopes."""
    import urllib.parse
    scopes_str = scopes or "instagram_basic,instagram_content_publish,pages_show_list,pages_read_engagement,pages_manage_posts,public_profile"
    ver = get_meta_graph_version()
    params = {
        "client_id": app_id.strip(),
        "redirect_uri": redirect_uri.strip(),
        "scope": scopes_str,
        "response_type": "code"
    }
    return f"https://www.facebook.com/{ver}/dialog/oauth?{urllib.parse.urlencode(params)}"


def exchange_meta_oauth_code(code: str, app_id: str, app_secret: str, redirect_uri: str) -> Dict[str, Any]:
    """
    1-Click Automated Token Exchange & Account Discovery:
      1. Exchanges OAuth authorization code for short-lived User Access Token.
      2. Upgrades short-lived token to 60-day Long-Lived User Access Token.
      3. Calls /me/accounts to discover Facebook Pages and extract non-expiring Page Access Tokens.
      4. Queries each page for linked Instagram Business Accounts.
      5. Returns full discovery dictionary with primary IDs and tokens ready for publishing.
    """
    import urllib.parse
    ver = get_meta_graph_version()
    clean_code = code.strip()

    # Strip full redirect URL if user pasted the entire redirected URL
    if "code=" in clean_code:
        parsed = urllib.parse.urlparse(clean_code)
        q = urllib.parse.parse_qs(parsed.query)
        clean_code = q.get("code", [clean_code])[0]
    if "#" in clean_code:
        clean_code = clean_code.split("#")[0]

    with httpx.Client(timeout=30.0) as client:
        # Step 1: Exchange code for short-lived user token
        tok_url = f"https://graph.facebook.com/{ver}/oauth/access_token"
        tok_params = {
            "client_id": app_id.strip(),
            "client_secret": app_secret.strip(),
            "redirect_uri": redirect_uri.strip(),
            "code": clean_code
        }
        res1 = client.get(tok_url, params=tok_params)
        data1 = res1.json()
        if "error" in data1 or "access_token" not in data1:
            err_msg = data1.get("error", {}).get("message", str(data1))
            return {"success": False, "error": f"Failed to exchange code for token: {err_msg}"}

        short_lived_token = data1["access_token"]

        # Step 2: Upgrade to 60-day Long-Lived Token
        ext_params = {
            "grant_type": "fb_exchange_token",
            "client_id": app_id.strip(),
            "client_secret": app_secret.strip(),
            "fb_exchange_token": short_lived_token
        }
        res2 = client.get(tok_url, params=ext_params)
        data2 = res2.json()
        long_lived_token = data2.get("access_token", short_lived_token)

        # Step 3: Discover Facebook Pages & Page Tokens
        pages_url = f"https://graph.facebook.com/{ver}/me/accounts"
        res3 = client.get(pages_url, params={"access_token": long_lived_token})
        data3 = res3.json()
        raw_pages = data3.get("data", [])

        if not raw_pages:
            return {
                "success": False,
                "error": "No Facebook Pages found for this account. Ensure your Facebook account owns or manages a Page."
            }

        pages_info = []
        primary_match = None

        for p in raw_pages:
            p_id = p.get("id")
            p_name = p.get("name")
            p_token = p.get("access_token")

            # Step 4: Query linked Instagram Business Account
            ig_url = f"https://graph.facebook.com/{ver}/{p_id}"
            res_ig = client.get(ig_url, params={
                "fields": "instagram_business_account{id,username,name}",
                "access_token": p_token
            })
            data_ig = res_ig.json()
            ig_data = data_ig.get("instagram_business_account") or {}
            ig_id = ig_data.get("id")
            ig_username = ig_data.get("username")

            p_entry = {
                "page_id": p_id,
                "page_name": p_name,
                "page_token": p_token,
                "instagram_id": ig_id,
                "instagram_username": ig_username
            }
            pages_info.append(p_entry)

            if ig_id and not primary_match:
                primary_match = p_entry

        if not primary_match:
            primary_match = pages_info[0]

        return {
            "success": True,
            "long_lived_user_token": long_lived_token,
            "pages": pages_info,
            "primary": primary_match,
            "meta_page_id": primary_match.get("page_id"),
            "meta_page_token": primary_match.get("page_token"),
            "ig_business_id": primary_match.get("instagram_id"),
            "ig_business_token": primary_match.get("page_token") or long_lived_token,
            "ig_username": primary_match.get("instagram_username"),
            "page_name": primary_match.get("page_name"),
        }


def save_and_sync_meta_credentials(auth_result: Dict[str, Any], user_id_str: Optional[str] = None) -> Tuple[bool, str]:
    """
    Persists discovered Meta & Instagram credentials:
    - Admin users: updates root canonical .env, in-memory os.environ, and root GitHub Secrets.
    - Non-admin users: strictly updates USER_<id>_ environment variables, personal telegram_users.json
      record, and USER_<id>_ GitHub Secrets. NEVER touches root admin credentials.
    """
    if not auth_result.get("success"):
        return False, auth_result.get("error", "Unknown auth failure")

    p_id = auth_result.get("meta_page_id") or ""
    p_token = auth_result.get("meta_page_token") or ""
    ig_id = auth_result.get("ig_business_id") or ""
    ig_token = auth_result.get("ig_business_token") or p_token

    if not p_id or not p_token:
        return False, "Missing Page ID or Page Token in discovery result."

    from Telegram_Storage_Modules.telegram_user_manager import (
        _is_admin_user,
        _persist_env_var_to_disk,
        load_all_users,
        save_all_users,
        sync_user_secret_to_github,
    )

    clean_uid = str(user_id_str).strip() if user_id_str else ""
    is_admin = _is_admin_user(clean_uid) if clean_uid else True

    # 1. Update in-memory runtime & disk
    if is_admin:
        os.environ["META_PAGE_ID"] = p_id
        os.environ["META_PAGE_TOKEN"] = p_token
        if ig_id:
            os.environ["IG_BUSINESS_ID"] = ig_id
            os.environ["IG_BUSINESS_TOKEN"] = ig_token

        # Persist to local .env and Credentials/.env strictly for admin
        _persist_env_var_to_disk("META_PAGE_ID", p_id)
        _persist_env_var_to_disk("META_PAGE_TOKEN", p_token)
        if ig_id:
            _persist_env_var_to_disk("IG_BUSINESS_ID", ig_id)
            _persist_env_var_to_disk("IG_BUSINESS_TOKEN", ig_token)
    else:
        # Non-admin: isolate to user-prefixed in-memory environment, NEVER touch root .env or root os.environ
        os.environ[f"USER_{clean_uid}_META_PAGE_ID"] = p_id
        os.environ[f"USER_{clean_uid}_META_PAGE_TOKEN"] = p_token
        if ig_id:
            os.environ[f"USER_{clean_uid}_IG_BUSINESS_ID"] = ig_id
            os.environ[f"USER_{clean_uid}_IG_BUSINESS_TOKEN"] = ig_token

    # 2. Persist in telegram_users.json under user's isolated record
    try:
        users = load_all_users()
        uid_key = clean_uid if clean_uid else "admin"
        u_rec = users.setdefault(uid_key, {})
        u_rec["meta_page_id"] = p_id
        u_rec["meta_page_token"] = p_token
        if ig_id:
            u_rec["ig_business_id"] = ig_id
            u_rec["ig_business_token"] = ig_token
        save_all_users(users)
    except Exception as _e:
        logger.warning("Could not persist to telegram_users.json: %s", _e)

    # 3. Sync to GitHub Secrets via sync_user_secret_to_github
    # Admin -> updates root canonical secret (META_PAGE_ID, etc.)
    # Non-admin -> updates USER_<id>_ prefixed secret (USER_7822881619_META_PAGE_ID, etc.)
    try:
        target_uid = clean_uid if clean_uid else "admin"
        sync_user_secret_to_github(target_uid, "META_PAGE_ID", p_id)
        sync_user_secret_to_github(target_uid, "META_PAGE_TOKEN", p_token)
        if ig_id:
            sync_user_secret_to_github(target_uid, "IG_BUSINESS_ID", ig_id)
            sync_user_secret_to_github(target_uid, "IG_BUSINESS_TOKEN", ig_token)
    except Exception as _ge:
        logger.warning("GitHub Secret sync notice: %s", _ge)

    msg = f"Connected Facebook Page '{auth_result.get('page_name')}' (ID: {p_id})"
    if ig_id:
        msg += f" and Instagram '@{auth_result.get('ig_username', 'linked')}' (ID: {ig_id})"
    return True, msg


__all__ = [
    "AsyncMetaUploader",
    "get_meta_graph_version",
    "set_meta_graph_version",
    "get_graph_api_url",
    "get_rupload_video_url",
    "GRAPH_API_URL",
    "DEFAULT_META_GRAPH_VERSION",
    "build_meta_oauth_url",
    "exchange_meta_oauth_code",
    "save_and_sync_meta_credentials",
]

