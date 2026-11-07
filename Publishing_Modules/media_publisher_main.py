"""
media_publisher_main.py — Phase 4 Multi-Platform Publishing Orchestrator
================================================================================
Orchestrates multi-platform broadcasting for approved master video reels:
  1. YouTube Shorts (via uploader.py using dynamic YOUTUBE_API_VERSION)
  2. Instagram Reels & Facebook Pages (via meta_uploader.py using dynamic META_GRAPH_VERSION)
  3. Make.com Webhook Pipeline (optional secondary broadcast if MAKE_WEBHOOK_URL is set)
  4. Telegram Channel / Public Group (via Telegram Bot API)

Called directly after a user approves a reel title or taps 'Post Immediately' in Telegram.
"""

import os
import re
import sys
import time
import json
import logging
import asyncio
from typing import Dict, Any, Optional

from Publishing_Modules.make_publisher import publish_to_make, is_make_configured

logger = logging.getLogger("media_publisher_main")
logger.setLevel(logging.INFO)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(_REPO_ROOT) == "simpler update":
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)
    load_dotenv(os.path.join(_REPO_ROOT, "Credentials", ".env"), override=False)
except ImportError:
    pass


def _resolve_user_credentials(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolves per-user social media credentials from telegram_users.json first,
    then USER_<id>_* environment variables, then root .env (admin fallback only).

    For non-admin users: NEVER falls back to admin root .env credentials.
    For admin users: falls back to root .env only if no personal record found.
    """
    cred_file = os.getenv("CLIENT_SECRET_FILE", os.path.join(_REPO_ROOT, "Credentials", "youtube", "client_secret.json"))
    if not os.path.exists(cred_file):
        fallback_cred = os.path.join(_REPO_ROOT, "Credentials", "client_secret.json")
        if os.path.exists(fallback_cred):
            cred_file = fallback_cred

    token_file = os.getenv("YOUTUBE_TOKEN_FILE", os.path.join(_REPO_ROOT, "Credentials", "youtube", "token.json"))
    if not os.path.exists(token_file):
        fallback_token = os.path.join(_REPO_ROOT, "Credentials", "token.json")
        if os.path.exists(fallback_token):
            token_file = fallback_token

    has_yt = os.path.exists(cred_file) or os.path.exists(token_file) or bool(os.getenv("YOUTUBE_TOKEN_JSON"))

    # No user_id: use root admin credentials
    if not user_id:
        return {
            "meta_token": os.getenv("IG_BUSINESS_TOKEN") or os.getenv("META_PAGE_TOKEN"),
            "meta_id": os.getenv("IG_BUSINESS_ACCOUNT_ID") or os.getenv("IG_BUSINESS_ID") or os.getenv("META_PAGE_ID"),
            "meta_page_id": os.getenv("META_PAGE_ID"),
            "meta_page_token": os.getenv("META_PAGE_TOKEN"),
            "yt_token": token_file if has_yt else None,
            "yt_token_json": os.getenv("TOKEN_JSON") or os.getenv("YOUTUBE_TOKEN_JSON"),
            "tiktok_token": os.getenv("TIKTOK_ACCESS_TOKEN"),
            "tg_pub_group": os.getenv("TELEGRAM_PUBLIC_GROUP_ID"),
            "is_admin": True
        }

    try:
        from Telegram_Storage_Modules.telegram_user_manager import (
            _is_admin_user,
            get_user_instagram_token,
            get_user_instagram_id,
            get_user_facebook_id,
            get_user_facebook_token,
            get_user_youtube_token_json,
            get_user_tiktok_token,
            get_user_public_group
        )
        clean_uid = str(user_id).strip()
        is_admin = _is_admin_user(clean_uid)

        meta_tok = get_user_instagram_token(clean_uid)
        meta_id = get_user_instagram_id(clean_uid)
        meta_page_id = get_user_facebook_id(clean_uid)
        meta_page_tok = get_user_facebook_token(clean_uid)

        yt_tok_json = get_user_youtube_token_json(clean_uid)
        yt_tok = yt_tok_json if yt_tok_json else (token_file if (has_yt and is_admin) else None)

        tt_tok = get_user_tiktok_token(clean_uid)
        tg_pub_group = get_user_public_group(clean_uid)

        return {
            "meta_token": meta_tok,
            "meta_id": meta_id,
            "meta_page_id": meta_page_id,
            "meta_page_token": meta_page_tok,
            "yt_token": yt_tok,
            "yt_token_json": yt_tok_json,  # raw JSON string for in-memory auth
            "tiktok_token": tt_tok,
            "tg_pub_group": tg_pub_group,
            "is_admin": is_admin
        }
    except Exception as ex:
        logger.warning("[CREDENTIAL RESOLVE] Error for user %s: %s. Returning empty credentials.", user_id, ex)
        return {
            "meta_token": None,
            "meta_id": None,
            "meta_page_id": None,
            "meta_page_token": None,
            "yt_token": None,
            "yt_token_json": None,
            "tiktok_token": None,
            "tg_pub_group": None,
            "is_admin": False
        }


async def publish_to_youtube(
    video_path: str,
    title: str,
    description: str = "",
    tags: str = "",
    niche: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Uploads video reel to YouTube Shorts via YouTube Data API (dynamic version)."""
    logger.info("🎬 [PUBLISHER 1/4] Uploading to YouTube Shorts (user_id=%s)...", user_id or "admin")
    try:
        user_creds = _resolve_user_credentials(user_id)
        if user_id and not user_creds.get("is_admin") and not user_creds.get("yt_token"):
            logger.warning("YouTube API credentials missing for non-admin User %s. Skipping YouTube upload.", user_id)
            return {"status": "skipped", "message": "Personal YouTube credentials not configured for non-admin user"}

        from Publishing_Modules.uploader import upload_to_youtube
        video_id = await upload_to_youtube(
            file_path=video_path,
            title=title,
            description=description or f"{title}\n\n#shorts #viral #trending",
            hashtags=tags or "#shorts #viral #trending",
            privacy="public",
            niche=niche,
            user_yt_token_json=user_creds.get("yt_token_json"),
            user_id=user_id,
        )
        if video_id:
            logger.info("✅ [YOUTUBE SUCCESS] Video ID: %s", video_id)
            clean_url = video_id if str(video_id).startswith("http") else f"https://youtu.be/{video_id}"
            return {"status": "success", "video_id": video_id, "url": clean_url}
        else:
            logger.warning("YouTube upload completed without returning Video ID.")
            return {"status": "failed", "message": "No video ID returned (possible rate limit or lock)"}
    except Exception as e:
        logger.error("❌ [YOUTUBE ERROR] %s", e)
        return {"status": "failed", "error": str(e)}


async def publish_to_meta(
    video_path: str,
    caption: str,
    niche: Optional[str] = None,
    user_id: Optional[str] = None,
    trail_reels: bool = False,
    meta_features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Uploads video reel to Instagram & Facebook via Meta Graph API using dynamic META_GRAPH_VERSION."""
    logger.info("📸 [PUBLISHER 2/4] Uploading to Meta Instagram & Facebook (user_id=%s, trial_reel=%s)...", user_id or "admin", trail_reels)
    try:
        from Publishing_Modules.meta_uploader import AsyncMetaUploader
        user_creds = _resolve_user_credentials(user_id)
        token = user_creds.get("meta_token")
        ig_user_id = user_creds.get("meta_id")

        if user_id and not user_creds.get("is_admin") and (not token or not ig_user_id):
            logger.warning("Meta credentials missing for non-admin User %s. Skipping Meta upload.", user_id)
            return {
                "instagram": {"status": "skipped", "message": "Personal Instagram credentials not configured for user"},
                "facebook": {"status": "skipped", "message": "Personal Facebook credentials not configured for user"}
            }

        m_feat = dict(meta_features or {})
        if trail_reels:
            m_feat["trail_reels"] = True

        if token or ig_user_id:
            meta_config_override = {
                "IG_BUSINESS_ID": ig_user_id or "",
                "IG_BUSINESS_TOKEN": token or "",
                "META_PAGE_ID": user_creds.get("meta_page_id") or "",
                "META_PAGE_TOKEN": user_creds.get("meta_page_token") or token or "",
                "trail_reels": m_feat.get("trail_reels", False),
                "meta_features": m_feat,
            }
            logger.info("[META CRED] Using user-resolved config: IG_ID=%s", (ig_user_id or "")[:8] + "...")
        else:
            meta_config_override = None

        uploader = AsyncMetaUploader()
        res = await uploader.upload_to_meta(
            video_path=video_path,
            caption=caption,
            upload_type="Reels",
            niche=niche or "General_Fallback",
            meta_config_override=meta_config_override
        )
        return res
    except Exception as e:
        logger.error("❌ [META ERROR] %s", e)
        return {"instagram": {"status": "failed", "error": str(e)}, "facebook": {"status": "failed", "error": str(e)}}


async def publish_to_instagram(
    video_path: str,
    caption: str,
    niche: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Uploads video reel to Instagram via Meta Graph API."""
    meta_res = await publish_to_meta(video_path, caption, niche, user_id=user_id)
    ig_result = meta_res.get("instagram", {})
    if ig_result.get("status") == "success":
        media_id = ig_result.get("id")
        link = ig_result.get("link") or (f"https://www.instagram.com/p/{media_id}/" if media_id else "Uploaded successfully")
        logger.info("✅ [INSTAGRAM SUCCESS] Media ID: %s, Link: %s", media_id, link)
        return {"status": "success", "media_id": media_id, "link": link, "url": link}
    else:
        logger.warning("⚠️ Instagram upload failed or skipped: %s", ig_result.get("status"))
        return ig_result


# ── Telegram Channel Broadcasting ─────────────────────────────────────────────

async def publish_to_telegram(
    video_path: str,
    title: str = "",
    caption: str = "",
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Publishes the approved reel to the configured Telegram Public Channel / Group."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    from Telegram_Storage_Modules.telegram_user_manager import get_user_public_group
    target_chat = get_user_public_group(user_id) if user_id else os.getenv("TELEGRAM_PUBLIC_GROUP_ID")

    if not bot_token or not target_chat:
        return {"status": "skipped", "reason": "TELEGRAM_PUBLIC_GROUP_ID not configured"}

    if not os.path.exists(video_path):
        return {"status": "failed", "error": f"Video file not found: {video_path}"}

    try:
        from telegram import Bot
        from telegram.request import HTTPXRequest

        req = HTTPXRequest(read_timeout=600.0, write_timeout=600.0)
        bot = Bot(token=bot_token, request=req)

        display_caption = caption or title or "Viral Reel"
        with open(video_path, "rb") as vf:
            msg = await bot.send_video(
                chat_id=int(target_chat),
                video=vf,
                caption=display_caption[:1024],
                read_timeout=600,
                write_timeout=600
            )

        msg_link = f"https://t.me/c/{str(target_chat).replace('-100', '')}/{msg.message_id}" if str(target_chat).startswith("-100") else f"msg_id_{msg.message_id}"
        return {
            "status": "success",
            "message_id": msg.message_id,
            "url": msg_link,
            "link": msg_link
        }
    except Exception as e:
        logger.error(f"❌ Telegram broadcast error: {e}")
        return {"status": "failed", "error": str(e)}


# ── Phase 4 Main Orchestrator ─────────────────────────────────────────────────

async def run_phase4_publishing_async(
    video_path: str,
    title: str,
    description: str = "",
    tags: str = "#viral #shorts #trending",
    niche: Optional[str] = None,
    user_id: Optional[str] = None,
    seo_data: Optional[Dict[str, Any]] = None,
    trail_reels: bool = False,
) -> Dict[str, Any]:
    """
    Asynchronous master multi-platform publishing orchestrator:
      1. YouTube Shorts (via uploader.py)
      2. Meta Instagram Reels & Facebook (via meta_uploader.py)
      3. Make.com Webhook Pipeline (if MAKE_WEBHOOK_URL is set)
      4. Telegram Channel / Group (via Bot API)
    """
    logger.info("==================================================================")
    logger.info("🚀 [PHASE 4 MEDIA PUBLISHER] Starting Multi-Platform Broadcast")
    logger.info("Reel: %s | User: %s", os.path.basename(video_path), user_id or "admin")
    logger.info("Title: '%s'", title)
    logger.info("==================================================================")

    if not os.path.exists(video_path):
        logger.error("[PUBLISHER FAILED] Video file not found: %s", video_path)
        return {"success": False, "error": f"File not found: {video_path}"}

    results: Dict[str, Any] = {
        "video_path": video_path,
        "title": title,
        "platforms": {}
    }

    platforms_payload = (seo_data.get("platforms", {}) if isinstance(seo_data, dict) else {}) or {}

    # ── Resolve trail_reels flag (may come from caller arg or seo_data.meta_features) ──
    if not trail_reels and isinstance(seo_data, dict):
        trail_reels = bool(seo_data.get("meta_features", {}).get("trail_reels", False))
    if trail_reels:
        logger.info("📸 [PHASE 4] Trail Reels mode active — will inject trial_params for Instagram.")

    # ── Native Master Pipeline: Guaranteed Direct Uploaders ─────────────────
    # YouTube (Official Data API), Meta (Instagram & Facebook Graph API), and
    # Telegram ALWAYS run directly to guarantee high quality, live URL reporting,
    # and zero third-party file leakage.
    logger.info("🔄 [MASTER SOCIAL PIPELINE] Executing direct YouTube & Meta uploaders...")

    # 1. YouTube Shorts (Direct Python Data API)
    yt_payload = platforms_payload.get("youtube", {})
    yt_title = yt_payload.get("title") or title
    yt_desc = yt_payload.get("description") or description or f"{title}\n\n{tags}"
    yt_tags = " ".join(yt_payload.get("hashtags", [])) if yt_payload.get("hashtags") else tags

    try:
        yt_res = await publish_to_youtube(
            video_path=video_path,
            title=yt_title,
            description=yt_desc,
            tags=yt_tags,
            niche=niche,
            user_id=user_id
        )
        if yt_res.get("status") != "skipped":
            results["platforms"]["youtube"] = yt_res
    except Exception as e:
        logger.error("❌ YouTube publish error: %s", e)
        results["platforms"]["youtube"] = {"status": "failed", "error": str(e)}

    # 2. Meta (Instagram Reels & Facebook Pages)
    ig_payload = platforms_payload.get("instagram", {})
    ig_caption = ig_payload.get("caption") or ig_payload.get("description") or description or f"{title}\n\n{tags}"
    target_tags = ig_payload.get("hashtags") or (tags.split() if tags else [])
    if target_tags:
        existing_tags = set(re.findall(r"#\w+", ig_caption.lower()))
        missing_tags = [t for t in target_tags if t.lower() not in existing_tags and t.lower() not in ("#ad", "#affiliate")]
        if missing_tags:
            ig_caption = ig_caption.rstrip() + "\n\n" + " ".join(missing_tags)

    try:
        _meta_feats = (seo_data.get("meta_features", {}) if isinstance(seo_data, dict) else {}) or {}
        meta_res = await publish_to_meta(
            video_path=video_path,
            caption=ig_caption,
            niche=niche,
            user_id=user_id,
            trail_reels=trail_reels,
            meta_features=_meta_feats,
        )
        ig_result = meta_res.get("instagram", {})
        if ig_result.get("status") == "success":
            media_id = ig_result.get("id")
            link = ig_result.get("link") or (f"https://www.instagram.com/p/{media_id}/" if media_id else "Uploaded successfully")
            results["platforms"]["instagram"] = {"status": "success", "media_id": media_id, "link": link, "url": link}
        elif ig_result.get("status") not in ("skipped", None):
            results["platforms"]["instagram"] = {"status": "failed", "response": ig_result}

        fb_result = meta_res.get("facebook", {})
        if fb_result.get("status") == "success":
            fb_link = fb_result.get("link") or fb_result.get("url") or "Uploaded successfully"
            results["platforms"]["facebook"] = {"status": "success", "link": fb_link, "url": fb_link}
        elif fb_result.get("status") not in ("skipped", "disabled/skipped", None):
            results["platforms"]["facebook"] = {"status": "failed", "response": fb_result}
    except Exception as e:
        logger.error("❌ Meta publish error: %s", e)
        results["platforms"]["instagram"] = {"status": "failed", "error": str(e)}

    # 3. Optional Auxiliary Webhook: Make.com (Non-blocking notification only)
    if is_make_configured(user_id=user_id):
        try:
            logger.info("📡 [AUXILIARY WEBHOOK] Firing optional background Make.com dispatch...")
            make_res = publish_to_make(
                media_path=video_path,
                title=title,
                caption=description,
                tags=tags,
                post_type="reel",
                media_type="video",
                seo_data=seo_data,
                user_id=user_id
            )
            if make_res.get("status") == "success":
                results["platforms"]["make"] = {
                    "status": "success",
                    "url": make_res.get("media_url"),
                    "message": "Dispatched to Make.com webhook as auxiliary feed."
                }
            else:
                results["platforms"]["make"] = {
                    "status": "skipped",
                    "error": make_res.get("error", "Make webhook notice")
                }
        except Exception as _mk_err:
            logger.warning("Optional Make.com dispatch notice: %s", _mk_err)
    else:
        results["platforms"]["make"] = {
            "status": "skipped",
            "message": "MAKE_WEBHOOK_URL not configured (optional)."
        }

    # 4. Telegram Channel Broadcasting
    tg_payload = platforms_payload.get("telegram", {})
    tg_desc = tg_payload.get("description")
    tg_title = tg_payload.get("title") or title
    tg_tags = " ".join(tg_payload.get("hashtags", [])) if tg_payload.get("hashtags") else tags

    if tg_desc:
        tg_caption = tg_desc if (tg_tags and tg_tags.split()[0] in tg_desc) else f"{tg_desc}\n\n{tg_tags}"
    elif description:
        tg_caption = description if (tags and tags.split()[0] in description) else f"{description}\n\n{tags}"
    else:
        tg_caption = f"{title}\n\n{tags}" if title else tg_tags

    try:
        tg_res = await publish_to_telegram(video_path=video_path, title=tg_title, caption=tg_caption, user_id=user_id)
        if tg_res.get("status") != "skipped":
            results["platforms"]["telegram"] = tg_res
    except Exception as e:
        logger.error(f"❌ Telegram publish error: {e}")
        results["platforms"]["telegram"] = {"status": "failed", "error": str(e)}

    # Determine overall success
    success_count = sum(1 for p in results["platforms"].values() if p.get("status") == "success")
    results["success"] = success_count > 0
    results["published_count"] = success_count

    logger.info("==================================================================")
    logger.info("🎉 [PHASE 4 COMPLETE] Status: %s (Successes: %d)", "SUCCESS" if results["success"] else "FAILED", success_count)
    for p_name, p_data in results["platforms"].items():
        logger.info("   • %s: %s (%s)", p_name.upper(), p_data.get("status"), p_data.get("url") or p_data.get("error") or "ok")
    logger.info("==================================================================")

    return results


def run_phase4_publishing(
    video_path: str,
    title: str,
    description: str = "",
    tags: str = "#viral #shorts #trending",
    niche: Optional[str] = None,
    user_id: Optional[str] = None,
    seo_data: Optional[Dict[str, Any]] = None,
    trail_reels: bool = False,
) -> Dict[str, Any]:
    """
    Synchronous Master Orchestration Entry Point for Phase 4 Publishing.
    Safely executes run_phase4_publishing_async in any running event loop or thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                run_phase4_publishing_async(
                    video_path=video_path,
                    title=title,
                    description=description,
                    tags=tags,
                    niche=niche,
                    user_id=user_id,
                    seo_data=seo_data,
                    trail_reels=trail_reels,
                )
            )
            return future.result()
    else:
        return asyncio.run(
            run_phase4_publishing_async(
                video_path=video_path,
                title=title,
                description=description,
                tags=tags,
                niche=niche,
                user_id=user_id,
                seo_data=seo_data,
                trail_reels=trail_reels,
            )
        )


__all__ = [
    "run_phase4_publishing",
    "run_phase4_publishing_async",
    "publish_to_youtube",
    "publish_to_meta",
    "publish_to_instagram",
    "publish_to_telegram",
    "_resolve_user_credentials",
]
