"""
media_publisher_main.py — Phase 4 Standalone Multi-Platform Publishing Orchestrator
=====================================================================================
Orchestrates sequential 4-platform broadcasting for approved master video reels:
  1. YouTube Shorts   (via uploader.py)
  2. Instagram Reels  (via meta_uploader.py)
  3. TikTok Creator   (via tiktok_uploader.py)
  4. Telegram Channel (via Bot API)

Called directly after a user approves a reel title or taps 'Post Immediately' in Telegram.
"""

import os
import sys
import time
import json
import logging
import asyncio
from typing import Dict, Any, Optional

logger = logging.getLogger("media_publisher_main")
logger.setLevel(logging.INFO)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# If we're in "simpler update", go up one more level to get to actual AMTCE root
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
            "tiktok_token": os.getenv("TIKTOK_ACCESS_TOKEN"),
            "is_admin": True
        }

    try:
        from Telegram_Storage_Modules.telegram_user_manager import load_all_users, _is_admin_user
        users = load_all_users()
        u_rec = users.get(str(user_id), {})
        is_admin = _is_admin_user(str(user_id)) if user_id else True
        user_id_clean = str(user_id).strip() if user_id else ""

        def _pick(json_key, env_suffixes, root_env_key=None):
            """Resolve credential: user record -> USER_<id>_ENV -> root_env (admin only)."""
            val = (u_rec.get(json_key) or "").strip()
            if val:
                return val
            if isinstance(env_suffixes, str):
                env_suffixes = [env_suffixes]
            if user_id_clean:
                for suffix in env_suffixes:
                    prefixed = os.getenv(f"USER_{user_id_clean}_{suffix}", "").strip()
                    if prefixed:
                        return prefixed
            if is_admin and root_env_key:
                return (os.getenv(root_env_key) or "").strip() or None
            return None

        meta_tok = _pick("ig_business_token", ["IG_BUSINESS_TOKEN", "META_PAGE_TOKEN", "META_TOKEN"], "IG_BUSINESS_TOKEN")
        if not meta_tok and is_admin:
            meta_tok = (os.getenv("META_PAGE_TOKEN") or os.getenv("META_TOKEN") or "").strip() or None

        meta_id = _pick("ig_business_id", ["IG_BUSINESS_ID", "IG_BUSINESS_ACCOUNT_ID", "META_PAGE_ID"], "IG_BUSINESS_ID")
        if not meta_id and is_admin:
            meta_id = (os.getenv("IG_BUSINESS_ACCOUNT_ID") or os.getenv("META_PAGE_ID") or "").strip() or None

        meta_page_id = _pick("meta_page_id", ["META_PAGE_ID", "IG_BUSINESS_ID"], "META_PAGE_ID")
        meta_page_tok = _pick("meta_page_token", ["META_PAGE_TOKEN", "IG_BUSINESS_TOKEN", "META_TOKEN"], "META_PAGE_TOKEN")

        yt_tok_json = (u_rec.get("youtube_token_json") or "").strip() or None
        if not yt_tok_json and user_id_clean:
            for y_suf in ["TOKEN_JSON", "YOUTUBE_TOKEN_JSON"]:
                v = os.getenv(f"USER_{user_id_clean}_{y_suf}", "").strip()
                if v:
                    yt_tok_json = v
                    break
        if not yt_tok_json and is_admin:
            yt_tok_json = os.getenv("TOKEN_JSON", "").strip() or os.getenv("YOUTUBE_TOKEN_JSON", "").strip() or None

        yt_tok = yt_tok_json if yt_tok_json else (token_file if (has_yt and is_admin) else None)

        tt_tok = _pick("tiktok_access_token", ["TIKTOK_ACCESS_TOKEN", "TIKTOK_TOKEN"], "TIKTOK_ACCESS_TOKEN")
        tg_pub_group = _pick("telegram_public_group_id", ["TELEGRAM_PUBLIC_GROUP_ID"], "TELEGRAM_PUBLIC_GROUP_ID")

        return {
            "meta_token": meta_tok,
            "meta_id": meta_id,
            "meta_page_id": meta_page_id,
            "meta_page_token": meta_page_tok,
            "yt_token": yt_tok,
            "yt_token_json": yt_tok_json,  # raw JSON string for in-memory auth
            "tiktok_token": tt_tok,
            "tg_public_group": tg_pub_group,
            "is_admin": is_admin
        }
    except Exception as ex:
        logger.warning("[CREDENTIAL RESOLVE] Error for user %s: %s. Returning empty credentials.", user_id, ex)
        # Never silently fall back to admin credentials for an unknown user
        return {
            "meta_token": None,
            "meta_id": None,
            "meta_page_id": None,
            "meta_page_token": None,
            "yt_token": None,
            "yt_token_json": None,
            "tiktok_token": None,
            "tg_public_group": None,
            "is_admin": False
        }


def publish_to_youtube(video_path: str, title: str, description: str = "", tags: str = "", niche: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Uploads video reel to YouTube Shorts via YouTube Data API v3."""
    logger.info("[PUBLISHER 1/4] Uploading to YouTube Shorts (user_id=%s)...", user_id or "admin")
    try:
        user_creds = _resolve_user_credentials(user_id)
        if user_id and not user_creds.get("is_admin") and not user_creds.get("yt_token"):
            logger.warning("YouTube API credentials missing for non-admin User %s. Skipping YouTube upload.", user_id)
            return {"status": "skipped", "message": "Personal YouTube credentials not configured for non-admin user"}

        from Publishing_Modules.uploader import _upload_sync
        video_id = _upload_sync(
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
            logger.info("[YOUTUBE SUCCESS] Video ID: %s", video_id)
            clean_url = video_id if str(video_id).startswith("http") else f"https://youtu.be/{video_id}"
            return {"status": "success", "video_id": video_id, "url": clean_url}
        else:
            logger.warning("YouTube upload completed without returning Video ID.")
            return {"status": "failed", "message": "No video ID returned (possible rate limit or lock)"}
    except Exception as e:
        logger.error("[YOUTUBE ERROR] %s", e)
        return {"status": "failed", "error": str(e)}


async def publish_to_meta(video_path: str, caption: str, niche: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Uploads video reel to Instagram & Facebook via Meta Graph API using per-user credentials."""
    logger.info("[PUBLISHER 2/4] Uploading to Meta Instagram & Facebook (user_id=%s)...", user_id or "admin")
    try:
        from Publishing_Modules.meta_uploader import AsyncMetaUploader
        user_creds = _resolve_user_credentials(user_id)
        token = user_creds.get("meta_token")
        ig_user_id = user_creds.get("meta_id")

        # Non-admin without credentials: skip — do NOT fall back to admin account
        if user_id and not user_creds.get("is_admin") and (not token or not ig_user_id):
            logger.warning("Meta credentials missing for non-admin User %s. Skipping Meta upload.", user_id)
            return {
                "instagram": {"status": "skipped", "message": "Personal Instagram credentials not configured for user"},
                "facebook": {"status": "skipped", "message": "Personal Facebook credentials not configured for user"}
            }

        # Build explicit meta_config from resolved user credentials.
        # This bypasses _resolve_meta_config() reading from root .env so the correct user account is used.
        if token or ig_user_id:
            meta_config_override = {
                "IG_BUSINESS_ID": ig_user_id or "",
                "IG_BUSINESS_TOKEN": token or "",
                "META_PAGE_ID": user_creds.get("meta_page_id") or "",
                "META_PAGE_TOKEN": user_creds.get("meta_page_token") or token or "",
            }
            logger.info("[META CRED] Using user-resolved config: IG_ID=%s", (ig_user_id or "")[:8] + "...")
        else:
            meta_config_override = None  # Admin fallback: let meta_uploader resolve from niche/env

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
        logger.error("[META ERROR] %s", e)
        return {"instagram": {"status": "failed", "error": str(e)}, "facebook": {"status": "failed", "error": str(e)}}


async def publish_to_instagram(video_path: str, caption: str, niche: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
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


async def publish_to_tiktok(video_path: str, title: str, tags: str = "", niche: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Uploads video reel to TikTok Creator account via TikTok Direct Post API."""
    logger.info("🎵 [PUBLISHER 3/4] Uploading to TikTok...")
    try:
        user_creds = _resolve_user_credentials(user_id)
        if not user_creds.get("tiktok_token"):
            logger.warning("⚠️ TikTok credentials missing for User %s. Skipping TikTok upload.", user_id or "default")
            return {"status": "skipped", "message": "Personal TikTok credentials not configured for user"}

        from Publishing_Modules.tiktok_uploader import upload_to_tiktok
        res = await upload_to_tiktok(
            file_path=video_path,
            title=title,
            hashtags=tags or "#viral #shorts #trending",
            niche=niche
        )
        if res.get("status") == "success":
            logger.info("✅ [TIKTOK SUCCESS] Publish ID: %s", res.get("id"))
            return {"status": "success", "publish_id": res.get("id")}
        else:
            logger.warning("⚠️ TikTok upload skipped or failed: %s", res.get("error"))
            return {"status": res.get("status", "failed"), "message": res.get("error")}
    except Exception as e:
        logger.error("❌ [TIKTOK ERROR] %s", e)
        return {"status": "failed", "error": str(e)}


async def publish_to_telegram(video_path: str, title: str, caption: str = "", user_id: Optional[str] = None) -> Dict[str, Any]:
    """Dispatches published master reel to Telegram Public Group & Vault Storage Group."""
    logger.info("✈️ [PUBLISHER 4/4] Publishing to Telegram Public Group / Channel (user_id=%s)...", user_id or "admin")
    try:
        from telegram import Bot
        from telegram.request import HTTPXRequest
        from telegram.error import TimedOut, NetworkError, RetryAfter
        import asyncio

        user_creds = _resolve_user_credentials(user_id)
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        public_group = user_creds.get("tg_public_group") or os.getenv("TELEGRAM_PUBLIC_GROUP_ID")
        storage_group = os.getenv("TELEGRAM_STORAGE_GROUP_ID") or os.getenv("TELEGRAM_STORAGE_CHAT_ID")

        if not bot_token:
            logger.warning("⚠️ Telegram Bot Token missing. Skipping Telegram broadcast.")
            return {"status": "skipped", "message": "Bot token missing"}

        if not public_group:
            logger.warning("⚠️ TELEGRAM_PUBLIC_GROUP_ID not configured in .env / GitHub Secrets. Skipping Public Telegram upload.")
            return {"status": "skipped", "message": "TELEGRAM_PUBLIC_GROUP_ID not configured"}

        # Format public group target ID (@channel_name, -100xxxxxxxxxx, or channel id)
        raw_pub = str(public_group).strip()
        if raw_pub.startswith("-"):
            target_public = int(raw_pub)
        elif raw_pub.isdigit():
            target_public = int(f"-100{raw_pub}") if not raw_pub.startswith("100") else int(f"-{raw_pub}")
        else:
            target_public = raw_pub if raw_pub.startswith("@") else f"@{raw_pub}"

        req = HTTPXRequest(
            connection_pool_size=8,
            connect_timeout=60.0,
            pool_timeout=60.0,
            read_timeout=300.0,
            write_timeout=300.0
        )
        bot = Bot(token=bot_token, request=req)
        clean_title = (title or "").strip()
        clean_caption = (caption or "").strip()

        # Deduplicate title: if clean_caption already starts with or contains clean_title in first 80 chars, don't duplicate!
        if clean_title and (clean_title.lower() in clean_caption[:len(clean_title) + 40].lower() or clean_caption.startswith("🔥")):
            full_caption = clean_caption
        elif clean_title:
            full_caption = f"🔥 **{clean_title}**\n\n{clean_caption}"
        else:
            full_caption = clean_caption or '#viral #shorts #reels'

        async def _send_video_safe(target_cid, file_or_id, text_caption, max_retries=3):
            for attempt in range(1, max_retries + 1):
                try:
                    if isinstance(file_or_id, str):
                        return await bot.send_video(
                            chat_id=target_cid,
                            video=file_or_id,
                            caption=text_caption,
                            supports_streaming=True,
                            read_timeout=300,
                            write_timeout=300
                        )
                    else:
                        with open(video_path, "rb") as vf:
                            return await bot.send_video(
                                chat_id=target_cid,
                                video=vf,
                                caption=text_caption,
                                supports_streaming=True,
                                read_timeout=300,
                                write_timeout=300
                            )
                except RetryAfter as ra:
                    logger.warning(f"⏳ Rate-limited by Telegram. Sleeping {ra.retry_after}s (attempt {attempt}/{max_retries})")
                    await asyncio.sleep(ra.retry_after + 1)
                except (TimedOut, NetworkError, TimeoutError) as te:
                    logger.warning(f"🔄 Telegram upload transient timeout/network error (attempt {attempt}/{max_retries}): {te}")
                    if attempt >= max_retries:
                        raise
                    await asyncio.sleep(2 * attempt)
                except Exception as ex:
                    err_txt = str(ex)
                    if "Forbidden" in err_txt or "can't initiate conversation" in err_txt.lower():
                        logger.warning(f"⚠️ Cannot send to chat ID {target_cid}: Bot cannot initiate conversation with user.")
                    raise

        # 1. Upload reel to Public Telegram Group / Channel
        sent_msg = None
        try:
            sent_msg = await _send_video_safe(target_public, None, full_caption)
            logger.info("📢 ✅ [TELEGRAM SUCCESS] Dispatched approved reel to Public Telegram Group (%s)", public_group)
        except Exception as pg_e:
            err_s = str(pg_e)
            if "Chat not found" in err_s or "chat not found" in err_s.lower():
                logger.error("❌ Public Telegram Group upload failed: Chat not found (%s). Make sure the Bot is added as Admin in your public group/channel!", public_group)
                return {"status": "failed", "error": f"Chat not found for {public_group}. Add bot to group as admin!"}
            else:
                logger.error("❌ Public Telegram Group upload failed (%s): %s", public_group, pg_e)
                return {"status": "failed", "error": str(pg_e)}

        cached_file_id = sent_msg.video.file_id if (sent_msg and sent_msg.video) else None

        # 2. Backup to Vault Storage Group if configured and distinct
        if storage_group and str(storage_group).strip() != str(public_group).strip():
            try:
                raw_sg = str(storage_group).strip()
                target_sg = int(raw_sg) if (raw_sg.startswith("-") or raw_sg.isdigit()) else raw_sg
                video_payload = cached_file_id if cached_file_id else None
                await _send_video_safe(
                    target_sg,
                    video_payload,
                    f"📦 **[VAULT PUBLISHED BACKUP]**\n📌 `{title}`\n📁 `{os.path.basename(video_path)}`"
                )
            except Exception as sg_e:
                logger.warning("⚠️ Vault storage group backup warning: %s", sg_e)

        msg_id = sent_msg.message_id if sent_msg else None
        return {"status": "success", "message_id": msg_id, "url": f"Public Group ({public_group})", "link": f"Public Group ({public_group})"}
    except Exception as e:
        logger.error("❌ [TELEGRAM ERROR] %s", e)
        return {"status": "failed", "error": str(e)}


async def run_phase4_publishing_async(
    video_path: str,
    title: str,
    description: str = "",
    tags: str = "#viral #shorts #trending",
    niche: Optional[str] = None,
    user_id: Optional[str] = None,
    seo_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Async implementation of Phase 4 Multi-Platform Publishing."""
    logger.info("==================================================================")
    logger.info("[PHASE 4 MEDIA PUBLISHER] Starting Multi-Platform Publishing Workflow")
    logger.info("Reel: %s | User: %s", os.path.basename(video_path), user_id or "admin")
    logger.info("Title: '%s'", title)
    logger.info("===========================================================================")

    if not os.path.exists(video_path):
        logger.error("[PUBLISHER FAILED] Video file not found: %s", video_path)
        return {"success": False, "error": f"File not found: {video_path}"}

    results = {
        "video_path": video_path,
        "title": title,
        "platforms": {}
    }

    platforms_payload = (seo_data.get("platforms", {}) if isinstance(seo_data, dict) else {}) or {}

    # 1. YouTube Shorts (sync function)
    yt_payload = platforms_payload.get("youtube", {})
    yt_title = yt_payload.get("title") or title
    yt_desc = yt_payload.get("description") or description
    yt_tags = " ".join(yt_payload.get("hashtags", [])) if yt_payload.get("hashtags") else tags
    yt_res = publish_to_youtube(video_path=video_path, title=yt_title, description=yt_desc, tags=yt_tags, niche=niche, user_id=user_id)
    results["platforms"]["youtube"] = yt_res

    # 2. Meta (Instagram Reels & Facebook Reels)
    ig_payload = platforms_payload.get("instagram", {})
    ig_desc = ig_payload.get("description")
    ig_tags = " ".join(ig_payload.get("hashtags", [])) if ig_payload.get("hashtags") else tags

    if ig_desc:
        # Include hashtags cleanly if not already present in the description body
        meta_caption = ig_desc if (ig_tags and ig_tags.split()[0] in ig_desc) else f"{ig_desc}\n\n{ig_tags}"
    elif description:
        meta_caption = description if (tags and tags.split()[0] in description) else f"{description}\n\n{tags}"
    else:
        meta_caption = f"{title}\n\n{tags}"

    try:
        meta_res = await publish_to_meta(video_path=video_path, caption=meta_caption, niche=niche, user_id=user_id)
        
        # Instagram
        ig_result = meta_res.get("instagram", {})
        if ig_result.get("status") == "success":
            media_id = ig_result.get("id")
            link = ig_result.get("link") or (f"https://www.instagram.com/p/{media_id}/" if media_id else "Uploaded successfully")
            results["platforms"]["instagram"] = {"status": "success", "media_id": media_id, "link": link, "url": link}
        elif ig_result.get("status") not in ("skipped", None):
            results["platforms"]["instagram"] = {"status": "failed", "response": ig_result}

        # Facebook
        fb_result = meta_res.get("facebook", {})
        if fb_result.get("status") == "success":
            fb_link = fb_result.get("link") or fb_result.get("url") or "Uploaded successfully"
            results["platforms"]["facebook"] = {"status": "success", "link": fb_link, "url": fb_link}
        elif fb_result.get("status") not in ("skipped", "disabled/skipped", None):
            results["platforms"]["facebook"] = {"status": "failed", "response": fb_result}
    except Exception as e:
        logger.error("❌ Meta publish error: %s", e)
        results["platforms"]["instagram"] = {"status": "failed", "error": str(e)}

    # 3. TikTok Creator (Modular Feature Flag: ENABLE_TIKTOK=yes)
    enable_tiktok = os.getenv("ENABLE_TIKTOK", "no").lower() in ("yes", "true", "1")
    if enable_tiktok:
        try:
            tt_res = await publish_to_tiktok(video_path=video_path, title=title, tags=tags, niche=niche, user_id=user_id)
            results["platforms"]["tiktok"] = tt_res
        except Exception as e:
            logger.error("TikTok publish error: %s", e)
            results["platforms"]["tiktok"] = {"status": "failed", "error": str(e)}

    # 4. Telegram Channel
    tg_payload = platforms_payload.get("telegram", {})
    tg_desc = tg_payload.get("description")
    tg_title = tg_payload.get("title") or title
    tg_tags = " ".join(tg_payload.get("hashtags", [])) if tg_payload.get("hashtags") else tags

    if tg_desc:
        tg_caption = tg_desc if (tg_tags and tg_tags.split()[0] in tg_desc) else f"{tg_desc}\n\n{tg_tags}"
    elif description:
        tg_caption = description if (tags and tags.split()[0] in description) else f"{description}\n\n{tags}"
    else:
        tg_caption = tg_tags

    try:
        tg_res = await publish_to_telegram(video_path=video_path, title=tg_title, caption=tg_caption, user_id=user_id)
        results["platforms"]["telegram"] = tg_res
    except Exception as e:
        logger.error("❌ Telegram publish error: %s", e)
        results["platforms"]["telegram"] = {"status": "failed", "error": str(e)}

    # Determine overall success
    success_count = sum(1 for p in results["platforms"].values() if p.get("status") == "success")
    total_active_platforms = len(results["platforms"])
    results["success"] = success_count > 0
    results["published_count"] = success_count

    logger.info("==================================================================")
    logger.info("🎉 [PHASE 4 COMPLETE] Published across %d/%d platforms!", success_count, total_active_platforms)
    for p_name, p_data in results["platforms"].items():
        logger.info("   • %s: %s (%s)", p_name.upper(), p_data.get("status"), p_data.get("url") or p_data.get("message") or "ok")
    logger.info("==================================================================")

    return results


def run_phase4_publishing(
    video_path: str,
    title: str,
    description: str = "",
    tags: str = "#viral #shorts #trending",
    niche: Optional[str] = None,
    user_id: Optional[str] = None,
    seo_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Master Orchestration Entry Point for Phase 4 Multi-Platform Publishing.
    Executes sequential 4-platform broadcasting: YouTube -> Instagram -> TikTok -> Telegram.
    Handles running event loops safely.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Event loop is running — execute async coroutine in existing loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                lambda: asyncio.run(
                    run_phase4_publishing_async(
                        video_path=video_path, title=title, description=description, tags=tags, niche=niche, user_id=user_id, seo_data=seo_data
                    )
                )
            )
            return future.result()
    else:
        return asyncio.run(
            run_phase4_publishing_async(
                video_path=video_path, title=title, description=description, tags=tags, niche=niche, user_id=user_id, seo_data=seo_data
            )
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4 Multi-Platform Video Publisher")
    parser.add_argument("video", help="Path to .mp4 video file to publish")
    parser.add_argument("--title", "-t", required=True, help="Video title/caption")
    parser.add_argument("--tags", default="#viral #shorts #trending", help="Hashtags")
    args = parser.parse_args()

    res = run_phase4_publishing(video_path=args.video, title=args.title, tags=args.tags)
    print(json.dumps(res, indent=2))
