"""
media_publisher_main.py — Phase 4 Multi-Platform Publishing Orchestrator (Make.com & Telegram)
================================================================================
Orchestrates multi-platform broadcasting for approved master video reels:
  1. Social Channels (YouTube Shorts, Instagram Reels, Facebook) via Make.com Webhook
  2. Telegram Channel / Public Group (via Telegram Bot API)

Called directly after a user approves a reel title or taps 'Post Immediately' in Telegram.
"""

import os
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
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)
    load_dotenv(os.path.join(_REPO_ROOT, "Credentials", ".env"), override=False)
except ImportError:
    pass


# ── Telegram Channel Broadcasting ─────────────────────────────────────────────

async def publish_to_telegram(
    video_path: str,
    title: str = "",
    caption: str = "",
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Publishes the approved reel to the configured Telegram Public Channel / Group."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    target_chat = os.getenv("TELEGRAM_PUBLIC_GROUP_ID") or os.getenv("TELEGRAM_STORAGE_GROUP_ID")

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
    seo_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Asynchronous multi-platform publishing orchestrator via Make.com & Telegram.
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

    # 1. Make.com Universal Social Dispatch (Instagram Reels, YouTube Shorts, Stories, Feed)
    if is_make_configured():
        logger.info("📡 Dispatching universal social broadcasting to Make.com Webhook...")
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
                "message": "Broadcasted successfully to Make.com Webhook pipeline!"
            }
        else:
            results["platforms"]["make"] = {
                "status": "failed",
                "error": make_res.get("error", "Unknown Make.com error")
            }
    else:
        results["platforms"]["make"] = {
            "status": "skipped",
            "error": "MAKE_WEBHOOK_URL is not configured. Run /setmakewebhookurl or add to GitHub Secrets."
        }

    # 2. Telegram Channel Broadcasting
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
    seo_data: Optional[Dict[str, Any]] = None
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
                    seo_data=seo_data
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
                seo_data=seo_data
            )
        )


__all__ = [
    "run_phase4_publishing",
    "run_phase4_publishing_async",
    "publish_to_telegram",
]
