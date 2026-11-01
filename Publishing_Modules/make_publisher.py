"""
Publishing_Modules / make_publisher.py
================================================================================
Universal Make.com Webhook Multi-Platform Publishing Client.

Enables 1-click publishing for:
  - Instagram Reels, Feed Photos, Carousels, and Stories (via Meta Graph API)
  - YouTube Shorts and Long-Form Videos (via YouTube Data API v3)
  - Facebook Page Reels and Videos
  - LinkedIn & Pinterest Video Pins

Key Features:
  - Supports Videos (Reels / Shorts / Feed), Single Photos, Carousels, and Stories.
  - Full Affiliate Link Support:
      * Links formatted with compliance disclosures (#ad #affiliate).
      * Included in main payload (`affiliate_links`, `primary_link`).
      * Auto-generated `first_comment` payload for Instagram clickable links.
      * Story link sticker / YouTube description link attachment (`story_link`).
  - Automatic Public URL Resolution:
      * Automatically generates direct public URLs via Telegram Storage Group Bot API
        (https://api.telegram.org/file/bot.../{file_path}) so Make.com and Meta can
        ingest media files without local server hosting.
  - Drop-in `PublishQueue` adapter for seamless integration with main.py.

Configuration (via .env or GitHub Secrets):
  MAKE_WEBHOOK_URL = "https://hook.eu2.make.com/your-private-webhook-token"
"""

import os
import sys
import json
import time
import logging
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger("make_publisher")
logger.setLevel(logging.INFO)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL", "").strip()


def is_make_configured() -> bool:
    """Returns True if MAKE_WEBHOOK_URL is set in environment or secrets."""
    url = os.getenv("MAKE_WEBHOOK_URL", MAKE_WEBHOOK_URL).strip()
    return bool(url and url.startswith("http"))


# ── Public URL Resolution via Telegram Vault ──────────────────────────────────

def resolve_public_media_url(local_path_or_url: str) -> Optional[str]:
    """
    Ensures media is accessible via a public HTTP(S) URL for Make.com / Meta.
    If already a URL, returns as-is.
    If a local file, obtains its Telegram Bot API direct download link:
      https://api.telegram.org/file/bot{TOKEN}/{file_path}
    """
    if not local_path_or_url:
        return None

    if str(local_path_or_url).startswith(("http://", "https://")):
        return str(local_path_or_url)

    if not os.path.isfile(local_path_or_url):
        logger.warning(f"⚠️ [MakePublisher] File does not exist locally: {local_path_or_url}")
        return None

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    storage_chat_id = os.getenv("TELEGRAM_STORAGE_GROUP_ID", "").strip()

    if not bot_token or not storage_chat_id:
        logger.warning("⚠️ [MakePublisher] TELEGRAM_BOT_TOKEN or TELEGRAM_STORAGE_GROUP_ID missing. Cannot resolve public URL.")
        return None

    try:
        # Step 1: Upload to Telegram Storage Group to get file_id
        import urllib.request
        from Telegram_Storage_Modules.telegram_http import send_document

        doc_res = send_document(storage_chat_id, local_path_or_url, caption="📦 Media for Make.com Dispatch")
        file_id = None
        if doc_res and doc_res.get("ok"):
            res_obj = doc_res.get("result", {})
            file_id = (
                res_obj.get("document", {}).get("file_id") or
                res_obj.get("video", {}).get("file_id") or
                res_obj.get("photo", [{}])[-1].get("file_id")
            )

        if not file_id:
            logger.warning("⚠️ [MakePublisher] Failed to upload media to Telegram for public URL resolution.")
            return None

        # Step 2: Get direct file_path from Telegram Bot API
        get_file_url = f"https://api.telegram.org/bot{bot_token}/getFile?file_id={file_id}"
        req = urllib.request.Request(get_file_url, headers={"User-Agent": "MakePublisher/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if data.get("ok") and data.get("result", {}).get("file_path"):
            remote_file_path = data["result"]["file_path"]
            public_url = f"https://api.telegram.org/file/bot{bot_token}/{remote_file_path}"
            logger.info("🌐 [MakePublisher] Resolved public media URL: %s", public_url[:40] + "...")
            return public_url
    except Exception as e:
        logger.warning(f"⚠️ [MakePublisher] Public URL resolution error: {e}")

    return None


# ── Payload Builder ────────────────────────────────────────────────────────────

def build_make_payload(
    media_path_or_url: Union[str, List[str]],
    title: str = "",
    caption: str = "",
    tags: str = "",
    post_type: str = "reel",  # "reel", "post", "story", "carousel"
    media_type: str = "video",  # "video", "image", "carousel"
    affiliate_links: Optional[List[str]] = None,
    story_link: Optional[str] = None,
    first_comment: Optional[str] = None,
    seo_data: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    custom_fields: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Constructs a rich, normalized JSON payload ready for Make.com routing.
    """
    # 1. Resolve Media URL(s)
    public_media_urls: List[str] = []
    if isinstance(media_path_or_url, list):
        for p in media_path_or_url:
            resolved = resolve_public_media_url(p)
            if resolved:
                public_media_urls.append(resolved)
        primary_media_url = public_media_urls[0] if public_media_urls else ""
    else:
        primary_media_url = resolve_public_media_url(media_path_or_url) or ""
        public_media_urls = [primary_media_url] if primary_media_url else []

    # 2. Extract Affiliate Links from SEO data if not explicitly provided
    aff_links = list(affiliate_links or [])
    if not aff_links and isinstance(seo_data, dict):
        aff_links = list(seo_data.get("_affiliate_links", []) or seo_data.get("affiliate_links", []))

    primary_aff_link = aff_links[0] if aff_links else (story_link or "")

    # 3. Clean Caption & Hashtags
    tags_clean = tags.strip() if tags else "#viral #trending"
    if tags_clean and not tags_clean.startswith("#"):
        tags_clean = " ".join([f"#{t.strip()}" for t in tags_clean.split() if t.strip()])

    full_caption = caption.strip()
    if not full_caption and title:
        full_caption = f"{title}\n\n{tags_clean}"
    elif full_caption and tags_clean and tags_clean.split()[0] not in full_caption:
        full_caption = f"{full_caption}\n\n{tags_clean}"

    # Auto-add affiliate disclosure if links exist and disclosure is missing
    if aff_links and "#ad" not in full_caption.lower() and "#affiliate" not in full_caption.lower():
        full_caption = f"{full_caption}\n\n(Disclosure: #ad #affiliate)"

    # 4. Generate First Comment (for Instagram clickable links)
    comment_text = first_comment or ""
    if not comment_text and aff_links:
        if len(aff_links) == 1:
            comment_text = f"Featured item link: {aff_links[0]} (#ad)"
        else:
            links_formatted = "\n".join([f"• {u}" for u in aff_links])
            comment_text = f"Featured items:\n{links_formatted}\n(#ad #affiliate)"

    # 5. Extract Platform Specific Overrides from seo_data
    platforms = seo_data.get("platforms", {}) if isinstance(seo_data, dict) else {}
    yt_seo = platforms.get("youtube", {})
    ig_seo = platforms.get("instagram", {})
    fb_seo = platforms.get("facebook", {})

    yt_title = yt_seo.get("title") or title
    yt_desc = yt_seo.get("description") or full_caption
    ig_caption = ig_seo.get("description") or full_caption
    fb_caption = fb_seo.get("description") or full_caption

    # Ensure YouTube description includes clickable affiliate links
    if aff_links and not any(u in yt_desc for u in aff_links):
        aff_section = "\n\nProduct Links:\n" + "\n".join([f"- {u}" for u in aff_links]) + "\n(Disclosure: #ad #affiliate)"
        yt_desc = f"{yt_desc}{aff_section}"

    # 6. Final Unified Payload
    payload = {
        "event": "gemini_editor_publish",
        "timestamp": time.time(),
        "user_id": user_id or "admin",
        "post_type": post_type,        # "reel", "post", "story", "carousel"
        "media_type": media_type,      # "video", "image", "carousel"
        "video_url": primary_media_url,
        "image_url": primary_media_url if media_type == "image" else "",
        "media_urls": public_media_urls,
        "title": title or yt_title,
        "caption": full_caption,
        "hashtags": tags_clean,
        "story_link": story_link or primary_aff_link,
        "primary_link": primary_aff_link,
        "affiliate_links": aff_links,
        "first_comment": comment_text,
        "youtube": {
            "title": yt_title[:100],
            "description": yt_desc[:5000],
            "tags": [t.lstrip("#") for t in tags_clean.split() if t.startswith("#")],
            "privacy_status": "public"
        },
        "instagram": {
            "caption": ig_caption[:2200],
            "media_type": "REELS" if post_type == "reel" else ("STORIES" if post_type == "story" else "FEED"),
            "first_comment": comment_text
        },
        "facebook": {
            "caption": fb_caption[:5000],
            "media_type": "REELS" if post_type == "reel" else "VIDEO"
        }
    }

    if custom_fields and isinstance(custom_fields, dict):
        payload.update(custom_fields)

    return payload


# ── Publishing Dispatcher ──────────────────────────────────────────────────────

def publish_to_make(
    media_path: Union[str, List[str]],
    title: str = "",
    caption: str = "",
    tags: str = "",
    post_type: str = "reel",
    media_type: str = "video",
    affiliate_links: Optional[List[str]] = None,
    story_link: Optional[str] = None,
    first_comment: Optional[str] = None,
    seo_data: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    webhook_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Submits a publishing dispatch to Make.com Webhook with retry logic.
    Returns:
      {
        "status": "success" | "failed",
        "post_type": post_type,
        "payload": payload_summary,
        "url": webhook_url
      }
    """
    active_url = (webhook_url or os.getenv("MAKE_WEBHOOK_URL", MAKE_WEBHOOK_URL)).strip()
    if not active_url:
        return {
            "status": "failed",
            "error": "MAKE_WEBHOOK_URL is not set. Please add it to your .env or GitHub Secrets."
        }

    logger.info("📡 [MakePublisher] Constructing payload for Make.com dispatch...")
    payload = build_make_payload(
        media_path_or_url=media_path,
        title=title,
        caption=caption,
        tags=tags,
        post_type=post_type,
        media_type=media_type,
        affiliate_links=affiliate_links,
        story_link=story_link,
        first_comment=first_comment,
        seo_data=seo_data,
        user_id=user_id
    )

    if not payload.get("video_url") and not payload.get("image_url") and not payload.get("media_urls"):
        return {
            "status": "failed",
            "error": "Failed to resolve a public media URL for Make.com."
        }

    encoded_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        active_url,
        data=encoded_data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "GeminiEditor-MakePublisher/1.0"
        },
        method="POST"
    )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("🚀 [MakePublisher] Sending dispatch to Make.com (Attempt %d/%d)...", attempt, max_retries)
            with urllib.request.urlopen(req, timeout=30) as resp:
                status_code = resp.getcode()
                resp_text = resp.read().decode("utf-8")

            if status_code in (200, 201, 202, 204):
                logger.info("✅ [MakePublisher] Successfully accepted by Make.com! Response: %s", resp_text[:100])
                return {
                    "status": "success",
                    "webhook_url": active_url[:40] + "...",
                    "post_type": post_type,
                    "title": title,
                    "media_url": payload.get("video_url") or payload.get("image_url"),
                    "response": resp_text
                }
            else:
                logger.warning("⚠️ [MakePublisher] Unexpected HTTP status %d: %s", status_code, resp_text)
        except Exception as err:
            logger.error("❌ [MakePublisher] Dispatch error on attempt %d: %s", attempt, err)
            if attempt < max_retries:
                time.sleep(2.0 * attempt)

    return {
        "status": "failed",
        "error": f"Failed to deliver payload to Make.com webhook after {max_retries} attempts."
    }


# ── Drop-In PublishQueue Compatibility Class ──────────────────────────────────

class PublishQueue:
    """Drop-in adapter replacing legacy queue publishers with Make.com webhook dispatch."""

    @classmethod
    def add(
        cls,
        video_path: str,
        channel_title: str = "",
        channel_folder: str = "General",
        caption: str = "",
        tags: str = "",
        seo_data: Optional[Dict[str, Any]] = None,
        post_type: str = "reel"
    ) -> bool:
        """Enqueues and immediately dispatches reel to Make.com."""
        logger.info("📥 [Make.com Queue] Dispatched reel '%s' (title: '%s')", os.path.basename(video_path), channel_title)
        res = publish_to_make(
            media_path=video_path,
            title=channel_title,
            caption=caption,
            tags=tags,
            post_type=post_type,
            seo_data=seo_data
        )
        return res.get("status") == "success"

    @classmethod
    def status(cls) -> Dict[str, Any]:
        return {
            "queue_type": "Make.com Webhook Publisher",
            "configured": is_make_configured(),
            "webhook_set": bool(os.getenv("MAKE_WEBHOOK_URL", MAKE_WEBHOOK_URL))
        }
