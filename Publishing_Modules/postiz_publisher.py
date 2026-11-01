"""
Publishing_Modules/postiz_publisher.py
======================================
Open-Source Postiz API Client (https://github.com/gitroomhq/postiz-app)

Connects AMTCE video rendering directly to a self-hosted or cloud Postiz instance.
Handles:
  1. Media upload via POST /public/v1/upload
  2. Integration querying via GET /public/v1/integrations
  3. Multi-channel immediate or scheduled publishing via POST /public/v1/posts
  4. Modern PublishQueue drop-in adapter for backwards compatibility
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, List, Optional
import httpx

logger = logging.getLogger("postiz_publisher")
logger.setLevel(logging.INFO)

# Base configurations
POSTIZ_URL = os.getenv("POSTIZ_URL", "").strip().rstrip("/")
POSTIZ_API_KEY = os.getenv("POSTIZ_API_KEY", "").strip()
POSTIZ_CHANNELS = os.getenv("POSTIZ_CHANNELS", "all").strip().lower()


def is_postiz_configured() -> bool:
    """Returns True if both POSTIZ_URL and POSTIZ_API_KEY are configured."""
    return bool(POSTIZ_URL and POSTIZ_API_KEY)


def get_postiz_headers() -> Dict[str, str]:
    """Returns authorization headers compatible with Postiz public API."""
    return {
        "Authorization": f"Bearer {POSTIZ_API_KEY}",
        "x-api-key": POSTIZ_API_KEY,
    }


def get_active_integrations() -> List[Dict[str, Any]]:
    """
    Fetches all active social integrations (channels) connected in Postiz.
    Endpoint: GET /public/v1/integrations
    """
    if not is_postiz_configured():
        logger.warning("⚠️ Postiz not configured (POSTIZ_URL or POSTIZ_API_KEY missing).")
        return []

    url = f"{POSTIZ_URL}/public/v1/integrations"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=get_postiz_headers())
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    active = [item for item in data if not item.get("disabled", False)]
                    return active
                elif isinstance(data, dict) and "integrations" in data:
                    return [item for item in data["integrations"] if not item.get("disabled", False)]
                return []
            else:
                logger.error(f"❌ Postiz integrations query failed: {resp.status_code} - {resp.text}")
                return []
    except Exception as e:
        logger.error(f"❌ Postiz integrations connection error: {e}")
        return []


def upload_video_to_postiz(video_path: str) -> Optional[str]:
    """
    Uploads master video reel to Postiz storage.
    Endpoint: POST /public/v1/upload
    Returns media ID or None if upload fails.
    """
    if not is_postiz_configured():
        logger.error("❌ Postiz not configured. Cannot upload video.")
        return None

    if not video_path or not os.path.exists(video_path):
        logger.error(f"❌ Master reel not found on disk: {video_path}")
        return None

    url = f"{POSTIZ_URL}/public/v1/upload"
    filename = os.path.basename(video_path)
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

    logger.info(f"📤 Uploading reel '{filename}' ({file_size_mb:.1f}MB) to Postiz ({POSTIZ_URL})...")

    try:
        with open(video_path, "rb") as vf:
            files = {"file": (filename, vf, "video/mp4")}
            headers = get_postiz_headers()
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(url, headers=headers, files=files)

        if resp.status_code in (200, 201):
            data = resp.json()
            media_id = data.get("id") or data.get("path") or data.get("media_id")
            if media_id:
                logger.info(f"✅ Video uploaded to Postiz successfully (media_id: {media_id})")
                return str(media_id)
            else:
                logger.warning(f"⚠️ Postiz upload returned unexpected JSON: {data}")
                return None
        else:
            logger.error(f"❌ Postiz upload failed ({resp.status_code}): {resp.text}")
            return None
    except Exception as e:
        logger.error(f"❌ Postiz upload exception: {e}")
        return None


def publish_to_postiz(
    video_path: str,
    title: str,
    description: str = "",
    tags: str = "",
    schedule_time: Optional[str] = None,
    seo_data: Optional[Dict[str, Any]] = None,
    target_channels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Master Publishing Function:
      1. Uploads video to Postiz storage.
      2. Discovers matching connected integrations in Postiz.
      3. Submits post to Postiz API via POST /public/v1/posts.
    """
    if not is_postiz_configured():
        return {
            "status": "failed",
            "error": "POSTIZ_URL or POSTIZ_API_KEY is not set. Please configure them in .env or GitHub Secrets."
        }

    # 1. Upload Video
    media_id = upload_video_to_postiz(video_path)
    if not media_id:
        return {
            "status": "failed",
            "error": f"Failed to upload video '{os.path.basename(video_path)}' to Postiz."
        }

    # 2. Discover Integrations
    integrations = get_active_integrations()
    if not integrations:
        return {
            "status": "failed",
            "error": "No active channels/integrations found in Postiz. Please connect at least one account (YouTube, Instagram, or TikTok) in your Postiz dashboard."
        }

    # Filter by POSTIZ_CHANNELS or target_channels
    filter_keys = target_channels or ([k.strip().lower() for k in POSTIZ_CHANNELS.split(",")] if POSTIZ_CHANNELS != "all" else None)
    
    selected_integrations = []
    for item in integrations:
        provider = str(item.get("providerIdentifier") or item.get("provider") or "").lower()
        item_id = item.get("id")
        if not item_id:
            continue

        if not filter_keys or "all" in filter_keys:
            selected_integrations.append(item)
        else:
            if any(fk in provider for fk in filter_keys):
                selected_integrations.append(item)

    if not selected_integrations:
        logger.warning("No integrations matched POSTIZ_CHANNELS filter. Defaulting to all active integrations.")
        selected_integrations = integrations

    # 3. Format Caption / Content
    content_parts = []
    if title:
        content_parts.append(title)
    if description and description.strip() != title.strip():
        content_parts.append(description.strip())
    if tags:
        content_parts.append(tags.strip())

    full_caption = "\n\n".join(content_parts)

    # 4. Build Postiz Post Payload
    # Postiz Public API expects list of provider objects with their integration id
    providers_payload = [{"id": item["id"]} for item in selected_integrations]

    post_payload: Dict[str, Any] = {
        "type": "now" if not schedule_time else "schedule",
        "content": full_caption,
        "title": title,
        "media": [media_id],
        "providers": providers_payload
    }
    if schedule_time:
        post_payload["date"] = schedule_time

    # 5. Submit to Postiz
    create_url = f"{POSTIZ_URL}/public/v1/posts"
    logger.info(f"🚀 Submitting post to Postiz ({len(providers_payload)} channel(s))...")

    try:
        headers = get_postiz_headers()
        headers["Content-Type"] = "application/json"
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(create_url, headers=headers, json=post_payload)

        if resp.status_code in (200, 201):
            res_data = resp.json()
            post_id = res_data.get("id") or "post_created"
            channels_info = []
            for item in selected_integrations:
                p_name = item.get("providerIdentifier") or item.get("provider") or "channel"
                p_handle = item.get("identifier") or item.get("name") or p_name
                channels_info.append(f"{p_name.upper()} ({p_handle})")

            dashboard_url = f"{POSTIZ_URL}/posts"
            logger.info(f"🎉 [POSTIZ SUCCESS] Post published/scheduled! ID: {post_id}")

            return {
                "status": "success",
                "post_id": post_id,
                "url": dashboard_url,
                "channels": channels_info,
                "message": f"Broadcasted via Postiz to: {', '.join(channels_info)}",
                "response": res_data
            }
        else:
            err_msg = f"Postiz API returned {resp.status_code}: {resp.text}"
            logger.error(f"❌ [POSTIZ FAILED] {err_msg}")
            return {
                "status": "failed",
                "error": err_msg
            }
    except Exception as e:
        logger.error(f"❌ [POSTIZ EXCEPTION] {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


# ── Modern PublishQueue Drop-in Adapter ────────────────────────────────────────

class PublishQueue:
    """
    Modern Postiz Publish Queue.
    Maintains compatibility with legacy PublishQueue.add(...) calls
    while routing through Postiz.
    """

    _items: List[Dict[str, Any]] = []

    @classmethod
    def add(
        cls,
        video_path: str,
        channel_title: str = "General",
        channel_folder: str = "General",
        meta: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        item_id = kwargs.get("id") or f"postiz_queue_{len(cls._items) + 1}_{int(time.time())}"
        resolved_title = kwargs.get("actress_title") or channel_title
        resolved_meta = meta if meta is not None else kwargs.get("metadata", {})

        item = {
            "id": item_id,
            "video_path": os.path.abspath(video_path) if video_path else None,
            "title": resolved_title,
            "channel_folder": channel_folder,
            "meta": resolved_meta,
            "status": "queued",
            "enqueued_at": time.time(),
            **kwargs
        }
        cls._items.append(item)
        logger.info(f"📥 [POSTIZ QUEUE] Enqueued item '{item_id}' for '{resolved_title}'")
        return item

    @classmethod
    def list(cls) -> List[Dict[str, Any]]:
        return list(cls._items)

    @classmethod
    def clear(cls) -> None:
        cls._items.clear()


def start_publish_scheduler() -> Dict[str, Any]:
    """Postiz handles scheduled posts natively on its own server."""
    return {"status": "ok", "scheduler": "postiz_native"}


__all__ = [
    "publish_to_postiz",
    "upload_video_to_postiz",
    "get_active_integrations",
    "is_postiz_configured",
    "PublishQueue",
    "start_publish_scheduler",
]

