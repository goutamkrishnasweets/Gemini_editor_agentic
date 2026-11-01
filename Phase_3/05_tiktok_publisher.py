"""
Phase 3 — Step 05: TikTok Publisher (Make.com Routing)
======================================================
Handles publishing short-form video content to TikTok via Make.com.
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger("phase3.step05_tiktok_publisher")


def publish_to_tiktok(
    video_path: str,
    title: str,
    caption: str
) -> Dict[str, Any]:
    """
    Publish video to TikTok platform via Make.com.
    """
    if not os.path.exists(video_path):
        return {"status": "error", "error": f"Video file not found: {video_path}"}

    try:
        from Publishing_Modules.make_publisher import publish_to_make, is_make_configured
        if is_make_configured():
            res = publish_to_make(
                media_path=video_path,
                title=title,
                caption=caption,
                post_type="reel"
            )
            logger.info("✅ Step 05 TikTok Publisher: Dispatched to Make.com for TikTok")
            return {"status": "published", "platform": "tiktok", "response": res}
        else:
            return {
                "status": "simulated",
                "platform": "tiktok",
                "message": "Make.com not configured. Run /setmakewebhookurl or set MAKE_WEBHOOK_URL."
            }
    except Exception as e:
        logger.error(f"❌ Step 05 TikTok Publisher failed: {e}")
        return {"status": "error", "platform": "tiktok", "error": str(e)}
