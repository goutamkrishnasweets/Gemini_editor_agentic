"""
Phase 3 — Step 05: TikTok Publisher (Buffer Multi-Channel API)
==============================================================
Handles publishing short-form video content to TikTok via Buffer API.
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger("phase3.step05_tiktok_publisher")


def publish_to_tiktok(
    video_path: str,
    title: str,
    caption: str,
    user_id: str = None
) -> Dict[str, Any]:
    """
    Publish video to TikTok platform via Buffer API.
    """
    if not os.path.exists(video_path):
        return {"status": "error", "error": f"Video file not found: {video_path}"}

    try:
        from Publishing_Modules.buffer_publisher import publish_to_buffer, is_buffer_configured
        if is_buffer_configured(user_id=user_id):
            res = publish_to_buffer(
                video_path=video_path,
                title=title,
                description=caption,
                channels=["tiktok"],
                user_id=user_id
            )
            logger.info("✅ Step 05 TikTok Publisher: Dispatched to Buffer API for TikTok")
            return {"status": "published", "platform": "tiktok", "response": res}
        else:
            return {
                "status": "simulated",
                "platform": "tiktok",
                "message": "Buffer API not configured. Run /setbuffer or set BUFFER_ACCESS_TOKEN."
            }
    except Exception as e:
        logger.error(f"❌ Step 05 TikTok Publisher failed: {e}")
        return {"status": "error", "platform": "tiktok", "error": str(e)}
