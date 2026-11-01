"""
Phase 3 — Step 05: TikTok Publisher (Postiz Routing)
====================================================
Handles publishing short-form video content to TikTok via Postiz.
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
    Publish video to TikTok platform via Postiz.
    """
    if not os.path.exists(video_path):
        return {"status": "error", "error": f"Video file not found: {video_path}"}

    try:
        from Publishing_Modules.postiz_publisher import publish_to_postiz, is_postiz_configured
        if is_postiz_configured():
            res = publish_to_postiz(
                video_path=video_path,
                title=title,
                description=caption,
                target_channels=["tiktok"]
            )
            logger.info("✅ Step 05 TikTok Publisher: Dispatched to Postiz for TikTok")
            return {"status": "published", "platform": "tiktok", "response": res}
        else:
            return {
                "status": "simulated",
                "platform": "tiktok",
                "message": "Postiz not configured. Please set POSTIZ_URL and POSTIZ_API_KEY."
            }
    except Exception as e:
        logger.error(f"❌ Step 05 TikTok Publisher failed: {e}")
        return {"status": "error", "platform": "tiktok", "error": str(e)}
