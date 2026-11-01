"""
Phase 3 — Step 04: Meta Reels Publisher (Postiz Routing)
=========================================================
Handles publishing short-form vertical video content to Instagram & Facebook Reels via Postiz.
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger("phase3.step04_meta_publisher")


def publish_to_meta_reels(
    video_path: str,
    caption: str,
    target_platform: str = "instagram"
) -> Dict[str, Any]:
    """
    Publish vertical video to Instagram Reels or Facebook Reels via Postiz.
    """
    if not os.path.exists(video_path):
        return {"status": "error", "error": f"Video file not found: {video_path}"}

    try:
        from Publishing_Modules.postiz_publisher import publish_to_postiz, is_postiz_configured
        if is_postiz_configured():
            res = publish_to_postiz(
                video_path=video_path,
                title=caption[:80],
                description=caption,
                target_channels=["instagram", "facebook"]
            )
            logger.info(f"✅ Step 04 Meta Publisher: Dispatched to Postiz for {target_platform}")
            return {"status": "published", "platform": target_platform, "response": res}
        else:
            return {
                "status": "simulated",
                "platform": target_platform,
                "message": "Postiz not configured. Please set POSTIZ_URL and POSTIZ_API_KEY."
            }
    except Exception as e:
        logger.error(f"❌ Step 04 Meta Publisher failed: {e}")
        return {"status": "error", "platform": target_platform, "error": str(e)}
