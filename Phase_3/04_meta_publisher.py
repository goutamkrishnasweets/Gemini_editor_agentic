"""
Phase 3 — Step 04: Meta Reels Publisher (Make.com Routing)
==========================================================
Handles publishing short-form vertical video content to Instagram & Facebook Reels via Make.com.
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
    Publish vertical video to Instagram Reels or Facebook Reels via Make.com.
    """
    if not os.path.exists(video_path):
        return {"status": "error", "error": f"Video file not found: {video_path}"}

    try:
        from Publishing_Modules.make_publisher import publish_to_make, is_make_configured
        if is_make_configured():
            res = publish_to_make(
                media_path=video_path,
                title=caption[:80],
                caption=caption,
                post_type="reel"
            )
            logger.info(f"✅ Step 04 Meta Publisher: Dispatched to Make.com for {target_platform}")
            return {"status": "published", "platform": target_platform, "response": res}
        else:
            return {
                "status": "simulated",
                "platform": target_platform,
                "message": "Make.com not configured. Run /setmakewebhookurl or set MAKE_WEBHOOK_URL."
            }
    except Exception as e:
        logger.error(f"❌ Step 04 Meta Publisher failed: {e}")
        return {"status": "error", "platform": target_platform, "error": str(e)}
