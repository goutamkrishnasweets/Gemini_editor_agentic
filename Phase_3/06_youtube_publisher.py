"""
Phase 3 — Step 06: YouTube Shorts Publisher (Make.com Routing)
==============================================================
Handles publishing short-form vertical video content to YouTube Shorts via Make.com.
"""

import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("phase3.step06_youtube_publisher")


def publish_to_youtube_shorts(
    video_path: str,
    title: str,
    description: str,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Publish vertical video to YouTube Shorts via Make.com.
    """
    if not os.path.exists(video_path):
        return {"status": "error", "error": f"Video file not found: {video_path}"}

    try:
        from Publishing_Modules.make_publisher import publish_to_make, is_make_configured
        if is_make_configured():
            tag_str = " ".join(tags) if tags else ""
            res = publish_to_make(
                media_path=video_path,
                title=title,
                caption=description,
                tags=tag_str,
                post_type="reel"
            )
            logger.info("✅ Step 06 YouTube Publisher: Dispatched to Make.com for YouTube Shorts")
            return {"status": "published", "platform": "youtube_shorts", "response": res}
        else:
            return {
                "status": "simulated",
                "platform": "youtube_shorts",
                "message": "Make.com not configured. Run /setmakewebhookurl or set MAKE_WEBHOOK_URL."
            }
    except Exception as e:
        logger.error(f"❌ Step 06 YouTube Publisher failed: {e}")
        return {"status": "error", "platform": "youtube_shorts", "error": str(e)}
