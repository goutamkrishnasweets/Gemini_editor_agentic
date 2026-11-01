"""
Phase 3 — Step 06: YouTube Shorts Publisher (Postiz Routing)
============================================================
Handles publishing short-form vertical video content to YouTube Shorts via Postiz.
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
    Publish vertical video to YouTube Shorts via Postiz.
    """
    if not os.path.exists(video_path):
        return {"status": "error", "error": f"Video file not found: {video_path}"}

    try:
        from Publishing_Modules.postiz_publisher import publish_to_postiz, is_postiz_configured
        if is_postiz_configured():
            tag_str = " ".join(tags) if tags else ""
            res = publish_to_postiz(
                video_path=video_path,
                title=title,
                description=description,
                tags=tag_str,
                target_channels=["youtube"]
            )
            logger.info("✅ Step 06 YouTube Publisher: Dispatched to Postiz for YouTube Shorts")
            return {"status": "published", "platform": "youtube_shorts", "response": res}
        else:
            return {
                "status": "simulated",
                "platform": "youtube_shorts",
                "message": "Postiz not configured. Please set POSTIZ_URL and POSTIZ_API_KEY."
            }
    except Exception as e:
        logger.error(f"❌ Step 06 YouTube Publisher failed: {e}")
        return {"status": "error", "platform": "youtube_shorts", "error": str(e)}
