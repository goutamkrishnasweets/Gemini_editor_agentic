"""
Phase 3 — Step 06: YouTube Shorts Publisher (Direct YouTube Data API v3)
========================================================================
Handles publishing short-form vertical video content to YouTube Shorts via YouTube Data API.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("phase3.step06_youtube_publisher")


def publish_to_youtube_shorts(
    video_path: str,
    title: str,
    description: str,
    tags: Optional[List[str]] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Publish vertical video to YouTube Shorts via YouTube Data API.
    """
    if not os.path.exists(video_path):
        return {"status": "error", "error": f"Video file not found: {video_path}"}

    try:
        from Publishing_Modules.media_publisher_main import publish_to_youtube
        tag_str = " ".join(tags) if tags else ""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                res = pool.submit(
                    asyncio.run,
                    publish_to_youtube(video_path, title=title, description=description, tags=tag_str, user_id=user_id)
                ).result()
        else:
            res = loop.run_until_complete(
                publish_to_youtube(video_path, title=title, description=description, tags=tag_str, user_id=user_id)
            )

        logger.info("✅ Step 06 YouTube Publisher: Dispatched to YouTube Data API for YouTube Shorts")
        return {"status": "published", "platform": "youtube_shorts", "response": res}
    except Exception as e:
        logger.error(f"❌ Step 06 YouTube Publisher failed: {e}")
        return {"status": "error", "platform": "youtube_shorts", "error": str(e)}
