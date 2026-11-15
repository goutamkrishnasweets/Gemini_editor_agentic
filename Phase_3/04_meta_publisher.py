"""
Phase 3 — Step 04: Meta Reels Publisher (Direct Meta Graph API)
================================================================
Handles publishing short-form vertical video content to Instagram & Facebook Reels via Meta Graph API.
"""

import os
import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger("phase3.step04_meta_publisher")


def publish_to_meta_reels(
    video_path: str,
    caption: str,
    target_platform: str = "instagram",
    user_id: str = None
) -> Dict[str, Any]:
    """
    Publish vertical video to Instagram Reels or Facebook Reels via Meta Graph API.
    """
    if not os.path.exists(video_path):
        return {"status": "error", "error": f"Video file not found: {video_path}"}

    try:
        from Publishing_Modules.media_publisher_main import publish_to_meta
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                res = pool.submit(asyncio.run, publish_to_meta(video_path, caption=caption, user_id=user_id)).result()
        else:
            res = loop.run_until_complete(publish_to_meta(video_path, caption=caption, user_id=user_id))

        logger.info(f"✅ Step 04 Meta Publisher: Dispatched to Meta Graph API for {target_platform}")
        return {"status": "published", "platform": target_platform, "response": res}
    except Exception as e:
        logger.error(f"❌ Step 04 Meta Publisher failed: {e}")
        return {"status": "error", "platform": target_platform, "error": str(e)}
