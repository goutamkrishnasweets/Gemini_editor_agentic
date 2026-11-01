"""
Phase 3 — Step 01: Queue Ingest Manager
========================================
Ingests master rendered video files from Phase 2 into publish_queue.json
and initializes their publishing queue status.
"""

import os
import logging
from typing import Dict, Any, Optional
from Publishing_Modules.postiz_publisher import PublishQueue

logger = logging.getLogger("phase3.step01_queue_ingest")

def ingest_to_publish_queue(
    video_path: str,
    title: Optional[str] = None,
    channel_folder: str = "General",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Ingest a rendered video file into publish_queue.json.

    Args:
        video_path: Absolute path to rendered master MP4.
        title: Optional custom title for publishing.
        channel_folder: Creator / category channel name.
        metadata: Optional visual/audio intelligence dict.

    Returns:
        Dict containing queue item details and success status.
    """
    if not os.path.exists(video_path):
        logger.error(f"❌ Step 01 Queue Ingest: Video file not found: {video_path}")
        return {"status": "error", "error": f"Video not found: {video_path}"}

    clean_title = title or os.path.splitext(os.path.basename(video_path))[0]
    
    try:
        try:
            item = PublishQueue.add(
                video_path=video_path,
                channel_title=clean_title,
                channel_folder=channel_folder,
                meta=metadata or {}
            )
        except TypeError:
            # Fallback for environments or shims where PublishQueue.add does not accept 'meta'
            try:
                item = PublishQueue.add(
                    video_path=video_path,
                    channel_title=clean_title,
                    channel_folder=channel_folder
                )
            except TypeError:
                item = PublishQueue.add(video_path=video_path)
            if isinstance(item, dict) and metadata:
                item["meta"] = metadata

        item_id = item.get('id', 'enqueued') if isinstance(item, dict) else 'enqueued'
        logger.info(f"✅ Step 01 Queue Ingest: Enqueued '{clean_title}' -> queue item {item_id}")
        return {
            "status": "success",
            "queue_item": item,
            "video_path": video_path,
            "title": clean_title,
            "channel_folder": channel_folder
        }
    except Exception as e:
        logger.error(f"❌ Step 01 Queue Ingest error: {e}")
        return {"status": "error", "error": str(e), "video_path": video_path}
