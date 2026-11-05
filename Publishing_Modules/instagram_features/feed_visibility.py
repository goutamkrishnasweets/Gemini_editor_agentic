"""
Publishing_Modules/instagram_features/feed_visibility.py
========================================================
Instagram Profile Grid Visibility Feature Manager
Kingdom: Instagram Publishing Features
Purpose: Manage share_to_feed toggle. Default is True (shares to both Profile Feed
         Grid and Reels Tab). If /no_grid or /reels_only command is present, sets
         share_to_feed to False (Reels Tab only).

Usage:
    from Publishing_Modules.instagram_features.feed_visibility import FeedVisibilityManager

    mgr = FeedVisibilityManager()
    clean_title, share_to_feed = mgr.extract_feed_visibility(raw_text)
    init_data = mgr.apply_feed_visibility(init_data, share_to_feed)
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Matches /no_grid or /reels_only or /nogrid
_NO_GRID_PATTERN = re.compile(
    r"\s*/(?:no_?grid|reels_?only)\b",
    re.IGNORECASE,
)


class FeedVisibilityManager:
    """
    Manages whether an Instagram Reel is shared to the creator's main profile grid.
    """

    def extract_feed_visibility(self, raw_text: str) -> Tuple[str, bool]:
        """
        Check for /no_grid or /reels_only command.

        Returns:
            (clean_text, share_to_feed: bool)

        Default is True (standard Instagram Reel behavior).
        If /no_grid is detected, returns False and strips the command.
        """
        if not raw_text or not isinstance(raw_text, str):
            return ("", True)

        has_no_grid = bool(_NO_GRID_PATTERN.search(raw_text))
        clean_text = _NO_GRID_PATTERN.sub("", raw_text).strip()
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        share_to_feed = not has_no_grid
        logger.info(
            "[FeedVisibility] Parsed: share_to_feed=%s | clean_text='%s'",
            share_to_feed, clean_text[:60],
        )
        return (clean_text, share_to_feed)

    def apply_feed_visibility(
        self, init_data: Dict[str, Any], share_to_feed: bool
    ) -> Dict[str, Any]:
        """
        Inject share_to_feed into Meta container init_data.
        """
        init_data["share_to_feed"] = share_to_feed
        logger.info("[FeedVisibility] Injected share_to_feed=%s into container init_data.", share_to_feed)
        return init_data
