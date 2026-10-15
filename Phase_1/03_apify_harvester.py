"""
03_apify_harvester.py — Phase 1 Step 3: Apify Reel Scraper & Pre-screener
==========================================================================
Queries Apify Instagram scraper actor for target accounts.
Applies:
  - Metadata filtering (views, likes, freshness)
  - Optional Gemini Vision thumbnail pre-screening before full video download
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger("Phase1.Step03")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def harvest_reels_from_apify(
    target_accounts: List[str],
    limit_per_account: int = 3,
    bypass_cooldown: bool = False,
    callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None
) -> List[Dict[str, Any]]:
    """
    Step 3 Execution: Scrapes target reels via Apify actor.
    """
    if callback:
        callback("step_03", "running", {
            "message": f"Scraping top reels via Apify for accounts: {', '.join(target_accounts)}..."
        })

    approved_reels = []
    try:
        from Downloader_Modules.apify_downloader import apify_scrape_actress_accounts
        reels = apify_scrape_actress_accounts(
            "General",
            target_accounts,
            limit_per_account=limit_per_account,
            bypass_cooldown=bypass_cooldown
        )
        if reels:
            approved_reels = reels
            logger.info(f"🤖 [STEP 03] Apify harvested {len(approved_reels)} approved reel candidate(s).")
    except Exception as e:
        logger.error(f"❌ [STEP 03] Apify harvest failed: {e}")
        if callback:
            callback("step_03", "failed", {"message": f"Apify scraper error: {e}"})
        return []

    if callback:
        callback("step_03", "success", {
            "message": f"Harvested {len(approved_reels)} reel candidate(s) from Apify.",
            "count": len(approved_reels),
            "reels": [r.get("shortcode") for r in approved_reels if isinstance(r, dict)]
        })

    return approved_reels
