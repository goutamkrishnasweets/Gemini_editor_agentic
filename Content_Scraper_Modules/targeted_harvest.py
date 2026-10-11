"""
Content_Scraper_Modules/targeted_harvest.py
===========================================
Targeted Harvest Module for Specific ID Lists

Scrapes content from specific account IDs with configurable:
- Number of clips per ID (default: 11)
- Skip first N clips (default: 3, for pinned posts)
- Process remaining clips through AMTCE pipeline
- Publish 10 minutes before scheduled time
- Time range selection (today/week/month/year)

Author: AMTCE Targeted Harvest v1.0
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("targeted_harvest")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_CLIPS_PER_ID = 5
SKIP_FIRST_N = 3  # Skip pinned posts
PUBLISH_ADVANCE_MINUTES = 10  # Publish 10 minutes before scheduled time

# Time range configurations
TIME_RANGES = {
    "today": timedelta(days=1),
    "week": timedelta(weeks=1),
    "month": timedelta(days=30),
    "year": timedelta(days=365)
}

# Scheduled publish times from .env
PUBLISH_TIMES = ["06:00", "19:00"]


class TargetedHarvest:
    """
    Targeted harvest for specific account IDs.
    """
    
    def __init__(self, target_ids: List[str], time_range: str = "today"):
        """
        Initialize targeted harvest.
        
        Args:
            target_ids: List of account IDs/handles to harvest
            time_range: Time range for scheduling (today/week/month/year)
        """
        self.target_ids = target_ids
        self.time_range = time_range
        self.clips_per_id = DEFAULT_CLIPS_PER_ID
        self.skip_first = SKIP_FIRST_N
        
        if time_range not in TIME_RANGES:
            raise ValueError(f"Invalid time_range: {time_range}. Must be one of: {list(TIME_RANGES.keys())}")
        
        logger.info(f"🎯 [TARGETED HARVEST] Initialized for {len(target_ids)} IDs, range: {time_range}")
    
    def calculate_next_publish_time(self) -> datetime:
        """
        Calculate the next publish time based on current time and scheduled slots.
        Returns time 10 minutes before the scheduled slot.
        """
        now = datetime.utcnow()
        
        # Get publish times from env if available
        env_times = os.getenv("AUTO_INPUT_SCHEDULE_TIMES") or os.getenv("PUBLISH_STATIC_TIMES") or "06:00,19:00"
        scheduled_times = [t.strip() for t in env_times.split(",")]
        
        # Find next scheduled time
        for time_str in scheduled_times:
            hour, minute = map(int, time_str.split(":"))
            scheduled = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            if scheduled > now:
                # Publish 10 minutes before scheduled time
                publish_time = scheduled - timedelta(minutes=PUBLISH_ADVANCE_MINUTES)
                logger.info(f"📅 [TARGETED HARVEST] Next publish: {publish_time} (10 min before {time_str})")
                return publish_time
        
        # If no more slots today, schedule for first slot tomorrow
        first_time = scheduled_times[0]
        hour, minute = map(int, first_time.split(":"))
        tomorrow = now + timedelta(days=1)
        scheduled = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
        publish_time = scheduled - timedelta(minutes=PUBLISH_ADVANCE_MINUTES)
        
        logger.info(f"📅 [TARGETED HARVEST] Next publish: {publish_time} (tomorrow 10 min before {first_time})")
        return publish_time
    
    async def harvest_target_ids(self) -> Dict[str, Any]:
        """
        Harvest clips from target IDs using Apify.
        
        Returns:
            Dict with harvest results
        """
        logger.info(f"🎯 [TARGETED HARVEST] Starting harvest for {len(self.target_ids)} IDs")
        
        results = {
            "target_ids": self.target_ids,
            "clips_per_id": self.clips_per_id,
            "skip_first": self.skip_first,
            "harvested": [],
            "skipped": [],
            "errors": []
        }
        
        try:
            from Downloader_Modules.apify_downloader import apify_scrape_creator_accounts
            
            for target_id in self.target_ids:
                logger.info(f"📥 [TARGETED HARVEST] Harvesting from {target_id}")
                
                try:
                    # Scrape clips from this ID
                    reels = apify_scrape_creator_accounts(
                        creator_name=target_id,
                        source_accounts=[target_id],
                        limit_per_account=self.clips_per_id,
                        bypass_cooldown=True,
                    )
                    
                    if not reels:
                        logger.warning(f"⚠️ [TARGETED HARVEST] No reels found for {target_id}")
                        results["errors"].append(f"No reels found for {target_id}")
                        continue
                    
                    # Skip first N (likely pinned)
                    usable_reels = reels[self.skip_first:]
                    skipped = reels[:self.skip_first]
                    
                    logger.info(f"✅ [TARGETED HARVEST] {target_id}: {len(usable_reels)} usable, {len(skipped)} skipped")
                    
                    results["harvested"].extend(usable_reels)
                    results["skipped"].extend(skipped)
                    
                except Exception as e:
                    logger.error(f"❌ [TARGETED HARVEST] Error harvesting {target_id}: {e}")
                    results["errors"].append(f"Error harvesting {target_id}: {str(e)}")
            
            logger.info(f"🎯 [TARGETED HARVEST] Complete: {len(results['harvested'])} total clips harvested")
            
        except ImportError:
            logger.error("❌ [TARGETED HARVEST] apify_downloader not available")
            results["errors"].append("apify_downloader module not available")
        
        return results
    
    async def process_through_pipeline(self, reels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process harvested reels through AMTCE pipeline.
        
        Args:
            reels: List of reel metadata dicts from Apify
        
        Returns:
            Dict with processing results
        """
        logger.info(f"🔄 [TARGETED HARVEST] Processing {len(reels)} reels through pipeline")
        
        results = {
            "processed": [],
            "failed": [],
            "errors": []
        }
        
        try:
            from Downloader_Modules.downloader_main import run_phase1_ingestion
            
            for reel in reels:
                try:
                    url = reel.get("url") or reel.get("videoUrl")
                    if not url:
                        logger.warning(f"⚠️ [TARGETED HARVEST] No URL in reel: {reel.get('shortcode')}")
                        continue
                    
                    logger.info(f"🔄 [TARGETED HARVEST] Processing: {url}")
                    
                    # Run through Phase 1 pipeline
                    result = await asyncio.to_thread(
                        run_phase1_ingestion,
                        mode="manual",
                        url=url
                    )
                    
                    if result.get("success"):
                        results["processed"].append({
                            "url": url,
                            "shortcode": reel.get("shortcode"),
                            "result": result
                        })
                        logger.info(f"✅ [TARGETED HARVEST] Processed: {reel.get('shortcode')}")
                    else:
                        results["failed"].append({
                            "url": url,
                            "shortcode": reel.get("shortcode"),
                            "error": result.get("error")
                        })
                        logger.warning(f"⚠️ [TARGETED HARVEST] Failed: {reel.get('shortcode')}")
                
                except Exception as e:
                    logger.error(f"❌ [TARGETED HARVEST] Error processing reel: {e}")
                    results["errors"].append(str(e))
        
        except ImportError:
            logger.error("❌ [TARGETED HARVEST] downloader_main not available")
            results["errors"].append("downloader_main module not available")
        
        return results
    
    async def schedule_publish(self, processed_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Schedule processed items for publishing.
        
        Args:
            processed_items: List of processed reel items
        
        Returns:
            Dict with scheduling results
        """
        logger.info(f"📅 [TARGETED HARVEST] Scheduling {len(processed_items)} items for publish")
        
        publish_time = self.calculate_next_publish_time()
        
        results = {
            "scheduled_at": publish_time.isoformat(),
            "items": [],
            "errors": []
        }
        
        try:
            from Publishing_Modules.queue_publisher import PublishQueue
            
            for item in processed_items:
                try:
                    shortcode = item.get("shortcode")
                    result = item.get("result", {})
                    downloaded_files = result.get("downloaded_files", [])
                    
                    if downloaded_files:
                        video_path = downloaded_files[0] if isinstance(downloaded_files, list) else downloaded_files
                        
                        # Add to publish queue
                        PublishQueue.add(
                            video_path=video_path,
                            niche="targeted_harvest",
                            shortcode=shortcode,
                            scheduled_for=publish_time.isoformat()
                        )
                        
                        results["items"].append({
                            "shortcode": shortcode,
                            "video_path": video_path,
                            "scheduled_for": publish_time.isoformat()
                        })
                        
                        logger.info(f"📅 [TARGETED HARVEST] Scheduled {shortcode} for {publish_time}")
                
                except Exception as e:
                    logger.error(f"❌ [TARGETED HARVEST] Error scheduling {item.get('shortcode')}: {e}")
                    results["errors"].append(str(e))
        
        except ImportError:
            logger.error("❌ [TARGETED HARVEST] queue_publisher not available")
            results["errors"].append("queue_publisher module not available")
        
        return results
    
    async def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run complete targeted harvest pipeline:
        1. Harvest from target IDs
        2. Process through AMTCE pipeline
        3. Schedule for publishing
        
        Returns:
            Dict with complete pipeline results
        """
        logger.info(f"🚀 [TARGETED HARVEST] Starting full pipeline for {len(self.target_ids)} IDs")
        
        # Step 1: Harvest
        harvest_results = await self.harvest_target_ids()
        
        if not harvest_results["harvested"]:
            logger.warning("⚠️ [TARGETED HARVEST] No clips harvested, aborting pipeline")
            return {
                "success": False,
                "stage": "harvest",
                "results": harvest_results
            }
        
        # Step 2: Process through pipeline
        process_results = await self.process_through_pipeline(harvest_results["harvested"])
        
        if not process_results["processed"]:
            logger.warning("⚠️ [TARGETED HARVEST] No clips processed, aborting pipeline")
            return {
                "success": False,
                "stage": "process",
                "harvest_results": harvest_results,
                "process_results": process_results
            }
        
        # Step 3: Schedule for publish
        schedule_results = await self.schedule_publish(process_results["processed"])
        
        logger.info(f"✅ [TARGETED HARVEST] Pipeline complete: {len(schedule_results['items'])} items scheduled")
        
        return {
            "success": True,
            "harvest_results": harvest_results,
            "process_results": process_results,
            "schedule_results": schedule_results
        }


# ── Helper Functions ─────────────────────────────────────────────────────────

def format_harvest_summary(results: Dict[str, Any]) -> str:
    """Format harvest results for display."""
    lines = [
        "🎯 *Targeted Harvest Summary*\n",
        f"Target IDs: {', '.join(results.get('target_ids', []))}",
        f"Clips per ID: {results.get('clips_per_id', 0)}",
        f"Skip First: {results.get('skip_first', 0)}",
        f"Total Harvested: {len(results.get('harvested', []))}",
        f"Total Skipped: {len(results.get('skipped', []))}",
        f"Errors: {len(results.get('errors', []))}"
    ]
    
    if results.get("errors"):
        lines.append(f"\n❌ Errors:")
        for error in results["errors"][:5]:  # Show first 5 errors
            lines.append(f"  • {error}")
    
    return "\n".join(lines)


def format_pipeline_summary(results: Dict[str, Any]) -> str:
    """Format full pipeline results for display."""
    if not results.get("success"):
        return f"❌ Pipeline failed at stage: {results.get('stage')}"
    
    harvest = results.get("harvest_results", {})
    process = results.get("process_results", {})
    schedule = results.get("schedule_results", {})
    
    lines = [
        "✅ *Targeted Harvest Pipeline Complete*\n",
        f"📥 Harvested: {len(harvest.get('harvested', []))} clips",
        f"🔄 Processed: {len(process.get('processed', []))} clips",
        f"📅 Scheduled: {len(schedule.get('items', []))} clips",
        f"⏰ Publish Time: {schedule.get('scheduled_at', 'N/A')}"
    ]
    
    return "\n".join(lines)


# ── CLI Entry Point ───────────────────────────────────────────────────────────

async def main():
    """CLI entry point for testing."""
    import sys
    
    # Example usage
    target_ids = sys.argv[1:] if len(sys.argv) > 1 else ["actress1", "actress2"]
    time_range = "today"
    
    harvester = TargetedHarvest(target_ids, time_range)
    results = await harvester.run_full_pipeline()
    
    print(format_pipeline_summary(results))


if __name__ == "__main__":
    asyncio.run(main())
