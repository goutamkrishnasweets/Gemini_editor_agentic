"""
phase1_orchestrator.py — Phase 1 Master Pipeline Orchestrator
===============================================================
Coordinates the complete Phase 1 ingestion flow across indexed steps:
  Step 1 (01_source_config.py)   -> Resolve target account pool
  Step 2 (02_dedup_ledger.py)    -> Check deduplication ledger
  Step 3 (03_apify_harvester.py) -> Apify scrape & Gemini pre-screen
  Step 4 (04_core_downloader.py) -> Multi-platform stream downloader
  Step 5 (05_proxy_encoder.py)   -> 480p proxy encoder
  Step 6 (06_audio_extractor.py) -> Mono 16kHz WAV audio extractor
  Step 7 (07_beat_analyzer.py)   -> BeatEngine rhythm & drop analyzer

Supports optional `event_callback(step_id, status, data)` for live WebSockets status tracking.
"""

import os
import sys
import time
import json
import logging
import importlib
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger("Phase1.Orchestrator")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import indexed step modules
spec_01 = importlib.import_module("Phase_1.01_source_config")
resolve_target_accounts = spec_01.resolve_target_accounts

spec_02 = importlib.import_module("Phase_1.02_dedup_ledger")
check_deduplication = spec_02.check_deduplication

spec_03 = importlib.import_module("Phase_1.03_apify_harvester")
harvest_reels_from_apify = spec_03.harvest_reels_from_apify

spec_04 = importlib.import_module("Phase_1.04_core_downloader")
download_stream = spec_04.download_stream

spec_05 = importlib.import_module("Phase_1.05_proxy_encoder")
encode_proxy_video = spec_05.encode_proxy_video

spec_06 = importlib.import_module("Phase_1.06_audio_extractor")
extract_clip_audio = spec_06.extract_clip_audio

spec_07 = importlib.import_module("Phase_1.07_beat_analyzer")
analyze_rhythm_and_beats = spec_07.analyze_rhythm_and_beats

try:
    from Import_Modules.tracker_notifier import notify_tracker
except ImportError:
    notify_tracker = None


def run_phase1_pipeline(
    mode: str = "auto",
    url: Optional[str] = None,
    limit_per_account: Optional[int] = None,
    target_accounts: Optional[List[str]] = None,
    platform: str = "instagram",
    event_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Executes Phase 1 Pipeline through indexed steps 01 -> 07.
    """
    if event_callback is None and notify_tracker is not None:
        event_callback = notify_tracker

    logger.info(f"🚀 [PHASE 1 ORCHESTRATOR] Starting Phase 1 Ingestion Pipeline (mode='{mode}', platform='{platform}')")
    downloads_dir = os.path.join(_REPO_ROOT, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    downloaded_files: List[str] = []

    # ── WORKER 2: Manual Input Downloader (Direct URL / Any Platform) ─────────
    if mode == "manual" or url:
        if not url:
            raise ValueError("Manual mode requires a valid video URL (--url).")

        logger.info(f"👤 [WORKER 2 - MANUAL] Ingesting video URL: {url}")
        try:
            import re
            m_ig = re.search(r"/(?:p|reel|reels)/([A-Za-z0-9_-]+)", url)
            shortcode = m_ig.group(1) if m_ig else f"manual_{int(time.time())}"
            owner = "manual"

            # Step 1: Config
            resolve_target_accounts(target_accounts=[owner], max_limit=1, callback=event_callback)

            # Step 2: Deduplication Check & Local Disk Verification
            dedup_info = check_deduplication(shortcode, owner=owner, downloads_dir=downloads_dir, callback=event_callback)
            clip_dir = dedup_info["clip_dir"]
            video_path = dedup_info.get("video_path")

            # In Manual mode, the user specifically requested this video.
            # If it already exists on local disk with content, reuse the local media.
            # If NOT on local disk, ALWAYS download it regardless of past vault/pool records.
            if dedup_info.get("already_on_disk") and video_path and os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                logger.info(f"📂 [WORKER 2 - CACHE HIT] Clip '{shortcode}' already exists locally at '{video_path}'. Reusing local media.")
            else:
                # Step 3: Harvester (Skipped in Manual Mode)
                if event_callback:
                    event_callback("step_03", "success", {"message": "Manual URL mode: Skipping Apify scraper step."})

                # Step 4: Download Video Stream
                meta = {"shortcode": shortcode, "url": url, "ownerUsername": owner, "platform": platform}
                dl_info = download_stream(url, clip_dir, metadata=meta, callback=event_callback)
                video_path = dl_info.get("video_path")

            if video_path and os.path.exists(video_path):
                downloaded_files.append(os.path.abspath(video_path))

                # Step 5: Proxy Video Encoder
                encode_proxy_video(video_path, callback=event_callback)

                # Step 6: Audio Extractor
                extract_clip_audio(video_path, callback=event_callback)

                # Step 7: Beat Analyzer
                analyze_rhythm_and_beats(video_path, callback=event_callback)

            return {
                "success": len(downloaded_files) > 0,
                "mode": "manual",
                "count": len(downloaded_files),
                "downloaded_files": downloaded_files,
                "downloads_dir": downloads_dir,
                "clip_dir": clip_dir,
                "error": None if downloaded_files else f"Failed to download or locate video stream for '{shortcode}'."
            }

        except Exception as err:
            logger.error(f"❌ [WORKER 2 FAILED] Manual download error: {err}")
            if event_callback:
                event_callback("phase1", "failed", {"error": str(err)})
            return {"success": False, "mode": "manual", "downloaded_files": [], "error": str(err)}

    # ── WORKER 1: Automated Account Harvester ─────────────────────────────────
    else:
        logger.info("🤖 [WORKER 1 - AUTOMATED] Executing Step 1 through Step 7...")

        # Step 1: Target Source Config
        cfg_res = resolve_target_accounts(target_accounts=target_accounts, max_limit=2, callback=event_callback)
        sources = cfg_res.get("accounts", [])

        # Step 3: Apify Harvest & Pre-screen
        limit = limit_per_account if limit_per_account else int(os.getenv("AUTO_SCRAPE_LIMIT", "8"))
        should_bypass = bool(target_accounts)
        reels = harvest_reels_from_apify(
            sources,
            limit_per_account=limit,
            bypass_cooldown=should_bypass,
            callback=event_callback
        )

        for item in (reels or []):
            if isinstance(item, dict):
                shortcode = item.get("shortcode") or f"clip_{int(time.time())}"
                owner = item.get("ownerUsername") or item.get("uploader") or "actress"
                video_url = item.get("videoUrl") or item.get("url") or f"https://www.instagram.com/reel/{shortcode}/"

                # Step 2: Deduplication check per clip
                dedup_info = check_deduplication(shortcode, owner=owner, downloads_dir=downloads_dir, callback=event_callback)
                if dedup_info.get("is_duplicate"):
                    logger.info(f"♻️ [DEDUP SKIP] Reel '{shortcode}' already processed/published. Skipping stream download & Phase 1 processing.")
                    continue
                clip_dir = dedup_info["clip_dir"]

                # Step 4: Download Video Stream
                dl_info = download_stream(video_url, clip_dir, metadata=item, callback=event_callback)
                video_path = dl_info.get("video_path")

                if video_path and os.path.exists(video_path):
                    downloaded_files.append(os.path.abspath(video_path))

                    # Step 5: Proxy Video Encoder
                    encode_proxy_video(video_path, callback=event_callback)

                    # Step 6: Audio Extractor
                    extract_clip_audio(video_path, callback=event_callback)

                    # Step 7: Beat Analyzer
                    analyze_rhythm_and_beats(video_path, callback=event_callback)

        return {
            "success": True,
            "mode": "auto",
            "count": len(downloaded_files),
            "downloaded_files": downloaded_files,
            "downloads_dir": downloads_dir
        }


# Backwards compatibility alias
run_phase1_ingestion = run_phase1_pipeline
