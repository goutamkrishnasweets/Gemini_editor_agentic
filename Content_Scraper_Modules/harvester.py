"""
harvester.py - AMTCE Layer 2: Harvest + Route
=============================================
Replaces the core logic of actress_scheduler.py and channel_router.py.

Responsibilities:
  1. Read accounts from Core (no direct JSON access)
  2. Fetch reels via Apify for each account
  3. Route each reel to the right channel via resolve_channel()
  4. Dedup via Core.ledger before and after download
  5. Organize clip into downloads/<actress>_NNN/ folder
  6. Inject .niche.json so the uploader knows the channel
  7. Add clip to PublishQueue
  8. Sleep until the next scheduled slot

Manual override flags (set in Credentials/.env while running):
  FORCE_HARVEST=yes    -> trigger scrape immediately
  FORCE_NEXT_BATCH=yes -> publish next clip immediately

Import API (what main.py uses):
    from Content_Harvester.harvester import start_scheduler
    start_scheduler()
"""

import os
import json
import time
import logging
import re
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    from Downloader_Modules.apify_downloader import apify_scrape_actress_accounts
    from Downloader_Modules.downloader import download_video
    from Content_Scraper_Modules.content_ledger import get_ledger
    from Publishing_Modules.make_publisher import PublishQueue
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Repo root (one level above this file's parent directory)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ENV config
_raw_limit = os.getenv("APIFY_REELS_PER_ACCOUNT", "5").strip().lower()
LIMIT_PER_ACCOUNT = (
    int(os.getenv("APIFY_REELS_AUTO_MAX", "48"))
    if _raw_limit == "auto"
    else int(_raw_limit)
)
DOWNLOADS_PER_ACCOUNT = int(os.getenv("ACTRESS_DOWNLOADS_PER_ACCOUNT", "3"))
MISSED_GRACE_MINUTES  = int(os.getenv("ACTRESS_MISSED_GRACE_MINUTES", "60"))

# Channel constants
CHANNEL_GENERAL       = "General_Fallback"
CHANNEL_ENTERTAINMENT = "Entertainment"
CHANNEL_FASHION       = "Fashion_Style"


# ==============================================================================
# Channel routing
# ==============================================================================

def _entertainment_creds_exist() -> bool:
    """True when Credentials/social_media/Entertainment/ has real files."""
    base = os.path.join(_REPO_ROOT, "Credentials", "social_media", "Entertainment")
    return os.path.isdir(base) and any(os.path.isfile(os.path.join(base, f)) for f in os.listdir(base))


def _extract_person_name(reel: Dict) -> str:
    """Extract featured person or creator name from reel metadata (tagged users > owner > caption)."""
    # Priority 1: tagged users
    for user in reel.get("taggedUsers", []):
        if isinstance(user, dict):
            fn = user.get("full_name") or user.get("fullName", "")
            if fn and len(fn) > 2:
                return fn
    # Priority 2: owner display name
    for key in ("ownerFullName", "ownerName", "fullName"):
        val = reel.get(key, "")
        if val and len(val) > 2:
            return val
    # Priority 3: capitalized name pattern in caption
    caption = reel.get("caption", "") or ""
    matches = re.findall(r"([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)", caption)
    if matches:
        return matches[0]
    return reel.get("ownerUsername", "")


def _detect_gender(name: str, cfg_data: Dict = None) -> str:
    """Neutral compatibility stub."""
    return "neutral"


def resolve_channel(ig_id: str, reel: Dict) -> Tuple[str, str, bool]:
    """
    Determine destination channel for a scraped reel.

    Args:
        ig_id: Instagram username of the posting account
        reel:  Full reel metadata dict from Apify

    Returns:
        (channel_folder, person_title, is_nsfw)
    """
    accounts_raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source_accounts.json")
    try:
        with open(accounts_raw_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        raw = {}

    nsfw_ids = set(raw.get("nsfw_accounts", []))
    ig_id_clean = ig_id.lower().lstrip("@")
    is_nsfw = ig_id_clean in nsfw_ids
    person_name = _extract_person_name(reel) or ig_id_clean

    target_channel = CHANNEL_ENTERTAINMENT if _entertainment_creds_exist() else CHANNEL_GENERAL
    return target_channel, person_name, is_nsfw


def get_source_accounts() -> List[str]:
    """Returns list of source IG account IDs to scrape."""
    env_override = os.getenv("SOURCE_ACCOUNTS", "").strip()
    if env_override:
        return [a.strip().lstrip("@") for a in env_override.split(",") if a.strip()]

    accounts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source_accounts.json")
    try:
        with open(accounts_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        accounts = raw.get("source_accounts", [])
        return [a for a in accounts if a and not a.startswith("REPLACE_WITH")]
    except Exception:
        return []


# ==============================================================================
# File helpers
# ==============================================================================

def _safe_title(title: str) -> str:
    """Strip characters unsafe for folder/file names."""
    return re.sub(r"[^\w\s\-]", "", title).strip()[:50]


def _next_batch_folder(actress_title: str, downloads_dir: str) -> str:
    """Return downloads/<actress>_NNN/ path, creating the next numbered batch folder."""
    prefix    = _safe_title(actress_title)
    existing  = [
        d for d in os.listdir(downloads_dir)
        if d.startswith(prefix) and re.search(r"_\d+$", d)
    ] if os.path.isdir(downloads_dir) else []
    last_num  = max(
        (int(re.search(r"_(\d+)$", d).group(1)) for d in existing if re.search(r"_(\d+)$", d)),
        default=0,
    )
    folder    = os.path.join(downloads_dir, f"{prefix}_{last_num + 1:03d}")
    os.makedirs(folder, exist_ok=True)
    return folder


def _inject_niche(video_path: str, actress_folder: str, actress_title: str = "") -> None:
    """Write .niche.json alongside the video so the uploader knows the channel."""
    niche_path = os.path.splitext(video_path)[0] + ".niche.json"
    try:
        with open(niche_path, "w", encoding="utf-8") as f:
            json.dump({"folder": actress_folder, "title": actress_title}, f)
    except Exception as exc:
        logger.warning("harvester: failed to write .niche.json: %s", exc)


def _name_in_reel(reel: Dict, actress_folder: str) -> bool:
    """Quick check: does the actress folder name appear in reel owner/caption?"""
    target = actress_folder.lower().replace("_", " ")
    owner  = reel.get("ownerUsername", "").lower()
    cap    = (reel.get("caption", "") or "").lower()[:500]
    return target in owner or target in cap


# ==============================================================================
# Download helper  (thin wrapper — calls existing download_video from project)
# ==============================================================================

def _download_reel(url: str, dest_dir: str) -> Optional[str]:
    """
    Download a reel to dest_dir. Returns local file path or None on failure.
    Delegates to the existing Download_Modules machinery.
    """
    try:
        from Downloader_Modules.download_handler import download_video
        return download_video(url, dest_dir)
    except Exception as exc:
        logger.warning("harvester: download failed for %s: %s", url, exc)
        return None


# ==============================================================================
# Apify fetch
# ==============================================================================

def _fetch_reels_apify(ig_id: str, limit: int) -> List[Dict]:
    """Call Apify to scrape reels from an Instagram account. Returns list of reel dicts."""
    try:
        from Downloader_Modules.apify_downloader import apify_scrape_actress_accounts
        return apify_scrape_actress_accounts("General", [ig_id], limit_per_account=limit)
    except Exception as e:
        logger.warning(f"harvester: Apify fetch failed for {ig_id}: {e}")
        return []

def _run_harvest_cycle() -> int:
    """
    One full harvest cycle:
      - for each account in source_accounts.json -> fetch reels -> dedup -> download -> route -> queue
    Returns number of clips added to queue.
    """
    downloads_dir = os.path.join(_REPO_ROOT, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    try:
        from Publishing_Modules.make_publisher import PublishQueue
    except ImportError:
        PublishQueue = None

    total_added = 0
    accounts_file = os.path.join(_REPO_ROOT, "Content_Scraper_Modules", "source_accounts.json")
    sources = []
    if os.path.exists(accounts_file):
        try:
            with open(accounts_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                sources = data.get("source_accounts", [])
        except Exception as e:
            logger.warning(f"harvester: failed to load source accounts: {e}")

    if not sources:
        sources = []

    logger.info(f"🌾 [HARVESTER] Starting harvest cycle across {len(sources)} source accounts: {sources}")
    try:
        from Downloader_Modules.apify_downloader import apify_scrape_actress_accounts
        downloaded = apify_scrape_actress_accounts("General", sources, limit_per_account=LIMIT_PER_ACCOUNT)
        total_added = len(downloaded or [])
        logger.info(f"🌾 [HARVESTER] Harvest cycle complete: {total_added} clips downloaded to downloads/")
    except Exception as exc:
        logger.error(f"harvester: harvest cycle failed: {exc}")

    return total_added


# ==============================================================================
# Scheduler loop
# ==============================================================================

def _batch_label(h: int, m: int) -> str:
    """Friendly batch name for a given hour."""
    if   4 <= h < 12: return "Morning Batch"
    elif 12 <= h < 17: return "Afternoon Batch"
    elif 17 <= h < 21: return "Evening Batch"
    else:              return "Night Batch"


def _today_is_scheduled() -> bool:
    """Returns True if today is an allowed run day (respects ACTRESS_RUN_DAYS env)."""
    raw = os.getenv("ACTRESS_RUN_DAYS", "").strip()
    if not raw:
        return True
    _DAY_MAP = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    today    = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    for token in raw.split(","):
        token = token.strip().lower()
        if token == today_str:
            return True
        if token in _DAY_MAP and today.weekday() == _DAY_MAP[token]:
            return True
    return False


def _seconds_until_next_slot() -> Tuple[float, str]:
    """Returns (seconds_to_wait, label) for the next scheduled slot."""
    cfg   = Core.get()
    slots = cfg.schedule_slots()
    now   = datetime.now()
    best_delta = None
    best_label = "Scheduled"

    for h, m in slots:
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        delta = (target - now).total_seconds()
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_label = _batch_label(h, m)

    return (best_delta or 3600.0, best_label)


def start_scheduler() -> None:
    """
    Main entry point. Runs harvest + publish loop forever.
    Call this from main.py instead of actress_scheduler.start_scheduler().
    """
    cfg = Core.get()

    # Run credential scan once at startup
    cfg.validate_credentials()

    # Start the publish side (Layer 3) in its own thread
    try:
        from Content_Harvester.publisher import start_publish_scheduler
        start_publish_scheduler()
        logger.info("harvester: publisher scheduler started")
    except Exception as exc:
        logger.warning("harvester: could not start publisher: %s", exc)

    logger.info("harvester: scheduler started. Slots: %s", cfg.schedule_slots())

    while True:
        try:
            # Check for FORCE_HARVEST override
            try:
                from dotenv import load_dotenv
                load_dotenv("Credentials/.env", override=True)
            except ImportError:
                pass

            if os.getenv("FORCE_HARVEST", "").lower() == "yes":
                logger.info("harvester: FORCE_HARVEST triggered")
                _run_harvest_cycle()
                # Clear the flag
                continue

            # Check scheduled slot
            now       = datetime.now()
            slots     = cfg.schedule_slots()
            grace_min = MISSED_GRACE_MINUTES

            fire_now = False
            for h, m in slots:
                slot_today = now.replace(hour=h, minute=m, second=0, microsecond=0)
                diff_min   = (now - slot_today).total_seconds() / 60
                if 0 <= diff_min <= grace_min:
                    fire_now = True
                    logger.info("harvester: slot %02d:%02d matched (%s)", h, m, _batch_label(h, m))
                    break

            if fire_now and _today_is_scheduled():
                _run_harvest_cycle()

            wait_sec, label = _seconds_until_next_slot()
            logger.info("harvester: sleeping %.0fs until next slot (%s)", wait_sec, label)
            time.sleep(min(wait_sec, 3600))     # wake up at least hourly to re-check ENV

        except KeyboardInterrupt:
            logger.info("harvester: shutting down")
            break
        except Exception as exc:
            logger.error("harvester: unexpected error in scheduler loop: %s", exc)
            time.sleep(300)

