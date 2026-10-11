"""
Core_Modules/salesman_state.py
==============================
Disk-persisted state tracking for:
  1. Apify daily API quotas (survives restarts)
  2. Per-account 24h scrape cooldown throttling
  3. Scraped Instagram reels shortcodes deduplication registry
"""

import os
import json
import time
import logging
from typing import List, Dict, Tuple, Any, Optional

logger = logging.getLogger("salesman_state")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_STATE_FILE = os.path.join(_DATA_DIR, "salesman_state.json")
_REGISTRY_FILE = os.path.join(_DATA_DIR, "scraped_posts_registry.json")


class ApifyQuotaTracker:
    """Tracks daily consumed Apify units, persisting to disk."""

    def __init__(self, state_file: str = _STATE_FILE):
        self.state_file = state_file
        self.daily_quota = int(os.getenv("APIFY_DAILY_QUOTA", "100"))
        self._load()

    def _load(self):
        self.quota_used = 0
        self.quota_date = time.strftime("%Y-%m-%d")
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    q = data.get("apify_quota", {})
                    if q.get("date") == self.quota_date:
                        self.quota_used = int(q.get("used", 0))
            except Exception as e:
                logger.debug(f"Could not load apify quota from state: {e}")

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            data = {}
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}
            data["apify_quota"] = {
                "date": self.quota_date,
                "used": self.quota_used,
                "limit": self.daily_quota,
                "updated_at": time.time(),
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save apify quota to state: {e}")

    def check(self, needed: int = 1) -> bool:
        today = time.strftime("%Y-%m-%d")
        if self.quota_date != today:
            self.quota_date = today
            self.quota_used = 0
            self._save()

        if self.quota_used + needed > self.daily_quota:
            logger.warning(
                "🛑 Apify daily quota exhausted (%d/%d). Sleeping until tomorrow. 💤",
                self.quota_used, self.daily_quota
            )
            return False
        return True

    def consume(self, amount: int = 1) -> None:
        self.quota_used += amount
        logger.info("💰 [APIFY QUOTA] Used: %d/%d today", self.quota_used, self.daily_quota)
        self._save()


class AccountScrapeThrottle:
    """Manages scrape cooldowns for creator/source accounts."""

    def __init__(self, state_file: str = _STATE_FILE, cooldown_hours: Optional[float] = None):
        self.state_file = state_file
        if cooldown_hours is None:
            try:
                cooldown_hours = float(os.getenv("APIFY_ACCOUNT_COOLDOWN_HOURS", "6.0"))
            except (ValueError, TypeError):
                cooldown_hours = 6.0
        self.cooldown_seconds = cooldown_hours * 3600.0

    def _load_history(self) -> Dict[str, float]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("account_scrape_timestamps", {})
            except Exception:
                pass
        return {}

    def _save_history(self, history: Dict[str, float]) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            data = {}
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}
            data["account_scrape_timestamps"] = history
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save account scrape timestamps: {e}")

    def filter_ready(self, accounts: List[str]) -> Tuple[List[str], List[str]]:
        now = time.time()
        history = self._load_history()
        ready = []
        blocked = []

        for acc in accounts:
            key = acc.lower().lstrip("@").strip()
            last_scraped = history.get(key, 0.0)
            if now - last_scraped < self.cooldown_seconds:
                blocked.append(acc)
            else:
                ready.append(acc)

        return ready, blocked

    def mark_scraped(self, accounts: List[str]) -> None:
        now = time.time()
        history = self._load_history()
        for acc in accounts:
            key = acc.lower().lstrip("@").strip()
            history[key] = now
        self._save_history(history)


class ScrapedPostsRegistry:
    """Disk-persisted registry of Instagram reel shortcodes to prevent re-downloads."""

    def __init__(self, registry_file: str = _REGISTRY_FILE):
        self.registry_file = registry_file
        self.seen_shortcodes = set()
        self._load()

    def _load(self):
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.seen_shortcodes = set(data)
                    elif isinstance(data, dict):
                        self.seen_shortcodes = set(data.get("seen_shortcodes", []))
            except Exception as e:
                logger.debug(f"Could not load scraped posts registry: {e}")

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump({
                    "version": 1,
                    "updated_at": time.time(),
                    "total_count": len(self.seen_shortcodes),
                    "seen_shortcodes": sorted(list(self.seen_shortcodes)),
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save scraped posts registry: {e}")

    def filter_new(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        new_items = []
        for item in items:
            sc = item.get("shortcode")
            if sc and sc in self.seen_shortcodes:
                continue
            new_items.append(item)
        return new_items

    def mark_seen(self, shortcodes: List[str]) -> None:
        added = False
        for sc in shortcodes:
            if sc and sc not in self.seen_shortcodes:
                self.seen_shortcodes.add(sc)
                added = True
        if added:
            self._save()


# ── Module Singletons ─────────────────────────────────────────────────────────

_quota_tracker: Optional[ApifyQuotaTracker] = None
_scrape_throttle: Optional[AccountScrapeThrottle] = None
_posts_registry: Optional[ScrapedPostsRegistry] = None


def get_apify_quota() -> ApifyQuotaTracker:
    global _quota_tracker
    if _quota_tracker is None:
        _quota_tracker = ApifyQuotaTracker()
    return _quota_tracker


def get_account_scrape_throttle() -> AccountScrapeThrottle:
    global _scrape_throttle
    if _scrape_throttle is None:
        _scrape_throttle = AccountScrapeThrottle()
    return _scrape_throttle


def get_scraped_posts_registry() -> ScrapedPostsRegistry:
    global _posts_registry
    if _posts_registry is None:
        _posts_registry = ScrapedPostsRegistry()
    return _posts_registry
