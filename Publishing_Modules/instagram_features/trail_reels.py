"""
Publishing_Modules/instagram_features/trail_reels.py
=====================================================
Trail Reels Feature Manager
Kingdom: Instagram Publishing Features
Purpose: Parse /trail_reels command from user title input, inject Meta trial_params
         into the upload container, handle ineligible-account fallback, and enforce
         daily velocity warnings (never hard-block — user always has bypass choice).

Graduation strategy is always MANUAL (user taps the phone to graduate).
No SS_PERFORMANCE mode is exposed to the user.

Usage:
    from Publishing_Modules.instagram_features.trail_reels import TrailReelsManager

    mgr = TrailReelsManager()
    clean_title, affiliate_url, is_trial = mgr.extract_trail_command(raw_text)
    init_data = mgr.apply_trial_params(init_data, is_trial)
    count, warned = mgr.check_daily_velocity(user_id)
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
# Accept both spelling variants so typos don't break the workflow
_TRAIL_PATTERN = re.compile(
    r"\s*/tr(?:ai|ia)l_reels\b",   # /trail_reels OR /trial_reels (typo alias)
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")

DAILY_VELOCITY_WARN_THRESHOLD = 10   # warn at >= this many reels in a single day

# Key inside telegram_users.json per-user dict for the daily velocity counters
_VEL_DATE_KEY = "trail_reels_velocity_date"
_VEL_COUNT_KEY = "trail_reels_velocity_count"


class TrailReelsManager:
    """
    Encapsulates all Trail Reels logic:
      - Command parsing from raw user input
      - Injecting Meta trial_params into the upload container init_data
      - Auto-fallback when the account is ineligible for trial mode
      - Daily publish velocity tracking with user-bypass-able warnings
    """

    # ------------------------------------------------------------------
    # 1. Command Parsing
    # ------------------------------------------------------------------

    def extract_trail_command(self, raw_text: str) -> Tuple[str, Optional[str], bool]:
        """
        Parse the raw user title string for /trail_reels (or typo /trial_reels).

        Returns:
            clean_title   (str)  — title with /trail_reels token and URLs stripped
            affiliate_url (str | None) — first detected URL, or None
            is_trial      (bool) — True if /trail_reels was present

        Rules:
            • /trail_reels must be stripped BEFORE the title reaches any SEO generator
              or YouTube/Instagram output — it must NEVER leak into published content.
            • Both /trail_reels and /trial_reels (common typo) are accepted.
            • URL extraction happens after command stripping.
        """
        if not raw_text or not isinstance(raw_text, str):
            return ("", None, False)

        # 1a. Detect and strip /trail_reels (or /trial_reels)
        is_trial = bool(_TRAIL_PATTERN.search(raw_text))
        text = _TRAIL_PATTERN.sub("", raw_text)

        # 1b. Extract affiliate URLs
        urls = _URL_PATTERN.findall(text)
        for u in urls:
            text = text.replace(u, "")
        text = re.sub(r"\s+", " ", text).strip()

        affiliate_url: Optional[str] = "\n".join(urls) if len(urls) > 1 else (urls[0] if urls else None)

        logger.info(
            "[TrailReels] Parsed: is_trial=%s | clean_title='%s' | affiliate_url=%s",
            is_trial, text[:60], affiliate_url,
        )
        return (text, affiliate_url, is_trial)

    # ------------------------------------------------------------------
    # 2. Meta Container Injection
    # ------------------------------------------------------------------

    def apply_trial_params(self, init_data: Dict[str, Any], is_trial: bool) -> Dict[str, Any]:
        """
        Inject Meta trial_params into the IG media container init payload.

        Only modifies init_data when is_trial is True.
        Graduation strategy is permanently MANUAL — the user graduates on their phone.

        Args:
            init_data:  The dict passed to POST /{ig-user-id}/media
            is_trial:   Flag from extract_trail_command()

        Returns:
            Modified init_data (mutates and returns the same dict for convenience).
        """
        if not is_trial:
            return init_data

        trial_payload = {"graduation_strategy": "MANUAL"}
        init_data["trial_params"] = json.dumps(trial_payload)
        logger.info(
            "[TrailReels] Injected trial_params into IG container: %s",
            init_data["trial_params"],
        )
        return init_data

    # ------------------------------------------------------------------
    # 3. Ineligible-Account Fallback
    # ------------------------------------------------------------------

    def handle_trial_ineligible_fallback(self, init_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove trial_params from init_data for re-submission as a standard Reel.

        Called when Meta returns an error indicating the account is ineligible for
        Trial Reels (e.g., error_code 100 / subcode 2804003 or similar).
        Never raises — always returns a clean dict ready for immediate re-submit.

        Args:
            init_data: The dict that was rejected by Meta

        Returns:
            init_data with trial_params removed (safe for standard Reel re-submit)
        """
        if "trial_params" in init_data:
            del init_data["trial_params"]
            logger.warning(
                "⚠️ [TRIAL REELS] Account not eligible for trial_params. "
                "Falling back to standard Reel upload."
            )
        return init_data

    @staticmethod
    def _is_trial_rejection_error(error_response: Any) -> bool:
        """
        Heuristic: returns True if Meta's error response looks like a trial_params
        eligibility rejection, so the caller can decide to fallback automatically.

        Meta may return various error shapes; we check for common indicators.
        """
        if isinstance(error_response, dict):
            # Check for 'error' sub-dict (Graph API style)
            err = error_response.get("error", {})
            if isinstance(err, dict):
                msg = str(err.get("message", "")).lower()
                code = err.get("code", 0)
                subcode = err.get("error_subcode", 0)
                if (
                    "trial" in msg
                    or "trial_params" in msg
                    or subcode in (2804003, 2804004)
                    or (code == 100 and "trial" in msg)
                ):
                    return True
            # Flat error string check (some versions return this)
            flat = json.dumps(error_response).lower()
            if "trial_params" in flat or "trial reel" in flat:
                return True
        return False

    # ------------------------------------------------------------------
    # 4. Daily Velocity Tracking (warn only — never block)
    # ------------------------------------------------------------------

    def check_daily_velocity(self, user_id: str) -> Tuple[int, bool]:
        """
        Check how many reels this user has published today.

        Returns:
            (daily_publish_count, is_warning_threshold_hit)

        is_warning_threshold_hit is True when count >= DAILY_VELOCITY_WARN_THRESHOLD.
        The caller is responsible for showing the warning and bypass button.
        This method NEVER hard-blocks anything — it only reports.
        """
        try:
            count = self._load_daily_count(user_id)
            is_warned = count >= DAILY_VELOCITY_WARN_THRESHOLD
            if is_warned:
                logger.warning(
                    "[TrailReels] Velocity warning for user %s: %d reels today (threshold=%d)",
                    user_id, count, DAILY_VELOCITY_WARN_THRESHOLD,
                )
            return (count, is_warned)
        except Exception as _e:
            logger.warning("[TrailReels] Could not check daily velocity for %s: %s", user_id, _e)
            return (0, False)

    def increment_daily_publish_count(self, user_id: str) -> int:
        """
        Increment today's reel publish count for the user and persist it.

        Returns the new count after increment.
        """
        try:
            from Telegram_Storage_Modules.telegram_user_manager import (
                get_user_field,
                set_user_field,
            )
            today = str(date.today())
            stored_date = get_user_field(user_id, _VEL_DATE_KEY) or ""
            if stored_date != today:
                # New day — reset counter
                new_count = 1
            else:
                raw = get_user_field(user_id, _VEL_COUNT_KEY) or 0
                new_count = int(raw) + 1
            set_user_field(user_id, _VEL_DATE_KEY, today)
            set_user_field(user_id, _VEL_COUNT_KEY, new_count)
            logger.info("[TrailReels] Incremented daily count for user %s → %d", user_id, new_count)
            return new_count
        except Exception as _e:
            logger.warning("[TrailReels] Could not increment daily count for %s: %s", user_id, _e)
            return 0

    def _load_daily_count(self, user_id: str) -> int:
        """Load today's reel publish count from persistent user storage."""
        try:
            from Telegram_Storage_Modules.telegram_user_manager import get_user_field
            today = str(date.today())
            stored_date = get_user_field(user_id, _VEL_DATE_KEY) or ""
            if stored_date != today:
                return 0
            raw = get_user_field(user_id, _VEL_COUNT_KEY) or 0
            return int(raw)
        except Exception as _e:
            logger.warning("[TrailReels] Could not load daily count for %s: %s", user_id, _e)
            return 0

