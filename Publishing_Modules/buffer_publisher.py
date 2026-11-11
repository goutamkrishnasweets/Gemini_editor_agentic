"""
Publishing_Modules / buffer_publisher.py
================================================================================
Buffer API v1 Publishing Client — TikTok, Pinterest & X (Twitter) Channels.

Automatically discovers your connected Buffer channels and dispatches approved
video reels to TikTok, Pinterest, and X (Twitter) using Buffer's REST API.

Configuration (via .env or GitHub Secrets):
  BUFFER_ACCESS_TOKEN = "<your_personal_access_token>"
  (Generate from: https://publish.buffer.com/settings/api?tab=personal-access)

Optional per-channel overrides (auto-discovered from Buffer API if not set):
  BUFFER_TIKTOK_PROFILE_ID  = "..."  # Buffer profile ID for your TikTok channel
  BUFFER_PINTEREST_PROFILE_ID = "..." # Buffer profile ID for your Pinterest channel
  BUFFER_X_PROFILE_ID        = "..."  # Buffer profile ID for your X/Twitter channel

Platform Caption Limits enforced:
  X / Twitter : 280 chars max
  TikTok      : 2200 chars max (uses video upload URL, not direct file)
  Pinterest   : 500 chars max pin description; 100 chars max title

Usage in Phase 4 (via media_publisher_main.py):
  from Publishing_Modules.buffer_publisher import publish_to_buffer, is_buffer_configured
"""

import os
import sys
import json
import time
import logging
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional, List

logger = logging.getLogger("buffer_publisher")
logger.setLevel(logging.INFO)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)
    load_dotenv(os.path.join(_REPO_ROOT, "Credentials", ".env"), override=False)
except ImportError:
    pass

# ── Buffer API Base ────────────────────────────────────────────────────────────
_BUFFER_API_BASE = "https://api.bufferapp.com/1"

# X/Twitter hard character ceiling (enforced by platform)
_X_CAPTION_LIMIT = 280

# Platform service types as reported by Buffer API
_TIKTOK_SERVICE_TYPES  = {"tiktok"}
_PINTEREST_SERVICE_TYPES = {"pinterest"}
_X_SERVICE_TYPES       = {"twitter"}

# Cache discovered profiles for the process lifetime to avoid repeated API calls
_profile_cache: Optional[List[Dict[str, Any]]] = None


# ── Token Resolution ───────────────────────────────────────────────────────────

def _get_buffer_token(user_id: Optional[str] = None) -> Optional[str]:
    """
    Resolves Buffer Personal Access Token.
    Non-admin users get their own token from telegram_users.json if set.
    Admin falls back to BUFFER_ACCESS_TOKEN env var.
    """
    if user_id:
        try:
            from Telegram_Storage_Modules.telegram_user_manager import get_user_field, _is_admin_user
            tok = get_user_field(str(user_id), "buffer_access_token")
            if tok and str(tok).strip():
                return str(tok).strip()
            # Non-admin users do NOT inherit admin token
            if not _is_admin_user(str(user_id)):
                return None
        except Exception:
            pass
    return os.getenv("BUFFER_ACCESS_TOKEN", "").strip() or None


def is_buffer_configured(user_id: Optional[str] = None) -> bool:
    """Returns True if a Buffer access token is available for this user/admin."""
    return bool(_get_buffer_token(user_id))


# ── Channel Discovery ──────────────────────────────────────────────────────────

def get_buffer_profiles(token: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Fetches all connected Buffer profiles using GET /1/profiles.json.
    Results are cached for the process lifetime to avoid redundant calls.
    """
    global _profile_cache
    if _profile_cache is not None and not force_refresh:
        return _profile_cache

    url = f"{_BUFFER_API_BASE}/profiles.json?access_token={urllib.parse.quote(token)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GeminiEditor-BufferPublisher/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            profiles = json.loads(resp.read().decode("utf-8"))

        if isinstance(profiles, list):
            logger.info(
                "🔍 [Buffer] Discovered %d connected channel(s): %s",
                len(profiles),
                [f"{p.get('service')}:{p.get('id', '')[:8]}" for p in profiles]
            )
            _profile_cache = profiles
            return profiles

        logger.warning("⚠️ [Buffer] Unexpected profiles response type: %s", type(profiles))
        return []
    except Exception as e:
        logger.error("❌ [Buffer] Profile discovery error: %s", e)
        return []


def _filter_profiles(profiles: List[Dict[str, Any]], service_types: set) -> List[str]:
    """Extracts Buffer profile IDs matching the given service type(s)."""
    return [
        p["id"] for p in profiles
        if p.get("service", "").lower() in service_types and p.get("id")
    ]


def _resolve_profile_ids(
    token: str,
    tiktok_pid: Optional[str] = None,
    pinterest_pid: Optional[str] = None,
    x_pid: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Returns resolved profile IDs for each Buffer-connected channel.
    Explicit overrides from env/args take precedence; falls back to auto-discovery.
    """
    resolved = {"tiktok": [], "pinterest": [], "x": []}

    # Use explicit env overrides if present (skips API call entirely for that channel)
    env_tiktok   = tiktok_pid   or os.getenv("BUFFER_TIKTOK_PROFILE_ID", "").strip()
    env_pinterest = pinterest_pid or os.getenv("BUFFER_PINTEREST_PROFILE_ID", "").strip()
    env_x        = x_pid        or os.getenv("BUFFER_X_PROFILE_ID", "").strip()

    if env_tiktok:
        resolved["tiktok"] = [env_tiktok]
    if env_pinterest:
        resolved["pinterest"] = [env_pinterest]
    if env_x:
        resolved["x"] = [env_x]

    # If any channel is still unresolved, fetch from Buffer API once
    if not resolved["tiktok"] or not resolved["pinterest"] or not resolved["x"]:
        profiles = get_buffer_profiles(token)
        if profiles:
            if not resolved["tiktok"]:
                resolved["tiktok"] = _filter_profiles(profiles, _TIKTOK_SERVICE_TYPES)
            if not resolved["pinterest"]:
                resolved["pinterest"] = _filter_profiles(profiles, _PINTEREST_SERVICE_TYPES)
            if not resolved["x"]:
                resolved["x"] = _filter_profiles(profiles, _X_SERVICE_TYPES)

    return resolved


# ── Caption Formatters ─────────────────────────────────────────────────────────

def _format_x_caption(title: str, caption: str, tags: str, affiliate_link: Optional[str]) -> str:
    """
    Builds an X (Twitter) post text that fits inside 280 characters.
    Priority: punchy hook from title → primary hashtags → affiliate link.
    """
    # Extract up to 2 hashtags
    all_tags = [t for t in (tags or "").split() if t.startswith("#")]
    top_tags = " ".join(all_tags[:2])

    # Base: short title + top tags
    base = f"{title} {top_tags}".strip()

    # Append affiliate link if space allows (URLs count as ~23 chars in Twitter)
    link_part = f"\n{affiliate_link}" if affiliate_link else ""

    text = f"{base}{link_part}"
    if len(text) <= _X_CAPTION_LIMIT:
        return text

    # Trim title to fit
    budget = _X_CAPTION_LIMIT - len(top_tags) - len(link_part) - 2  # 2 = spaces
    trimmed_title = title[:max(budget, 20)].rstrip()
    if len(trimmed_title) < len(title):
        trimmed_title = trimmed_title[:trimmed_title.rfind(" ")] + "…" if " " in trimmed_title else trimmed_title + "…"
    return f"{trimmed_title} {top_tags}{link_part}".strip()[:_X_CAPTION_LIMIT]


def _format_pinterest_caption(title: str, caption: str, tags: str) -> Dict[str, str]:
    """Returns pin title (≤100 chars) and pin description (≤500 chars)."""
    pin_title = title[:100].strip()
    pin_desc = caption[:500].strip() if caption else f"{title}\n\n{tags}"[:500]
    return {"title": pin_title, "description": pin_desc}


def _format_tiktok_caption(title: str, caption: str, tags: str, affiliate_link: Optional[str]) -> str:
    """Builds TikTok caption ≤2200 chars with energy hook + hashtags + affiliate link."""
    base = caption.strip() if caption.strip() else f"🔥 {title}"
    if tags:
        all_tags = [t for t in tags.split() if t.startswith("#")]
        tag_str = " ".join(all_tags[:20])  # TikTok: 20 hashtags is a practical limit
        if tag_str not in base:
            base = f"{base}\n\n{tag_str}"
    if affiliate_link and affiliate_link not in base:
        base = f"{base}\n🛒 {affiliate_link}"
    return base[:2200]


# ── Buffer API Dispatcher ──────────────────────────────────────────────────────

def _post_to_buffer(
    token: str,
    profile_ids: List[str],
    text: str,
    media: Optional[Dict[str, str]] = None,
    link: Optional[str] = None,
    now: bool = True,
) -> Dict[str, Any]:
    """
    Posts to Buffer via POST /1/updates/create.json.
    Args:
        token       : Buffer Personal Access Token.
        profile_ids : List of Buffer profile IDs to broadcast to.
        text        : Caption / post text.
        media       : Optional dict with 'link' (video URL) for video posts.
        link        : Optional URL to attach as a web link (Pinterest pin dest URL).
        now         : If True, publishes immediately (not queued in Buffer calendar).
    Returns:
        dict with 'status', 'updates', 'raw_response'.
    """
    if not profile_ids:
        return {"status": "skipped", "reason": "No matching Buffer profile IDs found for this channel."}

    params: Dict[str, Any] = {
        "access_token": token,
        "text": text,
        "now": "true" if now else "false",
    }
    for i, pid in enumerate(profile_ids):
        params[f"profile_ids[{i}]"] = pid

    if media:
        for k, v in media.items():
            params[f"media[{k}]"] = v

    if link:
        params["media[link]"] = link

    endpoint = f"{_BUFFER_API_BASE}/updates/create.json"
    encoded_data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=encoded_data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "GeminiEditor-BufferPublisher/1.0",
        },
        method="POST",
    )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                status_code = resp.getcode()
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)

            if status_code in (200, 201, 202) and data.get("success"):
                updates = data.get("updates", [])
                update_id = updates[0].get("id", "") if updates else ""
                logger.info("✅ [Buffer] Post accepted (attempt %d). Update ID: %s", attempt, update_id)
                return {"status": "success", "update_id": update_id, "updates": updates, "raw_response": data}

            logger.warning(
                "⚠️ [Buffer] Unexpected HTTP %d response (attempt %d): %s",
                status_code, attempt, raw[:200],
            )
            return {"status": "failed", "error": raw[:300], "status_code": status_code}

        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8", errors="ignore") if http_err.fp else ""
            logger.error(
                "❌ [Buffer] HTTP %d error (attempt %d): %s",
                http_err.code, attempt, err_body[:300],
            )
            if http_err.code in (401, 403):
                return {
                    "status": "failed",
                    "error": "Buffer auth failed — check BUFFER_ACCESS_TOKEN.",
                    "status_code": http_err.code,
                }
            if attempt < max_retries:
                time.sleep(2.0 * attempt)
        except Exception as e:
            logger.error("❌ [Buffer] Dispatch error (attempt %d): %s", attempt, e)
            if attempt < max_retries:
                time.sleep(2.0 * attempt)

    return {"status": "failed", "error": f"Failed to post to Buffer after {max_retries} attempts."}


# ── Public Media URL ───────────────────────────────────────────────────────────

def _resolve_media_url(video_path: str) -> Optional[str]:
    """
    Resolves a public video URL for Buffer's media[link] parameter.
    Reuses make_publisher's battle-tested Telegram vault URL resolver.
    """
    if str(video_path).startswith(("http://", "https://")):
        return video_path
    try:
        from Publishing_Modules.make_publisher import resolve_public_media_url
        return resolve_public_media_url(video_path)
    except Exception as e:
        logger.warning("⚠️ [Buffer] Media URL resolution error: %s", e)
        return None


# ── Main Publish Entry Point ───────────────────────────────────────────────────

def publish_to_buffer(
    video_path: str,
    title: str = "",
    description: str = "",
    tags: str = "",
    seo_data: Optional[Dict[str, Any]] = None,
    affiliate_link: Optional[str] = None,
    user_id: Optional[str] = None,
    channels: Optional[List[str]] = None,
    now: bool = True,
) -> Dict[str, Any]:
    """
    Publishes an approved reel to Buffer-connected channels: TikTok, Pinterest, X (Twitter).

    Args:
        video_path     : Local path to the master reel video file.
        title          : Human-approved video title (required).
        description    : Full caption / description text.
        tags           : Space-separated hashtag string (e.g. "#fashion #viral").
        seo_data       : Optional SEO dict from generate_platform_seo() — extracts
                         platform-specific captions and hashtags if present.
        affiliate_link : Primary affiliate/product link (optional).
        user_id        : Telegram user ID for per-user credential resolution.
        channels       : Explicit list of channels to target e.g. ['tiktok', 'x'].
                         Defaults to all three: ['tiktok', 'pinterest', 'x'].
        now            : If True, posts immediately. If False, adds to Buffer queue.

    Returns:
        dict: {
            "status": "success" | "partial" | "failed" | "skipped",
            "published_to": ["tiktok", ...],
            "failed_on": [...],
            "results": {
                "tiktok":    {"status": "success", "update_id": ...},
                "pinterest": {"status": "skipped", ...},
                "x":         {"status": "success", "update_id": ...},
            }
        }
    """
    token = _get_buffer_token(user_id)
    if not token:
        logger.info("ℹ️ [Buffer] BUFFER_ACCESS_TOKEN not configured. Skipping Buffer broadcast.")
        return {"status": "skipped", "reason": "BUFFER_ACCESS_TOKEN not set. Set via /setbuffer <token>."}

    target_channels = set(channels or ["tiktok", "pinterest", "x"])
    logger.info(
        "🚀 [Buffer] Starting broadcast to: %s (user=%s)",
        ", ".join(sorted(target_channels)), user_id or "admin"
    )

    # ── 1. Resolve public video URL ───────────────────────────────────────────
    video_url = _resolve_media_url(video_path)
    if not video_url:
        logger.error("❌ [Buffer] Could not resolve a public video URL for: %s", video_path)
        return {"status": "failed", "error": "Failed to resolve a public video URL for Buffer."}

    # ── 2. Extract platform-specific SEO content ──────────────────────────────
    platforms_seo = (seo_data or {}).get("platforms", {})
    aff_links = list((seo_data or {}).get("_affiliate_links", []) or [])
    if affiliate_link and affiliate_link not in aff_links:
        aff_links.insert(0, affiliate_link)
    primary_aff = aff_links[0] if aff_links else None

    # TikTok: use tiktok-specific caption from SEO, fallback to general description
    tt_seo  = platforms_seo.get("tiktok", {})
    tt_caption = (tt_seo.get("description") or tt_seo.get("caption") or description or f"{title}\n\n{tags}").strip()
    tt_tags = " ".join(tt_seo.get("hashtags", [])) if tt_seo.get("hashtags") else tags

    # Pinterest: use a dedicated pin description from SEO if present
    pin_seo = platforms_seo.get("pinterest", {})
    pin_desc_raw = (pin_seo.get("description") or pin_seo.get("caption") or description or "").strip()
    pin_tags = " ".join(pin_seo.get("hashtags", [])) if pin_seo.get("hashtags") else tags

    # X/Twitter: short hook from SEO title if available
    x_seo   = platforms_seo.get("x", {}) or platforms_seo.get("twitter", {})
    x_title = (x_seo.get("title") or title).strip()
    x_tags  = " ".join(x_seo.get("hashtags", [])) if x_seo.get("hashtags") else tags

    # ── 3. Discover Buffer profile IDs ────────────────────────────────────────
    profile_ids = _resolve_profile_ids(token)
    logger.info(
        "📋 [Buffer] Profile IDs — TikTok: %s | Pinterest: %s | X: %s",
        profile_ids["tiktok"] or "none",
        profile_ids["pinterest"] or "none",
        profile_ids["x"] or "none",
    )

    results: Dict[str, Dict[str, Any]] = {}
    published_to: List[str] = []
    failed_on: List[str] = []

    # ── 4. TikTok ─────────────────────────────────────────────────────────────
    if "tiktok" in target_channels:
        if not profile_ids["tiktok"]:
            logger.warning("⚠️ [Buffer] No TikTok profile found in your Buffer account.")
            results["tiktok"] = {"status": "skipped", "reason": "No TikTok channel found in Buffer account."}
        else:
            caption_tt = _format_tiktok_caption(title, tt_caption, tt_tags, primary_aff)
            media_tt = {"link": video_url, "picture": video_url, "title": title[:100]}
            logger.info("📱 [Buffer → TikTok] Posting: '%s'...", caption_tt[:60])
            tt_res = _post_to_buffer(
                token=token,
                profile_ids=profile_ids["tiktok"],
                text=caption_tt,
                media=media_tt,
                now=now,
            )
            results["tiktok"] = tt_res
            (published_to if tt_res.get("status") == "success" else failed_on).append("tiktok")

    # ── 5. Pinterest ──────────────────────────────────────────────────────────
    if "pinterest" in target_channels:
        if not profile_ids["pinterest"]:
            logger.warning("⚠️ [Buffer] No Pinterest profile found in your Buffer account.")
            results["pinterest"] = {"status": "skipped", "reason": "No Pinterest channel found in Buffer account."}
        else:
            pin_texts = _format_pinterest_caption(title, pin_desc_raw or description, pin_tags)
            media_pin = {
                "link": video_url,
                "picture": video_url,
                "title": pin_texts["title"],
                "description": pin_texts["description"],
            }
            # Pinterest: set the affiliate link as the destination URL for the pin
            pin_dest_link = primary_aff or video_url
            logger.info("📌 [Buffer → Pinterest] Posting pin: '%s'...", pin_texts["title"])
            pin_res = _post_to_buffer(
                token=token,
                profile_ids=profile_ids["pinterest"],
                text=pin_texts["description"],
                media=media_pin,
                link=pin_dest_link,
                now=now,
            )
            results["pinterest"] = pin_res
            (published_to if pin_res.get("status") == "success" else failed_on).append("pinterest")

    # ── 6. X (Twitter) ────────────────────────────────────────────────────────
    if "x" in target_channels:
        if not profile_ids["x"]:
            logger.warning("⚠️ [Buffer] No X/Twitter profile found in your Buffer account.")
            results["x"] = {"status": "skipped", "reason": "No X/Twitter channel found in Buffer account."}
        else:
            caption_x = _format_x_caption(x_title, description, x_tags, primary_aff)
            media_x = {"link": video_url, "picture": video_url}
            logger.info("🐦 [Buffer → X] Posting: '%s'...", caption_x[:60])
            x_res = _post_to_buffer(
                token=token,
                profile_ids=profile_ids["x"],
                text=caption_x,
                media=media_x,
                now=now,
            )
            results["x"] = x_res
            (published_to if x_res.get("status") == "success" else failed_on).append("x")

    # ── 7. Build Final Summary ─────────────────────────────────────────────────
    if published_to and not failed_on:
        overall = "success"
    elif published_to and failed_on:
        overall = "partial"
    elif not published_to and not failed_on:
        overall = "skipped"  # all channels were missing
    else:
        overall = "failed"

    logger.info(
        "🎯 [Buffer] Broadcast complete — Status: %s | Published: %s | Failed: %s",
        overall.upper(), published_to or "none", failed_on or "none"
    )

    return {
        "status": overall,
        "published_to": published_to,
        "failed_on": failed_on,
        "results": results,
        "video_url": video_url,
    }


# ── Credential Helpers for /setbuffer Telegram command ────────────────────────

def set_user_buffer_token(user_id_str: str, token: str) -> bool:
    """
    Saves a user's Buffer Personal Access Token to telegram_users.json and GitHub Secrets.
    Called by the /setbuffer <token> Telegram command handler in main.py.
    """
    try:
        from Telegram_Storage_Modules.telegram_user_manager import (
            load_all_users, save_all_users, _is_admin_user, _persist_env_var_to_disk,
            sync_user_secret_to_github,
        )
        clean_tok = token.strip()
        if not clean_tok:
            return False

        users = load_all_users()
        uid = str(user_id_str).strip()
        users.setdefault(uid, {})["buffer_access_token"] = clean_tok
        save_all_users(users)

        if _is_admin_user(uid):
            os.environ["BUFFER_ACCESS_TOKEN"] = clean_tok
            _persist_env_var_to_disk("BUFFER_ACCESS_TOKEN", clean_tok)

        sync_user_secret_to_github(uid, "BUFFER_ACCESS_TOKEN", clean_tok)

        # Invalidate profile cache so next run re-discovers channels with new token
        global _profile_cache
        _profile_cache = None

        logger.info("🔑 [Buffer] Access Token saved for User ID %s", uid)
        return True
    except Exception as e:
        logger.error("❌ [Buffer] Failed to save token for User %s: %s", user_id_str, e)
        return False


def get_user_buffer_token(user_id_str: Optional[str] = None) -> Optional[str]:
    """
    Returns the Buffer Access Token for a user (or admin fallback).
    Thin wrapper for external callers.
    """
    return _get_buffer_token(user_id_str)


__all__ = [
    "publish_to_buffer",
    "is_buffer_configured",
    "get_buffer_profiles",
    "set_user_buffer_token",
    "get_user_buffer_token",
]

