"""
Intelligence_Modules / news_trend_fetcher.py
============================================
Auto-Input-Only News/Trend Intelligence & Zero-Download Video Metadata Scout.

News Provider Cascade (0 Gemini calls, zero extra pip packages):
  1. NewsData.io      (NEWSDATA_API_KEY — 200 credits/day, primary)
  2. GNews API        (GNEWS_API_KEY    — 100 req/day)
  3. The Guardian     (GUARDIAN_API_KEY — 500 calls/day)
  4. Google News RSS  (zero-key real-time fallback — 0h delay)
  5. NewsAPI.org      (NEWSAPI_ORG_KEY  — optional dev-only fallback)

Zero-Download Metadata Scout:
  Uses `yt-dlp --flat-playlist --dump-json --playlist-end <limit>` to inspect
  the latest N videos (controllable via Telegram chat, default 20) and score
  keyword matches in title/description/tags BEFORE downloading any .mp4 file.
"""

import json
import logging
import os
import re
import subprocess
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Union

import requests

logger = logging.getLogger("news_trend_fetcher")

DEFAULT_METADATA_SCOUT_LIMIT = 20
MAX_METADATA_SCOUT_LIMIT = 50


def _clean_html(raw: str) -> str:
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(raw))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())[:80]


class NewsTrendFetcher:
    """Fetches verified recent news & trending updates for a target keyword/niche."""

    def __init__(
        self,
        newsdata_api_key: Optional[str] = None,
        gnews_api_key: Optional[str] = None,
        guardian_api_key: Optional[str] = None,
        newsapi_org_key: Optional[str] = None,
        user_id: Optional[Union[str, int]] = None,
        timeout: float = 10.0,
    ):
        self.timeout = timeout
        self.user_id = str(user_id).strip() if user_id is not None else None

        # Resolve per-user credentials from telegram_user_manager first, then env vars
        user_keys: Dict[str, Optional[str]] = {}
        try:
            from Telegram_Storage_Modules.telegram_user_manager import get_user_news_api_keys
            user_keys = get_user_news_api_keys(self.user_id)
        except Exception:
            user_keys = {}

        self.newsdata_api_key = (
            newsdata_api_key
            or user_keys.get("newsdata_api_key")
            or os.getenv("NEWSDATA_API_KEY")
        )
        self.gnews_api_key = (
            gnews_api_key
            or user_keys.get("gnews_api_key")
            or os.getenv("GNEWS_API_KEY")
        )
        self.guardian_api_key = (
            guardian_api_key
            or user_keys.get("guardian_api_key")
            or os.getenv("GUARDIAN_API_KEY")
        )
        self.newsapi_org_key = (
            newsapi_org_key
            or user_keys.get("newsapi_org_key")
            or os.getenv("NEWSAPI_ORG_KEY")
        )

    # ── Provider 1: NewsData.io ───────────────────────────────────────────────
    def _fetch_newsdata(self, query: str, max_results: int = 8, language: str = "en") -> List[Dict[str, Any]]:
        if not self.newsdata_api_key:
            return []
        try:
            resp = requests.get(
                "https://newsdata.io/api/1/latest",
                params={
                    "apikey": self.newsdata_api_key,
                    "q": query,
                    "language": language,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            items = []
            for art in (payload.get("results") or [])[:max_results]:
                title = (art.get("title") or "").strip()
                if not title:
                    continue
                items.append({
                    "title": title,
                    "summary": _clean_html(art.get("description") or art.get("content") or "")[:400],
                    "source": art.get("source_name") or art.get("source_id") or "NewsData",
                    "published_at": art.get("pubDate") or "",
                    "url": art.get("link") or "",
                    "provider": "newsdata",
                })
            return items
        except Exception as exc:
            logger.warning("📰 [NewsData.io] fetch failed for '%s': %s", query, exc)
            return []

    # ── Provider 2: GNews API ─────────────────────────────────────────────────
    def _fetch_gnews(self, query: str, max_results: int = 8, language: str = "en") -> List[Dict[str, Any]]:
        if not self.gnews_api_key:
            return []
        try:
            resp = requests.get(
                "https://gnews.io/api/v4/search",
                params={
                    "q": query,
                    "lang": language,
                    "max": min(max_results, 10),
                    "apikey": self.gnews_api_key,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            items = []
            for art in (payload.get("articles") or [])[:max_results]:
                title = (art.get("title") or "").strip()
                if not title:
                    continue
                src = art.get("source")
                src_name = src.get("name") if isinstance(src, dict) else str(src or "GNews")
                items.append({
                    "title": title,
                    "summary": _clean_html(art.get("description") or art.get("content") or "")[:400],
                    "source": src_name,
                    "published_at": art.get("publishedAt") or "",
                    "url": art.get("url") or "",
                    "provider": "gnews",
                })
            return items
        except Exception as exc:
            logger.warning("📰 [GNews] fetch failed for '%s': %s", query, exc)
            return []

    # ── Provider 3: The Guardian Open Platform ────────────────────────────────
    def _fetch_guardian(self, query: str, max_results: int = 8) -> List[Dict[str, Any]]:
        if not self.guardian_api_key:
            return []
        try:
            resp = requests.get(
                "https://content.guardianapis.com/search",
                params={
                    "q": query,
                    "api-key": self.guardian_api_key,
                    "page-size": min(max_results, 10),
                    "show-fields": "trailText",
                    "order-by": "newest",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            results = (payload.get("response") or {}).get("results") or []
            items = []
            for art in results[:max_results]:
                title = (art.get("webTitle") or "").strip()
                if not title:
                    continue
                fields = art.get("fields") or {}
                items.append({
                    "title": title,
                    "summary": _clean_html(fields.get("trailText") or "")[:400],
                    "source": "The Guardian",
                    "published_at": art.get("webPublicationDate") or "",
                    "url": art.get("webUrl") or "",
                    "provider": "guardian",
                })
            return items
        except Exception as exc:
            logger.warning("📰 [Guardian] fetch failed for '%s': %s", query, exc)
            return []

    # ── Provider 4: Google News RSS (zero-key real-time fallback) ─────────────
    def _fetch_google_news_rss(self, query: str, max_results: int = 8, language: str = "en") -> List[Dict[str, Any]]:
        try:
            encoded = urllib.parse.quote_plus(query)
            url = f"https://news.google.com/rss/search?q={encoded}&hl={language}-IN&gl=IN&ceid=IN:{language}"
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; GeminiEditorNewsBot/1.0)"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            items = []
            for item in root.findall(".//item")[:max_results]:
                title = (item.findtext("title") or "").strip()
                if not title:
                    continue
                source_el = item.find("source")
                src_name = (source_el.text or "").strip() if source_el is not None else "Google News"
                items.append({
                    "title": title,
                    "summary": _clean_html(item.findtext("description") or "")[:400],
                    "source": src_name,
                    "published_at": (item.findtext("pubDate") or "").strip(),
                    "url": (item.findtext("link") or "").strip(),
                    "provider": "google_news_rss",
                })
            return items
        except Exception as exc:
            logger.warning("📰 [Google News RSS] fetch failed for '%s': %s", query, exc)
            return []

    # ── Provider 5: NewsAPI.org (optional dev-only fallback) ──────────────────
    def _fetch_newsapi_org(self, query: str, max_results: int = 8, language: str = "en") -> List[Dict[str, Any]]:
        if not self.newsapi_org_key:
            return []
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "language": language,
                    "sortBy": "publishedAt",
                    "pageSize": min(max_results, 10),
                    "apiKey": self.newsapi_org_key,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            items = []
            for art in (payload.get("articles") or [])[:max_results]:
                title = (art.get("title") or "").strip()
                if not title:
                    continue
                src = art.get("source")
                src_name = src.get("name") if isinstance(src, dict) else str(src or "NewsAPI")
                items.append({
                    "title": title,
                    "summary": _clean_html(art.get("description") or "")[:400],
                    "source": src_name,
                    "published_at": art.get("publishedAt") or "",
                    "url": art.get("url") or "",
                    "provider": "newsapi_org",
                })
            return items
        except Exception as exc:
            logger.warning("📰 [NewsAPI.org] fetch failed for '%s': %s", query, exc)
            return []

    def fetch(self, target_keyword: str, max_results: int = 8, language: str = "en") -> List[Dict[str, Any]]:
        """
        Queries configured news providers in priority order and returns deduplicated,
        verified news items for `target_keyword`.
        """
        query = (target_keyword or "").strip()
        if not query:
            return []

        collected: List[Dict[str, Any]] = []
        seen_titles: set = set()
        seen_urls: set = set()

        provider_calls = [
            ("newsdata", lambda: self._fetch_newsdata(query, max_results=max_results, language=language)),
            ("gnews", lambda: self._fetch_gnews(query, max_results=max_results, language=language)),
            ("guardian", lambda: self._fetch_guardian(query, max_results=max_results)),
            ("google_news_rss", lambda: self._fetch_google_news_rss(query, max_results=max_results, language=language)),
            ("newsapi_org", lambda: self._fetch_newsapi_org(query, max_results=max_results, language=language)),
        ]

        for prov_name, fn in provider_calls:
            if len(collected) >= max_results:
                break
            batch = fn()
            added = 0
            for item in batch:
                tkey = _normalize_title_key(item.get("title", ""))
                url = (item.get("url") or "").strip()
                if not tkey or tkey in seen_titles or (url and url in seen_urls):
                    continue
                seen_titles.add(tkey)
                if url:
                    seen_urls.add(url)
                collected.append(item)
                added += 1
                if len(collected) >= max_results:
                    break
            if added:
                logger.info("📰 [NEWS INTEL] %s returned %d unique item(s) for '%s'", prov_name, added, query)

        return collected


def fetch_news_for_target(
    target_keyword: str,
    max_results: int = 8,
    user_id: Optional[Union[str, int]] = None,
    language: str = "en",
) -> List[Dict[str, Any]]:
    """Convenience wrapper around NewsTrendFetcher."""
    return NewsTrendFetcher(user_id=user_id).fetch(
        target_keyword=target_keyword,
        max_results=max_results,
        language=language,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Zero-Download Video Metadata Scout
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_scout_limit(limit: Optional[int] = None, user_id: Optional[Union[str, int]] = None) -> int:
    """Resolves metadata scout limit from explicit arg -> user chat setting -> env -> 20."""
    val = limit
    if val is None:
        try:
            from Telegram_Storage_Modules.telegram_user_manager import get_user_metadata_scout_limit
            val = get_user_metadata_scout_limit(user_id)
        except Exception:
            val = None
    if val is None:
        try:
            val = int(os.getenv("AUTO_INPUT_METADATA_SCOUT_LIMIT", str(DEFAULT_METADATA_SCOUT_LIMIT)))
        except (ValueError, TypeError):
            val = DEFAULT_METADATA_SCOUT_LIMIT
    return max(1, min(MAX_METADATA_SCOUT_LIMIT, int(val)))


def _score_metadata_against_keywords(meta: Dict[str, Any], keywords: Sequence[str]) -> Dict[str, Any]:
    """Scores a video metadata record against target keywords without downloading the file."""
    haystack_title = (meta.get("title") or "").lower()
    haystack_desc = (meta.get("description") or "").lower()
    tags = meta.get("tags") or []
    haystack_tags = " ".join(str(t).lower() for t in tags) if isinstance(tags, list) else str(tags).lower()

    matched_terms: List[str] = []
    score = 0.0

    for kw in keywords:
        clean_kw = (kw or "").strip().lower()
        if not clean_kw:
            continue
        # Full phrase match gets highest weight
        if clean_kw in haystack_title:
            score += 5.0
            matched_terms.append(clean_kw)
        elif clean_kw in haystack_tags:
            score += 3.0
            matched_terms.append(clean_kw)
        elif clean_kw in haystack_desc:
            score += 2.0
            matched_terms.append(clean_kw)
        else:
            # Partial token match for multi-word names/phrases
            tokens = [t for t in re.split(r"\s+", clean_kw) if len(t) >= 3]
            if tokens:
                hits = sum(1 for tok in tokens if tok in haystack_title or tok in haystack_desc or tok in haystack_tags)
                if hits == len(tokens):
                    score += 3.5
                    matched_terms.append(clean_kw)
                elif hits > 0:
                    score += 0.8 * hits

    return {
        "keyword_score": round(score, 2),
        "matched_keywords": sorted(set(matched_terms)),
    }


def _build_scout_target_url(source: str, limit: int) -> str:
    """Converts an account handle, channel URL, or search query into a yt-dlp flat-playlist target."""
    src = (source or "").strip()
    if src.startswith(("http://", "https://", "ytsearch")):
        return src
    if src.startswith("@"):
        return f"https://www.youtube.com/{src}/shorts"
    # Plain keyword -> YouTube search (fast, metadata-only)
    return f"ytsearch{limit}:{src}"


def scout_candidate_videos_metadata(
    keywords: Union[str, Sequence[str]],
    sources: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    user_id: Optional[Union[str, int]] = None,
    timeout_sec: int = 45,
) -> List[Dict[str, Any]]:
    """
    Zero-Download Metadata Scout:
    Inspects the latest `limit` videos (default 20, controllable via Telegram chat)
    from given account URLs/handles or keyword search WITHOUT downloading any .mp4.

    Returns candidate dicts sorted by keyword relevance score descending:
      {id, title, description, url, duration, uploader, view_count, keyword_score, matched_keywords}
    """
    kw_list = [keywords] if isinstance(keywords, str) else list(keywords or [])
    kw_list = [k.strip() for k in kw_list if k and str(k).strip()]
    eff_limit = _resolve_scout_limit(limit=limit, user_id=user_id)

    target_sources = list(sources) if sources else ([f"ytsearch{eff_limit}:{' '.join(kw_list)} shorts"] if kw_list else [])
    if not target_sources:
        return []

    candidates: List[Dict[str, Any]] = []
    seen_urls: set = set()

    for raw_src in target_sources:
        target_url = _build_scout_target_url(raw_src, eff_limit)
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--flat-playlist",
            "--dump-json",
            "--no-warnings",
            "--skip-download",
            "--playlist-end", str(eff_limit),
            target_url,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            if proc.returncode != 0 and not proc.stdout.strip():
                logger.warning("🔍 [META SCOUT] yt-dlp metadata failed for '%s': %s", raw_src, (proc.stderr or "")[:180])
                continue

            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                vid_id = entry.get("id") or ""
                vid_url = entry.get("webpage_url") or entry.get("url") or ""
                if vid_url and not vid_url.startswith("http") and vid_id:
                    vid_url = f"https://www.youtube.com/watch?v={vid_id}"
                if not vid_url or vid_url in seen_urls:
                    continue
                seen_urls.add(vid_url)

                rec = {
                    "id": vid_id,
                    "title": (entry.get("title") or "").strip(),
                    "description": (entry.get("description") or "")[:400],
                    "tags": entry.get("tags") or [],
                    "url": vid_url,
                    "duration": float(entry.get("duration") or 0.0),
                    "uploader": entry.get("uploader") or entry.get("channel") or raw_src,
                    "view_count": int(entry.get("view_count") or 0),
                }
                rec.update(_score_metadata_against_keywords(rec, kw_list))
                candidates.append(rec)
        except subprocess.TimeoutExpired:
            logger.warning("🔍 [META SCOUT] Timed out scouting '%s' (limit=%d)", raw_src, eff_limit)
        except Exception as exc:
            logger.warning("🔍 [META SCOUT] Error scouting '%s': %s", raw_src, exc)

    candidates.sort(key=lambda c: (c.get("keyword_score", 0.0), c.get("view_count", 0)), reverse=True)
    logger.info(
        "🔍 [META SCOUT] Scanned %d candidate video(s) (limit=%d, zero downloads) for keywords=%s",
        len(candidates), eff_limit, kw_list,
    )
    return candidates[:eff_limit]
