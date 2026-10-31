"""
Gemini_Modules/platform_seo_generator.py   (v3.0)
=================================================
Self-updating, commercially-optimized, source-private SEO generator for reused short-form video.

PIPELINE
  1. ProvenanceGuard   Learns every identifier of the SOURCE channel from metadata (handles, tagged users,
                       URLs, shortcodes, @mentions found in captions) and scrubs them from the evidence
                       BEFORE the model sees it, and from the output AFTER the model writes it.
  2. build_evidence()  One compact packet: transcript, visual/audio intel, edit plan, creator-behavior model,
                       scrubbed source caption/tags, analytics, affiliate slots, learned patterns.
  3. GENERATE (Gemini) Strategy first (buyer intent, buyer queries, value props, what this edit ADDS), then
                       3 title candidates per platform, pick the best, write caption + hashtags.
  4. FINALIZE (code)   Restore affiliate URLs byte-for-byte, route links per platform, place the ad disclosure
                       where Indian ASCI rules need it (first two lines), scrub source leaks, enforce limits
                       without cutting URLs.
  5. LINT (code)       seo_score is computed from verifiable checks: limits, disclosure position, source leaks,
                       ungrounded numbers/prices, ownership + endorsement claims, engagement bait, keyword
                       placement, hashtag budget.
  6. CRITIQUE (Gemini) Editor pass; a rewrite is kept only if it is hard-lint-clean and does not score lower.
  7. LEARN             Every output is logged. record_performance(seo_id, platform, metrics) feeds real results
                       back; measured high-lift patterns + top performers are injected into the next prompts
                       and an optional LLM distillation step rewrites per-niche lessons every N results.

RESEARCH BASIS (researched 2026-10-31; platforms change rules often -> override via SEO_RULES_PATH json)
  YouTube   Shorts feed is the main distribution surface; title/description/spoken words drive search; the first
            ~150 chars of the description are the snippet; first 3 hashtags surface above the title; links in
            Shorts descriptions/comments are NOT clickable (since Aug 2023); reused-content policy applies even
            with the original creator's permission, and (Oct 2026) Shorts recommendations now favor original work.
  Instagram Hashtag cap is 5 per post (rollout since Dec 2025, counted across caption + comments; some older
            accounts may still see 30). Captions, spoken words and on-screen text are the SEO surface. Caption links
            are not clickable. Mosseri (Apr 2026): accounts mostly posting others' content stop being recommended
            to non-followers; crediting alone does not count as transformation.
  Facebook  Meta: captions with no links, minimal capitals and <=5 hashtags perform best; minor edits (captions,
            borders, speed) of someone else's video are "unoriginal" and deprioritized; engagement bait penalized.
  Telegram  Public-channel posts are indexed by Telegram search + Google; keywords in the first line help;
            hashtags are category labels (2-5), global hashtag search only covers big public channels.
  India     ASCI: disclosure label (#ad etc.) within the FIRST TWO LINES, no "see more"; affiliate programs count;
            the platform's own paid-partnership tool is not sufficient on its own.

Backward compatible with v1/v2 public names. New kwargs on generate_platform_seo are optional and trailing.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import re
import string
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("platform_seo_generator")

__all__ = [
    "PLATFORM_SPECS", "PLATFORM_LIMITS", "reload_rules",
    "generate_platform_seo", "record_performance", "approve_and_finalize",
    "format_for_telegram_preview", "extract_cache_for_regeneration",
    "lint_platform", "build_evidence", "ProvenanceGuard",
    "clean_entity_name", "strip_system_and_tracking_tokens",
    "extract_main_subject_and_context", "extract_celebrity_human_name", "sanitize_raw_handles_out",
]

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

GEN_TASK_TYPE = "seo_generation"   # reuse a task type the governor already knows
GEN_TEMPERATURE = 0.6
CRITIC_TEMPERATURE = 0.2
TARGET_SCORE = 90
RULES_RESEARCHED = "2026-10-31"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "yes", "true", "on")


_COMMON_CRITERIA = [
    "evidence_grounding", "keyword_placement", "hook_strength",
    "scannability", "cta_fit", "hashtag_relevance", "commercial_intent_match",
]

# Link routing modes: inline | first_comment | dm_keyword | none
PLATFORM_SPECS: Dict[str, Dict[str, Any]] = {
    "youtube": {
        "label": "YouTube Shorts (YouTube + Google search, Shorts feed)",
        "title_max": 70, "title_suffix": "#Shorts",
        "description_max": 5000,
        "hashtags_min": 3, "hashtags_max": 5, "hashtags_total_max": 15,
        "hashtag_placement": "Put the first 3 in the first line of the description; YouTube shows them above the title.",
        "disclosure": "(Disclosure: #ad #affiliate — qualifying purchases earn commission)",
        "disclosure_placement": "line2",
        "links_label": "🛍️ FEATURED PRODUCTS / LINKS:",
        "link_mode_default": "inline",
        "caption_links_clickable": False,
        "caption_link_note": "Shorts description/comment links are not clickable. Clickable routes: channel profile links, "
                             "Related Video link, YouTube Shopping affiliate links.",
        "cta_re": r"(comment|below|drop|which|would you|thoughts|your (?:take|pick|opinion)|\?|👇)",
        "rules": [
            "Title: primary keyword in the first ~40 chars + a curiosity/value element; mirror a real buyer search "
            "(X review, X vs Y, is X worth it, best X for <use-case>). Price words ONLY if the price is in EVIDENCE. Ends with #Shorts.",
            "Description line 1 = ONE natural sentence containing the primary keyword (this is the search snippet, ~150 chars).",
            "Line 2 = the disclosure line exactly as given (when product slots exist).",
            "Then 1-2 short blocks selling the specific value props shown in the video, then the links block or pointer, "
            "then ONE genuine question. Keywords live inside sentences; never a visible keyword list.",
        ],
    },
    "instagram": {
        "label": "Instagram Reels (caption/spoken/on-screen text search + sends)",
        "title_max": 80,
        "description_max": 2200, "hook_visible_chars": 125,
        "hashtags_min": 3, "hashtags_max": 5, "hashtags_total_max": 5,
        "hashtag_placement": "End of caption. The cap of 5 counts every # in caption + comments, including #ad.",
        "disclosure": "#ad",
        "disclosure_placement": "prefix",
        "links_label": "🔗 FEATURED ITEM / SHOP LINK:",
        "link_mode_default": "dm_keyword",
        "caption_links_clickable": False,
        "caption_link_note": "Caption (and comment) URLs are not clickable. Working routes: DM auto-reply, bio link, "
                             "Story link sticker, native product tags.",
        "cta_re": r"(save|share|send|comment|rate|thoughts|👇|\?)",
        "rules": [
            "Title = the hook: a sharp curiosity gap tied to a buyer problem; no corporate summary.",
            "Caption line 1 (starts with '#ad ' when product slots exist) must work ALONE: only ~125 chars show before '… more'. "
            "Primary keyword in the first sentence.",
            "Instagram search reads caption text, spoken words and on-screen text: keep them aligned. Short, scannable, "
            "keywords inside natural sentences, never a keyword list.",
            "Hashtags are categories, not reach: 3-5 max INCLUDING #ad. Never write #affiliate on Instagram.",
            "End with ONE save/send trigger that fits the video.",
        ],
    },
    "facebook": {
        "label": "Facebook Reels (watch time, shares, original-content filter)",
        "title_max": 255,
        "description_max": 63206,
        "hashtags_min": 3, "hashtags_max": 5, "hashtags_total_max": 5,
        "hashtag_placement": "End of caption (Meta: no more than 5).",
        "disclosure": "#ad",
        "disclosure_placement": "prefix",
        "links_label": "🛍️ Featured Items:",
        "link_mode_default": "first_comment",
        "caption_links_clickable": True,
        "caption_link_note": "Meta reports captions with no links perform best; put the link in the first comment.",
        "cta_re": r"(which|would you|comment|below|thoughts|👇|\?)",
        "rules": [
            "Title: conversational and specific; the video must pay it off.",
            "Caption: short, relevant, minimal CAPITAL LETTERS, no links in the caption. Starts with '#ad ' when product slots exist.",
            "Close with a genuine either/or or opinion question. No engagement bait ('like if', 'tag a friend', 'comment YES').",
        ],
    },
    "telegram": {
        "label": "Telegram broadcast (public-channel search + direct UX)",
        "title_max": 255,
        "description_max": 1000, "soft_text_max": 300,
        "hashtags_min": 2, "hashtags_max": 5, "hashtags_total_max": 8,
        "hashtag_placement": "End of post. Use 2-5 stable category tags; they are navigation labels, not a reach hack.",
        "disclosure": "(#ad #affiliate)",
        "disclosure_placement": "line2",
        "links_label": "",
        "link_mode_default": "inline",
        "caption_links_clickable": True,
        "caption_link_note": "Links are clickable in Telegram captions.",
        "cta_re": None,
        "rules": [
            "Format: line 1 = bold headline with the primary keyword (🔥 **Title**); line 2 = the disclosure; then ONE punchy "
            "takeaway sentence naming the concrete benefit; then product slots one per line. Zero filler.",
        ],
    },
}

PLATFORM_LIMITS: Dict[str, Dict[str, Any]] = {}


def _rebuild_limits() -> None:
    PLATFORM_LIMITS.clear()
    for p, s in PLATFORM_SPECS.items():
        PLATFORM_LIMITS[p] = {"title_max": s["title_max"], "description_max": s["description_max"],
                              "hashtags_max": s["hashtags_max"], "hashtags_min": s["hashtags_min"]}


def reload_rules() -> None:
    """Apply overrides without a code change. Sources: SEO_RULES_PATH (json: {"instagram": {"hashtags_max": 30,
    "hashtags_total_max": 30}}), and SEO_IG_HASHTAG_MAX for accounts that still have the old 30-tag allowance."""
    path = os.getenv("SEO_RULES_PATH")
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                for plat, patch in (json.load(fh) or {}).items():
                    if plat in PLATFORM_SPECS and isinstance(patch, dict):
                        PLATFORM_SPECS[plat].update(patch)
        except Exception as exc:
            logger.warning("[PlatformSEO] could not load SEO_RULES_PATH: %s", exc)
    ig_max = os.getenv("SEO_IG_HASHTAG_MAX")
    if ig_max and ig_max.isdigit():
        PLATFORM_SPECS["instagram"]["hashtags_max"] = int(ig_max)
        PLATFORM_SPECS["instagram"]["hashtags_total_max"] = int(ig_max)
    _rebuild_limits()


reload_rules()

_GENERIC_TAGS = frozenset({
    "viral", "trending", "reels", "reel", "fyp", "foryou", "foryoupage", "explore", "instagood",
    "love", "followforfollow", "like4like", "instadaily", "viralvideo", "viralreels", "trend",
})


def _link_mode(platform: str, override: Optional[Dict[str, str]] = None) -> str:
    """Explicit override/env wins. Otherwise the researched default applies ONLY if the publisher can act on
    routes (SEO_PUBLISHER_SUPPORTS_ROUTES=1); else fall back to inline so existing publishers don't silently lose links."""
    explicit = (override or {}).get(platform) or os.getenv(f"SEO_LINK_MODE_{platform.upper()}")
    if explicit in ("inline", "first_comment", "dm_keyword", "none"):
        return explicit
    default = PLATFORM_SPECS[platform].get("link_mode_default", "inline")
    if default != "inline" and not _env_flag("SEO_PUBLISHER_SUPPORTS_ROUTES"):
        return "inline"
    return default


def _dm_keyword() -> str:
    return re.sub(r"[^A-Za-z0-9]", "", os.getenv("SEO_DM_KEYWORD", "LINK")).upper() or "LINK"


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER (lazy)
# ══════════════════════════════════════════════════════════════════════════════

_ROUTER: Any = None
_ROUTER_TRIED = False


def _get_router():
    global _ROUTER, _ROUTER_TRIED
    if _ROUTER_TRIED:
        return _ROUTER
    _ROUTER_TRIED = True
    for mod in ("Gemini_Modules.gemini_router_module.gemini_governor", "gemini_router_module.gemini_governor"):
        try:
            _ROUTER = importlib.import_module(mod).GeminiGovernor()
            return _ROUTER
        except Exception as exc:
            logger.debug("[PlatformSEO] router import via %s failed: %s", mod, exc)
    logger.warning("⚠️ GeminiGovernor not available. SEO generator will use heuristic fallback.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# TEXT UTILITIES (URL/slot-safe)
# ══════════════════════════════════════════════════════════════════════════════

_URL_RE = re.compile(r'https?://[^\s<>"]+')
_SLOT_RE = re.compile(r"\[\[AFF_LINK_\d+\]\]")
_ML, _MR = "\ue000", "\ue001"
_MASK_RE = re.compile(_ML + r"(\d+)" + _MR)


def _mask(text: str) -> Tuple[str, List[str]]:
    saved: List[str] = []

    def _sub(m: "re.Match[str]") -> str:
        saved.append(m.group(0))
        return f"{_ML}{len(saved) - 1}{_MR}"

    return _SLOT_RE.sub(_sub, _URL_RE.sub(_sub, text)), saved


def _unmask(text: str, saved: List[str]) -> str:
    return _MASK_RE.sub(lambda m: saved[int(m.group(1))], text)


_ENTITY_PREFIXES = ("celebrity|outfit|accessory|environment|concept|niche|person|actor|actress|"
                    "model|influencer|brand|subject|star|celeb|tag")
_RE_IG_TRACK = re.compile(r"#?AQ(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{10,}")
_RE_HASH_SIG = re.compile(r"#(?=[A-Za-z0-9_-]{26,})(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]+")
_RE_PLACEHOLDER = re.compile(r"(?i)#?(?:creator_?unknown|niche_?unknown)\b|#(?:none|null|unknown)\b")
_RE_ENTITY_TAG = re.compile(r"(?i)#(?:celebrity|model|actress|actor|influencer|star|person)[_:]\s*([A-Za-z0-9_]+)")
_RE_ENTITY_PREFIX = re.compile(rf"(?i)\b(?:{_ENTITY_PREFIXES}):(?=\S)")
_RE_NULL_LITERAL = re.compile(r"(?<![\w#])(?:None|null|NaN)(?!\w)")
_SNAKE = re.compile(r"^[^\W\d_]+(?:_[^\W\d_]+)+[.,!?:;]*$")
_NONWORD = re.compile(r"[^\w\u0900-\u0DFF]+")   # keep Devanagari..Malayalam so Indic tags survive


def _sanitize_str(text: str) -> str:
    masked, saved = _mask(text)
    t = _RE_IG_TRACK.sub("", masked)
    t = _RE_HASH_SIG.sub("", t)
    t = _RE_PLACEHOLDER.sub("", t)
    t = _RE_ENTITY_TAG.sub(r"#\1", t)
    t = _RE_ENTITY_PREFIX.sub("", t)
    t = _RE_NULL_LITERAL.sub("", t)
    lines_out: List[str] = []
    for line in t.split("\n"):
        seen: set = set()
        toks: List[str] = []
        for tok in line.split():
            if tok.startswith("#"):
                inner = tok[1:].replace("_", "")
                if not inner or inner.lower() in seen:
                    continue
                seen.add(inner.lower())
                toks.append("#" + inner)
            elif _SNAKE.match(tok):
                toks.append(tok.replace("_", " "))
            else:
                toks.append(tok)
        lines_out.append(" ".join(toks))
    t = re.sub(r"\n{3,}", "\n\n", "\n".join(lines_out)).strip()
    return _unmask(t, saved)


def strip_system_and_tracking_tokens(text_or_obj: Any) -> Any:
    """Scrub IG tracking tokens, placeholder tags, entity prefixes, leaked None/null, duplicate hashtags.
    URLs and [[AFF_LINK_n]] slots are never touched; newlines are preserved."""
    if isinstance(text_or_obj, str):
        return _sanitize_str(text_or_obj)
    if isinstance(text_or_obj, list):
        out: List[Any] = []
        seen: set = set()
        for item in text_or_obj:
            res = strip_system_and_tracking_tokens(item)
            if isinstance(res, str) and res.startswith("#"):
                if res.lower() in seen:
                    continue
                seen.add(res.lower())
                out.append(res)
            elif res or isinstance(res, (bool, int, float)):
                out.append(res)
        return out
    if isinstance(text_or_obj, set):
        return {r for r in (strip_system_and_tracking_tokens(i) for i in text_or_obj)
                if r or isinstance(r, (bool, int, float))}
    if isinstance(text_or_obj, tuple):
        return tuple(strip_system_and_tracking_tokens(list(text_or_obj)))
    if isinstance(text_or_obj, dict):
        return {k: strip_system_and_tracking_tokens(v) for k, v in text_or_obj.items()}
    return text_or_obj


def clean_entity_name(entity_str: str) -> str:
    """'celebrity:Avneet_Kaur' -> 'Avneet Kaur'; only all-lowercase words are capitalized (iPhone/OnePlus survive)."""
    if not entity_str or not isinstance(entity_str, str):
        return ""
    text = re.sub(rf"(?i)^(?:{_ENTITY_PREFIXES}):\s*", "", entity_str.strip())
    text = re.sub(r"(?<=[a-z0-9]{2})(?:None|null)\b", "", text).replace("_", " ")
    words = []
    for w in text.split():
        if w.lower() in {"none", "null", "unknown"}:
            continue
        words.append(w.capitalize() if w.islower() else w)
    return " ".join(words)


def extract_celebrity_human_name(handle: str) -> str:
    if not handle:
        return ""
    clean = re.sub(r"[._\-\d]+", " ", handle.strip().lstrip("@")).strip()
    noise = {"official", "real", "daily", "page", "fp", "club", "fan"}
    return " ".join(w.capitalize() for w in clean.split() if w.lower() not in noise)


def sanitize_raw_handles_out(text_or_obj: Any, raw_handle: str = "", discovered_subject: str = "") -> Any:
    """v1-compatible helper. v3 uses ProvenanceGuard internally (broader); kept for callers that import it."""
    handle = (raw_handle or "").strip().lstrip("@")
    if len(handle) < 3:
        return text_or_obj
    name = (discovered_subject or "").strip()
    tag_name = re.sub(r"[^\w]", "", name)
    h = re.escape(handle)
    id_like = bool(re.search(r"[_.\d]", handle))

    def _clean(text: str) -> str:
        masked, saved = _mask(text)
        masked = re.sub(rf"#{h}(?!\w)", f"#{tag_name}" if tag_name else "", masked, flags=re.I)
        masked = re.sub(rf"@{h}(?!\w)", name, masked, flags=re.I)
        if id_like:
            masked = re.sub(rf"(?<![\w@#]){h}(?!\w)", name, masked, flags=re.I)
        masked = re.sub(r"[ \t]{2,}", " ", masked)
        masked = re.sub(r" +\n", "\n", masked)
        return _unmask(masked, saved).strip()

    if isinstance(text_or_obj, str):
        return _clean(text_or_obj)
    if isinstance(text_or_obj, list):
        return [sanitize_raw_handles_out(i, raw_handle, discovered_subject) for i in text_or_obj if i]
    if isinstance(text_or_obj, dict):
        return {k: sanitize_raw_handles_out(v, raw_handle, discovered_subject) for k, v in text_or_obj.items()}
    return text_or_obj


def _to_tag(text: str) -> str:
    words = [w for w in _NONWORD.sub(" ", (text or "").replace("_", " ")).split()
             if w.lower() not in {"none", "null", "unknown"}]
    if not words:
        return ""
    if len(words) == 1:
        return "#" + words[0]
    return "#" + "".join(w[:1].upper() + w[1:] for w in words)


def _cut_words(text: str, n: int) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    if n <= 1:
        return text[:max(n, 0)]
    cut = text[: n - 1]
    if " " in cut and not text[n - 1:n].isspace():
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-–—|/") + "…"


def _clip(s: Any, n: int) -> str:
    s = "" if s is None else str(s).strip()
    return s if len(s) <= n else s[:n] + "…"


def _clip_obj(obj: Any, n: int) -> Any:
    if not obj:
        return None
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return obj if len(s) <= n else s[:n] + "…"


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# ══════════════════════════════════════════════════════════════════════════════
# PROVENANCE GUARD  (never leak the source channel's credit)
# ══════════════════════════════════════════════════════════════════════════════

_ID_KEYS = ("creator_handle", "owner_username", "ownerUsername", "username", "uploader", "uploader_id",
            "channel", "channel_name", "channel_title", "channel_id", "author", "author_name", "authorName",
            "ownerFullName", "owner_full_name", "creator", "creator_name", "page_name", "source_handle",
            "source_channel", "account", "handle", "shortcode", "source_id", "video_id", "clip_id", "session_id")
_LIST_ID_KEYS = ("taggedUsers", "tagged_users", "mentions", "coauthors", "coAuthors", "collaborators")
_URL_KEYS = ("source_url", "url", "permalink", "webpage_url", "original_url", "input_url", "inputUrl",
             "post_url", "video_url")
_CAPTION_KEYS = ("caption", "raw_caption", "title", "source_title", "description")

_CREDIT_LINE_RE = re.compile(
    r"(?i)(?:\bcredits?\b\s*(?:to|goes|go|:|-|—)|\bcredit\b.*@|\bvia\s*@|\bvia\s*:|\bsource\s*:|\bcr\s*:|\bctto\b|\bh/t\b"
    r"|\bdm\s+(?:me\s+)?(?:for|4)\s+(?:credit|removal)|\b(?:re)?posted\s+(?:from|by)|\brepost(?:ed)?\b|\bregram\b"
    r"|\bfollow\s+@|\boriginal\s+(?:video|content|creator|owner|post|audio|sound)\b|\ball\s+rights?\s+(?:belong|reserved)"
    r"|\bno\s+copyright\s+(?:infringement\s+)?intended|\binspired\s+by\s+@|📸\s*[:\-]|🎥\s*[:\-]|🎬\s*[:\-])")
_OUTRO_RE = re.compile(
    r"(?i)\b(?:subscribe|follow\s+(?:me|us|for|my|our)|like\s+(?:and|&)\s+share|hit\s+the\s+bell|"
    r"(?:my|our)\s+(?:channel|page)|dm\s+me|check\s+out\s+(?:my|our)|link\s+in\s+(?:my\s+)?bio)\b")
_ATTRIB_TAGS = frozenset({
    "credit", "credits", "creditgoestoowner", "creditstotheowner", "creditsgoestotheowner", "creditsgoestoowner",
    "creditgoesto", "credittotheowner", "credittorightfulowner", "creditstorightfulowner", "dmforcredit",
    "dmforcredits", "dm4credit", "dmforremoval", "dmforremove", "repost", "reposted", "repostfromcreator",
    "regram", "ctto", "via", "dc", "cr", "nocopyrightintended", "nocopyrightinfringementintended",
})
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_MENTION_RE = re.compile(r"@[\w.]{2,}")


def _ids_from_url(u: str) -> List[str]:
    try:
        p = urlparse(u)
    except Exception:
        return []
    host = (p.hostname or "").lower()
    segs = [s for s in p.path.split("/") if s]
    out: List[str] = []
    if "instagram.com" in host:
        kinds = {"p", "reel", "reels", "tv", "stories", "explore"}
        if segs and segs[0] not in kinds:
            out.append(segs[0])
        for i, s in enumerate(segs[:-1]):
            if s in kinds and s not in ("stories", "explore"):
                out.append(segs[i + 1])
    elif "tiktok.com" in host:
        out += [s[1:] for s in segs if s.startswith("@")]
        if "video" in segs and segs.index("video") + 1 < len(segs):
            out.append(segs[segs.index("video") + 1])
    elif "youtube.com" in host or "youtu.be" in host:
        out += [s[1:] for s in segs if s.startswith("@")]
        if segs and segs[0] in ("channel", "c", "user", "shorts", "embed", "live") and len(segs) > 1:
            out.append(segs[1])
        if "youtu.be" in host and segs:
            out.append(segs[0])
        out += parse_qs(p.query).get("v", [])
    elif "facebook.com" in host or "fb.watch" in host:
        if segs and segs[0] not in ("reel", "watch", "share", "videos", "permalink.php", "story.php"):
            out.append(segs[0])
    elif host in ("t.me", "telegram.me") and segs:
        out.append(segs[0])
    return out


class ProvenanceGuard:
    """Learns the source channel's identifiers and keeps them out of everything that gets published."""

    def __init__(self, identifiers: Optional[List[str]] = None, allow_text: Optional[List[str]] = None,
                 allowed_urls: Optional[List[str]] = None):
        self.allowed_urls = set(allowed_urls or [])
        allow = {_norm_key(a) for a in (allow_text or []) if a}
        variants: set = set()
        for raw in identifiers or []:
            raw = str(raw).strip().lstrip("@")
            if not raw:
                continue
            spaced = re.sub(r"[._\-]+", " ", raw).strip()
            nodigits = re.sub(r"\d+", "", spaced).strip()
            for v in {raw, spaced, nodigits, re.sub(r"\s+", "", spaced), re.sub(r"\s+", "", nodigits)}:
                v = v.strip()
                if len(v) >= 4 and _norm_key(v) not in allow:
                    variants.add(v)
        self.blocked = sorted(variants, key=len, reverse=True)
        pats = []
        for v in self.blocked:
            pats.append(re.escape(v).replace(r"\ ", r"[\s._\-]*"))
        self._rx = re.compile(r"(?<![\w])#?(?:" + "|".join(pats) + r")(?![\w])", re.I) if pats else None

    @classmethod
    def from_inputs(cls, metadata: Optional[Dict[str, Any]], video_context: str, cache: Optional[Dict[str, Any]],
                    user_title: str = "", user_hint: str = "", links: Optional[List[str]] = None) -> "ProvenanceGuard":
        md = _as_dict(metadata)
        ids: List[str] = []
        for k in _ID_KEYS:
            if isinstance(md.get(k), str):
                ids.append(md[k])
        for k in _LIST_ID_KEYS:
            v = md.get(k)
            for item in (v if isinstance(v, list) else [v] if v else []):
                if isinstance(item, str):
                    ids.append(item)
                elif isinstance(item, dict):
                    ids += [str(item[x]) for x in ("username", "full_name", "fullName", "name") if item.get(x)]
        for k in _URL_KEYS:
            if isinstance(md.get(k), str):
                ids += _ids_from_url(md[k])
        texts = [str(md.get(k) or "") for k in _CAPTION_KEYS] + [video_context or ""]
        for t in texts:
            ids += re.findall(r"@([A-Za-z0-9._]{3,})", t)
            ids += re.findall(r"(?:creator_handle|raw_handle|Handle|channel|uploader|author)\s*[:=]\s*@?([A-Za-z0-9._-]{3,})", t)
        # Names produced by OUR OWN analysis are legitimate subject text, never "source credit".
        c = _as_dict(cache)
        vis = _as_dict(c.get("visual_data"))
        vc = _as_dict(c.get("visual_context") or vis.get("gemini_visual_output") or vis.get("clip_intelligence"))
        allow = [user_title, user_hint, vc.get("person_name"), c.get("person_name"), vc.get("main_subject"),
                 c.get("main_subject")]
        det = vc.get("detected_entities") or c.get("detected_entities") or []
        allow += [det] if isinstance(det, str) else list(det)
        allow = [clean_entity_name(str(a)) for a in allow if a]
        return cls(ids, allow, links)

    # -- scrubbing ---------------------------------------------------------------------------------
    def scrub_text(self, text: Any) -> Any:
        if not isinstance(text, str) or not text:
            return text
        t = _URL_RE.sub(lambda m: m.group(0) if m.group(0) in self.allowed_urls else "", text)
        t = _EMAIL_RE.sub("", t)
        masked, saved = _mask(t)
        kept = []
        for line in masked.split("\n"):
            if _ML not in line and _CREDIT_LINE_RE.search(line):
                continue          # whole credit/attribution line goes (never a line carrying an affiliate link)
            kept.append(line)
        t = "\n".join(kept)
        t = _MENTION_RE.sub("", t)
        if self._rx:
            t = self._rx.sub("", t)
        out_lines = []
        for line in t.split("\n"):
            toks = []
            for tok in line.split():
                if tok.startswith("#") and _norm_key(tok) in _ATTRIB_TAGS:
                    continue
                toks.append(tok)
            out_lines.append(re.sub(r"[ \t]{2,}", " ", " ".join(toks)))
        t = "\n".join(out_lines)
        t = re.sub(r"(?m)^[ \t]*(?:👉|•|-|\*|\||:)+[ \t]*$\n?", "", t)
        t = re.sub(r"\n{3,}", "\n\n", t).strip()
        return _unmask(t, saved)

    def scrub_evidence_text(self, text: Any) -> Any:
        """Evidence-only: also drops spoken/captioned outros ('subscribe to my channel')."""
        t = self.scrub_text(text)
        if not isinstance(t, str) or not t:
            return t
        parts = re.split(r"(?<=[.!?।])\s+", t)
        return " ".join(p for p in parts if not _OUTRO_RE.search(p)).strip() if len(parts) > 1 else \
            ("" if _OUTRO_RE.search(t) and len(t) < 160 else t)

    def scrub_hashtags(self, tags: Any) -> List[str]:
        items = tags if isinstance(tags, list) else str(tags or "").split()
        out = []
        for t in items:
            t = str(t).strip()
            if not t or _norm_key(t) in _ATTRIB_TAGS:
                continue
            if self._rx and self._rx.search(t):
                continue
            out.append(t)
        return out

    def scrub_obj(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.scrub_evidence_text(obj)
        if isinstance(obj, list):
            return [x for x in (self.scrub_obj(i) for i in obj) if x not in ("", None, [], {})]
        if isinstance(obj, dict):
            return {k: self.scrub_obj(v) for k, v in obj.items()}
        return obj

    def find_leaks(self, text: str) -> List[str]:
        if not text:
            return []
        leaks: List[str] = [m for m in _MENTION_RE.findall(_EMAIL_RE.sub("", text))]
        leaks += [u for u in _URL_RE.findall(text) if u not in self.allowed_urls]
        masked, _ = _mask(text)
        if self._rx:
            leaks += [m.group(0) for m in self._rx.finditer(masked)]
        leaks += [m.group(0) for line in masked.split("\n") if _ML not in line
                  for m in [_CREDIT_LINE_RE.search(line)] if m]
        leaks += [tok for tok in masked.split() if tok.startswith("#") and _norm_key(tok) in _ATTRIB_TAGS]
        return leaks


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_cache(cache: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cache = _as_dict(cache)
    vis = _as_dict(cache.get("visual_data"))
    aud = _as_dict(cache.get("audio_data"))
    visual_ctx = _as_dict(cache.get("visual_context") or vis.get("gemini_visual_output") or vis.get("clip_intelligence"))
    editing = _as_dict(cache.get("editing_plan") or cache.get("video_editing_plan"))
    audio_ctx = _as_dict(aud.get("context") or aud.get("gemini_audio_output"))
    w = aud.get("whisper_transcript") or cache.get("polished_transcript") or cache.get("transcript")
    if isinstance(w, dict):
        transcript = w.get("text") or w.get("transcript") or json.dumps(w, ensure_ascii=False)
    elif isinstance(w, str):
        transcript = w
    else:
        transcript = ""
    return {"visual_ctx": visual_ctx, "editing": editing, "audio_ctx": audio_ctx,
            "audio_math": aud.get("audio_math"), "transcript": transcript,
            "cbm": _as_dict(cache.get("creator_behavior_model"))}


def extract_main_subject_and_context(
    video_context: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
    user_hint: Optional[str] = None,
    affiliate_link: Optional[str] = None,
) -> Dict[str, Any]:
    """Heuristic subject discovery (+ subject_source / subject_confidence). The LLM may override a weak subject."""
    metadata = _as_dict(metadata)
    rc = _resolve_cache(cache)
    cache = _as_dict(cache)
    vc, ep, audio_ctx = rc["visual_ctx"], rc["editing"], rc["audio_ctx"]

    hint_clean = ""
    if user_hint:
        hint_clean = clean_entity_name(re.sub(r"\s+", " ", _URL_RE.sub("", user_hint)).strip())

    raw_caption = metadata.get("raw_caption") or metadata.get("caption") or ""
    source_title = metadata.get("title") or metadata.get("source_title") or ""
    tags = metadata.get("hashtags") or []
    tags_str = " ".join(tags) if isinstance(tags, list) else str(tags)

    raw_detected = vc.get("detected_entities") or cache.get("detected_entities") or []
    if isinstance(raw_detected, str):
        raw_detected = [raw_detected]
    detected = [c for c in (clean_entity_name(e) for e in raw_detected) if c]
    person_name = clean_entity_name(vc.get("person_name") or cache.get("person_name") or "")
    cached_subject = clean_entity_name(vc.get("main_subject") or cache.get("main_subject") or "")
    full_text = f"{hint_clean} {video_context} {source_title} {raw_caption} {tags_str} {' '.join(detected)}".strip()

    subject, source, confidence = "", "", "none"
    if hint_clean:
        subject, source, confidence = hint_clean, "user_hint", "high"
    elif person_name:
        subject, source, confidence = person_name, "cache.person_name", "medium"
    elif cached_subject:
        subject, source, confidence = cached_subject, "cache.main_subject", "medium"
    elif detected:
        subject, source, confidence = detected[0], "cache.detected_entities", "medium"
    elif full_text:
        caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", full_text)
        if caps:
            subject, source, confidence = clean_entity_name(caps[0]), "capitalized_phrase", "low"
    if not subject:
        subject, source, confidence = "Featured Reel", "fallback", "none"

    descriptors: List[str] = []
    if vc.get("intent"):
        descriptors.append(str(vc["intent"]).replace("_", " "))
    if vc.get("tone"):
        descriptors.append(str(vc["tone"]))
    if ep.get("vibe_summary"):
        descriptors.append(str(ep["vibe_summary"]))
    snippets = [s.strip() for s in re.split(r"[\n\r\t,#|.]+", f"{source_title} {raw_caption} {video_context}")
                if len(s.strip()) > 3]
    stop = {"video", "reels", "shorts", "trending", "viral", "daily inspiration", "viral moment",
            "daily special", "none", "null"}
    for snip in snippets[:4]:
        sl = snip.lower()
        if subject.lower() not in sl and sl not in subject.lower() and sl not in stop:
            descriptors.append(snip)
    fluff = {"daily inspiration", "viral moment", "daily special", "trending feature", "viral reel", "daily", "none", "null"}
    descriptors = [d for d in descriptors if d.lower().strip() not in fluff]
    context = ", ".join(dict.fromkeys(descriptors)) if descriptors else (
        video_context[:100].strip() if len((video_context or "").strip()) > 3 else "video highlights")

    return {
        "main_subject": strip_system_and_tracking_tokens(clean_entity_name(subject)) or "Featured Reel",
        "applicable_context": strip_system_and_tracking_tokens(context) or "video highlights",
        "subject_source": source, "subject_confidence": confidence,
        "raw_caption": raw_caption, "source_title": source_title,
        "detected_entities": detected,
        "intent": vc.get("intent", "general"), "tone": vc.get("tone", "engaging"),
        "audio_vibe": audio_ctx.get("dominant_emotion", ""),
    }


_MERCHANTS = (("amazon", "Amazon"), ("amzn", "Amazon"), ("meesho", "Meesho"), ("myntra", "Myntra"),
              ("ajio", "Ajio"), ("flipkart", "Flipkart"), ("fkrt", "Flipkart"), ("nykaa", "Nykaa"))


def _merchant(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    for key, name in _MERCHANTS:
        if key in host:
            return name
    return "Product"


def _extract_links(affiliate_link: Any) -> List[str]:
    if not affiliate_link:
        return []
    items = affiliate_link if isinstance(affiliate_link, (list, tuple)) else [affiliate_link]
    found: List[str] = []
    for item in items:
        found.extend(_URL_RE.findall(str(item)))
    return list(dict.fromkeys(u.strip() for u in found if u.strip()))


def _compact(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}


def _clean_metadata(md: Dict[str, Any], guard: ProvenanceGuard) -> Dict[str, Any]:
    """Only whitelisted fields survive; every identifier/URL field is dropped; text fields are scrubbed."""
    out: Dict[str, Any] = {}
    for k in _CAPTION_KEYS:
        if isinstance(md.get(k), str):
            out[k] = guard.scrub_evidence_text(md[k])
    if md.get("hashtags") is not None:
        out["hashtags"] = guard.scrub_hashtags(md["hashtags"])
    return out


def build_evidence(
    video_context: str = "",
    user_title: str = "",
    brand_info: str = "",
    cache: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    links: Optional[List[str]] = None,
    user_hint: Optional[str] = None,
    performance_hints: Optional[List[Dict[str, Any]]] = None,
    previous: Optional[Dict[str, Any]] = None,
    guard: Optional[ProvenanceGuard] = None,
    content_origin: str = "reused",
) -> Dict[str, Any]:
    """One compact, source-scrubbed packet of everything the pipeline knows about this clip."""
    md = _as_dict(metadata)
    guard = guard or ProvenanceGuard.from_inputs(md, video_context, cache, user_title, user_hint or "", links)
    md_clean = _clean_metadata(md, guard)
    vc_clean = guard.scrub_evidence_text(video_context or "")
    rc = _resolve_cache(cache)
    ex = extract_main_subject_and_context(vc_clean, md_clean, cache, user_hint=user_hint)
    vc, ep = rc["visual_ctx"], rc["editing"]
    subject = guard.scrub_text(ex["main_subject"]) or "Featured Reel"
    tags = md_clean.get("hashtags") or []
    hint_txt = re.sub(r"\s+", " ", _URL_RE.sub("", user_hint or "")).strip()

    ev = {
        "content_origin": content_origin,
        "subject": {"value": subject, "confidence": ex["subject_confidence"], "source": ex["subject_source"]},
        "context": ex["applicable_context"],
        "user_title": _clip(user_title, 200),
        "user_hint": _clip(hint_txt, 200),
        "video_context": _clip(vc_clean, 1200),
        "original_caption": _clip(ex["raw_caption"], 600),
        "source_title": _clip(ex["source_title"], 200),
        "source_hashtags": [str(t) for t in tags[:25]],
        "transcript": _clip(guard.scrub_evidence_text(rc["transcript"]), 1500),
        "detected_entities": ex["detected_entities"][:10],
        "intent": ex["intent"] if ex["intent"] != "general" else "",
        "tone": ex["tone"] if ex["tone"] != "engaging" else "",
        "audio_vibe": ex["audio_vibe"],
        "visual_intel": _clip_obj(guard.scrub_obj(vc), 1200),
        "editing_vibe": ep.get("vibe_summary"),
        "editing_plan": _clip_obj(guard.scrub_obj({k: v for k, v in ep.items() if k != "vibe_summary"}), 600),
        "creator_behavior": _clip_obj(guard.scrub_obj(rc["cbm"]), 800),
        "audio_math": _clip_obj(rc["audio_math"], 400),
        "analytics": _compact({"likes": md.get("likesCount"), "views": md.get("videoViewCount")}),
        "brand": _clip(brand_info, 300),
        "product_links": [{"slot": f"[[AFF_LINK_{i}]]", "merchant": _merchant(u)} for i, u in enumerate(links or [], 1)],
        "past_performance": _clip_obj(performance_hints[:6] if performance_hints else None, 900),
        "previous_attempt": _clip_obj(previous, 900),
    }
    return _compact(ev)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-UPDATING LEARNING STORE
# ══════════════════════════════════════════════════════════════════════════════

_LEARN_LOCK = threading.Lock()
_METRIC_PRIORITY = ("revenue_per_1k", "conversions_per_1k", "clicks_per_1k", "engagement_pct")


def _learning_enabled() -> bool:
    return _env_flag("SEO_LEARNING", "1")


def _learn_path() -> str:
    return os.getenv("SEO_LEARNING_PATH") or os.path.join("data", "seo_learning.json")


def _learn_load() -> Dict[str, Any]:
    try:
        with open(_learn_path(), encoding="utf-8") as fh:
            d = json.load(fh)
        if isinstance(d, dict) and isinstance(d.get("records"), list):
            d.setdefault("lessons", {})
            return d
    except (FileNotFoundError, ValueError, OSError):
        pass
    return {"version": 1, "records": [], "lessons": {}}


def _learn_save(d: Dict[str, Any]) -> None:
    path = _learn_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    d["records"] = d["records"][-2000:]
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _slug(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s or "").lower()).strip("-")[:40] or "general"


def _bucket(n: int, edges: Tuple[int, ...]) -> str:
    for e in edges:
        if n < e:
            return f"<{e}"
    return f">={edges[-1]}"


def extract_features(platform: str, content: Dict[str, Any], analysis: Dict[str, Any], link_mode: str) -> Dict[str, Any]:
    title, desc = content.get("title", ""), content.get("description", "")
    kw = str(analysis.get("primary_keyword") or "")
    return {
        "title_len": _bucket(len(title), (30, 50, 70)),
        "kw_in_title_first40": bool(kw) and _kw_hit(title[:40], kw),
        "title_question": "?" in title,
        "title_number": bool(re.search(r"\d", title)),
        "title_vs": bool(re.search(r"\bvs\.?\b|\bversus\b", title, re.I)),
        "hashtag_count": len(content.get("hashtags") or []),
        "desc_len": _bucket(len(desc), (150, 400, 900)),
        "cta_question": "?" in desc,
        "link_mode": link_mode,
        "intent": str(analysis.get("commercial_intent") or analysis.get("search_intent") or ""),
    }


def performance_score(metrics: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """Best available commercial metric wins: revenue > conversions > link clicks > engagement."""
    m = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None) for k, v in metrics.items()}
    views = m.get("views") or 0.0
    if views < 100:
        return None, ""
    per_k = lambda x: 1000.0 * x / views
    if m.get("revenue") is not None and views >= 200:
        return per_k(m["revenue"]), "revenue_per_1k"
    if m.get("conversions") is not None and views >= 200:
        return per_k(m["conversions"]), "conversions_per_1k"
    if m.get("link_clicks") is not None and views >= 200:
        return per_k(m["link_clicks"]), "clicks_per_1k"
    eng = (m.get("likes") or 0) + 2 * (m.get("comments") or 0) + 3 * (m.get("shares") or 0) + 2 * (m.get("saves") or 0)
    return 100.0 * eng / views, "engagement_pct"


def _scored_group(records: List[Dict[str, Any]], platform: str, niche: Optional[str]) -> Tuple[List[Dict[str, Any]], str]:
    pool = [r for r in records if r.get("platform") == platform and r.get("score") is not None]
    if niche:
        niche_pool = [r for r in pool if r.get("niche") == niche]
        if len(niche_pool) >= 3:
            pool = niche_pool
    for mt in _METRIC_PRIORITY:
        grp = [r for r in pool if r.get("metric_type") == mt]
        if len(grp) >= 3:
            return grp, mt
    return [], ""


def _patterns(grp: List[Dict[str, Any]], min_support: int = 3) -> List[str]:
    if len(grp) < 2 * min_support:
        return []
    overall = sum(r["score"] for r in grp) / len(grp)
    if overall <= 0:
        return []
    found: List[Tuple[float, str]] = []
    keys = {k for r in grp for k in _as_dict(r.get("features"))}
    for k in keys:
        vals = {}
        for r in grp:
            vals.setdefault(str(_as_dict(r.get("features")).get(k)), []).append(r["score"])
        for v, scores in vals.items():
            if min_support <= len(scores) <= len(grp) - min_support:
                lift = (sum(scores) / len(scores)) / overall - 1.0
                if abs(lift) >= 0.15:
                    found.append((abs(lift) * math.sqrt(len(scores)), f"{k}={v}: {lift:+.0%} vs average (n={len(scores)})"))
    return [s for _, s in sorted(found, reverse=True)[:6]]


def learned_context(platform: str, niche: Optional[str] = None, k: int = 3) -> Dict[str, Any]:
    if not _learning_enabled():
        return {}
    try:
        with _LEARN_LOCK:
            data = _learn_load()
    except Exception as exc:
        logger.debug("[PlatformSEO] learning store unreadable: %s", exc)
        return {}
    grp, metric = _scored_group(data["records"], platform, niche)
    ctx: Dict[str, Any] = {}
    if grp:
        top = sorted(grp, key=lambda r: r["score"], reverse=True)[:k]
        ctx["metric"] = metric
        ctx["top_examples"] = [{"title": r.get("title"), "score": round(r["score"], 2),
                                "features": r.get("features")} for r in top]
        ctx["measured_patterns"] = _patterns(grp)
    lesson = data["lessons"].get(f"{platform}|{niche or '*'}") or data["lessons"].get(f"{platform}|*")
    if lesson:
        ctx["lessons"] = lesson.get("rules", [])
    return ctx


def _log_generation(seo_id: str, platform: str, niche: str, title: str, features: Dict[str, Any], link_mode: str) -> None:
    if not _learning_enabled():
        return
    try:
        with _LEARN_LOCK:
            d = _learn_load()
            d["records"].append({"id": f"{seo_id}|{platform}", "seo_id": seo_id, "platform": platform, "niche": niche,
                                 "ts": _now(), "title": title, "features": features, "link_mode": link_mode,
                                 "metrics": None, "score": None, "metric_type": None})
            _learn_save(d)
    except Exception as exc:
        logger.warning("[PlatformSEO] could not log generation: %s", exc)


def record_performance(seo_id: str, platform: str, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Feed real results back. metrics: views (required), plus any of likes, comments, shares, saves, link_clicks,
    conversions, revenue. The best available commercial metric becomes the score. Every SEO_DISTILL_EVERY scored
    results for a platform+niche, Gemini distills the data into updated rules for the next generations."""
    if not _learning_enabled():
        return None
    try:
        with _LEARN_LOCK:
            d = _learn_load()
            rec = next((r for r in d["records"] if r.get("id") == f"{seo_id}|{platform}"), None)
            if rec is None:
                return None
            score, mt = performance_score(metrics)
            rec.update({"metrics": metrics, "score": score, "metric_type": mt or None, "scored_at": _now()})
            _learn_save(d)
            niche = rec.get("niche")
            n_scored = sum(1 for r in d["records"] if r.get("platform") == platform and r.get("niche") == niche
                           and r.get("score") is not None)
        try:
            every = int(os.getenv("SEO_DISTILL_EVERY", "10"))
        except ValueError:
            every = 10
        if score is not None and every > 0 and n_scored > 0 and n_scored % every == 0:
            _distill(platform, niche)
        return rec
    except Exception as exc:
        logger.warning("[PlatformSEO] record_performance failed: %s", exc)
        return None


def _distill(platform: str, niche: Optional[str]) -> None:
    with _LEARN_LOCK:
        data = _learn_load()
    grp, metric = _scored_group(data["records"], platform, niche)
    if len(grp) < 6:
        return
    ranked = sorted(grp, key=lambda r: r["score"], reverse=True)
    sample = {"metric": metric, "n": len(grp),
              "top": [{"title": r["title"], "score": round(r["score"], 2), "features": r["features"]} for r in ranked[:5]],
              "bottom": [{"title": r["title"], "score": round(r["score"], 2), "features": r["features"]} for r in ranked[-5:]],
              "measured_patterns": _patterns(grp)}
    prompt = ("You analyze short-form video SEO results for ONE account.\n"
              f"PLATFORM: {platform}   NICHE: {niche}\nDATA:\n{json.dumps(sample, ensure_ascii=False, indent=1)}\n\n"
              "Write at most 6 concrete, testable rules for the NEXT titles/captions (hook type, length, keyword position, "
              "link route, CTA type). Only claim what the data supports; with n < 15 say 'weak evidence'. "
              'Return ONLY JSON: {"rules": ["..."]}')
    out = _llm_json(prompt, 0.2, "distill")
    rules = [str(r) for r in _as_dict(out).get("rules", []) if r][:6]
    if rules:
        with _LEARN_LOCK:
            d = _learn_load()
            d["lessons"][f"{platform}|{niche or '*'}"] = {"updated": _now(), "n": len(grp), "rules": rules}
            _learn_save(d)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_RULES = """\
GROUNDING
- Use ONLY facts in EVIDENCE. Never invent prices, discounts, ratings, specs, test results, "I tested/used/bought",
  sample sizes or timeframes. Every number above 10 that you print must appear in EVIDENCE.
- subject.confidence says how sure the heuristic was. If transcript/caption/entities contradict subject.value, trust them
  and put the corrected subject in analysis.subject_override.
- Never imply endorsement, sponsorship, partnership, or that a person in the video wears/uses/recommends the product,
  unless EVIDENCE says so.
- If learned exists (measured results on THIS account), prefer its high-lift patterns; they are correlations, not laws.
- If previous_attempt exists, do not reuse its titles verbatim.
SOURCE PRIVACY (hard rule)
- The footage may come from other creators. Never name, tag, thank, credit, link or hint at any source channel/creator/page:
  no @mentions, no handles, no credit/via/repost/source/original/inspired-by/DC wording, no non-affiliate URLs.
  EVIDENCE is already scrubbed; do not reconstruct names from context.
COMMERCIAL OPTIMIZATION
- Goal = qualified clicks and sales, not vanity reach. Match the title and first sentence to the query a BUYER types
  (review, vs, worth it, alternatives, for <use-case>). The primary keyword is in the first ~40 chars of the title and in
  the first sentence of the caption.
- Sell the specific value props actually shown or said in the video (analysis.value_props); answer one objection if the
  transcript does. ONE call to action per platform, matched to its link mode.
- No hype superlatives ("best ever", "guaranteed", "#1") unless in EVIDENCE. No engagement bait ("like if", "tag a friend",
  "comment YES"); a genuine question is fine.
REUSED / TRANSFORMED CONTENT
- If content_origin is "reused": never claim ownership or first-hand use ("my video", "I filmed/tested/reviewed",
  "exclusive", "original"). Foreground what this edit ADDS (analysis.transformation_angle: commentary, comparison,
  breakdown, context) and only if the transcript/edit plan supports it. Never merely describe the on-screen action.
LANGUAGE
- Write in the language/register of the user title, caption and transcript (English, Hinglish, Malayalam, ...).
  Keep search keywords in the form people type.
STYLE
- No motivational/corporate filler. Short lines, one blank line between blocks, emoji only as markers, mobile-scannable.
- Keywords live inside natural sentences. NEVER a visible keyword list. The "keywords" array is metadata only.
- Never write "link in bio", "in my bio", "check bio".
LINKS & DISCLOSURE
- Product slots like [[AFF_LINK_1]] are copied verbatim, never altered, never in a title, never invent a URL.
- If product_links exist, the affiliate relationship exists in EVERY link mode, so the disclosure line is always required,
  at the position the platform brief states. Follow each platform's link mode exactly.
HASHTAGS
- Only tags that describe THIS video's niche/topic/intent. Never #viral #trending #reels #fyp #foryou #explore.
"""

_GENERATE_TEMPLATE = """\
You are an elite commercial short-form SEO strategist. Write platform-native titles, captions, keywords and hashtags for ONE
video, grounded strictly in EVIDENCE, to maximize qualified buyer clicks.

WORK IN THIS ORDER (silently; output only the JSON):
1. STRATEGY: primary buyer keyword, niche, commercial intent + funnel stage, 4-8 buyer queries supported by the evidence,
   the value props the video actually shows/says, likely objections, and the transformation angle (what this edit adds).
2. For EACH platform: write 3 title candidates with different hook types (question / contrast / outcome / specific detail),
   choose the one that best serves THAT platform's goal, then the caption and hashtags following its brief and link mode.

__RULES__
PLATFORM BRIEFS
__BRIEFS__

EVIDENCE
__EVIDENCE__

Return ONLY this JSON (no prose, no code fences). Include only: __PLATFORMS__
{
  "analysis": {
    "primary_keyword": "", "secondary_keywords": [], "niche": "<2-3 word product/topic category>",
    "commercial_intent": "transactional|commercial_investigation|deal|informational|entertainment",
    "funnel_stage": "awareness|consideration|purchase",
    "buyer_queries": [], "value_props": [], "objections": [], "transformation_angle": "",
    "audience": "", "hook_angle": "", "subject_override": "", "on_screen_text": ["<<=6 word overlays that echo the primary keyword>"]
  },
  "platforms": {
    "<platform>": {"title": "", "title_alternates": ["", ""], "description": "", "keywords": [], "hashtags": ["#tag"]}
  },
  "content_category": "", "target_audience": ""
}
"""

_CRITIC_TEMPLATE = """\
You are a ruthless commercial SEO editor reviewing a junior's draft. Be harsh: 8+ means "I would ship this unchanged".

For EACH platform in CURRENT_DRAFT:
1. Score 0-10 on: __CRITERIA__.
2. List up to 3 concrete weaknesses (quote the offending text).
3. If any score is below 8 or LINT_ISSUES is non-empty for that platform, return "revised": a COMPLETE replacement
   {title, title_alternates, description, keywords, hashtags} that fixes them. Otherwise "revised": null.
Do not lengthen for its own sake. Do not add claims absent from EVIDENCE. Keep [[AFF_LINK_n]] slots verbatim.
'hard' lint issues are mandatory fixes. Never introduce source names, @mentions or credit wording.

__RULES__
PLATFORM BRIEFS
__BRIEFS__

EVIDENCE
__EVIDENCE__

ANALYSIS
__ANALYSIS__

CURRENT_DRAFT
__DRAFT__

LINT_ISSUES
__LINT__

Return ONLY this JSON (no prose, no code fences):
{
  "platforms": {
    "<platform>": {
      "scores": {"evidence_grounding": 0, "keyword_placement": 0, "hook_strength": 0, "scannability": 0, "cta_fit": 0, "hashtag_relevance": 0, "commercial_intent_match": 0},
      "weaknesses": [""], "revised": null
    }
  }
}
"""


def _link_instruction(mode: str) -> str:
    kw = _dm_keyword()
    return {
        "inline": "put the product slots (one per line) in a links block.",
        "first_comment": "put NO slot and NO URL in the caption; add one pointer line such as 'Link in the first comment 👇'.",
        "dm_keyword": f"put NO slot and NO URL in the caption; add one pointer line: 'Comment {kw} and I'll DM you the link 👇'.",
        "none": "put NO slot, NO URL and NO pointer line in the caption.",
    }[mode]


_PLACEMENT_TEXT = {
    "prefix": "the caption's FIRST LINE starts with the disclosure ('#ad ' + hook).",
    "line2": "the disclosure is line 2 (directly under line 1), so it is within the first two lines.",
}


def _render_brief(platform: str, modes: Dict[str, str]) -> str:
    s = PLATFORM_SPECS[platform]
    lines = [f"### {platform.upper()} — {s['label']}"]
    t = f"- Title ≤ {s['title_max']} characters" + (f", must end with {s['title_suffix']}" if s.get("title_suffix") else "")
    lines.append(t)
    d = f"- Description ≤ {s['description_max']} characters"
    if s.get("soft_text_max"):
        d += f"; text excluding URLs ≤ {s['soft_text_max']}"
    lines.append(d)
    lines.append(f"- Hashtags: {s['hashtags_min']}-{s['hashtags_max']} tags; total # across caption + list ≤ {s['hashtags_total_max']}")
    lines.append(f"- Disclosure (when product slots exist): exactly {s['disclosure']} — {_PLACEMENT_TEXT[s['disclosure_placement']]}")
    lines.append(f"- Link mode '{modes[platform]}': {_link_instruction(modes[platform])}  ({s['caption_link_note']})")
    lines.extend(f"- {r}" for r in s["rules"])
    return "\n".join(lines)


def _fill(template: str, **kw: str) -> str:
    out = template
    for k, v in kw.items():
        out = out.replace(f"__{k.upper()}__", v)
    return out


def _build_generation_prompt(evidence: Dict[str, Any], platforms: List[str], modes: Dict[str, str]) -> str:
    return _fill(_GENERATE_TEMPLATE, rules=_RULES,
                 briefs="\n\n".join(_render_brief(p, modes) for p in platforms),
                 evidence=json.dumps(evidence, ensure_ascii=False, indent=1), platforms=", ".join(platforms))


def _build_critic_prompt(evidence: Dict[str, Any], analysis: Dict[str, Any], draft: Dict[str, Any],
                         lint: Dict[str, Any], platforms: List[str], modes: Dict[str, str]) -> str:
    return _fill(_CRITIC_TEMPLATE, criteria=", ".join(_COMMON_CRITERIA), rules=_RULES,
                 briefs="\n\n".join(_render_brief(p, modes) for p in platforms),
                 evidence=json.dumps(evidence, ensure_ascii=False, indent=1),
                 analysis=json.dumps(analysis, ensure_ascii=False, indent=1),
                 draft=json.dumps(draft, ensure_ascii=False, indent=1),
                 lint=json.dumps(lint, ensure_ascii=False, indent=1))


# ══════════════════════════════════════════════════════════════════════════════
# LLM I/O
# ══════════════════════════════════════════════════════════════════════════════

def _clean_json(text: str) -> str:
    if not text:
        return "{}"
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        text = m.group(1) if m else text.replace("```json", "").replace("```", "")
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1].strip()
    return text.strip()


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    s = _clean_json(text)
    for cand in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            data = json.loads(cand)
            return data if isinstance(data, dict) else None
        except Exception:
            continue
    return None


def _llm_json(prompt: str, temperature: float, label: str) -> Optional[Dict[str, Any]]:
    router = _get_router()
    if router is None:
        return None
    suffix = ""
    for attempt in (1, 2):
        try:
            raw = router.generate(task_type=GEN_TASK_TYPE, prompt=prompt + suffix,
                                  module_name="platform_seo_generator", gen_config={"temperature": temperature})
        except Exception as exc:
            logger.warning("[PlatformSEO] %s call failed: %s", label, exc)
            return None
        if not raw:
            logger.warning("[PlatformSEO] %s returned empty.", label)
            return None
        text = raw if isinstance(raw, str) else (getattr(raw, "text", None) or str(raw))
        data = _parse_json(text)
        if data is not None:
            return data
        logger.warning("[PlatformSEO] %s: unparseable JSON (attempt %d)", label, attempt)
        suffix = ("\n\nYour previous reply was not valid JSON. Reply with ONE valid JSON object only: "
                  "no prose, no code fences, no trailing commas.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ENFORCEMENT + LINT
# ══════════════════════════════════════════════════════════════════════════════

_BIO_RE = re.compile(
    r"(?i)\b(?:tap\s+the\s+link\s+in\s+(?:my\s+)?bio(?:\s+or\s+visit\s+directly)?"
    r"|tap\s+(?:the\s+)?link\s+in\s+bio|link\s+in\s+(?:my\s+)?bio(?:\s+to\s+shop)?"
    r"|(?:check|see)\s+(?:the\s+|my\s+)?bio(?:\s+link)?|tap\s+bio|in\s+(?:my|the)\s+bio|bio\s+link)\b")
_BIO_AND_BELOW_RE = re.compile(r"(?i)\blink\s+in\s+bio\s*&?\s*below\b")
_FLUFF_RE = re.compile(r"(?i)(elevate your perspective|radiate (?:absolute )?confidence|true confidence comes from|"
                       r"mastering the essentials|unlock your (?:true )?potential)")
_DISC_ANY = re.compile(r"\(\s*(?:Disclosure:\s*)?#ad\b[^)]*\)|(?<!\w)#(?:ad|affiliate|sponsored|collab|promo|partnership)(?!\w)", re.I)
_URL_THEN_TEXT = re.compile(r'(https?://[^\s<>"]+)[ \t]+(?=\S)')
_DANGLING_BULLET = re.compile(r"(?m)^[ \t]*(?:👉|•|-|\*)[ \t]*$\n?")
_OWNERSHIP_RE = re.compile(
    r"(?i)\b(?:i|we)\s+(?:have\s+)?(?:tested|tried|filmed|shot|made|reviewed|used|bought|wore|tracked)\b"
    r"|\bmy\s+(?:review|test|video|footage|experience|unboxing)\b|\bexclusive\s+(?:footage|video)\b"
    r"|\boriginal\s+content\b|\bfirst[\s-]?hand\b|\bafter\s+\d+\s+(?:days|weeks|months)\b")
_ENDORSE_RE = re.compile(
    r"(?i)\b(?:officially|official\s+(?:partner|store|collab)|endorsed\s+by|sponsored\s+by|partnered\s+with|"
    r"collab(?:oration)?\s+with|worn\s+by|recommended\s+by|brand\s+ambassador)\b")
_BAIT_RE = re.compile(
    r"(?i)\b(?:like|share|react|comment|tag)\s+(?:this\s+)?if\b|\btag\s+(?:a|your|some|\d+)\s+(?:friend|friends|bestie)|"
    r"\bcomment\s+[\"“']?(?:yes|no|1|2|a|b)[\"”']?\s+(?:if|to)\b|\bshare\s+(?:this\s+)?(?:with|to)\s+\d+\s+friends")
_SUPERLATIVE_RE = re.compile(r"(?i)\b(?:best\s+ever|cheapest|lowest\s+price|guaranteed|number\s+one|100%\s*(?:original|genuine|safe))\b|#1\b")


def _fit_title(platform: str, title: str, fallback: str = "Untitled Video") -> str:
    spec = PLATFORM_SPECS[platform]
    t = " ".join(_SLOT_RE.sub("", _URL_RE.sub("", title or "")).split()) or fallback
    suffix = spec.get("title_suffix")
    if suffix:
        t = re.sub(rf"\s*{re.escape(suffix)}\s*$", "", t, flags=re.I).strip() or fallback
        return f"{_cut_words(t, spec['title_max'] - len(suffix) - 1)} {suffix}"
    return _cut_words(t, spec["title_max"])


def _fit_text(text: str, max_len: int) -> str:
    """Shrink without cutting a URL or dropping the disclosure/first two lines."""
    if len(text) <= max_len:
        return text
    paras = text.split("\n\n")

    def protected(p: str) -> bool:
        return bool(_URL_RE.search(p)) or bool(re.search(r"#ad\b", p, re.I))

    while len("\n\n".join(paras)) > max_len:
        idx = next((i for i in range(len(paras) - 1, 0, -1) if not protected(paras[i])), None)
        if idx is None:
            break
        paras.pop(idx)
    out = "\n\n".join(paras)
    if len(out) > max_len and not protected(paras[0]):
        keep = len(paras[0]) - (len(out) - max_len)
        if keep >= 12:
            paras[0] = _cut_words(paras[0], keep)
            out = "\n\n".join(paras)
    if len(out) > max_len:
        logger.warning("[PlatformSEO] text still %d chars > %d after fitting.", len(out), max_len)
    return out


def _clean_tag(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    t = _to_tag(raw.strip().lstrip("#"))
    key = t[1:].lower()
    if not t or len(key) < 2 or len(key) > 30 or key in _GENERIC_TAGS or _norm_key(key) in _ATTRIB_TAGS:
        return ""
    return t


def _fit_hashtags(platform: str, tags: Any, backfill: List[str], budget: int,
                  guard: Optional[ProvenanceGuard] = None) -> List[str]:
    spec = PLATFORM_SPECS[platform]
    cap = max(0, min(spec["hashtags_max"], budget))
    out: List[str] = []
    seen: set = set()
    for raw in (guard.scrub_hashtags(tags) if guard and isinstance(tags, list) else tags if isinstance(tags, list) else []):
        t = _clean_tag(raw)
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    out = out[:cap]
    floor = min(spec["hashtags_min"], cap)
    for raw in (guard.scrub_hashtags(backfill) if guard else backfill):
        if len(out) >= floor:
            break
        t = _clean_tag(raw)
        if t and t.lower() not in {x.lower() for x in out}:
            out.append(t)
    return out


def _ensure_links(platform: str, desc: str, links: List[str]) -> str:
    missing = [u for u in links if u not in desc]
    if not missing:
        return desc
    label = PLATFORM_SPECS[platform].get("links_label") or ""
    new_lines = [f"👉 {u}" for u in missing]
    if label and label in desc:
        lines = desc.split("\n")
        last = max((i for i, ln in enumerate(lines) if _URL_RE.search(ln) or label in ln), default=len(lines) - 1)
        lines[last + 1:last + 1] = new_lines
        return "\n".join(lines)
    return desc.rstrip() + "\n\n" + (label + "\n" if label else "") + "\n".join(new_lines)


def _ensure_pointer(desc: str, mode: str) -> str:
    kw = _dm_keyword()
    if mode == "first_comment":
        if re.search(r"first\s+comment", desc, re.I):
            return desc
        return desc.rstrip() + "\n\n👇 Link in the first comment"
    if mode == "dm_keyword":
        if re.search(rf"comment\s+{re.escape(kw)}\b", desc, re.I):
            return desc
        return desc.rstrip() + f"\n\n👇 Comment {kw} and I'll DM you the link"
    return desc


def _place_disclosure(platform: str, desc: str, has_links: bool) -> str:
    """ASCI: the label must sit within the first two lines (no 'see more'). v2 appended it at the END."""
    if not has_links:
        return desc
    spec = PLATFORM_SPECS[platform]
    disc = spec["disclosure"]
    desc = _DISC_ANY.sub("", desc)
    desc = re.sub(r"\(\s*\)", "", desc)
    desc = "\n".join(re.sub(r"[ \t]{2,}", " ", ln).rstrip() for ln in desc.split("\n")).lstrip("\n")
    if spec["disclosure_placement"] == "prefix":
        head, sep, rest = desc.partition("\n")
        return f"{disc} {head}".strip() + (sep + rest if sep else "")
    head, _, rest = desc.partition("\n")
    rest = rest.lstrip("\n")
    return head + "\n" + disc + ("\n\n" + rest if rest else "")


def _disclosure_ok(platform: str, desc: str) -> bool:
    if PLATFORM_SPECS[platform]["disclosure_placement"] == "prefix":
        return bool(re.match(r"\s*#ad\b", desc, re.I))
    return bool(re.search(r"#ad\b", "\n".join(desc.split("\n")[:2]), re.I))


def _count_desc_tags(desc: str) -> int:
    return len(re.findall(r"#\w+", _URL_RE.sub("", desc)))


def _kw_hit(text: str, kw: str) -> bool:
    if not kw:
        return True
    t = (text or "").lower()
    words = [w for w in (x.strip(string.punctuation + "…—–") for x in kw.lower().split()) if len(w) > 2]
    if not words:
        return kw.lower() in t
    return sum(1 for w in words if w in t) >= max(1, math.ceil(len(words) / 2))


def _keyword_density(desc: str, kw: str) -> str:
    if not kw:
        return "n/a"
    plain = _URL_RE.sub("", desc)
    n_words = max(1, len(plain.split()))
    return f"{100.0 * plain.lower().count(kw.lower()) * max(1, len(kw.split())) / n_words:.1f}%"


def _norm_num(s: str) -> str:
    s = s.replace(",", "")
    return s[:-2] if s.endswith(".0") else s


def _content_numbers(text: str) -> set:
    t = re.sub(r"#\w+", " ", _URL_RE.sub(" ", _SLOT_RE.sub(" ", text)))
    return {_norm_num(n) for n in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", t)}


def lint_platform(platform: str, content: Dict[str, Any], links: Optional[List[str]] = None,
                  primary_keyword: str = "", ctx: Optional[Dict[str, Any]] = None) -> Tuple[int, List[Dict[str, str]]]:
    """Verifiable checks only. Returns (score 0-100, issues). hard = -25, soft = -8."""
    spec = PLATFORM_SPECS[platform]
    ctx = ctx or {}
    links = links or []
    mode = ctx.get("link_mode", "inline")
    guard: Optional[ProvenanceGuard] = ctx.get("guard")
    issues: List[Dict[str, str]] = []

    def add(code: str, sev: str, msg: str) -> None:
        issues.append({"platform": platform, "code": code, "severity": sev, "message": msg})

    title = content.get("title") or ""
    desc = content.get("description") or ""
    tags = content.get("hashtags") or []
    both = f"{title}\n{desc}"
    both_no_tags = re.sub(r"#\w+", " ", both)

    if not title.strip():
        add("title_empty", "hard", "Title is empty.")
    if len(title) > spec["title_max"]:
        add("title_too_long", "hard", f"Title is {len(title)} chars; max {spec['title_max']}.")
    suffix = spec.get("title_suffix")
    if suffix and not title.rstrip().lower().endswith(suffix.lower()):
        add("title_suffix_missing", "hard", f"Title must end with {suffix}.")
    if not desc.strip():
        add("description_empty", "hard", "Description is empty.")
    if len(desc) > spec["description_max"]:
        add("description_too_long", "hard", f"Description is {len(desc)} chars; max {spec['description_max']}.")
    soft_max = spec.get("soft_text_max")
    if soft_max and len(_URL_RE.sub("", desc)) > soft_max:
        add("text_too_long", "soft", f"Text (excluding URLs) is {len(_URL_RE.sub('', desc))} chars; target ≤ {soft_max}.")
    if _BIO_RE.search(both):
        add("link_in_bio", "hard", "Contains a 'link in bio' style phrase.")
    if _FLUFF_RE.search(both):
        add("fluff", "hard", "Contains motivational/corporate filler.")

    # links + disclosure
    if links and mode == "inline":
        for u in links:
            if u not in desc:
                add("link_missing", "hard", f"Affiliate link missing from description: {u}")
    if links and mode != "inline" and _URL_RE.search(desc):
        add("url_in_caption", "hard", f"Link mode is '{mode}': no URL belongs in the caption.")
    if links and mode in ("first_comment", "dm_keyword"):
        pat = r"first\s+comment" if mode == "first_comment" else rf"comment\s+{re.escape(_dm_keyword())}\b"
        if not re.search(pat, desc, re.I):
            add("pointer_missing", "soft", "Caption lacks the pointer line for this link mode.")
    if links and not _disclosure_ok(platform, desc):
        where = "the first line" if spec["disclosure_placement"] == "prefix" else "the first two lines"
        add("disclosure_position", "hard", f"Affiliate disclosure must be in {where} (ASCI: no 'see more'). Use exactly {spec['disclosure']}.")

    # source privacy
    if guard:
        leaks = guard.find_leaks(both) + guard.find_leaks(" ".join(tags)) + \
            [x for k in content.get("keywords", []) for x in guard.find_leaks(str(k))]
        if leaks:
            add("source_leak", "hard", f"Source-channel credit leaked: {', '.join(sorted(set(leaks))[:3])}")

    # hashtags
    n_tags = len(tags)
    if n_tags > spec["hashtags_max"]:
        add("too_many_hashtags", "hard", f"{n_tags} hashtags; max {spec['hashtags_max']}.")
    if n_tags < spec["hashtags_min"]:
        add("too_few_hashtags", "soft", f"{n_tags} hashtags; min {spec['hashtags_min']}.")
    total = n_tags + _count_desc_tags(desc)
    if total > spec["hashtags_total_max"]:
        add("hashtag_budget", "hard", f"{total} total # across caption + list; platform cap {spec['hashtags_total_max']}.")
    generic = [t for t in tags if str(t).lstrip("#").lower() in _GENERIC_TAGS]
    if generic:
        add("generic_hashtags", "soft", f"Generic tags hurt classification: {', '.join(generic)}")

    # commercial-risk claims
    ev_text = ctx.get("evidence_text", "")
    if ev_text:
        ev_nums = {_norm_num(n) for n in re.findall(r"\d+(?:\.\d+)?", ev_text.replace(",", ""))}
        bad = []
        for n in _content_numbers(both_no_tags):
            try:
                v = float(n)
            except ValueError:
                continue
            if v > 10 and not (1990 <= v <= 2100) and n not in ev_nums:
                bad.append(n)
        if bad:
            add("ungrounded_number", "hard", f"Number(s) not in the evidence (invented price/claim?): {', '.join(sorted(bad)[:4])}")
        if _ENDORSE_RE.search(both_no_tags) and not _ENDORSE_RE.search(ev_text):
            add("endorsement_claim", "hard", "Implies endorsement/partnership not present in the evidence.")
        if _SUPERLATIVE_RE.search(both_no_tags) and not _SUPERLATIVE_RE.search(ev_text):
            add("unsupported_superlative", "soft", "Superlative/guarantee claim not supported by the evidence.")
    if ctx.get("content_origin", "reused") == "reused" and _OWNERSHIP_RE.search(both_no_tags):
        ev_has = bool(ev_text) and False
        if not ev_has:
            add("ownership_claim", "hard", "Claims ownership/first-hand use on reused footage ('I tested', 'my video', 'original').")
    if _BAIT_RE.search(both_no_tags):
        add("engagement_bait", "hard" if platform == "facebook" else "soft", "Engagement-bait phrasing (Meta/YouTube penalize it).")

    if primary_keyword:
        if not _kw_hit(title, primary_keyword):
            add("keyword_not_in_title", "soft", f"Primary keyword '{primary_keyword}' not in title.")
        elif platform == "youtube" and not _kw_hit(title[:45], primary_keyword):
            add("keyword_not_frontloaded", "soft", f"Move '{primary_keyword}' into the first ~45 chars of the title.")
        first_sentence = re.split(r"(?<=[.!?])\s|\n", re.sub(r"^\s*#ad\s+", "", desc), maxsplit=1)[0]
        if not _kw_hit(first_sentence, primary_keyword):
            add("keyword_not_in_first_sentence", "soft", f"Primary keyword '{primary_keyword}' missing from the first sentence.")

    hook_max = spec.get("hook_visible_chars")
    if hook_max and len(desc.split("\n", 1)[0]) > hook_max:
        add("hook_truncated", "soft", f"First line is {len(desc.split(chr(10), 1)[0])} chars; only ~{hook_max} show before '… more'.")
    if len(desc) > 300 and "\n\n" not in desc:
        add("wall_of_text", "soft", "No paragraph breaks in a long description.")
    cta = spec.get("cta_re")
    if cta and not re.search(cta, desc, re.I):
        add("cta_missing", "soft", "No platform-native CTA found.")

    hard = sum(1 for i in issues if i["severity"] == "hard")
    return max(0, 100 - 25 * hard - 8 * (len(issues) - hard)), issues


def _build_link_delivery(platform: str, mode: str, links: List[str]) -> Dict[str, Any]:
    spec = PLATFORM_SPECS[platform]
    d: Dict[str, Any] = {"mode": mode, "recommended_mode": spec["link_mode_default"],
                         "clickable_in_caption": spec["caption_links_clickable"], "note": spec["caption_link_note"],
                         "urls": links}
    if not links:
        return d
    block = "\n".join(f"👉 {u}" for u in links)
    if mode == "first_comment":
        d["first_comment"] = f"🛍️ Featured items:\n{block}\n\n(#ad #affiliate)"
    elif mode == "dm_keyword":
        d["dm_keyword"] = _dm_keyword()
        d["dm_reply"] = f"Here you go 👇\n{block}\n\n(#ad #affiliate)"
    return d


def _finalize_platform(
    platform: str, raw: Dict[str, Any], *, links: List[str], slot_to_url: Dict[str, str], backfill_tags: List[str],
    fallback_title: str = "Untitled Video", primary_keyword: str = "", mode: str = "inline",
    guard: Optional[ProvenanceGuard] = None, ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Raw model/heuristic output -> publishable content. Everything here is deterministic."""
    spec = PLATFORM_SPECS[platform]
    title = str(raw.get("title") or "")
    desc = str(raw.get("description") or "")

    for token, url in slot_to_url.items():
        desc = desc.replace(token, url)
    desc = _SLOT_RE.sub("", desc)
    desc = _DANGLING_BULLET.sub("", desc)
    if mode != "inline":                              # link lives elsewhere (comment/DM/profile)
        desc = _URL_RE.sub("", desc)
    if guard:
        desc, title = guard.scrub_text(desc), guard.scrub_text(title)
    desc = _BIO_RE.sub("link below" if mode == "inline" else "", _BIO_AND_BELOW_RE.sub("link below", desc))
    title = _BIO_RE.sub("", _BIO_AND_BELOW_RE.sub("link below", title))
    desc, title = strip_system_and_tracking_tokens(desc), strip_system_and_tracking_tokens(title)

    if mode == "inline":
        desc = _ensure_links(platform, desc, links)
    else:
        desc = _ensure_pointer(desc, mode) if links else desc
    desc = _place_disclosure(platform, desc, bool(links))
    desc = _URL_THEN_TEXT.sub(r"\1\n", desc)
    desc = re.sub(r"\n{3,}", "\n\n", desc).strip()

    if platform == "telegram":
        t0, d0 = title.strip(), desc.strip()
        if t0 and d0.lower().startswith(t0.lower()):
            desc = d0[len(t0):].lstrip(" :\n*-—") or desc
    desc = _fit_text(desc or "Check out this video!", spec["description_max"])
    title = _fit_title(platform, title, fallback_title)

    tags = strip_system_and_tracking_tokens(
        _fit_hashtags(platform, raw.get("hashtags"), backfill_tags, spec["hashtags_total_max"] - _count_desc_tags(desc), guard))

    alts: List[str] = []
    for a in (raw.get("title_alternates") or []):
        if isinstance(a, str) and a.strip():
            fa = _fit_title(platform, strip_system_and_tracking_tokens(guard.scrub_text(a) if guard else a), fallback_title)
            if fa.lower() != title.lower() and fa not in alts:
                alts.append(fa)
    kws = raw.get("keywords") or []
    kws = kws if isinstance(kws, list) else [str(kws)]
    kws = [strip_system_and_tracking_tokens(guard.scrub_text(str(k)) if guard else str(k)) for k in kws if str(k).strip()]
    kws = [k for k in kws if k][:12]

    content = {"title": title, "description": desc, "hashtags": tags, "keywords": kws}
    lint_ctx = dict(ctx or {}, link_mode=mode, guard=guard)
    score, issues = lint_platform(platform, content, links, primary_keyword, lint_ctx)
    return {
        "title": title, "description": desc, "hashtags": tags, "keywords": kws,
        "title_alternates": alts[:2],
        "hashtag_placement": spec["hashtag_placement"],
        "link_delivery": _build_link_delivery(platform, mode, links),
        "seo_score": score,
        "keyword_density": _keyword_density(desc, primary_keyword),
        "lint_issues": issues,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HEURISTIC FALLBACK (0 API calls; same enforcement path)
# ══════════════════════════════════════════════════════════════════════════════

def _backfill_tags(analysis: Dict[str, Any], evidence: Dict[str, Any]) -> List[str]:
    subj = _as_dict(evidence.get("subject"))
    subject = subj.get("value", "") if subj.get("confidence") != "none" else ""
    cands: List[str] = []
    for kw in [analysis.get("primary_keyword"), *(analysis.get("secondary_keywords") or []), subject]:
        if isinstance(kw, str) and kw.strip():
            cands.append(_to_tag(kw))
    cands.extend(evidence.get("source_hashtags") or [])
    return [c for c in dict.fromkeys(cands) if c]


def _heuristic_draft(evidence: Dict[str, Any], platforms: List[str]) -> Dict[str, Dict[str, Any]]:
    """No fabricated claims and no category bias."""
    subj = _as_dict(evidence.get("subject")).get("value") or "Featured Reel"
    ctx = [c.strip() for c in (evidence.get("context") or "").split(",") if c.strip()]
    ctx = [c for c in ctx if len(c) > 3 and c.lower() not in {"video highlights", "general", "n/a"}]
    user_title = re.sub(r"\s*#(?:Shorts?|Reels?)\b", "", evidence.get("user_title") or "", flags=re.I).strip()
    base = user_title or (f"{subj} — {ctx[0]}" if ctx and ctx[0].lower() not in subj.lower() else subj)
    kws = list(dict.fromkeys([k for k in [subj] + ctx[:5] + (evidence.get("detected_entities") or [])[:5] if k and len(k) > 2]))
    tags = [t for t in dict.fromkeys(_backfill_tags({}, evidence) + [_to_tag(c) for c in ctx]) if t]
    return {
        "youtube": {"title": base, "hashtags": tags, "keywords": kws,
                    "description": f"{base}\n\n💬 What's your take? Tell us in the comments 👇"},
        "instagram": {"title": base, "hashtags": tags, "keywords": kws,
                      "description": f"{base}\n\n💾 Save this for later or send it to someone who needs it."},
        "facebook": {"title": base, "hashtags": tags, "keywords": kws,
                     "description": f"{base}\n\nWould you pick this? Tell us below 👇"},
        "telegram": {"title": base, "hashtags": tags, "keywords": kws, "description": f"🔥 **{base}**"},
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str, slot_to_url: Dict[str, str]) -> str:
    for token, url in slot_to_url.items():
        text = text.replace(url, token)
    return text


def _mean_critic_score(scores: Any) -> Optional[float]:
    vals = [float(v) for v in _as_dict(scores).values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(10.0 * sum(vals) / len(vals), 1) if vals else None


def _hard_count(issues: List[Dict[str, str]]) -> int:
    return sum(1 for i in issues if i["severity"] == "hard")


def generate_platform_seo(
    video_context: str,
    user_title: str = "",
    brand_info: str = "",
    platforms: Optional[List[str]] = None,
    cache: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    affiliate_link: Optional[Any] = None,
    user_hint: Optional[str] = None,
    optimize_rounds: Optional[int] = None,
    performance_hints: Optional[List[Dict[str, Any]]] = None,
    previous: Optional[Dict[str, Any]] = None,
    content_origin: Optional[str] = None,
    link_modes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Generate commercial, source-private, platform-native SEO and self-optimize it.

    content_origin   "reused" (default, env SEO_CONTENT_ORIGIN) | "original". Reused footage may never claim ownership.
    link_modes       per-platform override of inline | first_comment | dm_keyword | none (see _link_mode).
    optimize_rounds  critique passes (env SEO_OPTIMIZE_ROUNDS, default 1; 0 = single shot).
    The result carries `seo_id`; later call record_performance(seo_id, platform, metrics) so the next runs learn.
    """
    platforms = [p for p in (platforms or list(PLATFORM_SPECS)) if p in PLATFORM_SPECS]
    if not platforms:
        return {"generated_at": _now(), "platforms": {}, "_source": "no_platforms"}
    content_origin = (content_origin or os.getenv("SEO_CONTENT_ORIGIN", "reused")).lower()

    links = _extract_links(affiliate_link)
    slot_to_url = {f"[[AFF_LINK_{i}]]": u for i, u in enumerate(links, 1)}
    modes = {p: _link_mode(p, link_modes) for p in platforms}
    md = _as_dict(metadata)
    guard = ProvenanceGuard.from_inputs(md, video_context, cache, user_title, user_hint or "", links)
    evidence = build_evidence(video_context, user_title, brand_info, cache, md, links, user_hint,
                              performance_hints, previous, guard, content_origin)
    subject = evidence["subject"]["value"]
    seo_id = str(md.get("clip_id") or md.get("shortcode") or md.get("session_id") or "") or uuid.uuid4().hex[:12]
    seo_id = f"{_slug(seo_id)}-{uuid.uuid4().hex[:6]}"

    def fin(platform: str, raw: Dict[str, Any], analysis: Dict[str, Any], backfill: List[str]) -> Dict[str, Any]:
        ev_text = json.dumps(evidence, ensure_ascii=False) + " " + (user_title or "") + " " + (user_hint or "")
        return _finalize_platform(
            platform, raw, links=links, slot_to_url=slot_to_url, backfill_tags=backfill, fallback_title=subject,
            primary_keyword=str(analysis.get("primary_keyword") or subject), mode=modes[platform], guard=guard,
            ctx={"evidence_text": ev_text, "content_origin": content_origin})

    offline = _env_flag("FAST_OFFLINE_SEO")
    heur = _heuristic_draft(evidence, platforms)
    draft = None
    if not offline:
        gen_evidence = dict(evidence)
        learned = {p: learned_context(p) for p in platforms}
        learned = {p: c for p, c in learned.items() if c}
        if learned:
            gen_evidence["learned"] = learned
        draft = _llm_json(_build_generation_prompt(gen_evidence, platforms, modes), GEN_TEMPERATURE, "generate")

    if draft is None:
        analysis: Dict[str, Any] = {"primary_keyword": subject}
        backfill = _backfill_tags(analysis, evidence)
        finals = {p: fin(p, heur[p], analysis, backfill) for p in platforms}
        return _assemble(finals, analysis, evidence, links, subject, "heuristic_fallback", seo_id, content_origin, modes,
                         {"llm_calls": 0, "rounds_run": 0, "accepted_rewrites": 0}, {}, guard)

    analysis = _as_dict(draft.get("analysis"))
    if isinstance(analysis.get("subject_override"), str) and analysis["subject_override"].strip():
        subject = guard.scrub_text(clean_entity_name(analysis["subject_override"])) or subject
    backfill = _backfill_tags(analysis, evidence)
    drafts_raw = _as_dict(draft.get("platforms"))
    finals: Dict[str, Dict[str, Any]] = {}
    for p in platforms:
        finals[p] = fin(p, _as_dict(drafts_raw.get(p)) or heur[p], analysis, backfill)

    if optimize_rounds is None:
        try:
            optimize_rounds = int(os.getenv("SEO_OPTIMIZE_ROUNDS", "1"))
        except ValueError:
            optimize_rounds = 1
    calls, accepted, rounds_run = 1, 0, 0
    critic_scores: Dict[str, Optional[float]] = {}
    weaknesses: Dict[str, List[str]] = {}
    niche = _slug(analysis.get("niche") or draft.get("content_category"))

    for rnd in range(max(0, optimize_rounds)):
        if rnd > 0 and all(f["seo_score"] >= TARGET_SCORE and _hard_count(f["lint_issues"]) == 0 for f in finals.values()):
            break
        crit_evidence = dict(evidence)
        learned_n = {p: learned_context(p, niche) for p in platforms}
        learned_n = {p: c for p, c in learned_n.items() if c}
        if learned_n:
            crit_evidence["learned"] = learned_n
        current = {p: {"title": _tokenize(f["title"], slot_to_url), "title_alternates": f.get("title_alternates", []),
                       "description": _tokenize(f["description"], slot_to_url), "keywords": f.get("keywords", []),
                       "hashtags": f["hashtags"]} for p, f in finals.items()}
        lint = {p: [{"severity": i["severity"], "code": i["code"], "message": _tokenize(i["message"], slot_to_url)}
                    for i in f["lint_issues"]] for p, f in finals.items()}
        crit = _llm_json(_build_critic_prompt(crit_evidence, analysis, current, lint, platforms, modes),
                         CRITIC_TEMPERATURE, f"critique#{rnd + 1}")
        calls += 1
        rounds_run += 1
        if crit is None:
            break
        for p in platforms:
            c = _as_dict(_as_dict(crit.get("platforms")).get(p))
            if not c:
                continue
            critic_scores[p] = _mean_critic_score(c.get("scores"))
            if isinstance(c.get("weaknesses"), list):
                weaknesses[p] = [str(w) for w in c["weaknesses"] if w][:3]
            rev = c.get("revised")
            if not isinstance(rev, dict) or not rev.get("description"):
                continue
            rev.setdefault("title_alternates", finals[p].get("title_alternates", []))
            cand = fin(p, rev, analysis, backfill)
            if _hard_count(cand["lint_issues"]) == 0 and cand["seo_score"] >= finals[p]["seo_score"]:
                finals[p] = cand
                accepted += 1
            else:
                logger.info("[PlatformSEO] %s rewrite rejected (score %s -> %s, hard=%d).", p,
                            finals[p]["seo_score"], cand["seo_score"], _hard_count(cand["lint_issues"]))

    for p, f in finals.items():
        if critic_scores.get(p) is not None:
            f["critic_score"] = critic_scores[p]     # model self-assessment, separate from the computed seo_score
        if weaknesses.get(p):
            f["critic_weaknesses"] = weaknesses[p]

    result = _assemble(finals, analysis, evidence, links, subject, "gemini_semantic", seo_id, content_origin, modes,
                       {"llm_calls": calls, "rounds_run": rounds_run, "accepted_rewrites": accepted},
                       {"content_category": draft.get("content_category"), "target_audience": draft.get("target_audience")},
                       guard)
    logger.info("✅ [PlatformSEO] %s | calls=%d accepted=%d scores=%s", subject, calls, accepted,
                {p: f["seo_score"] for p, f in finals.items()})
    return result


def _assemble(finals: Dict[str, Dict[str, Any]], analysis: Dict[str, Any], evidence: Dict[str, Any], links: List[str],
              subject: str, source: str, seo_id: str, content_origin: str, modes: Dict[str, str],
              opt: Dict[str, Any], llm_meta: Dict[str, Any], guard: ProvenanceGuard) -> Dict[str, Any]:
    scores = [f["seo_score"] for f in finals.values()]
    avg = round(sum(scores) / len(scores)) if scores else 0
    level = "high" if avg >= 90 else "medium" if avg >= 75 else "low"
    kws = [k for k in [analysis.get("primary_keyword"), *(analysis.get("secondary_keywords") or [])]
           if isinstance(k, str) and k.strip()]
    clean_analysis = guard.scrub_obj(analysis)
    niche = _slug(analysis.get("niche") or llm_meta.get("content_category"))
    for p, f in finals.items():                       # log for the learning loop
        _log_generation(seo_id, p, niche, f["title"], extract_features(p, f, analysis, modes.get(p, "inline")),
                        modes.get(p, "inline"))
    edit_directives: Dict[str, Any] = {}
    if analysis.get("on_screen_text"):
        edit_directives["on_screen_text"] = [str(guard.scrub_text(x)) for x in analysis["on_screen_text"] if x][:6]
    if links:
        edit_directives["disclosure_overlay"] = "Ad · Affiliate links"
        edit_directives["verbal_disclosure"] = "This video contains affiliate links."
        edit_directives["note"] = ("ASCI also expects the disclosure in the video itself (spoken early + on-screen text); "
                                   "the platform's paid-partnership toggle alone is not enough.")
    result = {
        "generated_at": _now(), "seo_id": seo_id, "platforms": finals,
        "global_keywords": strip_system_and_tracking_tokens(guard.scrub_obj(kws or [subject])),
        "content_category": llm_meta.get("content_category") or evidence.get("intent") or "general",
        "target_audience": llm_meta.get("target_audience") or "general",
        "engagement_prediction": f"{level} (rule_score={avg})",      # rule compliance, NOT a traffic forecast
        "main_subject": strip_system_and_tracking_tokens(clean_entity_name(subject)),
        "applicable_context": evidence.get("context", ""),
        "analysis": clean_analysis, "niche": niche, "content_origin": content_origin,
        "edit_directives": edit_directives, "link_modes": modes,
        "rules_researched": RULES_RESEARCHED,
        "_affiliate_links": links, "_optimization": opt, "_source": source,
    }
    for k in ("main_subject", "applicable_context", "content_category", "target_audience"):
        if isinstance(result.get(k), str):
            result[k] = strip_system_and_tracking_tokens(guard.scrub_text(result[k]))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# APPROVAL / PREVIEW / REGENERATION
# ══════════════════════════════════════════════════════════════════════════════

def approve_and_finalize(seo_data: Dict[str, Any], approved_title: str, platform: str,
                         custom_edits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Finalize after user approval. User edits win, but limits, URL integrity, disclosure position and hashtag
    caps are re-enforced. The source-privacy guard is NOT re-applied to the user's own edits (their text, their call),
    but lint reports anything that looks like a credit leak is not possible here without the source metadata."""
    if platform not in seo_data.get("platforms", {}) or platform not in PLATFORM_SPECS:
        logger.error("❌ [PlatformSEO] Platform %s not found in generated data", platform)
        return {}
    pdata = dict(seo_data["platforms"][platform])
    pdata["title"] = approved_title
    if custom_edits:
        for k in ("description", "hashtags", "keywords"):
            if k in custom_edits:
                pdata[k] = custom_edits[k]
    mode = _as_dict(seo_data.get("link_modes")).get(platform, "inline")
    kept = [u for u in (seo_data.get("_affiliate_links") or []) if u in str(pdata.get("description", ""))]
    links = kept if mode == "inline" else list(seo_data.get("_affiliate_links") or [])
    keyword = str(_as_dict(seo_data.get("analysis")).get("primary_keyword") or seo_data.get("main_subject") or "")
    final = _finalize_platform(platform, pdata, links=links, slot_to_url={}, backfill_tags=[],
                               fallback_title=approved_title or "Untitled Video", primary_keyword=keyword, mode=mode,
                               ctx={"content_origin": seo_data.get("content_origin", "reused")})
    final["approved_at"] = _now()
    final["status"] = "approved"
    final["seo_id"] = seo_data.get("seo_id")
    return final


def _md_escape(text: Any) -> str:
    return re.sub(r"([_*`\[])", r"\\\1", str(text))


def format_for_telegram_preview(seo_data: Dict[str, Any]) -> str:
    lines = ["🎯 *SEO Content Preview*\n"]
    for platform, data in seo_data.get("platforms", {}).items():
        lines.append(f"\n📱 *{platform.upper()}*")
        lines.append(f"📝 Title: {_md_escape(data.get('title', 'N/A'))}")
        for alt in (data.get("title_alternates") or [])[:2]:
            lines.append(f"   ↳ alt: {_md_escape(alt)}")
        lines.append(f"📊 Rule score: {data.get('seo_score', 'N/A')}/100")
        hard = [i for i in data.get("lint_issues", []) if i.get("severity") == "hard"]
        if hard:
            lines.append(f"⚠️ {len(hard)} hard issue(s): {_md_escape(hard[0].get('message', ''))}")
        ld = _as_dict(data.get("link_delivery"))
        if ld.get("urls"):
            lines.append(f"🔗 Link route: {_md_escape(ld.get('mode'))}")
        lines.append(f"🏷️ Hashtags: {_md_escape(' '.join(data.get('hashtags', [])[:15]))}")
        desc = data.get("description")
        if desc:
            lines.append(f"📄 Description: {_md_escape(desc[:100] + '...' if len(desc) > 100 else desc)}")
    if seo_data.get("global_keywords"):
        lines.append(f"\n🔑 Keywords: {_md_escape(', '.join(seo_data['global_keywords']))}")
    lines.append(f"\n📈 Status: {_md_escape(seo_data.get('engagement_prediction', 'N/A'))}")
    return "\n".join(lines)


def extract_cache_for_regeneration(seo_data: Dict[str, Any]) -> Dict[str, Any]:
    plats = seo_data.get("platforms", {})
    return {
        "analysis": seo_data.get("analysis", {}),
        "global_keywords": seo_data.get("global_keywords", []),
        "content_category": seo_data.get("content_category", ""),
        "target_audience": seo_data.get("target_audience", ""),
        "previous_titles": {p: d.get("title", "") for p, d in plats.items()},
        "previous_hashtags": {p: d.get("hashtags", []) for p, d in plats.items()},
    }