"""
Gemini_Modules/platform_seo_generator.py   (v2.0)
=================================================
Platform-specific SEO generator with a self-optimizing loop.

Flow
----
1. build_evidence()      Collapse every upstream signal (whisper transcript, forensic/visual intel,
                         editing plan, creator behavior model, source caption/hashtags, analytics,
                         user hint/title, affiliate slots) into ONE compact evidence packet.
2. GENERATE   (Gemini)   Analyze the evidence first (primary keyword, intent, hook angle), then write
                         3 title candidates per platform, pick the best, write description + hashtags.
3. FINALIZE   (code)     Deterministic enforcement: restore affiliate URLs byte-for-byte, scrub banned
                         phrases/tracking junk, enforce limits WITHOUT cutting URLs, force the
                         disclosure line, cap/clean hashtags.
4. LINT       (code)     Verifiable checks only (length, links, disclosure, hashtag budget, keyword
                         placement, whitespace, CTA). Produces a 0-100 `seo_score` that is computed,
                         not self-reported by the model.
5. CRITIQUE   (Gemini)   Editor pass: scores the draft against a rubric, lists weaknesses, rewrites.
                         A rewrite is ACCEPTED only if it is hard-lint-clean and does not score lower
                         than the current draft. Otherwise the previous version is kept.
6. Repeat 5 up to `optimize_rounds` (env SEO_OPTIMIZE_ROUNDS, default 1) or until every platform
   reaches TARGET_SCORE.

Cost: 1 generation call + `optimize_rounds` critique calls. FAST_OFFLINE_SEO=yes -> 0 calls (heuristic).

Public API is backward compatible with v1: generate_platform_seo, approve_and_finalize,
format_for_telegram_preview, extract_cache_for_regeneration, PLATFORM_LIMITS, clean_entity_name,
strip_system_and_tracking_tokens, extract_main_subject_and_context, extract_celebrity_human_name,
sanitize_raw_handles_out. New kwargs on generate_platform_seo are optional and trailing.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import re
import string
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("platform_seo_generator")

__all__ = [
    "PLATFORM_SPECS", "PLATFORM_LIMITS",
    "generate_platform_seo", "approve_and_finalize", "format_for_telegram_preview",
    "extract_cache_for_regeneration", "lint_platform", "build_evidence",
    "clean_entity_name", "strip_system_and_tracking_tokens",
    "extract_main_subject_and_context", "extract_celebrity_human_name",
    "sanitize_raw_handles_out",
]

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Reuse the routing key the governor already knows. Do NOT invent new task types
# (e.g. "seo_critique") unless GeminiGovernor has a profile for them.
GEN_TASK_TYPE = "seo_generation"
GEN_TEMPERATURE = 0.6      # hooks need some variance
CRITIC_TEMPERATURE = 0.2   # editing needs discipline
TARGET_SCORE = 90          # stop optimizing once every platform reaches this and has no hard issues

_COMMON_CRITERIA = [
    "evidence_grounding",   # every claim traceable to EVIDENCE, nothing invented
    "keyword_placement",    # primary keyword early + natural variants, no stuffing
    "hook_strength",        # would a stranger stop scrolling / click this
    "scannability",         # whitespace, short lines, mobile-readable
    "cta_fit",              # CTA native to the platform
    "hashtag_relevance",    # tags describe THIS video, nothing generic
]

# Single source of truth. The prompt, the validators and PLATFORM_LIMITS are all derived from this,
# so they can no longer drift apart (v1 had 100 vs 70 for YT titles, 30 vs 4-6 for IG tags, etc.).
PLATFORM_SPECS: Dict[str, Dict[str, Any]] = {
    "youtube": {
        "label": "YouTube Shorts (Google/YouTube search indexing + browse velocity)",
        "title_max": 70,
        "title_suffix": "#Shorts",
        "description_max": 5000,
        "hashtags_min": 3, "hashtags_max": 15, "hashtags_total_max": 15,
        "disclosure": "(Disclosure: #ad #affiliate — qualifying purchases earn commission)",
        "links_label": "🛍️ FEATURED PRODUCTS / LINKS:",
        "cta_re": r"(comment|below|drop|which|would you|thoughts|your (?:take|pick|opinion)|\?|👇)",
        "rules": [
            "Title: front-load the highest-intent search query for this product/topic (comparison, "
            "'best X', honest review, unboxing/setup, specific problem it solves). Price words ONLY if a price is in EVIDENCE.",
            "Description lines 1-2 are the search snippet: plain-language summary using primary + secondary keywords naturally.",
            "Include a structured gap with '🔍 Search Keywords:' containing 5-10 comma- or pipe-separated search queries derived from EVIDENCE.",
            "Then (only if product slots exist) the links block: the label, then one slot per line.",
            "Then the disclosure line exactly as given.",
            "Last line: ONE question that invites a comment (comment velocity ranks).",
            "Hashtags: LLM decides optimal count up to 15 relevant niche & intent tags strictly derived from EVIDENCE.",
        ],
    },
    "instagram": {
        "label": "Instagram Reels (above-the-fold hook + semantic caption indexing + structured gaps)",
        "title_max": 100,
        "description_max": 2200,
        "hook_visible_chars": 125,
        "hashtags_min": 5, "hashtags_max": 30, "hashtags_total_max": 30,
        "disclosure": "(#ad)",
        "links_label": "🔗 FEATURED ITEM / SHOP LINK:",
        "cta_re": r"(save|share|send|comment|rate|thoughts|👇|\?)",
        "rules": [
            "Title = the hook: a sharp curiosity gap, pattern interrupt or tension. No corporate or robotic summary.",
            "Caption line 1 must work ALONE: only ~125 characters show before '… more'. Make it a scroll-stopping hook.",
            "Use clean structured gaps (blank line between each block) for mobile readability and visual scannability.",
            "Second block: punchy context + debate/rating question to trigger comments (comment velocity drives explore distribution).",
            "Third block: include '🔍 Search Keywords:' with 8-15 high-intent search queries separated by ' | ' derived directly from the input process (transcript, detected entities, visual context, source caption).",
            "Then (only if product slots exist) the links block, then the disclosure line exactly as given.",
            "End with a save/share trigger.",
            "Hashtags: No artificial cap below platform capacity (up to 30). LLM decides optimal volume and density. "
            "Must be semantically connected to EVIDENCE (hero entity, niche community, style/action, long-tail intent).",
        ],
    },
    "facebook": {
        "label": "Facebook Reels (comment velocity + social graph)",
        "title_max": 255,
        "description_max": 63206,
        "hashtags_min": 3, "hashtags_max": 20, "hashtags_total_max": 25,
        "disclosure": "(#ad #affiliate)",
        "links_label": "🛍️ Featured Items:",
        "cta_re": r"(which|would you|comment|below|thoughts|vote|👇|\?)",
        "rules": [
            "Title: conversational, specific, no clickbait the video doesn't pay off.",
            "Description: a relatable binary-choice or opinion prompt that makes people argue in comments; keep it under ~600 chars.",
            "Structured spacing between hook, context, and links.",
            "Then (only if product slots exist) the links block and the disclosure line exactly as given.",
            "Hashtags: LLM decides optimal count (up to 20) derived from EVIDENCE.",
        ],
    },
    "telegram": {
        "label": "Telegram broadcast (direct-messenger UX)",
        "title_max": 255,
        "description_max": 1000,   # bot video captions cap at 1024; keep a margin
        "soft_text_max": 300,      # text length EXCLUDING URLs
        "hashtags_min": 1, "hashtags_max": 10, "hashtags_total_max": 15,
        "disclosure": "(#ad #affiliate)",
        "links_label": "",
        "cta_re": None,
        "rules": [
            "Format: one bold headline line (🔥 **Title**), ONE punchy takeaway/verdict sentence, product slots one per line, "
            "disclosure line exactly as given. Zero filler.",
            "Hashtags: 1-10 relevant niche tags.",
        ],
    },
}

# Back-compat view (v1 exposed this dict).
PLATFORM_LIMITS: Dict[str, Dict[str, Any]] = {
    p: {
        "title_max": s["title_max"],
        "description_max": s["description_max"],
        "hashtags_max": s["hashtags_max"],
        "hashtags_min": s["hashtags_min"],
    }
    for p, s in PLATFORM_SPECS.items()
}

_GENERIC_TAGS = frozenset({
    "viral", "trending", "reels", "reel", "fyp", "foryou", "foryoupage", "explore", "instagood",
    "love", "followforfollow", "like4like", "instadaily", "viralvideo", "viralreels", "trend",
})

# ══════════════════════════════════════════════════════════════════════════════
# ROUTER (lazy: importing this module no longer instantiates a Gemini client)
# ══════════════════════════════════════════════════════════════════════════════

_ROUTER: Any = None
_ROUTER_TRIED = False


def _get_router():
    global _ROUTER, _ROUTER_TRIED
    if _ROUTER_TRIED:
        return _ROUTER
    _ROUTER_TRIED = True
    for mod in ("Gemini_Modules.gemini_router_module.gemini_governor",
                "gemini_router_module.gemini_governor"):
        try:
            _ROUTER = importlib.import_module(mod).GeminiGovernor()
            return _ROUTER
        except Exception as exc:  # ImportError or constructor failure
            logger.debug("[PlatformSEO] router import via %s failed: %s", mod, exc)
    logger.warning("⚠️ GeminiGovernor not available. SEO generator will use heuristic fallback.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

_URL_RE = re.compile(r'https?://[^\s<>"]+')
_SLOT_RE = re.compile(r"\[\[AFF_LINK_\d+\]\]")

# URLs and slot tokens are swapped for private-use placeholders before ANY regex scrubbing,
# then restored. v1 ran its scrubbers over raw URLs (it could eat "none"/"AQ..." inside a link).
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
# '#celebrity_Avneet_Kaur' / '#model:Name' -> '#Avneet_Kaur'. Needs a separator, so '#celebrityNews' and
# '#personalfinance' survive (v1 mangled both).
_RE_ENTITY_TAG = re.compile(rf"(?i)#(?:celebrity|model|actress|actor|influencer|star|person)[_:]\s*([A-Za-z0-9_]+)")
# 'celebrity:Avneet_Kaur' (pipeline namespace, no space after the colon).
_RE_ENTITY_PREFIX = re.compile(rf"(?i)\b(?:{_ENTITY_PREFIXES}):(?=\S)")
_RE_NULL_LITERAL = re.compile(r"(?<![\w#])(?:None|null|NaN)(?!\w)")  # case-sensitive: leaked str(None)
_SNAKE = re.compile(r"^[^\W\d_]+(?:_[^\W\d_]+)+[.,!?:;]*$")
_NONWORD = re.compile(r"[^\w\u0900-\u0DFF]+")  # keep Devanagari..Malayalam so Indic tags survive


def _sanitize_str(text: str) -> str:
    masked, saved = _mask(text)
    t = _RE_IG_TRACK.sub("", masked)
    t = _RE_HASH_SIG.sub("", t)
    t = _RE_PLACEHOLDER.sub("", t)
    t = _RE_ENTITY_TAG.sub(r"#\1", t)
    t = _RE_ENTITY_PREFIX.sub("", t)
    t = _RE_NULL_LITERAL.sub("", t)

    lines_out: List[str] = []
    for line in t.split("\n"):          # line-by-line: paragraph structure is preserved
        seen: set = set()
        toks: List[str] = []
        for tok in line.split():
            if tok.startswith("#"):
                inner = tok[1:].replace("_", "")
                if not inner:
                    continue
                key = inner.lower()
                if key in seen:
                    continue
                seen.add(key)
                toks.append("#" + inner)
            elif _SNAKE.match(tok):
                toks.append(tok.replace("_", " "))
            else:
                toks.append(tok)
        lines_out.append(" ".join(toks))
    t = re.sub(r"\n{3,}", "\n\n", "\n".join(lines_out)).strip()
    return _unmask(t, saved)


def strip_system_and_tracking_tokens(text_or_obj: Any) -> Any:
    """Recursively scrub IG tracking tokens, placeholder tags, entity prefixes, leaked None/null,
    duplicate hashtags. URLs and [[AFF_LINK_n]] slots are never touched. Newlines are preserved."""
    if isinstance(text_or_obj, str):
        return _sanitize_str(text_or_obj)
    if isinstance(text_or_obj, list):
        out: List[Any] = []
        seen: set = set()
        for item in text_or_obj:
            res = strip_system_and_tracking_tokens(item)
            if isinstance(res, str) and res.startswith("#"):
                key = res.lower()
                if key in seen:
                    continue
                seen.add(key)
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
    """'celebrity:Avneet_Kaur' -> 'Avneet Kaur'. Only ALL-LOWERCASE words are capitalized, so
    'iPhone', 'OnePlus', 'USB-C', 'AirPods' keep their casing (v1 turned them into 'Iphone', 'Usb-c')."""
    if not entity_str or not isinstance(entity_str, str):
        return ""
    text = re.sub(rf"(?i)^(?:{_ENTITY_PREFIXES}):\s*", "", entity_str.strip())
    text = re.sub(r"(?<=[a-z0-9]{2})(?:None|null)\b", "", text)
    text = text.replace("_", " ")
    words = []
    for w in text.split():
        if w.lower() in {"none", "null", "unknown"}:
            continue
        words.append(w.capitalize() if w.islower() else w)
    return " ".join(words)


def extract_celebrity_human_name(handle: str) -> str:
    """Extract a human-ish name from an account handle."""
    if not handle:
        return ""
    clean = re.sub(r"[._\-\d]+", " ", handle.strip().lstrip("@")).strip()
    noise = {"official", "real", "daily", "page", "fp", "club", "fan"}
    return " ".join(w.capitalize() for w in clean.split() if w.lower() not in noise)


def sanitize_raw_handles_out(text_or_obj: Any, raw_handle: str = "", discovered_subject: str = "") -> Any:
    """Replace the creator's raw handle in titles/captions/tags with the discovered subject name.
    URL-safe, and keeps paragraph breaks (v1 collapsed ALL whitespace, flattening every description).
    Bare-word handles are only replaced when they look like IDs (contain _ . or digits) or carry @/#,
    so a handle like 'fashion' can't delete the word from the copy."""
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
        return "#" + words[0]            # keep 'iPhone15' as typed
    return "#" + "".join(w[:1].upper() + w[1:] for w in words)


def _cut_words(text: str, n: int) -> str:
    """Cut to <= n chars on a word boundary, adding an ellipsis. Never returns more than n chars."""
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


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE (everything the pipeline already knows, in one packet)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_cache(cache: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve both cache schemas: ClipIntelligenceStore (visual_context / editing_plan / audio_data.context)
    and pool v3 (visual_data.gemini_visual_output / video_editing_plan / audio_data.gemini_audio_output)."""
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

    return {
        "visual_ctx": visual_ctx,
        "editing": editing,
        "audio_ctx": audio_ctx,
        "audio_math": aud.get("audio_math"),
        "transcript": transcript,
        "cbm": _as_dict(cache.get("creator_behavior_model")),
    }


def extract_main_subject_and_context(
    video_context: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
    user_hint: Optional[str] = None,
    affiliate_link: Optional[str] = None,   # kept for signature compatibility; unused
) -> Dict[str, Any]:
    """Heuristic subject discovery. Same return keys as v1, plus `subject_source` and
    `subject_confidence`. The LLM receives the confidence and is told to override a weak subject
    (v1 fell back to 'first word longer than 3 chars', which is how you get 'Whisper' as a subject)."""
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
        code = metadata.get("shortcode") or metadata.get("clip_id") or metadata.get("session_id") or ""
        subject = str(code).replace("manual_", "").replace("sess_", "") if code else "Featured Reel"
        source, confidence = "fallback", "none"

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
    context = ", ".join(dict.fromkeys(descriptors)) if descriptors else (video_context[:100].strip() if len((video_context or "").strip()) > 3 else "video highlights")

    return {
        "main_subject": strip_system_and_tracking_tokens(clean_entity_name(subject)) or "Featured Reel",
        "applicable_context": strip_system_and_tracking_tokens(context) or "video highlights",
        "subject_source": source,
        "subject_confidence": confidence,
        "raw_caption": raw_caption,
        "source_title": source_title,
        "detected_entities": detected,
        "intent": vc.get("intent", "general"),
        "tone": vc.get("tone", "engaging"),
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
) -> Dict[str, Any]:
    """One compact, size-bounded packet of everything the pipeline knows about this clip.
    (v1 computed cache/CBM/editing-plan/analytics blocks and then never put them in the prompt.)"""
    md = _as_dict(metadata)
    rc = _resolve_cache(cache)
    ex = extract_main_subject_and_context(video_context, md, cache, user_hint=user_hint)
    vc, ep = rc["visual_ctx"], rc["editing"]

    tags = md.get("hashtags") or []
    tags = tags if isinstance(tags, list) else str(tags).split()
    hint_txt = re.sub(r"\s+", " ", _URL_RE.sub("", user_hint or "")).strip()

    return _compact({
        "subject": {"value": ex["main_subject"], "confidence": ex["subject_confidence"], "source": ex["subject_source"]},
        "context": ex["applicable_context"],
        "user_title": _clip(user_title, 200),
        "user_hint": _clip(hint_txt, 200),
        "video_context": _clip(video_context, 1200),
        "original_caption": _clip(ex["raw_caption"], 600),
        "source_title": _clip(ex["source_title"], 200),
        "source_hashtags": [str(t) for t in tags[:25]],
        "transcript": _clip(rc["transcript"], 1500),
        "detected_entities": ex["detected_entities"][:10],
        "intent": ex["intent"] if ex["intent"] != "general" else "",
        "tone": ex["tone"] if ex["tone"] != "engaging" else "",
        "audio_vibe": ex["audio_vibe"],
        "visual_intel": _clip_obj(vc, 1200),
        "editing_vibe": ep.get("vibe_summary"),
        "editing_plan": _clip_obj({k: v for k, v in ep.items() if k != "vibe_summary"}, 600),
        "creator_behavior": _clip_obj(rc["cbm"], 800),
        "audio_math": _clip_obj(rc["audio_math"], 400),
        "analytics": _compact({
            "likes": md.get("likesCount"),
            "views": md.get("videoViewCount"),
            "tagged_users": _clip_obj(md.get("taggedUsers"), 200),
        }),
        "creator_handle": md.get("creator_handle"),   # context only; never printed
        "brand": _clip(brand_info, 300),
        "product_links": [{"slot": f"[[AFF_LINK_{i}]]", "merchant": _merchant(u)}
                         for i, u in enumerate(links or [], 1)],
        "past_performance": _clip_obj(performance_hints[:6] if performance_hints else None, 900),
        "previous_attempt": _clip_obj(previous, 900),
    })


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_RULES = """\
GROUNDING & DATA EXTRACTION
- Extract all keywords, entities, and context directly from the data collected in the input process:
  EVIDENCE (transcript, detected entities, visual intel, source caption, source hashtags, editing vibe, audio vibe).
- Use ONLY facts present in EVIDENCE. Never invent prices, ratings, specs, discounts, test results, or personal experience.
- subject.confidence says how sure the heuristic was. If transcript/caption/entities contradict subject.value, trust them
  and put the corrected subject in analysis.subject_override.
- creator_handle and tagged_users are context only. Never print handles, @mentions or raw IDs.
- If past_performance exists, copy what worked structurally (hook type, length, keyword position), never the wording.
- If previous_attempt exists, do not reuse its titles verbatim; keep its keywords unless EVIDENCE contradicts them.
LANGUAGE
- Write in the language/register of the user title, caption and transcript (English, Hinglish, Malayalam, ...).
  Keep search keywords in the form people actually type into platform search bars.
STYLE & STRUCTURED GAPS
- No motivational/corporate filler ("elevate your perspective", "radiate confidence", "this reel showcases...").
- Mobile-scannable layout with STRUCTURED GAPS: exactly one blank line (`\\n\\n`) between distinct blocks:
  1. Hook / Pattern Interrupt (≤ 125 chars)
  2. Context & engagement/debate question (drives comments)
  3. Search Keywords block ('🔍 Search Keywords: [query1] | [query2] | ...') for algorithmic search indexing
  4. Product Links & Disclosure (only if product slots exist in EVIDENCE)
  5. Hashtags block
- Never write "link in bio", "in my bio", "check bio". Links are inline.
KEYWORDS & SEARCH INDEXING
- Derive high-intent search queries from the input process (transcript words, detected entities, visual context, source caption).
- Provide these in the 'keywords' array for each platform AND embed them as a clean search block inside the description.
HASHTAGS (LLM-OPTIMIZED DENSITY)
- There is NO low artificial maximum: you (the LLM) decide the optimal hashtag volume and density up to the platform ceiling
  (up to 30 for Instagram, up to 15 for YouTube, up to 20 for Facebook).
- Every single hashtag must be SEMANTICALLY CONNECTED to the actual content in EVIDENCE:
  * Primary Entity / Subject (e.g. #DishaPatani)
  * Niche Community (e.g. #BollywoodDance)
  * Style, Technique or Action from visual/audio context (e.g. #Choreography, #DanceReels, #Footwork)
  * Long-tail search intent (e.g. #DanceTutorial, #DanceInspo)
- Never use spammy generic tags (#viral #trending #reels #fyp #foryou #explore).
LINKS
- Product slots like [[AFF_LINK_1]] must be copied verbatim, one per line, never altered, never placed in a title,
  and you must never invent a URL. No product_links in EVIDENCE => no links block, no disclosure line.
"""

_GENERATE_TEMPLATE = """\
You are an elite short-form SEO strategist. Write platform-native titles, captions, keywords, and hashtags for ONE video,
grounded strictly in the EVIDENCE collected from the input process. Adapt vocabulary and search intent to whatever the subject/niche actually is.

WORK IN THIS ORDER (silently; output only the JSON):
1. ANALYZE the evidence: extract primary search query, secondary keywords, 8-15 long-tail search queries from transcript/caption/entities,
   search intent, target audience, and the scroll-stopping curiosity hook.
2. For EACH platform:
   - Write 3 title candidates (different hook angles: question / contrast / outcome / specific detail from evidence).
   - Write a structured description with clear visual gaps (\\n\\n) containing: Hook line, engagement trigger, '🔍 Search Keywords:' block, and product links (if any).
   - Generate a rich, semantically connected hashtag stack (up to platform ceiling) where you decide the optimal count.

__RULES__
PLATFORM BRIEFS
__BRIEFS__

EVIDENCE
__EVIDENCE__

Return ONLY this JSON (no prose, no code fences). Include only these platforms: __PLATFORMS__
{
  "analysis": {
    "primary_keyword": "",
    "secondary_keywords": [],
    "search_queries": [],
    "search_intent": "review|comparison|how-to|deal|discovery|entertainment",
    "audience": "",
    "hook_angle": "",
    "subject_override": ""
  },
  "platforms": {
    "<platform>": {
      "title": "",
      "title_alternates": ["", ""],
      "description": "",
      "keywords": ["query 1", "query 2"],
      "hashtags": ["#tag1", "#tag2"]
    }
  },
  "content_category": "",
  "target_audience": ""
}
"""

_CRITIC_TEMPLATE = """\
You are a ruthless SEO editor reviewing a junior's draft. Be harsh: 8+ means "I would ship this unchanged".

For EACH platform in CURRENT_DRAFT:
1. Score 0-10 on: __CRITERIA__.
2. List up to 3 concrete weaknesses (quote the offending text).
3. If any score is below 8 or LINT_ISSUES is non-empty for that platform, return "revised": a COMPLETE replacement
   {title, title_alternates, description, keywords, hashtags} that fixes them. Otherwise "revised": null.
Do not lengthen anything for its own sake. Do not add claims absent from EVIDENCE. Keep [[AFF_LINK_n]] slots verbatim.
Lint 'hard' issues are mandatory fixes; 'soft' issues are strong suggestions.

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
      "scores": {"evidence_grounding": 0, "keyword_placement": 0, "hook_strength": 0, "scannability": 0, "cta_fit": 0, "hashtag_relevance": 0},
      "weaknesses": [""],
      "revised": null
    }
  }
}
"""


def _render_brief(platform: str) -> str:
    s = PLATFORM_SPECS[platform]
    lines = [f"### {platform.upper()} — {s['label']}"]
    t = f"- Title ≤ {s['title_max']} characters"
    if s.get("title_suffix"):
        t += f", must end with {s['title_suffix']}"
    lines.append(t)
    d = f"- Description ≤ {s['description_max']} characters"
    if s.get("soft_text_max"):
        d += f"; text excluding URLs ≤ {s['soft_text_max']}"
    lines.append(d)
    lines.append(f"- Hashtags: optimal count up to {s['hashtags_max']} tags (platform ceiling ≤ {s['hashtags_total_max']}); you decide the optimal density based on evidence depth")
    lines.append(f"- If product slots exist, include this exact disclosure line: {s['disclosure']}")
    lines.extend(f"- {r}" for r in s["rules"])
    return "\n".join(lines)


def _fill(template: str, **kw: str) -> str:
    # str.replace instead of str.format: no brace-escaping landmines in the JSON schema.
    out = template
    for k, v in kw.items():
        out = out.replace(f"__{k.upper()}__", v)
    return out


def _build_generation_prompt(evidence: Dict[str, Any], platforms: List[str]) -> str:
    return _fill(
        _GENERATE_TEMPLATE,
        rules=_RULES,
        briefs="\n\n".join(_render_brief(p) for p in platforms),
        evidence=json.dumps(evidence, ensure_ascii=False, indent=1),
        platforms=", ".join(platforms),
    )


def _build_critic_prompt(evidence: Dict[str, Any], analysis: Dict[str, Any], draft: Dict[str, Any],
                         lint: Dict[str, List[Dict[str, Any]]], platforms: List[str]) -> str:
    return _fill(
        _CRITIC_TEMPLATE,
        criteria=", ".join(_COMMON_CRITERIA),
        rules=_RULES,
        briefs="\n\n".join(_render_brief(p) for p in platforms),
        evidence=json.dumps(evidence, ensure_ascii=False, indent=1),
        analysis=json.dumps(analysis, ensure_ascii=False, indent=1),
        draft=json.dumps(draft, ensure_ascii=False, indent=1),
        lint=json.dumps(lint, ensure_ascii=False, indent=1),
    )


# ══════════════════════════════════════════════════════════════════════════════
# LLM I/O
# ══════════════════════════════════════════════════════════════════════════════

def _clean_json(text: str) -> str:
    """Extract the JSON object from a Gemini response."""
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
    for cand in (s, re.sub(r",\s*([}\]])", r"\1", s)):   # second try forgives trailing commas
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
            raw = router.generate(
                task_type=GEN_TASK_TYPE,
                prompt=prompt + suffix,
                module_name="platform_seo_generator",
                gen_config={"temperature": temperature},
            )
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
# DETERMINISTIC ENFORCEMENT + LINT
# ══════════════════════════════════════════════════════════════════════════════

_BIO_RE = re.compile(
    r"(?i)\b(?:tap\s+the\s+link\s+in\s+(?:my\s+)?bio(?:\s+or\s+visit\s+directly)?"
    r"|tap\s+(?:the\s+)?link\s+in\s+bio|link\s+in\s+(?:my\s+)?bio(?:\s+to\s+shop)?"
    r"|(?:check|see)\s+(?:the\s+|my\s+)?bio(?:\s+link)?|tap\s+bio|in\s+(?:my|the)\s+bio|bio\s+link)\b")
_BIO_AND_BELOW_RE = re.compile(r"(?i)\blink\s+in\s+bio\s*&?\s*below\b")
_FLUFF_RE = re.compile(r"(?i)(elevate your perspective|radiate (?:absolute )?confidence|"
                       r"true confidence comes from|mastering the essentials|unlock your (?:true )?potential)")
_DISCLOSURE_GROUP = re.compile(r"\(\s*(?:Disclosure:\s*)?#ad\b[^)]*\)", re.I)
_URL_THEN_TEXT = re.compile(r'(https?://[^\s<>"]+)[ \t]+(?=\S)')
_DANGLING_BULLET = re.compile(r"(?m)^[ \t]*(?:👉|•|-|\*)[ \t]*$\n?")


def _fit_title(platform: str, title: str, fallback: str = "Untitled Video") -> str:
    spec = PLATFORM_SPECS[platform]
    t = " ".join(_SLOT_RE.sub("", _URL_RE.sub("", title or "")).split()) or fallback
    suffix = spec.get("title_suffix")
    if suffix:
        t = re.sub(rf"\s*{re.escape(suffix)}\s*$", "", t, flags=re.I).strip() or fallback
        return f"{_cut_words(t, spec['title_max'] - len(suffix) - 1)} {suffix}"
    return _cut_words(t, spec["title_max"])


def _fit_text(text: str, max_len: int) -> str:
    """Shrink to max_len WITHOUT cutting a URL or dropping the disclosure: drop whole low-priority
    paragraphs from the end first, then trim the opening paragraph on a word boundary.
    (v1 did text[:max-3] + '...' which could slice an affiliate link in half.)"""
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
        logger.warning("[PlatformSEO] text still %d chars > %d after fitting (protected links/disclosure).",
                       len(out), max_len)
    return out


def _clean_tag(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    t = _to_tag(raw.strip().lstrip("#"))
    key = t[1:].lower()
    if not t or len(key) < 2 or len(key) > 30 or key in _GENERIC_TAGS:
        return ""
    return t


def _fit_hashtags(platform: str, tags: Any, backfill: List[str], budget: int) -> List[str]:
    spec = PLATFORM_SPECS[platform]
    cap = max(0, min(spec["hashtags_max"], budget))
    out: List[str] = []
    seen: set = set()
    for raw in (tags if isinstance(tags, list) else []):
        t = _clean_tag(raw)
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    out = out[:cap]
    floor = min(spec["hashtags_min"], cap)
    for raw in backfill:
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
        # Label already there (the model wrote the block but dropped a link): extend that block, don't repeat the label.
        lines = desc.split("\n")
        last = max((i for i, ln in enumerate(lines) if _URL_RE.search(ln) or label in ln), default=len(lines) - 1)
        lines[last + 1:last + 1] = new_lines
        return "\n".join(lines)
    block = (label + "\n" if label else "") + "\n".join(new_lines)
    return desc.rstrip() + "\n\n" + block


def _normalize_disclosure(platform: str, desc: str, has_links: bool) -> str:
    if not has_links:
        return desc
    disclosure = PLATFORM_SPECS[platform]["disclosure"]
    state = {"first": True}

    def _rep(_m: "re.Match[str]") -> str:
        if state["first"]:
            state["first"] = False
            return disclosure
        return ""

    desc = _DISCLOSURE_GROUP.sub(_rep, desc)
    if platform == "instagram":      # #affiliate would burn a slot of the 5-hashtag cap
        desc = re.sub(r"(?i)[ \t]*#affiliate\b", "", desc)
    if state["first"] and not re.search(r"#ad\b", desc, re.I):
        desc = desc.rstrip() + "\n\n" + disclosure
    return desc


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
    n_words = max(1, len(_URL_RE.sub("", desc).split()))
    if not kw:
        return "n/a"
    occ = _URL_RE.sub("", desc).lower().count(kw.lower())
    return f"{100.0 * occ * max(1, len(kw.split())) / n_words:.1f}%"


def lint_platform(platform: str, content: Dict[str, Any], links: Optional[List[str]] = None,
                  primary_keyword: str = "") -> Tuple[int, List[Dict[str, str]]]:
    """Verifiable checks only. Returns (score 0-100, issues). hard = -25, soft = -8."""
    spec = PLATFORM_SPECS[platform]
    links = links or []
    issues: List[Dict[str, str]] = []

    def add(code: str, sev: str, msg: str) -> None:
        issues.append({"platform": platform, "code": code, "severity": sev, "message": msg})

    title = content.get("title") or ""
    desc = content.get("description") or ""
    tags = content.get("hashtags") or []

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
    if _BIO_RE.search(desc) or _BIO_RE.search(title):
        add("link_in_bio", "hard", "Contains a 'link in bio' style phrase. Links are inline.")
    if _FLUFF_RE.search(desc) or _FLUFF_RE.search(title):
        add("fluff", "hard", "Contains motivational/corporate filler. Replace with concrete, evidence-backed copy.")
    for u in links:
        if u not in desc:
            add("link_missing", "hard", f"Affiliate link missing from description: {u}")
    if links and not re.search(r"#ad\b", desc, re.I):
        add("disclosure_missing", "hard", f"Disclosure line missing; use exactly: {spec['disclosure']}")

    n_tags = len(tags)
    if n_tags > spec["hashtags_max"]:
        add("too_many_hashtags", "hard", f"{n_tags} hashtags; max {spec['hashtags_max']}.")
    if n_tags < spec["hashtags_min"]:
        add("too_few_hashtags", "soft", f"{n_tags} hashtags; min {spec['hashtags_min']}.")
    if n_tags + _count_desc_tags(desc) > spec["hashtags_total_max"]:
        add("hashtag_budget", "hard",
            f"{n_tags + _count_desc_tags(desc)} total # across caption + list; platform cap {spec['hashtags_total_max']}.")
    generic = [t for t in tags if str(t).lstrip("#").lower() in _GENERIC_TAGS]
    if generic:
        add("generic_hashtags", "soft", f"Generic tags hurt classification: {', '.join(generic)}")

    if primary_keyword:
        if not _kw_hit(title, primary_keyword):
            add("keyword_not_in_title", "soft", f"Primary keyword '{primary_keyword}' not in title.")
        elif platform == "youtube" and not _kw_hit(title[:45], primary_keyword):
            add("keyword_not_frontloaded", "soft", f"Move '{primary_keyword}' into the first ~45 chars of the title.")
        if not _kw_hit(desc, primary_keyword):
            add("keyword_not_in_description", "soft", f"Primary keyword '{primary_keyword}' not in description.")

    first_line = desc.split("\n", 1)[0]
    hook_max = spec.get("hook_visible_chars")
    if hook_max and len(first_line) > hook_max:
        add("hook_truncated", "soft", f"First line is {len(first_line)} chars; only ~{hook_max} show before '… more'.")
    if len(desc) > 300 and "\n\n" not in desc:
        add("wall_of_text", "soft", "No paragraph breaks in a long description.")
    cta = spec.get("cta_re")
    if cta and not re.search(cta, desc, re.I):
        add("cta_missing", "soft", "No platform-native CTA found.")

    hard = sum(1 for i in issues if i["severity"] == "hard")
    soft = len(issues) - hard
    return max(0, 100 - 25 * hard - 8 * soft), issues


def _finalize_platform(
    platform: str,
    raw: Dict[str, Any],
    *,
    links: List[str],
    slot_to_url: Dict[str, str],
    backfill_tags: List[str],
    fallback_title: str = "Untitled Video",
    raw_handle: str = "",
    handle_name: str = "",
    primary_keyword: str = "",
) -> Dict[str, Any]:
    """Raw model/heuristic output -> publishable content. Everything here is deterministic."""
    spec = PLATFORM_SPECS[platform]
    title = str(raw.get("title") or "")
    desc = str(raw.get("description") or "")

    for token, url in slot_to_url.items():            # restore affiliate URLs byte-for-byte
        desc = desc.replace(token, url)
    desc = _SLOT_RE.sub("", desc)                     # hallucinated slots like [[AFF_LINK_9]]
    desc = _DANGLING_BULLET.sub("", desc)

    desc = _BIO_RE.sub("link below", _BIO_AND_BELOW_RE.sub("link below", desc))
    title = _BIO_RE.sub("link below", _BIO_AND_BELOW_RE.sub("link below", title))
    if raw_handle:
        desc = sanitize_raw_handles_out(desc, raw_handle, handle_name)
        title = sanitize_raw_handles_out(title, raw_handle, handle_name)
    desc = strip_system_and_tracking_tokens(desc)
    title = strip_system_and_tracking_tokens(title)

    desc = _ensure_links(platform, desc, links)
    desc = _normalize_disclosure(platform, desc, bool(links))
    desc = _URL_THEN_TEXT.sub(r"\1\n", desc)          # keep URLs visually isolated (safe version)
    desc = re.sub(r"\n{3,}", "\n\n", desc).strip()

    if platform == "telegram":                        # v1 behaviour: drop a literally repeated headline
        t0, d0 = title.strip(), desc.strip()
        if t0 and d0.lower().startswith(t0.lower()):
            desc = d0[len(t0):].lstrip(" :\n*-—") or desc
    desc = _fit_text(desc or "Check out this video!", spec["description_max"])
    title = _fit_title(platform, title, fallback_title)

    budget = spec["hashtags_total_max"] - _count_desc_tags(desc)
    tags = strip_system_and_tracking_tokens(_fit_hashtags(platform, raw.get("hashtags"), backfill_tags, budget))

    alts: List[str] = []
    for a in (raw.get("title_alternates") or []):
        if isinstance(a, str) and a.strip():
            fa = _fit_title(platform, strip_system_and_tracking_tokens(a), fallback_title)
            if fa.lower() != title.lower() and fa not in alts:
                alts.append(fa)
    alts = alts[:2]

    content = {"title": title, "description": desc, "hashtags": tags}
    score, issues = lint_platform(platform, content, links, primary_keyword)
    kws = raw.get("keywords") or []
    if not isinstance(kws, list):
        kws = [str(kws)]
    kws = [strip_system_and_tracking_tokens(str(k)) for k in kws if str(k).strip()]
    return {
        **content,
        "title_alternates": alts,
        "keywords": kws,
        "seo_score": score,
        "keyword_density": _keyword_density(desc, primary_keyword),
        "lint_issues": issues,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HEURISTIC FALLBACK (0 API calls; same enforcement path as the LLM output)
# ══════════════════════════════════════════════════════════════════════════════

def _backfill_tags(analysis: Dict[str, Any], evidence: Dict[str, Any]) -> List[str]:
    subj = _as_dict(evidence.get("subject"))
    subject = subj.get("value", "") if subj.get("confidence") != "none" else ""   # 'Featured Reel' is not a tag
    candidates: List[str] = []
    for kw in [analysis.get("primary_keyword"), *(analysis.get("secondary_keywords") or []), subject]:
        if isinstance(kw, str) and kw.strip():
            candidates.append(_to_tag(kw))
    candidates.extend(evidence.get("source_hashtags") or [])
    return [c for c in dict.fromkeys(candidates) if c]


def _heuristic_draft(evidence: Dict[str, Any], platforms: List[str]) -> Dict[str, Dict[str, Any]]:
    """No fabricated claims ('tested', 'honest breakdown') and no fixed category bias."""
    subj = _as_dict(evidence.get("subject")).get("value") or "Featured Reel"
    ctx = [c.strip() for c in (evidence.get("context") or "").split(",") if c.strip()]
    ctx = [c for c in ctx if len(c) > 3 and c.lower() not in {"video highlights", "general", "n/a"}]
    user_title = re.sub(r"\s*#(?:Shorts?|Reels?)\b", "", evidence.get("user_title") or "", flags=re.I).strip()

    if user_title:
        base = user_title
    elif ctx and ctx[0].lower() not in subj.lower():
        base = f"{subj} — {ctx[0]}"
    else:
        base = subj

    # Extract semantic search queries directly from input process data
    search_kws = [k for k in [subj] + ctx[:5] + (evidence.get("detected_entities") or [])[:5] if k and len(k) > 2]
    search_kws = list(dict.fromkeys(search_kws))
    kw_block = f"\n\n🔍 Search Keywords:\n{' | '.join(search_kws)}" if search_kws else ""

    tags = _backfill_tags({}, evidence) + [_to_tag(c) for c in ctx]
    for ent in (evidence.get("detected_entities") or [])[:6]:
        ent_tag = _to_tag(ent)
        if ent_tag and ent_tag not in tags:
            tags.append(ent_tag)
    tags = [t for t in dict.fromkeys(tags) if t]

    return {
        "youtube": {
            "title": base,
            "hashtags": tags,
            "keywords": search_kws,
            "description": f"{base}\n\n💬 What's your take? Tell us in the comments 👇{kw_block}"
        },
        "instagram": {
            "title": base,
            "hashtags": tags,
            "keywords": search_kws,
            "description": f"{base}\n\nRate this 1-10 in the comments 👇\n\n💾 Save this for later & share with a friend!{kw_block}"
        },
        "facebook": {
            "title": base,
            "hashtags": tags,
            "keywords": search_kws,
            "description": f"{base}\n\nWould you pick this? Tell us below 👇{kw_block}"
        },
        "telegram": {
            "title": base,
            "hashtags": tags,
            "keywords": search_kws,
            "description": f"🔥 **{base}**{kw_block}"
        },
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
) -> Dict[str, Any]:
    """
    Generate platform-specific SEO content, then self-optimize it.

    optimize_rounds    critique/rewrite passes after the first draft. Default: env SEO_OPTIMIZE_ROUNDS or 1.
                       0 = single-shot generation.
    performance_hints  optional [{"platform","title","views","ctr",...}] of past winners on this channel.
                       This is the only way the loop learns from real results; without it "optimizing" means
                       "better against the rubric", not "ranks better".
    previous           output of extract_cache_for_regeneration(); wires the regenerate-with-context flow.
    """
    platforms = [p for p in (platforms or list(PLATFORM_SPECS)) if p in PLATFORM_SPECS]
    if not platforms:
        return {"generated_at": _now(), "platforms": {}, "_source": "no_platforms"}

    links = _extract_links(affiliate_link)
    slot_to_url = {f"[[AFF_LINK_{i}]]": u for i, u in enumerate(links, 1)}
    evidence = build_evidence(video_context, user_title, brand_info, cache, metadata, links, user_hint,
                              performance_hints, previous)
    subject = evidence["subject"]["value"]

    raw_handle = _as_dict(metadata).get("creator_handle") or ""
    if not raw_handle:
        m = re.search(r"(?:creator_handle|raw_handle|Handle):\s*([A-Za-z0-9._]+)", video_context or "")
        raw_handle = m.group(1) if m else ""
    handle_name = extract_celebrity_human_name(raw_handle) or subject

    def fin(platform: str, raw: Dict[str, Any], analysis: Dict[str, Any], backfill: List[str]) -> Dict[str, Any]:
        return _finalize_platform(
            platform, raw, links=links, slot_to_url=slot_to_url, backfill_tags=backfill,
            fallback_title=subject, raw_handle=raw_handle, handle_name=handle_name,
            primary_keyword=str(analysis.get("primary_keyword") or subject),
        )

    offline = os.getenv("FAST_OFFLINE_SEO", "auto").lower() in ("yes", "true", "on", "1")
    heur = _heuristic_draft(evidence, platforms)
    draft = None
    if not offline:
        logger.info("🎯 [PlatformSEO] Generating for subject='%s' platforms=%s", subject, ",".join(platforms))
        draft = _llm_json(_build_generation_prompt(evidence, platforms), GEN_TEMPERATURE, "generate")

    # ── No LLM available / failed: heuristic through the SAME enforcement path ─────────────────
    if draft is None:
        logger.info("⚡ [PlatformSEO] heuristic path (0 Gemini calls).")
        analysis: Dict[str, Any] = {"primary_keyword": subject}
        backfill = _backfill_tags(analysis, evidence)
        finals = {p: fin(p, heur[p], analysis, backfill) for p in platforms}
        return _assemble(finals, analysis, evidence, links, subject, "heuristic_fallback",
                         {"llm_calls": 0, "rounds_run": 0, "accepted_rewrites": 0}, {})

    analysis = _as_dict(draft.get("analysis"))
    if isinstance(analysis.get("subject_override"), str) and analysis["subject_override"].strip():
        subject = clean_entity_name(analysis["subject_override"]) or subject
    backfill = _backfill_tags(analysis, evidence)

    drafts_raw = _as_dict(draft.get("platforms"))
    finals: Dict[str, Dict[str, Any]] = {}
    for p in platforms:
        raw = _as_dict(drafts_raw.get(p)) or heur[p]     # model skipped a platform -> heuristic for that one
        finals[p] = fin(p, raw, analysis, backfill)

    # ── Self-optimization loop ─────────────────────────────────────────────────────────────────
    if optimize_rounds is None:
        try:
            optimize_rounds = int(os.getenv("SEO_OPTIMIZE_ROUNDS", "1"))
        except ValueError:
            optimize_rounds = 1
    calls, accepted, rounds_run = 1, 0, 0
    critic_scores: Dict[str, Optional[float]] = {}
    weaknesses: Dict[str, List[str]] = {}

    for rnd in range(max(0, optimize_rounds)):
        if rnd > 0 and all(f["seo_score"] >= TARGET_SCORE and _hard_count(f["lint_issues"]) == 0
                           for f in finals.values()):
            break   # round 1 always runs (that's the feature); later rounds stop once everything is clean
        current = {
            p: {
                "title": _tokenize(f["title"], slot_to_url),
                "title_alternates": f.get("title_alternates", []),
                "description": _tokenize(f["description"], slot_to_url),
                "hashtags": f["hashtags"],
            } for p, f in finals.items()
        }
        lint = {p: [{"severity": i["severity"], "code": i["code"], "message": _tokenize(i["message"], slot_to_url)}
                    for i in f["lint_issues"]] for p, f in finals.items()}
        crit = _llm_json(_build_critic_prompt(evidence, analysis, current, lint, platforms),
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
            better = cand["seo_score"] >= finals[p]["seo_score"]
            if _hard_count(cand["lint_issues"]) == 0 and better:
                finals[p] = cand
                accepted += 1
            else:
                logger.info("[PlatformSEO] %s rewrite rejected (score %s -> %s, hard=%d); keeping previous.",
                            p, finals[p]["seo_score"], cand["seo_score"], _hard_count(cand["lint_issues"]))

    for p, f in finals.items():                       # model self-assessment, kept SEPARATE from seo_score
        if critic_scores.get(p) is not None:
            f["critic_score"] = critic_scores[p]
        if weaknesses.get(p):
            f["critic_weaknesses"] = weaknesses[p]

    result = _assemble(finals, analysis, evidence, links, subject, "gemini_semantic",
                       {"llm_calls": calls, "rounds_run": rounds_run, "accepted_rewrites": accepted},
                       {"content_category": draft.get("content_category"),
                        "target_audience": draft.get("target_audience")})
    logger.info("✅ [PlatformSEO] %s | calls=%d accepted_rewrites=%d scores=%s", subject, calls, accepted,
                {p: f["seo_score"] for p, f in finals.items()})
    return result


def _assemble(finals: Dict[str, Dict[str, Any]], analysis: Dict[str, Any], evidence: Dict[str, Any],
              links: List[str], subject: str, source: str, opt: Dict[str, Any],
              llm_meta: Dict[str, Any]) -> Dict[str, Any]:
    scores = [f["seo_score"] for f in finals.values()]
    avg = round(sum(scores) / len(scores)) if scores else 0
    level = "high" if avg >= 90 else "medium" if avg >= 75 else "low"
    kws = [k for k in [analysis.get("primary_keyword"), *(analysis.get("secondary_keywords") or [])]
           if isinstance(k, str) and k.strip()]
    result = {
        "generated_at": _now(),
        "platforms": finals,
        "global_keywords": strip_system_and_tracking_tokens(kws or [subject]),
        "content_category": llm_meta.get("content_category") or evidence.get("intent") or "general",
        "target_audience": llm_meta.get("target_audience") or "general",
        # Rule compliance, NOT a traffic forecast. Key kept for v1 consumers.
        "engagement_prediction": f"{level} (rule_score={avg})",
        "main_subject": strip_system_and_tracking_tokens(clean_entity_name(subject)),
        "applicable_context": evidence.get("context", ""),
        "analysis": analysis,
        "_affiliate_links": links,
        "_optimization": opt,
        "_source": source,
    }
    return _sanitize_free_text_fields(result)


def _sanitize_free_text_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    # Platform content is already sanitized in _finalize_platform. Sanitize ONLY top-level free text:
    # running the sanitizer over the whole dict rewrites machine values ('heuristic_fallback' -> 'heuristic fallback').
    for k in ("main_subject", "applicable_context", "content_category", "target_audience"):
        if isinstance(result.get(k), str):
            result[k] = strip_system_and_tracking_tokens(result[k])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# APPROVAL / PREVIEW / REGENERATION
# ══════════════════════════════════════════════════════════════════════════════

def approve_and_finalize(
    seo_data: Dict[str, Any],
    approved_title: str,
    platform: str,
    custom_edits: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Finalize after user approval. The user's edits win (a link they deliberately removed stays removed),
    but limits, URL integrity, disclosure (if any affiliate link remains) and hashtag caps are re-enforced."""
    if platform not in seo_data.get("platforms", {}) or platform not in PLATFORM_SPECS:
        logger.error("❌ [PlatformSEO] Platform %s not found in generated data", platform)
        return {}

    pdata = dict(seo_data["platforms"][platform])
    pdata["title"] = approved_title
    if custom_edits:
        if "description" in custom_edits:
            pdata["description"] = custom_edits["description"]
        if "hashtags" in custom_edits:
            pdata["hashtags"] = custom_edits["hashtags"]

    kept_links = [u for u in (seo_data.get("_affiliate_links") or []) if u in str(pdata.get("description", ""))]
    keyword = str(_as_dict(seo_data.get("analysis")).get("primary_keyword") or seo_data.get("main_subject") or "")
    final = _finalize_platform(
        platform, pdata, links=kept_links, slot_to_url={}, backfill_tags=[],
        fallback_title=approved_title or "Untitled Video", primary_keyword=keyword,
    )
    final["approved_at"] = _now()
    final["status"] = "approved"
    logger.info("✅ [PlatformSEO] Finalized %s with title: '%s'", platform, final["title"][:50])
    return final


def _md_escape(text: Any) -> str:
    """Escape Telegram legacy-Markdown specials; unescaped _ * ` [ in a title makes the API reject the whole message."""
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
            lines.append(f"⚠️ {len(hard)} unresolved hard issue(s): {_md_escape(hard[0].get('message', ''))}")
        if data.get("keywords"):
            lines.append(f"🔍 Keywords: {_md_escape(' | '.join(data.get('keywords', [])[:8]))}")
        lines.append(f"🏷️ Hashtags: {_md_escape(' '.join(data.get('hashtags', [])[:15]))}")
        desc = data.get("description")
        if desc:
            preview = desc[:100] + "..." if len(desc) > 100 else desc
            lines.append(f"📄 Description: {_md_escape(preview)}")
    if seo_data.get("global_keywords"):
        lines.append(f"\n🔑 Keywords: {_md_escape(', '.join(seo_data['global_keywords']))}")
    lines.append(f"\n📈 Status: {_md_escape(seo_data.get('engagement_prediction', 'N/A'))}")
    return "\n".join(lines)


def extract_cache_for_regeneration(seo_data: Dict[str, Any]) -> Dict[str, Any]:
    """Feed this back as `previous=` on the next generate_platform_seo() call."""
    plats = seo_data.get("platforms", {})
    return {
        "analysis": seo_data.get("analysis", {}),
        "global_keywords": seo_data.get("global_keywords", []),
        "content_category": seo_data.get("content_category", ""),
        "target_audience": seo_data.get("target_audience", ""),
        "previous_titles": {p: d.get("title", "") for p, d in plats.items()},
        "previous_hashtags": {p: d.get("hashtags", []) for p, d in plats.items()},
    }
