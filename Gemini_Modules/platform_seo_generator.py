"""
Gemini_Modules/platform_seo_generator.py
=========================================
Platform-Specific SEO Content Generator

Generates optimized titles, hashtags, and descriptions for multiple platforms
(YouTube, Instagram, Facebook, Telegram) using Gemini AI with platform-specific
SEO algorithms and best practices.

Features:
- Platform-specific title optimization (character limits, keyword placement)
- Hashtag generation with platform-specific volume and relevance
- SEO-optimized descriptions with CTAs and engagement triggers
- Cache injection for context preservation
- User approval workflow with edit capability
- Multi-platform batch generation

Author: AMTCE Platform SEO Engine v1.0
"""

import json
import logging
import os
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger("platform_seo_generator")

# ── Router Import ─────────────────────────────────────────────────────────────
try:
    from Gemini_Modules.gemini_router_module.gemini_governor import GeminiGovernor
    _router = GeminiGovernor()
    _HAS_ROUTER = True
except ImportError:
    try:
        from gemini_router_module.gemini_governor import GeminiGovernor
        _router = GeminiGovernor()
        _HAS_ROUTER = True
    except ImportError:
        _router = None
        _HAS_ROUTER = False
        logger.warning("⚠️ GeminiGovernor not found. SEO generator will use heuristic fallback.")

# ── Platform SEO Constraints ───────────────────────────────────────────────────
PLATFORM_LIMITS = {
    "youtube": {
        "title_max": 100,
        "description_max": 5000,
        "hashtags_max": 15,
        "hashtag_style": "#",
        "title_emoji": "moderate",
        "description_style": "detailed",
        "cta_style": "subscribe",
    },
    "instagram": {
        "title_max": 2200,  # Caption limit
        "description_max": 2200,
        "hashtags_max": 30,
        "hashtag_style": "#",
        "title_emoji": "high",
        "description_style": "engaging",
        "cta_style": "link",
    },
    "facebook": {
        "title_max": 255,
        "description_max": 63206,
        "hashtags_max": 10,
        "hashtag_style": "#",
        "title_emoji": "moderate",
        "description_style": "conversational",
        "cta_style": "share",
    },
    "telegram": {
        "title_max": 255,
        "description_max": 4096,
        "hashtags_max": 10,
        "hashtag_style": "#",
        "title_emoji": "moderate",
        "description_style": "concise",
        "cta_style": "join",
    },
}

def clean_entity_name(entity_str: str) -> str:
    """
    Cleans entity strings like 'celebrity:Avneet_Kaur', 'outfit:olive_suit', 'person:John_Doe'
    into human-readable names like 'Avneet Kaur', 'Olive Suit', 'John Doe'.
    Strips entity category prefixes, removes underscores, and strips 'None'/null literals.
    """
    if not entity_str or not isinstance(entity_str, str):
        return ""
    text = entity_str.strip()
    text = re.sub(r'^(?:celebrity|outfit|accessory|environment|concept|niche|person|actor|actress|model|influencer|brand|subject|star|celeb|tag):\s*', '', text, flags=re.IGNORECASE)
    # Strip None / null / unknown attached at end of word or standalone
    text = re.sub(r'(?i)(?<=[a-zA-Z0-9_]{2})(?:None|null|unknown)\b', '', text)
    text = re.sub(r'(?i)\b(?:None|null|unknown)\b', '', text)
    text = text.replace("_", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w.capitalize() for w in text.split() if w.lower() not in {"none", "null", "unknown"}]
    return " ".join(words)

def strip_system_and_tracking_tokens(text_or_obj: Any) -> Any:
    """
    Recursively scrubs Instagram internal tracking tokens (#AQ...), raw system hashes,
    entity category prefixes (celebrity:), appended 'None' literals, and placeholder tags
    from text, lists, sets, tuples, and dictionary objects.
    """
    if isinstance(text_or_obj, str):
        text = text_or_obj

        # 1. Strip IG tracking tokens (#?AQ...)
        text = re.sub(r'#?AQ[a-zA-Z0-9_-]{10,}', '', text)

        # 2. Strip hashtags > 25 chars containing mixed alphanumeric hash signatures
        text = re.sub(r'#(?=[a-zA-Z0-9_-]{26,})(?=[a-zA-Z0-9_-]*\d)(?=[a-zA-Z0-9_-]*[a-zA-Z])[a-zA-Z0-9_-]+', '', text)

        # 3. Strip system placeholders & None / null tags (case-insensitive)
        text = re.sub(r'(?i)#?(?:creator_?unknown|niche_?unknown|unknown|none|null)\b', '', text)

        # 4. Strip entity prefixes from text and hashtags (e.g. celebrity:Avneet_Kaur -> Avneet Kaur, #celebrityAvneet_Kaur -> #AvneetKaur)
        text = re.sub(r'(?i)#(?:celebrity|model|actress|actor|influencer|star|person)_?:?\s*([A-Za-z0-9_]+)', r'#\1', text)
        text = re.sub(r'(?i)\b(?:celebrity|outfit|accessory|environment|concept|niche|person|actor|actress|model|influencer|brand|subject|star|celeb):\s*', '', text)

        # 5. Clean appended "None" or "null" from hashtags (e.g. #AvneetkaurNone -> #Avneetkaur)
        text = re.sub(r'(?i)(?<=#[a-zA-Z0-9_]{2})None\b', '', text)
        text = re.sub(r'(?i)(?<=#[a-zA-Z0-9_]{2})null\b', '', text)

        # 6. Clean orphaned '#' symbols and standalone 'None' strings in text
        text = re.sub(r'#(?![a-zA-Z0-9_])', '', text)
        text = re.sub(r'(?i)\bNone\b', '', text)

        # 7. Convert remaining underscores in entity names to spaces where appropriate, and deduplicate hashtags case-insensitively
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            tokens = line.split()
            cleaned_tokens = []
            seen_tags_lower = set()
            for tok in tokens:
                if tok.startswith("#"):
                    tag_inner = tok[1:].replace("_", "")
                    if tag_inner:
                        tag_lower = tag_inner.lower()
                        if tag_lower not in seen_tags_lower:
                            seen_tags_lower.add(tag_lower)
                            cleaned_tokens.append(f"#{tag_inner}")
                elif tok.startswith("http://") or tok.startswith("https://"):
                    cleaned_tokens.append(tok)
                else:
                    tok_clean = tok.replace("_", " ")
                    if tok_clean.strip():
                        cleaned_tokens.append(tok_clean)
            sub_l = " ".join(cleaned_tokens)
            sub_l = re.sub(r'[ \t]{2,}', ' ', sub_l).strip()
            cleaned_lines.append(sub_l)

        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text

    elif isinstance(text_or_obj, list):
        cleaned_list = []
        seen_tags_lower = set()
        for item in text_or_obj:
            res = strip_system_and_tracking_tokens(item)
            if res or isinstance(res, (bool, int, float)):
                if isinstance(res, str) and res.startswith("#"):
                    t_lower = res.lower().replace("_", "")
                    if t_lower not in seen_tags_lower:
                        seen_tags_lower.add(t_lower)
                        cleaned_list.append(res)
                else:
                    cleaned_list.append(res)
        return cleaned_list

    elif isinstance(text_or_obj, set):
        cleaned_set = set()
        for item in text_or_obj:
            res = strip_system_and_tracking_tokens(item)
            if res or isinstance(res, (bool, int, float)):
                cleaned_set.add(res)
        return cleaned_set

    elif isinstance(text_or_obj, tuple):
        return tuple(strip_system_and_tracking_tokens(list(text_or_obj)))

    elif isinstance(text_or_obj, dict):
        cleaned_dict = {}
        for key, val in text_or_obj.items():
            cleaned_dict[key] = strip_system_and_tracking_tokens(val)
        return cleaned_dict

    return text_or_obj

def extract_celebrity_human_name(handle: str) -> str:
    """Extract a human name or clean title from a handle string."""
    if not handle:
        return ""
    clean = handle.strip().lstrip("@")
    clean = re.sub(r"[._\-\d]+", " ", clean).strip()
    words = [w.capitalize() for w in clean.split() if w.lower() not in {"official", "real", "daily", "page", "fp", "club", "fan"}]
    return " ".join(words)

def extract_main_subject_and_context(
    video_context: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
    user_hint: Optional[str] = None,
    affiliate_link: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyzes video context, raw metadata (titles, captions, tags), prior Gemini call caches,
    and user-provided clues/hints/affiliate links to discover:
    1. main_subject: Primary hero entity, human name, pet name, brand name, or user hint
       (e.g., 'Core Topic', 'Pet Name', 'Brand Name', '2026').
    2. applicable_context: Supporting descriptors, secondary context, action details.
    """
    metadata = metadata or {}
    cache = cache or {}

    # 0. Check User Hint first for main subject clue
    hint_clean = ""
    if user_hint:
        hint_clean = re.sub(r'https?://[^\s<>"]+', "", user_hint).strip()
        hint_clean = re.sub(r"\s+", " ", hint_clean)
        hint_clean = clean_entity_name(hint_clean)

    # Extract fields from cache — support both ClipIntelligenceStore schema (visual_context,
    # editing_plan, audio_data.context) AND pool v3 social entry schema (visual_data.gemini_visual_output,
    # video_editing_plan, audio_data.gemini_audio_output). Additive alias resolution, nothing removed.
    _vc_raw = (cache.get("visual_context")
               or cache.get("visual_data", {}).get("gemini_visual_output")
               or cache.get("visual_data", {}).get("clip_intelligence"))
    vc = _vc_raw if isinstance(_vc_raw, dict) else {}

    _ep_raw = cache.get("editing_plan") or cache.get("video_editing_plan")
    ep = _ep_raw if isinstance(_ep_raw, dict) else {}

    _aud = cache.get("audio_data", {}) if isinstance(cache.get("audio_data"), dict) else {}
    audio_ctx = _aud.get("context") or _aud.get("gemini_audio_output") or {}
    if not isinstance(audio_ctx, dict):
        audio_ctx = {}

    raw_caption = metadata.get("raw_caption") or metadata.get("caption") or ""
    source_title = metadata.get("title") or metadata.get("source_title") or ""
    tags = metadata.get("hashtags") or []
    if isinstance(tags, list):
        tags_str = " ".join(tags)
    else:
        tags_str = str(tags)

    raw_detected = vc.get("detected_entities") or cache.get("detected_entities") or []
    if isinstance(raw_detected, str):
        raw_detected = [raw_detected]

    detected_entities = [clean_entity_name(e) for e in raw_detected if clean_entity_name(e)]
    person_name = clean_entity_name(vc.get("person_name") or cache.get("person_name") or "")
    cached_subject = clean_entity_name(vc.get("main_subject") or cache.get("main_subject") or "")

    full_text = f"{hint_clean} {video_context} {source_title} {raw_caption} {tags_str} {' '.join(detected_entities)}".strip()

    # 1. Main Subject Discovery
    main_subject = ""
    if hint_clean:
        # User provided explicit hint/title clue
        main_subject = hint_clean
    elif person_name:
        main_subject = person_name
    elif cached_subject:
        main_subject = cached_subject
    elif detected_entities and len(detected_entities) > 0:
        main_subject = detected_entities[0]

    if not main_subject and full_text:
        # Search for capitalized names/entities (e.g. "Main Subject", "Brand Name")
        cap_matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", full_text)
        if cap_matches:
            main_subject = clean_entity_name(cap_matches[0])

    if not main_subject and full_text:
        # Match single notable keywords
        words = [w for w in re.findall(r"\b[A-Za-z0-9_]+\b", full_text) if len(w) > 3 and w.lower() not in {"video", "short", "reels", "trending", "viral", "post", "none", "null"}]
        if words:
            main_subject = clean_entity_name(words[0])

    if not main_subject:
        # Check shortcode/session_id/clip_id in metadata
        shortcode = metadata.get("shortcode") or metadata.get("clip_id") or metadata.get("session_id") or ""
        if shortcode:
            main_subject = str(shortcode).replace("manual_", "").replace("sess_", "")
        else:
            main_subject = "Featured Reel"

    # 2. Applicable Context & Supporting Descriptors
    descriptors = []
    if vc.get("intent"):
        descriptors.append(str(vc.get("intent")).replace("_", " "))
    if vc.get("tone"):
        descriptors.append(str(vc.get("tone")))
    if ep.get("vibe_summary"):
        descriptors.append(str(ep.get("vibe_summary")))

    # Parse real snippets from source title, caption, and video_context
    caption_snippets = [s.strip() for s in re.split(r"[\n\r\t,#|.]+", f"{source_title} {raw_caption} {video_context}") if s.strip() and len(s.strip()) > 3]
    for snip in caption_snippets[:4]:
        if main_subject.lower() not in snip.lower() and snip.lower() not in main_subject.lower():
            if snip.lower() not in {"video", "reels", "shorts", "trending", "viral", "daily inspiration", "viral moment", "daily special", "none", "null"}:
                descriptors.append(snip)

    # Filter out generic fluff
    fluff_set = {"daily inspiration", "viral moment", "daily special", "trending feature", "viral reel", "daily", "none", "null"}
    clean_descriptors = [d for d in descriptors if d.lower().strip() not in fluff_set]

    applicable_context = ", ".join(dict.fromkeys(clean_descriptors)) if clean_descriptors else (video_context[:100].strip() if video_context else "video highlights")
    main_subject_clean = strip_system_and_tracking_tokens(clean_entity_name(main_subject)) or "Featured Reel"
    applicable_context_clean = strip_system_and_tracking_tokens(applicable_context) or "video highlights"

    return {
        "main_subject": main_subject_clean,
        "applicable_context": applicable_context_clean,
        "raw_caption": raw_caption,
        "source_title": source_title,
        "detected_entities": detected_entities,
        "intent": vc.get("intent", "general"),
        "tone": vc.get("tone", "engaging"),
        "audio_vibe": audio_ctx.get("dominant_emotion", "")
    }

def sanitize_raw_handles_out(text_or_obj: Any, raw_handle: str = "", discovered_subject: str = "") -> Any:
    """
    Sanitizes titles, descriptions, and hashtags to ensure raw account handles/IDs
    (e.g., 'creator_handle', '@username') are stripped out and replaced with
    the real discovered subject/star name or clean hashtags.
    """
    if not raw_handle or len(raw_handle) < 3:
        return text_or_obj

    handle_clean = raw_handle.strip().lstrip("@")
    replacement_text = discovered_subject if (discovered_subject and "celeb" not in discovered_subject.lower()) else ""

    if isinstance(text_or_obj, str):
        text = text_or_obj
        # Remove handle hashtags like #creator_handle
        text = re.sub(rf"#{re.escape(handle_clean)}\b", f"#{replacement_text.replace(' ', '')}" if replacement_text else "#viral", text, flags=re.IGNORECASE)
        # Remove handle mentions like @creator_handle
        text = re.sub(rf"@{re.escape(handle_clean)}\b", replacement_text or "", text, flags=re.IGNORECASE)
        # Remove standalone handle ID string
        text = re.sub(rf"\b{re.escape(handle_clean)}\b", replacement_text or "", text, flags=re.IGNORECASE)
        # Clean up double spaces or dangling punctuation
        text = re.sub(r"\s+", " ", text).strip()
        return text
    elif isinstance(text_or_obj, list):
        return [sanitize_raw_handles_out(item, raw_handle, discovered_subject) for item in text_or_obj if item]
    elif isinstance(text_or_obj, dict):
        return {k: sanitize_raw_handles_out(v, raw_handle, discovered_subject) for k, v in text_or_obj.items()}
    return text_or_obj


# ── SEO Generation Prompt ──────────────────────────────────────────────────────
_SEO_GENERATION_PROMPT = """\
You are an elite short-form algorithmic growth strategist & search indexing architect.
Your mission is to generate platform-sculpted, high-CTR titles, hashtags, and descriptions
for short-form video across ANY product, niche, or topic (Tech & Gadgets, Home & Kitchen,
Fitness & Health, Beauty & Skincare, Fashion & Apparel, Books & Productivity, Gaming, etc.).

Dynamically adapt all vocabulary, search intent, and conversion psychology to match
the specific product/niche provided. NEVER assume or default to fashion unless the product is apparel.

STRICT NEGATIVE CONSTRAINTS (ZERO TOLERANCE FOR FLUFF):
- ABSOLUTELY NO MOTIVATIONAL SPEAKER OR CORPORATE PR ESSAYS: Never write generic fluff like 'Most people focus on complex routines, but true confidence comes from mastering the essentials...' or 'Elevate your perspective and radiate absolute confidence'.
- NO CROWDED WALLS OF TEXT: Content must be high-contrast, visually uncluttered, generous with whitespace, and mobile-scannable.
- NO HASHTAG STUFFING: Maximum 3 to 6 targeted niche hashtags strictly matching the product category. Never output 15-30 generic tags like '#viral #trending #reels' which hurt algorithmic classification.
- NO 'LINK IN BIO' PHRASING: Never write 'link in bio', 'in my bio', or 'tap bio'. Links are directly embedded or listed inline.
- IMMUTABLE AFFILIATE SLOTS: If product slots like `[[AFF_LINK_1]]` are provided, you MUST preserve them verbatim. DO NOT modify, remove, or invent URLs.

INPUT SIGNALS:
- Primary Subject: {main_subject}
- Secondary Context & Niche: {applicable_context}
- User Provided Title / Clue: {user_title}
- Video Context: {video_context}
- Original Creator Caption: {original_caption}
- Whisper Audio Transcript: {whisper_transcript}
- Brand / Channel Info: {brand_info}
- Product Promotion Slots:
{affiliate_info}

═══════════════════════════════════════════════════════════════════════════════
PLATFORM-SPECIFIC ALGORITHMIC EXECUTION DIRECTIVES:
═══════════════════════════════════════════════════════════════════════════════

1. YOUTUBE SHORTS (Google Search Indexing + Browse Velocity):
   - Title: Strictly max 70 characters (prevents mobile truncation). Front-load high-intent Google Search queries tailored to the product category (e.g. '[Brand A] vs [Brand B]', 'Best [Category] Under ₹999', 'Honest Review: Is It Worth It?', 'Unboxing & Setup', or high-contrast comparison), ending with '#Shorts'.
   - Description:
     * Lines 1-2: Direct Google Search snippet summary matching what users search on YouTube/Google for this product or topic.
     * Blank line (\\n\\n).
     * '🛍️ FEATURED PRODUCTS / LINKS:' followed by bulleted [[AFF_LINK_X]] slots with concise item labels.
     * Blank line (\\n\\n).
     * Mandatory compliance line: '(Disclosure: #ad #affiliate — qualifying purchases earn commission)'.
     * 1-line viewer opinion prompt (boosts YouTube comment-velocity ranking).
   - Hashtags: 3 to 5 high-intent search tags strictly matching this product niche.

2. INSTAGRAM REELS (Vector Graph & Above-the-Fold Viewport):
   - Title / Hook: Strictly under 80 characters before the mobile '...more' truncation button. Must establish a sharp curiosity gap tailored to the product (e.g. 'Is this ₹499 find actually worth it? 👀' or 'Can you spot the difference between these two?').
   - Description:
     * 1-line scroll-stopping curiosity hook tailored to the product.
     * Blank line (\\n\\n).
     * '🔗 FEATURED ITEM / SHOP LINK:' followed by bulleted [[AFF_LINK_X]] slots.
     * Compliance line: '(#ad #affiliate)'.
     * Blank line (\\n\\n).
     * Save/Share action trigger ('💾 Save this for later & share with a friend!').
   - Hashtags: 4 to 6 hyper-relevant niche community tags matching the actual topic/product.

3. FACEBOOK REELS (Comment Velocity & Social Graph):
   - Title: Engaging, conversational headline.
   - Description: Relatable comparison or binary choice prompt that triggers comment debates (e.g. 'Which one would you choose: Option 1 or Option 2? 👇' or 'Would you buy this? Drop your thoughts below!') followed by [[AFF_LINK_X]] slots and (#ad #affiliate).
   - Hashtags: 3 to 5 broad interest tags.

4. TELEGRAM BROADCAST (Direct Messenger High-Speed UX):
   - Strictly under 300 characters total.
   - Format: Single bold headline (`🔥 **[Title]**`), 1-sentence punchy takeaway/verdict, bulleted [[AFF_LINK_X]] slots, `*(#ad #affiliate)*`, and 2 niche hashtags. Zero filler words!

OUTPUT SCHEMA — return ONLY this JSON:
{{
  "generated_at": "<ISO timestamp>",
  "platforms": {{
    "youtube": {{
      "title": "<Search-frontloaded title under 70 chars with #Shorts>",
      "description": "<Clean, search-indexed description with isolated slots>",
      "hashtags": ["#tag1", "#tag2", "#tag3"],
      "seo_score": 95,
      "keyword_density": "high_intent_search"
    }},
    "instagram": {{
      "title": "<Hook under 80 chars>",
      "description": "<Clean, spaced caption with slots, above-the-fold hook, and save CTA>",
      "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4"],
      "seo_score": 95,
      "keyword_density": "semantic_topic_cluster"
    }},
    "facebook": {{
      "title": "<Engaging conversational title>",
      "description": "<Comment-debate prompt with isolated slots and disclosure>",
      "hashtags": ["#tag1", "#tag2", "#tag3"],
      "seo_score": 90,
      "keyword_density": "conversational_social"
    }},
    "telegram": {{
      "title": "<Direct headline>",
      "description": "<Single bold title + 1 sentence + slots under 300 chars>",
      "hashtags": ["#tag1", "#tag2"],
      "seo_score": 95,
      "keyword_density": "direct_messenger_ux"
    }}
  }},
  "global_keywords": ["<keyword1>", "<keyword2>"],
  "content_category": "<category>",
  "target_audience": "<audience>",
  "engagement_prediction": "<high>"
}}
"""

def _clean_json(text: str) -> str:
    """Extract JSON from Gemini response."""
    if not text:
        return "{}"
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        else:
            text = text.replace("```json", "").replace("```", "")
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1].strip()
    return text.strip()

def _validate_platform_content(platform: str, content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and adjusts content based on platform limits.
    Truncates if exceeds limits, adds defaults if missing.
    """
    limits = PLATFORM_LIMITS.get(platform, {})
    validated = {}

    # Title validation
    title = content.get("title", "")
    max_title = limits.get("title_max", 255)
    if len(title) > max_title:
        title = title[:max_title-3] + "..."
    validated["title"] = strip_system_and_tracking_tokens(title or "Untitled Video")

    # Description validation
    description = content.get("description", "")
    max_desc = limits.get("description_max", 2200)

    # Universal sanitization: Strictly scrub any accidental "link in bio" phrases across ALL platforms
    description = re.sub(r'(?i)\b(?:tap\s+the\s+link\s+in\s+bio\s+or\s+visit\s+directly|tap\s+(?:the\s+)?link\s+in\s+bio|link\s+in\s+bio(?:\s+to\s+shop)?|check\s+(?:the\s+)?bio(?:\s+link)?|tap\s+bio|in\s+my\s+bio|in\s+the\s+bio|bio\s+link)\b', 'direct link', description)
    description = re.sub(r'(?i)\b(?:link\s+in\s+bio\s*&?\s*below)\b', 'link below', description)

    if platform == "telegram":
        max_desc = 1000  # Ensure Telegram video captions fit within Telegram's 1024 char limit
        # Deduplicate title if repeated at top of description
        clean_t = validated["title"].strip()
        desc_s = description.strip()
        if clean_t and desc_s.lower().startswith(clean_t.lower()):
            desc_s = desc_s[len(clean_t):].lstrip(" :\n*-—")
            description = desc_s

    # Ensure generous spacing around URLs (keep them visually isolated)
    description = re.sub(r'(?<!\n)(https?://[^\s<>"]+)', r'\n\1', description)
    description = re.sub(r'(https?://[^\s<>"]+)(?!\n)', r'\1\n', description)
    description = re.sub(r'\n{3,}', '\n\n', description).strip()

    if len(description) > max_desc:
        description = description[:max_desc-3] + "..."
    validated["description"] = strip_system_and_tracking_tokens(description or "Check out this amazing video!")

    # Hashtag validation
    hashtags = content.get("hashtags", [])
    max_tags = limits.get("hashtags_max", 15)
    if len(hashtags) > max_tags:
        hashtags = hashtags[:max_tags]
    # Ensure hashtags start with #
    hashtags = [tag if tag.startswith("#") else f"#{tag}" for tag in hashtags]
    validated["hashtags"] = strip_system_and_tracking_tokens(hashtags)

    # SEO score
    validated["seo_score"] = content.get("seo_score", 75)
    validated["keyword_density"] = content.get("keyword_density", "N/A")

    return strip_system_and_tracking_tokens(validated)

def _heuristic_fallback(
    video_context: str,
    user_title: str = "",
    brand_info: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
    affiliate_link: Optional[str] = None,
    user_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Heuristic fallback generating natural, storytelling creator captions & hashtags (0 API call cost).
    Features dynamic parallel vocabulary, storytelling hooks, analogies, and isolated affiliate links.
    """
    import random
    extracted = extract_main_subject_and_context(video_context, metadata, cache, user_hint=user_hint, affiliate_link=affiliate_link)
    main_subject = extracted["main_subject"] or "Product Spotlight"
    applicable = extracted["applicable_context"] or ""

    # Humanized title formatting
    clean_subj = main_subject.strip()
    if user_title:
        # Strip raw platform-suffix tags (#Shorts, #Reels, etc.) — keep user intent as anchor
        clean_title = re.sub(r'\s*#(?:Shorts?|Reels?)\b', '', user_title, flags=re.IGNORECASE).strip()
        # Enrich with applicable context if it adds information not already in the title
        if applicable and applicable.lower() not in {"video highlights", "general", "n/a"}:
            desc_part = applicable.split(",")[0].strip().title()
            if desc_part.lower() not in clean_title.lower():
                clean_title = f"{clean_title} — {desc_part} ✨"
    elif applicable and applicable.lower() not in {"video highlights", "general", "n/a"}:
        desc_clean = applicable.split(",")[0].strip().title()
        if desc_clean.lower() in clean_subj.lower():
            clean_title = f"{clean_subj} ✨"
        else:
            clean_title = f"{clean_subj} — {desc_clean} ✨"
    else:
        clean_title = f"{clean_subj} Review & Spotlight ✨"

    # Humanized hashtag builder (clean camel case, no underscores/None/numbers)
    def _to_camel_tag(text: str) -> str:
        words = [w.capitalize() for w in re.sub(r"\W+", " ", text).split() if w.lower() not in {"none", "null", "unknown"}]
        return "#" + "".join(words) if words else ""

    subj_tag = _to_camel_tag(clean_subj)
    desc_tags = []
    if applicable:
        for chunk in applicable.split(","):
            ctag = _to_camel_tag(chunk.strip())
            if ctag and len(ctag) > 3 and ctag not in desc_tags:
                desc_tags.append(ctag)

    # Dynamic universal tags derived from subject & context (no hardcoded category bias)
    base_tags = [subj_tag] if subj_tag else []
    context_words = [f"#{w.capitalize()}" for w in re.findall(r'\b[A-Za-z0-9]{3,}\b', applicable)
                     if w.lower() not in {"video", "short", "reels", "general", "n/a", "none", "featured", "highlights"}]
    universal_defaults = ["#MustHaves", "#ProductReview", "#BestFinds", "#HonestReview"]
    all_tags = list(dict.fromkeys([t for t in base_tags + desc_tags + context_words + universal_defaults if t]))[:5]

    # Clean YouTube Title (strictly <= 70 chars, front-load search query, end with #Shorts)
    yt_base = clean_title.replace("—", "-").strip()
    if not yt_base.lower().endswith("#shorts"):
        if len(yt_base) > 60:
            yt_title = f"{yt_base[:57]}... #Shorts"
        else:
            yt_title = f"{yt_base} #Shorts"
    else:
        yt_title = yt_base[:70]

    ig_title = clean_title[:80]

    # ── Formatted affiliate / non-affiliate blocks with generous spacing ──
    all_aff_links = []
    if affiliate_link:
        if isinstance(affiliate_link, (list, tuple)):
            for item in affiliate_link:
                all_aff_links.extend(re.findall(r'https?://[^\s<>"]+', str(item)))
        else:
            all_aff_links = re.findall(r'https?://[^\s<>"]+', str(affiliate_link))
        all_aff_links = list(dict.fromkeys([l.strip() for l in all_aff_links if l.strip()]))

    if all_aff_links:
        if len(all_aff_links) == 1:
            prod_links_yt = f"👉 {all_aff_links[0]}"
            prod_links_ig = f"👉 {all_aff_links[0]}"
            prod_links_tg = f"👉 {all_aff_links[0]}"
            items_label = "Item"
        else:
            prod_links_yt = "\n".join([f"👉 Option {i+1}: {lnk}" for i, lnk in enumerate(all_aff_links)])
            prod_links_ig = "\n".join([f"👉 Option {i+1}: {lnk}" for i, lnk in enumerate(all_aff_links)])
            prod_links_tg = "\n".join([f"👉 Item {i+1}: {lnk}" for i, lnk in enumerate(all_aff_links)])
            items_label = "Items"

        yt_desc = (
            f"Tested and reviewed: {clean_title}.\n\n"
            f"🛍️ FEATURED PRODUCTS / LINKS:\n"
            f"{prod_links_yt}\n\n"
            f"(Disclosure: #ad #affiliate — qualifying purchases earn commission)\n\n"
            f"💬 Which one is your top pick? Drop your thoughts below! 👇"
        )
        ig_caption = (
            f"{clean_title} ✨\n\n"
            f"🔗 FEATURED PRODUCTS & LINKS:\n"
            f"{prod_links_ig}\n\n"
            f"(#ad #affiliate)\n\n"
            f"💾 Save this for later & share with a friend!\n"
            f"💬 Drop your thoughts below! 👇"
        )
        fb_desc = (
            f"{clean_title} — Here is the honest breakdown!\n\n"
            f"🛍️ Featured Items:\n"
            f"{prod_links_yt}\n\n"
            f"(#ad #affiliate)\n\n"
            f"Which one would you choose? Drop a comment below! 👇"
        )
        tg_desc = (
            f"🔥 **{clean_title}**\n\n"
            f"Featured links & product details below:\n\n"
            f"{prod_links_tg}\n"
            f"*(#ad #affiliate)*\n\n"
            f"👉 Join our channel for daily updates!"
        )
    else:
        yt_desc = (
            f"Spotlight breakdown on {clean_title}.\n\n"
            f"💬 What's your take on this? Drop your thoughts below! 👇\n\n"
            f"⚡ Subscribe for daily curated reviews & breakdowns!"
        )
        ig_caption = (
            f"{clean_title} ✨\n\n"
            f"💬 On a scale of 1-10, how would you rate this? Drop it below! 👇\n\n"
            f"💾 Save this post for your daily inspiration & share with a friend!"
        )
        fb_desc = (
            f"{clean_title} ✨\n\n"
            f"What do you think about this? Share your opinion below! 👇"
        )
        tg_desc = (
            f"🔥 **{clean_title}**\n\n"
            f"Daily feature & spotlight breakdown.\n\n"
            f"👉 Join our channel for daily product spotlights & updates!"
        )

    platforms = {
        "youtube": {
            "title": yt_title,
            "description": yt_desc,
            "hashtags": all_tags[:5],
            "seo_score": 90,
            "keyword_density": "high_intent_search"
        },
        "instagram": {
            "title": ig_title,
            "description": ig_caption,
            "hashtags": all_tags[:5],
            "seo_score": 92,
            "keyword_density": "semantic_topic_cluster"
        },
        "facebook": {
            "title": clean_title[:255],
            "description": fb_desc,
            "hashtags": all_tags[:4],
            "seo_score": 88,
            "keyword_density": "conversational_social"
        },
        "telegram": {
            "title": clean_title[:255],
            "description": tg_desc,
            "hashtags": all_tags[:2],
            "seo_score": 95,
            "keyword_density": "direct_messenger_ux"
        }
    }

    return strip_system_and_tracking_tokens({
        "generated_at": datetime.utcnow().isoformat(),
        "platforms": platforms,
        "global_keywords": [main_subject] + desc_tags[:4],
        "content_category": extracted.get("intent", "general"),
        "target_audience": "general",
        "engagement_prediction": "medium (subject_heuristic)",
        "main_subject": main_subject,
        "_source": "heuristic_fallback"
    })

def generate_platform_seo(
    video_context: str,
    user_title: str = "",
    brand_info: str = "",
    platforms: Optional[List[str]] = None,
    cache: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    affiliate_link: Optional[str] = None,
    user_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate platform-specific SEO content using Gemini AI with Subject Extraction & Cache Injection.

    Args:
        video_context: Description of video content, scene analysis, or transcript
        user_title: User-provided title (optional, will be optimized)
        brand_info: Brand/channel information for personalization
        platforms: List of platforms to generate for (default: all)
        cache: Cached data from previous Gemini calls (Forensic, Audio, Editing Plan)
        metadata: Raw video metadata (titles, captions, hashtags, source info)
        affiliate_link: Commercial affiliate URL to promote (Amazon, Myntra, Ajio, etc.)
        user_hint: User-provided hint or subject clue

    Returns:
        Dict with platform-specific titles, descriptions, hashtags, and SEO scores
    """
    if platforms is None:
        platforms = ["youtube", "instagram", "facebook", "telegram"]

    # Sanitize affiliate link(s) as passive text
    clean_aff_links = []
    if affiliate_link:
        if isinstance(affiliate_link, (list, tuple)):
            for item in affiliate_link:
                clean_aff_links.extend(re.findall(r'https?://[^\s<>"]+', str(item)))
        else:
            clean_aff_links = re.findall(r'https?://[^\s<>"]+', str(affiliate_link))
        clean_aff_links = list(dict.fromkeys([l.strip() for l in clean_aff_links if l.strip()]))

    clean_aff_link = "\n".join(clean_aff_links) if clean_aff_links else ""

    extracted = extract_main_subject_and_context(video_context, metadata, cache, user_hint=user_hint, affiliate_link=clean_aff_link)
    main_subject = extracted["main_subject"]
    applicable_context = extracted["applicable_context"]

    # Fast Offline Heuristic SEO (0 API call cost) for shortform reels
    if os.getenv("FAST_OFFLINE_SEO", "auto").lower() in ("yes", "true", "on", "1") or not _HAS_ROUTER or _router is None:
        logger.info("⚡ [PlatformSEO] Using local fast SEO generator for title & hashtags (0 Gemini API calls).")
        result = _heuristic_fallback(video_context, user_title, brand_info, metadata, cache, affiliate_link=clean_aff_link, user_hint=user_hint)
        if platforms:
            result["platforms"] = {k: v for k, v in result["platforms"].items() if k in platforms}
        return result

    # ── Pool v3 enrichment fields ──────────────────────────────────────────────
    _aud_block = (cache or {}).get("audio_data", {}) if isinstance((cache or {}).get("audio_data"), dict) else {}
    _vis_block = (cache or {}).get("visual_data", {}) if isinstance((cache or {}).get("visual_data"), dict) else {}

    # Format cache & metadata context for prompt.
    # Prune cache_context to exclude sections that have dedicated prompt slots (CBM, editing plan,
    # whisper, captions, tags, analytics) and technical plumbing (hashes, telegram IDs, logs)
    # while preserving supplementary audio DSP math, vibe, and forensic perception tags.
    pruned_cache = {}
    if cache and isinstance(cache, dict):
        _aud_math = _aud_block.get("audio_math")
        _aud_gemini = _aud_block.get("gemini_audio_output") or _aud_block.get("context")
        _supp_audio = {}
        if _aud_math:
            _supp_audio["audio_math"] = _aud_math
        if _aud_gemini:
            _supp_audio["audio_context"] = _aud_gemini
        if _supp_audio:
            pruned_cache["supplementary_audio"] = _supp_audio

        _vis_intel = _vis_block.get("clip_intelligence") or cache.get("visual_context")
        if _vis_intel:
            pruned_cache["supplementary_visual_intel"] = _vis_intel

        for k in ("detected_entities", "intent", "tone", "vibe", "dominant_emotion"):
            if k in cache and cache[k]:
                pruned_cache[k] = cache[k]

    cache_context = json.dumps(pruned_cache, indent=2) if pruned_cache else "Supplementary intelligence integrated in dedicated sections above."
    # Build token slotting for 100% zero-touch affiliate link preservation
    slot_to_url = {}
    if clean_aff_links:
        slot_lines = []
        for i, url in enumerate(clean_aff_links, 1):
            token = f"[[AFF_LINK_{i}]]"
            slot_to_url[token] = url
            dest = "Product"
            if "amazon" in url.lower() or "amzn" in url.lower():
                dest = "Amazon Find"
            elif "meesho" in url.lower():
                dest = "Meesho Find"
            elif "myntra" in url.lower():
                dest = "Myntra Find"
            elif "ajio" in url.lower():
                dest = "Ajio Find"
            elif "flipkart" in url.lower():
                dest = "Flipkart Find"
            slot_lines.append(f"- Slot {i} ({dest}): {token}")
        aff_info_str = "Available Product Promotion Slots (Insert these EXACT tokens into your links block):\n" + "\n".join(slot_lines)
    else:
        aff_info_str = "None provided (Do NOT output product links or placeholder boxes)"

    # Whisper transcript
    _whisper = _aud_block.get("whisper_transcript", {})
    if isinstance(_whisper, dict):
        whisper_text = _whisper.get("text") or _whisper.get("transcript") or json.dumps(_whisper, ensure_ascii=False)[:600]
    elif isinstance(_whisper, str):
        whisper_text = _whisper[:600]
    else:
        whisper_text = "Not available"

    # Creator Behavior Model (CBM)
    _cbm = (cache or {}).get("creator_behavior_model", {})
    cbm_text = json.dumps(_cbm, indent=2, ensure_ascii=False)[:800] if _cbm else "Not available"

    # Video Editing Plan (pool v3 key alias)
    _vep = (cache or {}).get("video_editing_plan") or (cache or {}).get("editing_plan") or {}
    vep_text = json.dumps(_vep, indent=2, ensure_ascii=False)[:600] if _vep else "Not available"

    # Original caption from source post (independent of user_title / user_hint)
    original_caption_text = (metadata or {}).get("caption") or (metadata or {}).get("raw_caption") or "Not available"

    # Source hashtags (niche signal)
    _src_tags = (metadata or {}).get("hashtags", [])
    if isinstance(_src_tags, list):
        source_hashtags_text = " ".join(_src_tags[:25]) if _src_tags else "Not available"
    else:
        source_hashtags_text = str(_src_tags)[:200] if _src_tags else "Not available"

    # Analytics signals (strategy-only)
    _analytics = {
        "taggedUsers":    (metadata or {}).get("taggedUsers", []),
        "likesCount":     (metadata or {}).get("likesCount"),
        "videoViewCount": (metadata or {}).get("videoViewCount"),
        "scene_vectors":  _vis_block.get("vectors", [])[:10],
    }
    analytics_text = json.dumps(_analytics, indent=2, ensure_ascii=False)
    raw_metadata = json.dumps(metadata, indent=2) if metadata else f"Caption: {extracted['raw_caption']}"

    prompt = _SEO_GENERATION_PROMPT.format(
        main_subject=main_subject,
        applicable_context=applicable_context,
        raw_metadata=raw_metadata,
        video_context=video_context or "Video content not provided",
        user_title=user_title or user_hint or "No user title provided",
        brand_info=brand_info or "No brand info provided",
        affiliate_info=aff_info_str,
        cache_context=cache_context,
        platforms=", ".join(platforms),
        # ── new additive fields ──────────────────────────────────────────────
        whisper_transcript=whisper_text,
        creator_behavior_model=cbm_text,
        video_editing_plan=vep_text,
        original_caption=original_caption_text,
        source_hashtags=source_hashtags_text,
        analytics_signals=analytics_text,
    )

    try:
        logger.info(f"🎯 [PlatformSEO] Generating SEO content for subject='{main_subject}' platforms: {', '.join(platforms)}")
        raw_resp = _router.generate(
            task_type="seo_generation",
            prompt=prompt,
            module_name="platform_seo_generator",
            gen_config={"temperature": 0.3},
        )

        if not raw_resp:
            logger.warning("[PlatformSEO] Empty Gemini response — using heuristic fallback.")
            result = _heuristic_fallback(video_context, user_title, brand_info, metadata, cache, affiliate_link=clean_aff_link, user_hint=user_hint)
            if platforms:
                result["platforms"] = {k: v for k, v in result["platforms"].items() if k in platforms}
            return result

        clean = _clean_json(raw_resp)
        seo_data = json.loads(clean)

        # ── Zero-Touch Link Guardian: Restore exact byte-for-byte affiliate URLs from tokens ──
        if slot_to_url:
            for p_name, p_data in seo_data.get("platforms", {}).items():
                if not isinstance(p_data, dict):
                    continue
                desc = p_data.get("description", "")
                title_str = p_data.get("title", "")

                for token, real_url in slot_to_url.items():
                    if token in desc:
                        desc = desc.replace(token, real_url)
                    if token in title_str:
                        title_str = title_str.replace(token, "")

                # Safety check: If Gemini omitted any link, append it cleanly to the description
                missing_urls = [u for u in clean_aff_links if u not in desc]
                if missing_urls:
                    missing_block = "\n".join([f"👉 {u}" for u in missing_urls])
                    desc += f"\n\n🛍️ FEATURED PRODUCTS:\n{missing_block}"

                p_data["description"] = desc.strip()
                p_data["title"] = title_str.strip()

        # Validate and adjust each platform's content
        for platform in platforms:
            if platform in seo_data.get("platforms", {}):
                seo_data["platforms"][platform] = _validate_platform_content(
                    platform,
                    seo_data["platforms"][platform]
                )

        # Filter to requested platforms
        if platforms:
            seo_data["platforms"] = {
                k: v for k, v in seo_data.get("platforms", {}).items() if k in platforms
            }

        # Add metadata
        seo_data.setdefault("generated_at", datetime.utcnow().isoformat())
        seo_data.setdefault("global_keywords", [main_subject])
        seo_data.setdefault("content_category", extracted.get("intent", "general"))
        seo_data.setdefault("target_audience", "general")
        seo_data.setdefault("engagement_prediction", "high")
        seo_data["main_subject"] = strip_system_and_tracking_tokens(clean_entity_name(main_subject))
        seo_data["applicable_context"] = strip_system_and_tracking_tokens(applicable_context)
        seo_data["_source"] = "gemini_semantic"

        # Sanitize output to strip raw handle / ID text if present
        raw_h = (metadata or {}).get("creator_handle") or ""
        if not raw_h:
            m_h = re.search(r"(?:creator_handle|raw_handle|Handle):\s*([A-Za-z0-9._]+)", video_context)
            if m_h:
                raw_h = m_h.group(1)
        if raw_h:
            clean_name = extract_celebrity_human_name(raw_h)
            seo_data = sanitize_raw_handles_out(seo_data, raw_h, clean_name or main_subject)

        logger.info(f"✅ [PlatformSEO] Generated SEO content for subject='{main_subject}' across {len(seo_data['platforms'])} platforms")
        return strip_system_and_tracking_tokens(seo_data)

    except Exception as e:
        logger.warning(f"[PlatformSEO] Gemini call failed ({e}) — using heuristic fallback.")
        result = _heuristic_fallback(video_context, user_title, brand_info, metadata, cache, affiliate_link=clean_aff_link, user_hint=user_hint)
        if platforms:
            result["platforms"] = {k: v for k, v in result["platforms"].items() if k in platforms}
        return strip_system_and_tracking_tokens(result)

def approve_and_finalize(
    seo_data: Dict[str, Any],
    approved_title: str,
    platform: str,
    custom_edits: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Finalize SEO content after user approval.
    
    Args:
        seo_data: Original generated SEO data
        approved_title: User-approved title (will be used as base)
        platform: Platform to finalize for
        custom_edits: Optional custom edits to description/hashtags
    
    Returns:
        Finalized platform-specific content ready for publishing
    """
    if platform not in seo_data.get("platforms", {}):
        logger.error(f"❌ [PlatformSEO] Platform {platform} not found in generated data")
        return {}
    
    platform_data = seo_data["platforms"][platform].copy()
    
    # Update with approved title
    platform_data["title"] = approved_title
    
    # Apply custom edits if provided
    if custom_edits:
        if "description" in custom_edits:
            platform_data["description"] = custom_edits["description"]
        if "hashtags" in custom_edits:
            platform_data["hashtags"] = custom_edits["hashtags"]
    
    # Re-validate after edits
    platform_data = _validate_platform_content(platform, platform_data)
    
    # Add approval metadata
    platform_data["approved_at"] = datetime.utcnow().isoformat()
    platform_data["status"] = "approved"
    
    logger.info(f"✅ [PlatformSEO] Finalized {platform} content with title: '{approved_title[:50]}...'")
    return strip_system_and_tracking_tokens(platform_data)

def format_for_telegram_preview(seo_data: Dict[str, Any]) -> str:
    """
    Format SEO data for Telegram preview message.
    """
    lines = ["🎯 *SEO Content Preview*\n"]
    
    for platform, data in seo_data.get("platforms", {}).items():
        lines.append(f"\n📱 *{platform.upper()}*")
        lines.append(f"📝 Title: {data.get('title', 'N/A')}")
        lines.append(f"📊 SEO Score: {data.get('seo_score', 'N/A')}/100")
        lines.append(f"🏷️ Hashtags: {' '.join(data.get('hashtags', [])[:5])}")
        if data.get('description'):
            desc_preview = data['description'][:100] + "..." if len(data['description']) > 100 else data['description']
            lines.append(f"📄 Description: {desc_preview}")
    
    if seo_data.get("global_keywords"):
        lines.append(f"\n🔑 Keywords: {', '.join(seo_data['global_keywords'])}")
    
    lines.append(f"\n📈 Engagement Prediction: {seo_data.get('engagement_prediction', 'N/A')}")
    
    return "\n".join(lines)

def extract_cache_for_regeneration(seo_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relevant data to cache for future regeneration.
    This allows maintaining context across edits.
    """
    return {
        "global_keywords": seo_data.get("global_keywords", []),
        "content_category": seo_data.get("content_category", ""),
        "target_audience": seo_data.get("target_audience", ""),
        "previous_titles": {
            platform: data.get("title", "") 
            for platform, data in seo_data.get("platforms", {}).items()
        },
        "previous_hashtags": {
            platform: data.get("hashtags", []) 
            for platform, data in seo_data.get("platforms", {}).items()
        }
    }
