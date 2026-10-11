"""
Watermark and Inpainting Directive Helper
=========================================
Robust parsing and keyword extraction for human re-edit directives.
Provides typo-tolerant detection for watermark removal/inpainting requests
and extracts target watermark handles/branding (e.g. 'filmygyan', 'viralbhayani').
"""

import re
from typing import Optional

# Action keywords & typo patterns
INPAINT_ACTION_PATTERNS = [
    r"\b(?:in[-_ ]?a?p[ai]*nt(?:ing)?)\b",      # inpaint, inapint, in-paint, inpaintng, inpiant
    r"\b(?:delogo|de[-_ ]?logo)\b",             # delogo, de-logo
    r"\b(?:water[-_ ]?m(?:ar)?k(?:s)?)\b",      # watermark, water mark, watermrk, wtrmrk
    r"\b(?:logo(?:s)?)\b",                      # logo, logos
    r"\b(?:clean(?:ing|er|up)?)\b",             # clean, cleaning, cleaner, cleanup
    r"\b(?:erase(?:r|d|ing)?)\b",               # erase, eraser, erasing
    r"\b(?:remov(?:e|al|ed|ing))\b",            # remove, removal, removed, removing
]

# Known paparazzi / media agency watermark handles
KNOWN_WATERMARK_BRANDS = [
    r"\b(?:f[li]{1,2}m[yi]gyan)\b",             # filmygyan, flimygyan, filmigyan, flimigyan
    r"\b(?:viral\s*bhayani)\b",                 # viralbhayani, viral bhayani
    r"\b(?:voompla)\b",
    r"\b(?:manav\.?\s*manglani)\b",
    r"\b(?:instant\s*bollywood)\b",
    r"\b(?:yogen\s*shah)\b",
    r"\b(?:pinkvilla)\b",
    r"\b(?:bollywood\s*(?:hungama|bubble|life)?)\b",
    r"\b(?:paparazzi)\b",
    r"\b(?:getty\s*(?:images)?)\b",
    r"\b(?:shutterstock)\b",
]

_COMBINED_INPAINT_RE = re.compile(
    "|".join(INPAINT_ACTION_PATTERNS + KNOWN_WATERMARK_BRANDS),
    re.IGNORECASE
)

_STRIP_ACTION_WORDS_RE = re.compile(
    "|".join(INPAINT_ACTION_PATTERNS + [
        r"\b(?:please|pls|the|video|clip|from|this|that|and|to|for|with)\b"
    ]),
    re.IGNORECASE
)


def is_inpaint_directive(directive: Optional[str]) -> bool:
    """
    Returns True if the directive requests watermark removal, delogo, inpainting,
    or mentions a known media/watermark brand. Typo-tolerant.
    """
    if not directive or not isinstance(directive, str):
        return False
    return bool(_COMBINED_INPAINT_RE.search(directive))


def extract_watermark_keywords(directive: Optional[str]) -> str:
    """
    Extracts the specific watermark handle/branding from the directive.
    Strips out generic command words ("inpaint", "remove", "please") and
    canonicalizes known handles (e.g. 'flimygyan' -> 'filmygyan').
    Returns a cleaned keyword string suitable for forensic perception prompts.
    """
    if not directive or not isinstance(directive, str):
        return "watermark logo branding"

    # Check for known brand canonicalization first
    cleaned = _STRIP_ACTION_WORDS_RE.sub(" ", directive).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if re.search(r"\bf[li]{1,2}m[yi]gyan\b", directive, re.IGNORECASE):
        cleaned = re.sub(r"\bf[li]{1,2}m[yi]gyan\b", "filmygyan", cleaned, flags=re.IGNORECASE)
        if "filmygyan" not in cleaned.lower():
            cleaned = (cleaned + " filmygyan").strip()

    if re.search(r"\bviral\s*bhayani\b", directive, re.IGNORECASE):
        cleaned = re.sub(r"\bviral\s*bhayani\b", "viralbhayani", cleaned, flags=re.IGNORECASE)
        if "viralbhayani" not in cleaned.lower():
            cleaned = (cleaned + " viralbhayani").strip()

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else "watermark logo branding"
