"""
Watermark and Inpainting Directive Helper
=========================================
Robust parsing and keyword extraction for human re-edit directives.
Provides typo-tolerant detection for watermark removal/inpainting requests
and extracts target watermark handles/branding dynamically without hardcoded brands.
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

# ── Failure Feedback Detection Signals ────────────────────────────────────────
# Centralized here so any module (hybrid_watermark, gemini_enhance_for_watermark, etc.)
# can import this without duplicating the regex.
_FEEDBACK_SIGNALS_RE = re.compile(
    r"\b(?:missed|still|left|right|top|bottom|corner|position|2\s+watermark|two\s+watermark|"
    r"second\s+watermark|different\s+logo|not\s+removed|not\s+in\s+correct|wrong\s+spot|"
    r"wrong\s+position|inpaint(?:ing)?|blur(?:red)?|failed|attempt|retry|didn'?t\s+work|"
    r"still\s+there|visible|partially|incomplete)\b",
    re.IGNORECASE
)

_COMBINED_INPAINT_RE = re.compile(
    "|".join(INPAINT_ACTION_PATTERNS),
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
    or contains retry failure feedback signals. Typo-tolerant.
    """
    if not directive or not isinstance(directive, str):
        return False
    return bool(_COMBINED_INPAINT_RE.search(directive) or _FEEDBACK_SIGNALS_RE.search(directive))


def extract_watermark_keywords(directive: Optional[str]) -> str:
    """
    Extracts target branding, handle, or hint text from the directive by stripping
    out generic command words ("inpaint", "remove", "please").
    Returns a cleaned keyword string suitable for forensic perception prompts.
    """
    if not directive or not isinstance(directive, str):
        return "watermark logo branding"

    # Strip generic action words and return the brand/handle as-is.
    # Gemini's semantic vision handles typos natively — no manual canonicalization needed.
    cleaned = _STRIP_ACTION_WORDS_RE.sub(" ", directive).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else "watermark logo branding"


def is_feedback_directive(directive: Optional[str]) -> bool:
    """
    Returns True if the directive reads like a failure feedback report
    from a user inspecting a previous watermark removal attempt,
    rather than a brand/handle target hint.

    Examples that return True:
      "logo still visible in bottom right"
      "inpaint not in correct position"
      "there are 2 watermarks still"
      "missed the circular logo top left"

    Examples that return False:
      "my_brand_logo"
      "retro style watermark"
      "music beats"
    """
    if not directive or not isinstance(directive, str):
        return False
    return bool(_FEEDBACK_SIGNALS_RE.search(directive))
