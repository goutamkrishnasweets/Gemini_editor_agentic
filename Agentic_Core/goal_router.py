"""
Agentic_Core / goal_router.py
=============================
Deterministic (zero-LLM) goal classifier.

WHY THIS EXISTS
---------------
Ingest -> render -> audit -> publish is a FIXED workflow. Putting an LLM in the
middle of it adds 5+ network round trips, rate-limit exposure and failure modes,
and gains nothing. The LLM Director is only a good fit when the user's request is
genuinely open-ended free text.

This module is pure Python (no google-genai / no heavy imports) so it is
instant and trivially unit-testable.

Route kinds
-----------
  url      -> run the proven direct pipeline (mode="manual")
  handle   -> run the proven direct pipeline (mode="auto", target_accounts=[...])
  batch    -> run the proven direct pipeline for 1-2 creator handles / scheduled pool
  upload   -> run the proven direct pipeline on a local video file
  reedit   -> call tool_reedit_session directly (preset buttons / custom directive)
  freeform -> hand to the LLM Director (the only place it adds value)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_HANDLE_RE = re.compile(r"^@?([A-Za-z0-9._]{1,30})$")
_REEDIT_RE = re.compile(
    r"re-?edit\s+session\s+['\"]?([A-Za-z0-9_\-]+)['\"]?\s+with\s+directive\s*:\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_UPLOAD_RE = re.compile(r"process\s+uploaded\s+video\s+at\s+'([^']+)'", re.IGNORECASE)
_BATCH_ACCOUNTS_RE = re.compile(r"creator\s+accounts?\s*:\s*((?:@[A-Za-z0-9._]+[\s,]*)+)", re.IGNORECASE)
_SCHEDULED_BATCH_RE = re.compile(r"run\s+scheduled\s+content\s+scraper\s+batch", re.IGNORECASE)

# Max handles per run (matches existing 2-account anti-duplicate rotation policy)
MAX_BATCH_ACCOUNTS = 2


@dataclass
class Route:
    kind: str                                   # url|handle|batch|upload|reedit|freeform
    goal: str = ""
    url: Optional[str] = None
    handles: List[str] = field(default_factory=list)
    platform: str = "instagram"
    session_id: Optional[str] = None
    directive: str = ""
    path: Optional[str] = None
    scheduled_pool: bool = False                # batch with no explicit handles

    @property
    def is_direct(self) -> bool:
        """True when the proven non-LLM pipeline should handle this goal."""
        return self.kind != "freeform"


def detect_platform(url: str) -> str:
    u = (url or "").lower()
    if "youtube.com" in u or "youtu.be" in u:
        return "youtube"
    if "tiktok.com" in u:
        return "tiktok"
    if "facebook.com" in u or "fb.watch" in u:
        return "facebook"
    return "instagram"


def _strip_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    return s.strip()


def classify_goal(text: str) -> Route:
    """Classify a user goal string into a Route without calling any LLM."""
    goal = (text or "").strip()
    if not goal:
        return Route(kind="freeform", goal=goal)

    # 1) Re-edit (preset buttons + custom directive) — highest priority, exact format from main.py
    m = _REEDIT_RE.search(goal)
    if m:
        return Route(
            kind="reedit",
            goal=goal,
            session_id=m.group(1).strip(),
            directive=_strip_quotes(m.group(2)),
        )

    # 2) Uploaded local video
    m = _UPLOAD_RE.search(goal)
    if m:
        return Route(kind="upload", goal=goal, path=m.group(1).strip())

    # 3) Scheduled-pool batch trigger (menu button)
    if _SCHEDULED_BATCH_RE.search(goal):
        return Route(kind="batch", goal=goal, scheduled_pool=True)

    # 4) Batch with explicit handles ("... creator accounts: @a, @b, edit each ...")
    m = _BATCH_ACCOUNTS_RE.search(goal)
    if m:
        handles = [h.strip().lstrip("@") for h in re.split(r"[\s,]+", m.group(1)) if h.strip()]
        handles = [h for h in handles if h][:MAX_BATCH_ACCOUNTS]
        if handles:
            return Route(kind="batch", goal=goal, handles=handles)

    # 5) URL (optionally followed by free-text editing instructions -> becomes directive)
    urls = _URL_RE.findall(goal)
    if len(urls) == 1:
        url = urls[0].rstrip(".,;)")
        extra = _URL_RE.sub("", goal).strip(" \t\r\n-:,")
        return Route(
            kind="url",
            goal=goal,
            url=url,
            platform=detect_platform(url),
            directive=extra,
        )

    # 6) Explicit handle forms only: "@name", "scrape: name", "account: name".
    #    A bare word ("hello", "faster") is NOT assumed to be a handle.
    lowered = goal.lower()
    for prefix in ("scrape:", "account:"):
        if lowered.startswith(prefix):
            cand = goal[len(prefix):].strip()
            hm = _HANDLE_RE.match(cand)
            if hm:
                return Route(kind="handle", goal=goal, handles=[hm.group(1)])
    if goal.startswith("@"):
        hm = _HANDLE_RE.match(goal)
        if hm:
            return Route(kind="handle", goal=goal, handles=[hm.group(1)])

    # 7) Anything else is genuinely open-ended -> LLM Director
    return Route(kind="freeform", goal=goal)
