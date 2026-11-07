"""
Publishing_Modules/instagram_features/collaborators.py
======================================================
Instagram Collaborators (Co-Author) Feature Manager
Kingdom: Instagram Publishing Features
Purpose: Parse /collab commands from user input, validate and clean usernames,
         inject collaborator tags into the IG Reels media container, and provide
         an auto-fallback if Meta rejects the container due to invalid/private accounts.

Usage:
    from Publishing_Modules.instagram_features.collaborators import CollaboratorsManager

    mgr = CollaboratorsManager()
    clean_title, collabs = mgr.extract_collaborators(raw_text)
    init_data = mgr.apply_collaborators(init_data, collabs)
    # On Meta 400 error:
    init_data = mgr.handle_collaborators_ineligible_fallback(init_data)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# Pattern matches /collab or /collaborators followed by one or more usernames
# Examples:
#   /collab @alex
#   /collab @alex, @john
#   /collaborators alex john
_COLLAB_PATTERN = re.compile(
    r"\s*/collab(?:orators)?\s+([@a-zA-Z0-9_\.,\s]+)",
    re.IGNORECASE,
)
_USERNAME_EXTRACT = re.compile(r"[@]?([a-zA-Z0-9_\.]+)")


class CollaboratorsManager:
    """
    Manages Instagram Co-Author / Collaborators feature for Reels.
    Meta API limits: up to 3 collaborators, accounts must be public.
    """

    MAX_COLLABORATORS = 3

    def extract_collaborators(self, raw_text: str) -> Tuple[str, List[str]]:
        """
        Extract collaborator usernames and return the cleaned text.
        Supports /collab anywhere: in front, middle, or end.

        Returns:
            (clean_text, list_of_clean_usernames)

        Rules:
            • /collab token and arguments are stripped completely from clean_text.
            • Leading '@' symbols are removed.
            • Capped at MAX_COLLABORATORS (3 accounts).
            • Non-greedy: Never swallows title words (e.g. 'disha patani') following handles.
        """
        if not raw_text or not isinstance(raw_text, str):
            return ("", [])

        m = re.search(r"/(?:collab|collaborators)\b", raw_text, re.IGNORECASE)
        if not m:
            return (raw_text.strip(), [])

        pre = raw_text[:m.start()]
        post = raw_text[m.end():].lstrip()

        tokens = re.split(r"(\s+)", post)
        handles: List[str] = []
        consumed: List[str] = []
        prev_had_comma = False

        for i in range(0, len(tokens), 2):
            tok = tokens[i].strip()
            if not tok:
                continue
            if tok.startswith("/") or tok.startswith("http://") or tok.startswith("https://"):
                break

            is_at = tok.startswith("@")
            has_comma = tok.endswith(",")
            clean = tok.strip(",").lstrip("@")

            if is_at or prev_had_comma or len(handles) == 0:
                handles.append(clean)
                consumed.append(tokens[i])
                if i + 1 < len(tokens):
                    consumed.append(tokens[i + 1])
                prev_had_comma = has_comma
                if not is_at and not has_comma and not prev_had_comma:
                    break
                if len(handles) >= self.MAX_COLLABORATORS:
                    break
            else:
                break

        matched_len = len("".join(consumed))
        clean_text = (pre + " " + post[matched_len:]).strip()
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        logger.info(
            "[Collaborators] Extracted: %s | clean_text='%s'",
            handles, clean_text[:60],
        )
        return (clean_text, handles)

    def apply_collaborators(
        self, init_data: Dict[str, Any], collaborators: List[str]
    ) -> Dict[str, Any]:
        """
        Inject collaborators payload into Meta container init_data.

        Meta Graph API expects a JSON array of usernames:
            "collaborators": json.dumps(["user1", "user2"])
        """
        if not collaborators:
            return init_data

        valid_collabs = [c.lstrip("@").strip() for c in collaborators if c][:self.MAX_COLLABORATORS]
        if valid_collabs:
            init_data["collaborators"] = json.dumps(valid_collabs)
            logger.info("[Collaborators] Injected into container init_data: %s", init_data["collaborators"])
        return init_data

    def handle_collaborators_ineligible_fallback(
        self, init_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Remove collaborators from init_data for re-submission as standard Reel.
        Called when Meta rejects container initialization due to invalid/private collaborator accounts.
        """
        if "collaborators" in init_data:
            del init_data["collaborators"]
            logger.warning(
                "⚠️ [COLLABORATORS] Account ineligible or rejected by Meta. "
                "Stripped collaborators from init_data for standard re-upload."
            )
        return init_data

    @staticmethod
    def is_collaborator_error(error_response: Any) -> bool:
        """Heuristic to detect if Meta failed due to collaborator permissions/validity."""
        flat = json.dumps(error_response).lower() if isinstance(error_response, (dict, list)) else str(error_response).lower()
        return "collaborator" in flat or "collab" in flat or "coauthor" in flat

