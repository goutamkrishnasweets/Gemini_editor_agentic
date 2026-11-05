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

        Returns:
            (clean_text, list_of_clean_usernames)

        Rules:
            • /collab token and arguments are stripped completely from clean_text.
            • Leading '@' symbols are removed.
            • Capped at MAX_COLLABORATORS (3 accounts).
        """
        if not raw_text or not isinstance(raw_text, str):
            return ("", [])

        match = _COLLAB_PATTERN.search(raw_text)
        if not match:
            return (raw_text.strip(), [])

        matched_segment = match.group(0)
        args_segment = match.group(1)

        # Extract usernames from the arguments segment
        raw_names = _USERNAME_EXTRACT.findall(args_segment)
        clean_collabs: List[str] = []
        for name in raw_names:
            clean_name = name.lstrip("@").strip()
            # Prevent accidental match of subsequent commands or empty strings
            if clean_name and not clean_name.startswith("/") and clean_name not in clean_collabs:
                clean_collabs.append(clean_name)
            if len(clean_collabs) >= self.MAX_COLLABORATORS:
                break

        # Strip the full /collab command from the text
        clean_text = raw_text.replace(matched_segment, "").strip()
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        logger.info(
            "[Collaborators] Extracted: %s | clean_text='%s'",
            clean_collabs, clean_text[:60],
        )
        return (clean_text, clean_collabs)

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
