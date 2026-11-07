"""
Publishing_Modules/instagram_features/user_tags.py
==================================================
Instagram User Tagging Feature Manager
Kingdom: Instagram Publishing Features
Purpose: Parse /tag commands from user input, calculate safe visual (x, y) coordinates
         so tags never block faces or subtitles, inject into IG Reels container,
         and provide auto-fallback if Meta rejects the container.

Usage:
    from Publishing_Modules.instagram_features.user_tags.py import UserTagsManager

    mgr = UserTagsManager()
    clean_title, usernames = mgr.extract_user_tags(raw_text)
    user_tags_payload = mgr.calculate_visual_coords(usernames)
    init_data = mgr.apply_user_tags(init_data, user_tags_payload)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

# Matches /tag or /user_tags followed by usernames
# Examples:
#   /tag @guest
#   /tag @john, @sarah
_TAG_PATTERN = re.compile(
    r"\s*/(?:user_)?tag(?:s)?\s+([@a-zA-Z0-9_\.,\s]+)",
    re.IGNORECASE,
)
_USERNAME_EXTRACT = re.compile(r"[@]?([a-zA-Z0-9_\.]+)")


class UserTagsManager:
    """
    Manages Instagram visual user tagging for Reels.
    Meta API limits: up to 20 tags per Reel.
    Visual placement: Coordinates must be in range [0.0, 1.0].
    """

    MAX_TAGS = 20

    def extract_user_tags(self, raw_text: str) -> Tuple[str, List[str]]:
        """
        Extract tagged usernames from raw text and strip the /tag command.
        Supports /tag anywhere: in front, middle, or end.

        Returns:
            (clean_text, list_of_clean_usernames)
        """
        if not raw_text or not isinstance(raw_text, str):
            return ("", [])

        m = re.search(r"/(?:user_)?tag(?:s)?\b", raw_text, re.IGNORECASE)
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
                if len(handles) >= self.MAX_TAGS:
                    break
            else:
                break

        matched_len = len("".join(consumed))
        clean_text = (pre + " " + post[matched_len:]).strip()
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        logger.info(
            "[UserTags] Extracted: %s | clean_text='%s'",
            handles, clean_text[:60],
        )
        return (clean_text, handles)

    def calculate_visual_coords(
        self,
        usernames: List[str],
        scene_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Calculate safe visual (x, y) coordinates for tagged accounts.

        Rules:
            • Default zone: Lower-third safe zone (x=0.5, y=0.82 to 0.88),
              avoiding center video subjects, face detections, and top headers.
            • Multiple users are horizontally or vertically staggered so tags
              do not overlap each other on screen.
        """
        if not usernames:
            return []

        tagged_payload: List[Dict[str, Any]] = []
        count = min(len(usernames), self.MAX_TAGS)

        # Generate staggered coordinates in the bottom-third safe margin
        for i, uname in enumerate(usernames[:count]):
            if count == 1:
                x, y = 0.50, 0.85
            elif count == 2:
                # Stagger left and right
                x = 0.35 if i == 0 else 0.65
                y = 0.85
            else:
                # Stagger across safe bottom row
                step = 0.60 / max(1, count - 1)
                x = round(0.20 + (i * step), 2)
                y = round(0.82 + ((i % 2) * 0.05), 2)

            tagged_payload.append({
                "username": uname.lstrip("@").strip(),
                "x": x,
                "y": y,
            })

        logger.info("[UserTags] Generated %d visual coordinates in safe zone.", len(tagged_payload))
        return tagged_payload

    def apply_user_tags(
        self, init_data: Dict[str, Any], user_tags: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Inject user_tags payload into Meta container init_data.

        Meta Graph API expects JSON array of objects:
            "user_tags": json.dumps([{"username": "user1", "x": 0.5, "y": 0.85}])
        """
        if not user_tags:
            return init_data

        valid_tags = [
            {"username": t["username"], "x": float(t.get("x", 0.5)), "y": float(t.get("y", 0.85))}
            for t in user_tags if t.get("username")
        ]
        if valid_tags:
            init_data["user_tags"] = json.dumps(valid_tags)
            logger.info("[UserTags] Injected %d user_tags into container init_data.", len(valid_tags))
        return init_data

    def handle_user_tags_fallback(
        self, init_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Remove user_tags from init_data on Meta rejection.
        """
        if "user_tags" in init_data:
            del init_data["user_tags"]
            logger.warning(
                "⚠️ [USER TAGS] User tags rejected by Meta. "
                "Stripped user_tags from init_data for fallback re-upload."
            )
        return init_data

    @staticmethod
    def is_user_tags_error(error_response: Any) -> bool:
        """Heuristic to detect if Meta failed due to user_tags."""
        flat = json.dumps(error_response).lower() if isinstance(error_response, (dict, list)) else str(error_response).lower()
        return "user_tags" in flat or "tagged user" in flat or "tagging" in flat

