"""
Publishing_Modules/instagram_features/audio_namer.py
===================================================
Instagram Custom Audio Naming Feature Manager
Kingdom: Instagram Publishing Features
Purpose: Automatically brand and rename original audio tracks on Instagram Reels,
         transforming generic 'original audio - user' into branded titles like
         'Original Audio - My Topic'.

Usage:
    from Publishing_Modules.instagram_features.audio_namer import AudioNamerManager

    mgr = AudioNamerManager()
    audio_name = mgr.generate_audio_name(subject_or_title, brand_name)
    init_data = mgr.apply_audio_name(init_data, audio_name)
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AudioNamerManager:
    """
    Manages custom original audio naming for Instagram Reels.
    Meta API limits: max 100 chars, no special prohibited symbols.
    """

    MAX_AUDIO_NAME_LENGTH = 80

    def generate_audio_name(
        self,
        subject_or_title: str,
        brand_name: Optional[str] = None,
    ) -> str:
        """
        Generate a clean, branded audio title.

        Examples:
            "Original Audio - Sony A7IV Review"
            "TechLab - Sony A7IV Review"
        """
        clean_subj = re.sub(r"[^\w\s\-\.]", "", subject_or_title or "Reel").strip()
        clean_subj = re.sub(r"\s+", " ", clean_subj)

        if brand_name and brand_name.lower() not in ("general", "general_fallback", "none", ""):
            clean_brand = re.sub(r"[^\w\s\-\.]", "", brand_name).strip()
            name = f"{clean_brand} - {clean_subj}"
        else:
            name = f"Original Audio - {clean_subj}"

        if len(name) > self.MAX_AUDIO_NAME_LENGTH:
            name = name[:self.MAX_AUDIO_NAME_LENGTH].rstrip()

        logger.info("[AudioNamer] Generated audio name: '%s'", name)
        return name

    def apply_audio_name(
        self, init_data: Dict[str, Any], audio_name: str
    ) -> Dict[str, Any]:
        """
        Inject audio_name into Meta container init_data.
        """
        if audio_name and isinstance(audio_name, str) and audio_name.strip():
            init_data["audio_name"] = audio_name.strip()[:self.MAX_AUDIO_NAME_LENGTH]
            logger.info("[AudioNamer] Injected audio_name='%s' into container init_data.", init_data["audio_name"])
        return init_data

    def handle_audio_name_fallback(
        self, init_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Remove audio_name from init_data if Meta rejects custom audio naming.
        """
        if "audio_name" in init_data:
            del init_data["audio_name"]
            logger.warning(
                "⚠️ [AUDIO NAMER] audio_name rejected by Meta. "
                "Stripped audio_name from init_data for fallback re-upload."
            )
        return init_data

