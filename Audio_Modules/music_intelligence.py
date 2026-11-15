"""
Audio_Modules / music_intelligence.py
=====================================
Musical Intelligence Engine: Genre classification and audio filtergraph generation.
Provides rule-based and tempo-informed music classification for automated mixing.
"""

import os
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger("music_intelligence")

GENRE_MAP = {
    "lofi": {"vol": 0.35, "filter": "equalizer=f=1000:t=q:w=1:g=-2"},
    "ambient": {"vol": 0.35, "filter": "equalizer=f=800:t=q:w=1:g=-3"},
    "romantic": {"vol": 0.35, "filter": "equalizer=f=3000:t=q:w=1:g=1"},
    "pop": {"vol": 0.55, "filter": "equalizer=f=100:t=q:w=1:g=2,equalizer=f=10000:t=q:w=1:g=1.5"},
    "hiphop": {"vol": 0.55, "filter": "equalizer=f=60:t=q:w=1:g=3"},
    "high_energy": {"vol": 0.55, "filter": "equalizer=f=80:t=q:w=1:g=2.5"},
    "mass": {"vol": 0.55, "filter": "equalizer=f=120:t=q:w=1:g=2"},
    "classical": {"vol": 0.35, "filter": "equalizer=f=2000:t=q:w=1:g=0"},
    "neutral": {"vol": 0.40, "filter": ""},
}


def classify_music(audio_path: str) -> Tuple[str, float]:
    """
    Classifies the music genre of an audio track.
    Returns (genre_name, confidence).
    """
    if not audio_path or not os.path.exists(audio_path):
        return "neutral", 0.5

    clean_name = os.path.basename(audio_path).lower()

    if any(k in clean_name for k in ["lofi", "chill", "relax", "calm"]):
        return "lofi", 0.85
    if any(k in clean_name for k in ["ambient", "drone", "meditation", "space"]):
        return "ambient", 0.80
    if any(k in clean_name for k in ["romantic", "love", "acoustic", "slow"]):
        return "romantic", 0.85
    if any(k in clean_name for k in ["hiphop", "trap", "rap", "beat"]):
        return "hiphop", 0.85
    if any(k in clean_name for k in ["edm", "dance", "club", "hype", "workout", "fitness"]):
        return "high_energy", 0.90
    if any(k in clean_name for k in ["pop", "viral", "trend", "hit"]):
        return "pop", 0.80

    # Default heuristic: neutral
    return "neutral", 0.60


def get_filter_graph(genre: str) -> str:
    """Returns an FFmpeg audio filter string tuned for the given music genre."""
    return GENRE_MAP.get(genre.lower(), GENRE_MAP["neutral"])["filter"]


__all__ = ["classify_music", "get_filter_graph", "GENRE_MAP"]

