"""
audio_processing.py - Audio processing utility functions
"""

import os
import json
import subprocess
import logging

logger = logging.getLogger("AMTE.audio_processing")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")


def has_audio_stream(video_path: str) -> bool:
    """Returns True if the media file has at least one audio stream (ffprobe check)."""
    if not video_path or not os.path.exists(video_path):
        return False
    try:
        probe = subprocess.run(
            [
                FFPROBE_BIN, "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "json", video_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(probe.stdout or "{}")
        return bool(data.get("streams"))
    except Exception as e:
        logger.warning(f"ffprobe stream check failed for {video_path}: {e}")
        return True


def heavy_remix(input_path: str, output_path: str, original_volume: float = 1.0) -> bool:
    """
    Applies audio filters (subtle EQ / pitch adjustment) to bypass audio fingerprinting.
    """
    if not input_path or not os.path.exists(input_path):
        return False
    try:
        # Filter applies slight pitch shift + equalizer shift + volume adjustment
        filter_str = f"volume={original_volume},asetrate=44100*1.015,aresample=44100"
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", input_path,
            "-af", filter_str,
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            output_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return res.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        logger.warning(f"heavy_remix execution failed: {e}")
        return False

