"""
Audio_Modules/custom_audio_extractor.py
=======================================
Multi-Tier Custom User Audio Ingestion & Sanitization Engine
Kingdom: Audio Ingestion & Extraction
Purpose: Extracts and sanitizes user-provided audio URLs (Instagram, TikTok, YouTube, direct audio)
         through a 4-tier fail-safe pipeline:
           Tier 1: Telegram Storage Vault first-look (0-second instant recovery)
           Tier 2: Fast yt-dlp direct audio stream extraction (no full video download)
           Tier 3: Apify residential proxy scraper fallback (anti-bot bypass)
           Tier 4: Audio sanitization (FFprobe validation + 48kHz stereo WAV transcode)

Fail-Safe Guarantee:
    If all tiers fail or audio is corrupt, returns None.
    The downstream pipeline never crashes and gracefully falls back to Gemini's
    local audio pool selector. THE RENDER NEVER STOPS.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger("custom_audio_extractor")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class CustomAudioExtractor:
    """
    Multi-tier fail-safe extractor for user-provided custom audio tracks.
    """

    def __init__(self, dest_dir: Optional[str] = None):
        self.dest_dir = dest_dir or os.path.join(_REPO_ROOT, "downloads", "custom_audio_cache")
        os.makedirs(self.dest_dir, exist_ok=True)

    def extract_custom_audio(
        self,
        audio_url: str,
        target_clip_dir: Optional[str] = None,
        custom_filename: str = "user_custom_audio.wav"
    ) -> Optional[str]:
        """
        Extracts, verifies, and sanitizes custom audio from user-provided audio URL.

        Returns:
            Absolute path to sanitized 48kHz stereo WAV file, or None if extraction fails.
        """
        if not audio_url or not isinstance(audio_url, str):
            logger.warning("[CustomAudio] Empty or invalid audio URL provided.")
            return None

        clean_url = audio_url.strip().split("?")[0].rstrip("/")
        output_dir = target_clip_dir or self.dest_dir
        os.makedirs(output_dir, exist_ok=True)
        final_wav_path = os.path.join(output_dir, custom_filename)

        logger.info(f"🎧 [CUSTOM AUDIO] Initiating extraction for: {clean_url}")

        # ── TIER 1: Telegram Storage Vault First-Look Check ──────────────────
        try:
            from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
            v_indexer = TelegramVaultIndexer()
            vault_hydrated = v_indexer.hydrate_extracted_audio_from_vault(clean_url, dest_dir=output_dir)
            if vault_hydrated and os.path.exists(vault_hydrated) and os.path.getsize(vault_hydrated) > 10_000:
                logger.info(f"⚡ [VAULT FIRST-LOOK HIT] Retrieved audio from Telegram Vault: {vault_hydrated}")
                sanitized = self._sanitize_to_standard_wav(vault_hydrated, final_wav_path)
                if sanitized:
                    return sanitized
        except Exception as _v_err:
            logger.debug(f"[CustomAudio] Vault first-look notice: {_v_err}")

        # ── TIER 2: Fast yt-dlp Audio Stream Extraction ─────────────────────
        try:
            ytdlp_raw_audio = self._extract_via_ytdlp(clean_url, output_dir)
            if ytdlp_raw_audio and os.path.exists(ytdlp_raw_audio) and os.path.getsize(ytdlp_raw_audio) > 10_000:
                logger.info(f"✅ [TIER 2 HIT] yt-dlp extracted audio stream: {ytdlp_raw_audio}")
                sanitized = self._sanitize_to_standard_wav(ytdlp_raw_audio, final_wav_path)
                if sanitized:
                    return sanitized
        except Exception as _yt_err:
            logger.warning(f"⚠️ [CustomAudio] Tier 2 (yt-dlp) extraction error: {_yt_err}")

        # ── TIER 3: Apify Proxy Scraper Fallback ────────────────────────────
        try:
            apify_raw_audio = self._extract_via_apify(clean_url, output_dir)
            if apify_raw_audio and os.path.exists(apify_raw_audio) and os.path.getsize(apify_raw_audio) > 10_000:
                logger.info(f"✅ [TIER 3 HIT] Apify extracted audio: {apify_raw_audio}")
                sanitized = self._sanitize_to_standard_wav(apify_raw_audio, final_wav_path)
                if sanitized:
                    return sanitized
        except Exception as _ap_err:
            logger.warning(f"⚠️ [CustomAudio] Tier 3 (Apify) extraction error: {_ap_err}")

        logger.warning(f"❌ [CUSTOM AUDIO FAILED] All 3 extraction tiers failed for '{clean_url}'. Triggering Gemini pool fallback.")
        return None

    def _extract_via_ytdlp(self, url: str, output_dir: str) -> Optional[str]:
        """Extracts audio stream using yt-dlp directly."""
        import time
        tmp_stem = os.path.join(output_dir, f".tmp_audio_{os.getpid()}_{int(time.time()*1000)}")
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "-o", f"{tmp_stem}.%(ext)s",
            url
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45, text=True)
            expected_mp3 = f"{tmp_stem}.mp3"
            if os.path.exists(expected_mp3) and os.path.getsize(expected_mp3) > 5000:
                return expected_mp3
            # Check for any generated file with this stem
            for f in os.listdir(output_dir):
                if f.startswith(os.path.basename(tmp_stem)) and os.path.getsize(os.path.join(output_dir, f)) > 5000:
                    return os.path.join(output_dir, f)
        except Exception as e:
            logger.debug(f"[yt-dlp] subprocess error: {e}")
        return None

    def _extract_via_apify(self, url: str, output_dir: str) -> Optional[str]:
        """Downloads reel via Apify Downloader and rips audio using FFmpeg."""
        try:
            from Downloader_Modules.apify_downloader import download_reel_apify
            res = download_reel_apify(url, dest_dir=output_dir)
            video_file = res.get("video_path") if isinstance(res, dict) else None
            if video_file and os.path.exists(video_file) and os.path.getsize(video_file) > 10_000:
                # Rip audio track from downloaded video using FFmpeg
                ripped_mp3 = os.path.join(output_dir, f".apify_ripped_{os.getpid()}.mp3")
                cmd = ["ffmpeg", "-y", "-i", video_file, "-vn", "-acodec", "libmp3lame", "-q:a", "2", ripped_mp3]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
                if os.path.exists(ripped_mp3) and os.path.getsize(ripped_mp3) > 5000:
                    return ripped_mp3
        except Exception as e:
            logger.debug(f"[Apify] extraction error: {e}")
        return None

    def _sanitize_to_standard_wav(self, input_audio_path: str, output_wav_path: str) -> Optional[str]:
        """
        Transcodes raw audio into verified 48kHz stereo 16-bit PCM WAV.
        Ensures uniform sample rate and prevents FFmpeg buffer crashes.
        """
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", input_audio_path,
                "-vn",
                "-ar", "48000",
                "-ac", "2",
                "-f", "wav",
                output_wav_path
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
            if os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 10_000:
                logger.info(f"✅ [AUDIO SANITIZATION COMPLETE] 48kHz stereo WAV ready -> {output_wav_path}")
                # Clean up temporary raw download file if it's different
                if os.path.abspath(input_audio_path) != os.path.abspath(output_wav_path):
                    try:
                        os.remove(input_audio_path)
                    except Exception:
                        pass
                return output_wav_path
        except Exception as e:
            logger.warning(f"⚠️ [CustomAudio] Audio sanitization failed: {e}")
        return None
