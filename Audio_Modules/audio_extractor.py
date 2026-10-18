"""
audio_extractor.py — Phase 1 Audio Extraction + Beat Analysis
==============================================================
Called immediately after a clip is downloaded in Phase 1.

Workflow:
  1. ffprobe: check if video has an audio stream (skip silently if not)
  2. ffmpeg: extract mono 16 kHz WAV → {clip_dir}/{stem}_extracted.wav
     (saved in the CLIP folder, NOT in Original_audio/ which is the curated BGM pool)
  3. BeatEngine: run beat + drop analysis
  4. Persist → downloads/{owner}_{shortcode}/audio_analysis.json

Phase 2 reads audio_analysis.json directly — no re-extraction needed.

Audio Pool Architecture:
  Original_audio/active/   ← curated BGM tracks for Gemini selection
  Original_audio/cooldown/ ← recently used BGM (rotation cooldown)
  clip_dir/                ← per-clip extracted ambient audio + analysis (THIS file writes here)
"""

import os
import json
import subprocess
import logging
import threading
from typing import Optional

logger = logging.getLogger("AMTE.audio_extractor")

FFMPEG_BIN  = os.getenv("FFMPEG_BIN",  "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AUDIO_DIR   = os.path.join(_REPO_ROOT, "Original_audio")


def send_extracted_audio_to_telegram_vault_async(
    wav_path: str,
    clip_dir: str,
    stem: str = "clip"
) -> None:
    """
    Asynchronously uploads raw extracted audio WAV to Telegram Storage Vault without blocking main thread.
    Updates local audio_analysis.json and metadata.json with extracted_audio_file_id.
    """
    analysis_path = os.path.join(clip_dir, "audio_analysis.json")
    if os.path.exists(analysis_path):
        try:
            with open(analysis_path, "r", encoding="utf-8") as af:
                if json.load(af).get("extracted_audio_file_id"):
                    logger.info("ℹ️ [AUDIO VAULT ASYNC] Audio file_id already registered for %s — skipping duplicate upload", stem)
                    return
        except Exception:
            pass

    def _upload_task():
        try:
            logger.info("📤 [AUDIO VAULT ASYNC] Spawning background audio vault upload for: %s", wav_path)
            from Telegram_Storage_Modules.telegram_http import send_document, extract_file_id

            caption = f"🎵 Extracted Audio: {stem}"
            resp = send_document(wav_path, caption=caption)

            if resp and isinstance(resp, dict):
                file_id = extract_file_id(resp)
                if file_id:
                    logger.info("   ✓ [AUDIO VAULT ASYNC SUCCESS] Extracted audio uploaded to Vault (file_id: %s)", file_id)

                    # Update audio_analysis.json only
                    analysis_path = os.path.join(clip_dir, "audio_analysis.json")
                    adata = {}
                    if os.path.exists(analysis_path):
                        try:
                            with open(analysis_path, "r", encoding="utf-8") as af:
                                adata = json.load(af)
                        except Exception:
                            adata = {}
                    adata["extracted_audio_file_id"] = file_id
                    try:
                        with open(analysis_path, "w", encoding="utf-8") as af:
                            json.dump(adata, af, indent=2, ensure_ascii=False)
                    except Exception as ae:
                        logger.warning("   ⚠ Could not update audio_analysis.json with file_id: %s", ae)

                    try:
                        meta_path = os.path.join(clip_dir, "metadata.json")
                        if os.path.exists(meta_path):
                            with open(meta_path, "r", encoding="utf-8") as mf:
                                social_url = json.load(mf).get("url")
                            if social_url:
                                from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
                                TelegramVaultIndexer().enrich_and_update_pool_metadata(social_url, clip_dir)
                    except Exception as pe:
                        logger.warning("   ⚠ Could not sync pool_metadata.json with extracted audio file_id: %s", pe)
            else:
                logger.warning("   ⚠ [AUDIO VAULT ASYNC WARNING] Audio vault upload response not ok: %s", resp)
        except Exception as ve:
            logger.warning("   ⚠ [AUDIO VAULT ASYNC FAILED] Non-fatal audio vault upload error: %s", ve)

    thread = threading.Thread(target=_upload_task, name=f"AudioVaultUploader-{stem}", daemon=True)
    thread.start()
    logger.info("⚡ [AUDIO VAULT ASYNC] Background uploader thread started for audio %s (non-blocking).", stem)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _has_audio_stream(video_path: str) -> bool:
    """Returns True if the file has at least one audio stream (fast ffprobe check)."""
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
    except Exception:
        return True  # On probe failure, let FFmpeg try


def extract_audio(video_path: str, output_path: str) -> bool:
    """
    Extracts mono 16 kHz PCM WAV from video_path → output_path.
    Returns True on success, False if:
      - video has no audio stream
      - ffmpeg exits non-zero
    Captures FFmpeg stderr and logs it on failure so we know exactly what went wrong.
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        logger.info("♻️ [AUDIO] Extracted audio WAV already exists: %s (skipping duplicate FFmpeg)", os.path.basename(output_path))
        return True

    if not _has_audio_stream(video_path):
        logger.info("🔇 No audio stream in '%s' — skipping extraction.", os.path.basename(video_path))
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        FFMPEG_BIN, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path,
    ]

    logger.info("🎙️ Extracting audio from %s...", os.path.basename(video_path))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error(
                "❌ FFmpeg audio extraction failed (exit %d):\n%s",
                result.returncode,
                result.stderr[-800:],   # last 800 chars — enough to see the error
            )
            return False
        logger.info("✅ Extracted: %s (%.0f KB)", os.path.basename(output_path),
                    os.path.getsize(output_path) / 1024)
        return True
    except subprocess.TimeoutExpired:
        logger.error("❌ FFmpeg audio extraction timed out for '%s'", video_path)
        return False
    except Exception as exc:
        logger.error("❌ Unexpected error during audio extraction: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 Post-Download Hook
# ─────────────────────────────────────────────────────────────────────────────

def ensure_clip_audio_extracted(
    clip_dir: str,
    video_path: Optional[str] = None,
    clip_id: Optional[str] = None
) -> Optional[str]:
    """
    Directly retrieves the clip's extracted audio from Telegram Storage Vault
    using the file_id registered in pool_metadata.json.
    """
    if not clip_dir:
        return None

    clean_sc = (clip_id or os.path.basename(clip_dir)).replace("manual_", "").strip() or "clip"
    try:
        from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
        vault = TelegramVaultIndexer()
        hydrated_wav = vault.hydrate_extracted_audio_from_vault(clean_sc, dest_dir=clip_dir)
        if not hydrated_wav and clip_id:
            hydrated_wav = vault.hydrate_extracted_audio_from_vault(clip_id, dest_dir=clip_dir)
        return hydrated_wav
    except Exception as exc:
        logger.warning("⚠️ Vault extracted audio retrieval notice: %s", exc)
        return None


def run_phase1_audio_analysis(video_path: str, clip_dir: str) -> dict:
    """
    Phase 1 post-download hook. Called immediately after video.mp4 is saved.

    Steps:
      1. Extract audio WAV from video (skip if video is silent)
      2. Run BeatEngine analysis (tempo, energy, beats, drops, vibe)
      3. Save result to {clip_dir}/audio_analysis.json

    Returns the analysis dict (or a minimal silent-clip dict).
    """
    analysis_path = os.path.join(clip_dir, "audio_analysis.json")
    clean_sc = os.path.basename(clip_dir).replace("manual_", "").strip() or "clip"

    # Return cached result if already computed
    if os.path.exists(analysis_path):
        try:
            with open(analysis_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            logger.info("♻️ [AUDIO] Using cached audio_analysis.json for %s", os.path.basename(clip_dir))
            if not cached.get("extracted_audio_file_id"):
                wav_path = os.path.join(clip_dir, f"{clean_sc}.wav")
                if not os.path.exists(wav_path):
                    wav_path = os.path.join(clip_dir, "video_extracted.wav")
                if os.path.exists(wav_path):
                    send_extracted_audio_to_telegram_vault_async(wav_path, clip_dir, stem=clean_sc)
            return cached
        except Exception:
            pass  # Re-run if cache is corrupt

    os.makedirs(clip_dir, exist_ok=True)
    wav_path = os.path.join(clip_dir, f"{clean_sc}.wav")
    legacy_wav = os.path.join(clip_dir, "video_extracted.wav")

    # Step 1: Extract
    has_audio = extract_audio(video_path, wav_path)
    if has_audio and os.path.exists(wav_path):
        try:
            import shutil
            shutil.copy2(wav_path, legacy_wav)
        except Exception:
            pass

    # Step 1b: Asynchronous Telegram Storage Vault upload for raw extracted audio
    if has_audio and os.path.exists(wav_path):
        send_extracted_audio_to_telegram_vault_async(wav_path, clip_dir, stem=clean_sc)

    # Step 2: Beat analysis
    analysis: dict = {
        "has_audio":       has_audio,
        "has_vocals":      has_audio,
        "is_speech_vocal": has_audio,
        "wav_path":        wav_path if has_audio else None,
        "tempo_bpm":       0.0,
        "avg_energy":      0.0,
        "vibe":            "silent",
        "beats":           [],
        "beat_count":      0,
        "drops":           [],
        "drop_count":      0,
        "beat_score":      0.0,
        "source":          "phase1_extraction",
    }

    if has_audio and os.path.exists(wav_path):
        try:
            import sys
            sys.path.insert(0, _REPO_ROOT)
            from Audio_Modules.beat_engine import analyze_beats_with_drops

            beat_data = analyze_beats_with_drops(wav_path)
            vibe_str = str(beat_data.get("vibe", "unknown")).lower()
            has_v = beat_data.get("avg_energy", 0.0) > 0.01
            is_speech = has_v and ("speech" in vibe_str or "voice" in vibe_str or "dialogue" in vibe_str or "talk" in vibe_str or "vocal" in vibe_str or vibe_str in ("unknown", "silent"))

            analysis.update({
                "has_vocals":      has_v,
                "is_speech_vocal": is_speech,
                "tempo_bpm":  beat_data.get("tempo_bpm", 0.0),
                "avg_energy": beat_data.get("avg_energy", 0.0),
                "vibe":       beat_data.get("vibe", "unknown"),
                "beats":      beat_data.get("beats", []),
                "beat_count": len(beat_data.get("beats", [])),
                "drops":      beat_data.get("drops", []),
                "drop_count": len(beat_data.get("drops", [])),
                "beat_score": (
                    len(beat_data.get("beats", [])) * 1.0
                    + len(beat_data.get("drops", [])) * 3.0
                    + beat_data.get("avg_energy", 0.0) * 10.0
                ),
            })

            # Measure audio duration
            try:
                import wave
                with wave.open(wav_path, "rb") as wf:
                    dur_val = round(wf.getnframes() / float(wf.getframerate()), 2)
                    analysis["duration"] = dur_val
                    analysis["duration_sec"] = dur_val
            except Exception:
                pass

            # Step 2b: Extract word-level speech boundaries using faster-whisper
            try:
                from Audio_Modules.speech_boundary_detector import extract_speech_boundaries
                s_bounds_path = os.path.join(clip_dir, "speech_boundaries.json")
                s_data = extract_speech_boundaries(wav_path, output_json_path=s_bounds_path)
                if s_data.get("has_speech"):
                    analysis["speech_boundaries"] = {
                        "words_count": len(s_data.get("words", [])),
                        "clean_cut_timestamps": [c["timestamp_sec"] for c in s_data.get("clean_cut_timestamps", [])]
                    }
                    logger.info("🎙️ [SPEECH BOUNDARY] Found %d words, %d clean sentence cut boundaries.",
                                len(s_data.get("words", [])), len(s_data.get("clean_cut_timestamps", [])))
            except Exception as _sb_err:
                logger.warning("⚠️ [Speech Boundary] Extraction skipped/failed: %s", _sb_err)

        except Exception as exc:
            logger.warning("⚠️ Beat analysis failed: %s — storing raw extraction only.", exc)
    else:
        logger.info("🔇 [BEAT] Silent clip — skipping beat analysis.")

    # Step 3: Persist local clip analysis
    try:
        if os.path.exists(analysis_path):
            try:
                with open(analysis_path, "r", encoding="utf-8") as existing_f:
                    existing_data = json.load(existing_f)
                    for k, v in existing_data.items():
                        if k not in analysis or not analysis[k]:
                            analysis[k] = v
            except Exception:
                pass
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        logger.info("💾 [AUDIO] Saved audio_analysis.json → %s", clip_dir)
        try:
            meta_path = os.path.join(clip_dir, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as mf:
                    social_url = json.load(mf).get("url")
                if social_url:
                    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
                    TelegramVaultIndexer().enrich_and_update_pool_metadata(social_url, clip_dir)
        except Exception as _sync_err:
            logger.warning("   ⚠ Could not sync pool_metadata.json after audio analysis: %s", _sync_err)
    except Exception as exc:
        logger.warning("⚠️ Failed to write audio_analysis.json: %s", exc)

    # Step 4: Register extracted audio into active BGM pool for future selection.
    # Speech-only check: if the extracted audio is predominantly speech (podcast, dialogue,
    # talking-head, voiceover), do NOT ingest it into the BGM pool. It has no business being
    # offered as background music to any clip. Only musical audio enters the pool.
    _is_speech_only = bool(analysis.get("is_speech_vocal", False))
    if _is_speech_only:
        logger.info(
            "🗣️ [POOL_GATE] Skipping BGM pool ingestion for '%s' — detected as speech/dialogue audio. "
            "Only musical tracks are eligible for the BGM pool.",
            os.path.basename(wav_path)
        )
    elif has_audio and os.path.exists(wav_path) and analysis.get("tempo_bpm", 0) > 0:
        try:
            _ingest_clip_audio_to_pool(clean_sc, wav_path, analysis)
            logger.info("🎵 [AUDIO POOL] Registered harvested musical audio '%s' into Original_audio/active pool!", os.path.basename(wav_path))
        except Exception as pool_err:
            logger.warning("⚠️ Failed to register harvested audio into active pool: %s", pool_err)

        # Step 4: Preserve raw extracted audio in clip folder
        logger.info("🎵 [AUDIO] Extracted audio preserved in clip folder.")
        return analysis


def _ingest_clip_audio_to_pool(stem: str, wav_path: str, analysis: dict):
    """
    Ingests clean musical audio extracted from a Phase 1 clip into the central
    Original_audio/active/ BGM pool and updates indexed_audio_analysis.json.
    """
    try:
        active_dir = os.path.join(_AUDIO_DIR, "active")
        index_path = os.path.join(_AUDIO_DIR, "indexed_audio_analysis.json")
        os.makedirs(active_dir, exist_ok=True)

        clip_folder = os.path.basename(os.path.dirname(wav_path))
        clean_stem = stem
        if clean_stem.lower() in ("video", "video_extracted", "audio") and clip_folder and clip_folder != ".":
            clean_stem = f"bgm_{clip_folder}"

        # Duration Gate: never ingest audio shorter than 10 seconds into the BGM rotation pool
        clip_dur = float(analysis.get("duration") or analysis.get("duration_sec") or 0.0)
        if 0 < clip_dur < 10.0:
            logger.info("ℹ️ [POOL_INGEST] Extracted audio is too short (%.1fs < 10.0s) for BGM rotation pool. Preserving in clip folder only.", clip_dur)
            return

        target_wav = os.path.join(active_dir, f"{clean_stem}.wav")
        if not os.path.exists(target_wav):
            import shutil
            shutil.copy2(wav_path, target_wav)
            logger.info("🎶 [POOL_INGEST] Ingested clip audio into BGM pool: %s.wav", clean_stem)

        # Update global indexed_audio_analysis.json
        index_data = {}
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
            except Exception:
                index_data = {}

        index_data[stem] = {
            "video_id": stem,
            "file_name": f"{stem}.wav",
            "file_path": target_wav,
            "tempo_bpm": analysis.get("tempo_bpm", 0.0),
            "avg_energy": analysis.get("avg_energy", 0.0),
            "vibe": analysis.get("vibe", "unknown"),
            "beat_count": analysis.get("beat_count", 0),
            "drop_count": analysis.get("drop_count", 0),
            "drops": analysis.get("drops", []),
            "beat_score": analysis.get("beat_score", 0.0),
        }

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

        # Step 5: Upfront semantic speech analysis is handled cleanly downstream by
        # transcript_corrector in Step 02 (Forensic Perception) and lyric_rhythm_aligner
        # on the selected BGM in Step 05 (Rhythm Timeline).

        # Update pool_metadata.json (unified audio handler)
        try:
            from Audio_Modules.audio_pool_manager import AudioPoolManager
            pm = AudioPoolManager()
            track_name = f"{clean_stem}.wav"
            meta = pm.get_track_intelligence(track_name) or {}
            meta.update({
                "tempo_bpm": analysis.get("tempo_bpm", 0.0),
                "bpm": analysis.get("tempo_bpm", 0.0),
                "avg_energy": analysis.get("avg_energy", 0.0),
                "energy": analysis.get("avg_energy", 0.5),
                "vibe": analysis.get("vibe", "unknown"),
                "beat_count": analysis.get("beat_count", 0),
                "drop_count": analysis.get("drop_count", 0),
                "beat_score": analysis.get("beat_score", 0.0),
                "duration": clip_dur,
                "duration_sec": clip_dur,
                "audio_hash": pm._calculate_hash(target_wav),
                "version": pm.CURRENT_VERSION,
                "is_source_extract": True,
                "is_unusable_bgm": True,
            })
            pm._set_file_metadata(track_name, meta)
            pm._save_metadata()
            logger.info("📦 [POOL METADATA UPDATED] Audio intelligence for '%s' saved to pool_metadata.json", track_name)

            # ── TELEGRAM VAULT UPLOAD (capture file_id per track) ────────────────
            if not meta.get("file_id") and os.path.exists(target_wav):
                try:
                    synced = pm.sync_all_active_audios_to_telegram_vault(force=False)
                    captured_fid = synced.get(track_name) or pm.get_track_intelligence(track_name).get("file_id")
                    if captured_fid:
                        logger.info(
                            "✅ [VAULT AUDIO UPLOAD] Uploaded '%s' to Telegram Storage Group — file_id: %s",
                            track_name, captured_fid[:20]
                        )
                        meta["file_id"] = captured_fid
                        pm._set_file_metadata(track_name, meta)
                        pm._save_metadata()
                        # Sync file_id into Column 2 of master_vault_index.json
                        try:
                            from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
                            v_indexer = TelegramVaultIndexer()
                            c2_sess = v_indexer.vault_index.get("column_2_downloaded_sources", {}).get("by_session_id", {})
                            if clip_folder in c2_sess:
                                c2_sess[clip_folder]["extracted_audio_file_id"] = captured_fid
                                v_indexer._save_local_index()
                        except Exception:
                            pass

                        # Also sync extracted_audio_file_id under social_media_id in pool_metadata.json
                        try:
                            clips = pm.metadata.setdefault("files", {}).setdefault("social_media_id", {})
                            for k, entry in clips.items():
                                if isinstance(entry, dict):
                                    entry_sc = str(entry.get("shortcode") or "")
                                    if entry_sc.lower() == clean_stem.lower() or clean_stem.lower() in k.lower():
                                        entry.setdefault("media_file_ids", {})["extracted_audio_file_id"] = captured_fid
                                        entry["extracted_audio_file_id"] = captured_fid
                            pm._save_metadata(sync_to_vault=True)
                        except Exception:
                            pass
                    else:
                        logger.warning(
                            "⚠️ [VAULT AUDIO UPLOAD] '%s' upload returned no file_id — "
                            "Telegram vault retrieval will not work for this track.", track_name
                        )
                except Exception as _tg_err:
                    logger.warning("⚠️ [VAULT AUDIO UPLOAD] Could not upload '%s' to Telegram: %s", track_name, _tg_err)
            # ─────────────────────────────────────────────────────────────────────

        except Exception as _pm_err:
            logger.warning("⚠️ Could not update pool_metadata.json for '%s': %s", stem, _pm_err)

        # WAV path preserved for downstream reference
        analysis["wav_path"] = wav_path if os.path.exists(wav_path) else target_wav
        logger.info("💾 [AUDIO PRESERVED] Extracted WAV committed to active pool & Telegram Vault: %s", os.path.basename(wav_path))
    except Exception as err:
        logger.warning("⚠️ Pool ingestion warning for '%s': %s", stem, err)


def load_audio_analysis(clip_dir: str) -> Optional[dict]:
    """
    Phase 2 helper: loads pre-computed audio_analysis.json from clip_dir.
    Returns None if not found (Phase 2 will run its own fallback).
    """
    analysis_path = os.path.join(clip_dir, "audio_analysis.json")
    if not os.path.exists(analysis_path):
        return None
    try:
        with open(analysis_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and not data.get("extracted_audio_file_id"):
            wav_p = data.get("wav_path") or os.path.join(clip_dir, "video_extracted.wav")
            if os.path.exists(wav_p):
                stem = os.path.basename(clip_dir)
                try:
                    send_extracted_audio_to_telegram_vault_async(wav_p, clip_dir, stem=stem)
                except Exception:
                    pass
        return data
    except Exception as exc:
        logger.warning("⚠️ Failed to load audio_analysis.json: %s", exc)
        return None
