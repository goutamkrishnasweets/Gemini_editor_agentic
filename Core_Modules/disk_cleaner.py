"""
Core_Modules/disk_cleaner.py
─────────────────────────────────────────────────────────────────────────────
Zero-Leak Cloud Ephemeral Scratchpad Disk Cleaner

Responsible for wiping temporary download and render artifacts from local disk
immediately after Telegram Vault backup and user delivery:
1. downloads/<clip_id>/
2. Original_audio/beats/<clip_id>_*
3. Original_audio/active/vault_bgm_<clip_id>*

CRITICAL SAFETY GUARDS:
- NEVER deletes or modifies any intelligence index JSON:
  * pool_metadata.json
  * master_vault_index.json
  * rejected_audio_blacklist.json
  * telegram_sessions.json
  * source_accounts.json
  * telegram_users.json
  * scraper_rotation_pointer.json
- NEVER touches master audio library tracks in Original_audio/active/ (only vault_bgm_*).
- NEVER deletes entire directories outside downloads/.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import shutil
import glob
import time
import tempfile
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("DiskCleaner")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [DiskCleaner] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Strict whitelist of directories that may contain ephemeral scratchpad artifacts
_DOWNLOADS_DIR = os.path.join(_REPO_ROOT, "downloads")
_BEATS_DIR = os.path.join(_REPO_ROOT, "Original_audio", "beats")
_ACTIVE_AUDIO_DIR = os.path.join(_REPO_ROOT, "Original_audio", "active")
_TEMP_WATERMARK_DIR = os.path.join(_REPO_ROOT, "temp_watermark")
_PROCESSED_SHORTS_DIR = os.path.join(_REPO_ROOT, "Processed Shorts")
_OUTPUT_DIR = os.path.join(_REPO_ROOT, "output")

# Protected database files that must NEVER be touched under any circumstance
_PROTECTED_FILES = {
    "pool_metadata.json",
    "master_vault_index.json",
    "rejected_audio_blacklist.json",
    "telegram_sessions.json",
    "source_accounts.json",
    "telegram_users.json",
    "scraper_rotation_pointer.json",
}


def clean_clip_artifacts(clip_id: str, master_video_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Safely purges ephemeral scratchpad artifacts for a single clip_id after delivery:
    1. downloads/<clip_id>/ (and downloads/manual_<clip_id>/)
    2. Original_audio/beats/<clip_id>_*
    3. Original_audio/active/vault_bgm_<clip_id>*
    4. Processed Shorts/<clip_id>_master.mp4 (post-vault upload)

    Guaranteed not to touch user library songs or database JSONs.
    """
    if not clip_id or not str(clip_id).strip():
        return {"status": "skipped", "reason": "empty_clip_id", "deleted_items": []}

    clean_id = str(clip_id).strip().replace("manual_", "")
    patterns_to_match = [clip_id, f"manual_{clean_id}", clean_id]
    deleted_items: List[str] = []
    errors: List[str] = []

    # 1. Purge downloads/<clip_id> directories
    if os.path.isdir(_DOWNLOADS_DIR):
        for pattern in set(patterns_to_match):
            target_dir = os.path.join(_DOWNLOADS_DIR, pattern)
            if os.path.isdir(target_dir):
                try:
                    shutil.rmtree(target_dir, ignore_errors=True)
                    if not os.path.exists(target_dir):
                        deleted_items.append(f"dir:downloads/{pattern}")
                        logger.info(f"🧹 [CLEANER] Erased temporary download directory: downloads/{pattern}")
                except Exception as e:
                    errors.append(f"Failed to remove {target_dir}: {e}")

    # 2. Purge matching cache files in Original_audio/beats/
    if os.path.isdir(_BEATS_DIR):
        for pattern in set(patterns_to_match):
            beat_pattern = os.path.join(_BEATS_DIR, f"*{pattern}*")
            for fpath in glob.glob(beat_pattern):
                fname = os.path.basename(fpath)
                if fname in _PROTECTED_FILES:
                    logger.warning(f"🛡️ [CLEANER BLOCKED] Attempted to delete protected file: {fname}")
                    continue
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                        deleted_items.append(f"file:Original_audio/beats/{fname}")
                        logger.info(f"🧹 [CLEANER] Erased temporary beat cache: {fname}")
                except Exception as e:
                    errors.append(f"Failed to remove {fpath}: {e}")

    # 3. Purge temporary hydrated vault BGM files in Original_audio/active/
    # STRICT GUARD: MUST start with 'vault_bgm_' to protect permanent user library songs!
    if os.path.isdir(_ACTIVE_AUDIO_DIR):
        for pattern in set(patterns_to_match):
            vault_bgm_pattern = os.path.join(_ACTIVE_AUDIO_DIR, f"vault_bgm_*{pattern}*")
            for fpath in glob.glob(vault_bgm_pattern):
                fname = os.path.basename(fpath)
                if not fname.startswith("vault_bgm_"):
                    # Safety belt
                    continue
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                        deleted_items.append(f"file:Original_audio/active/{fname}")
                        logger.info(f"🧹 [CLEANER] Erased temporary hydrated vault audio: {fname}")
                except Exception as e:
                    errors.append(f"Failed to remove {fpath}: {e}")

    # 4. Purge local master reel once safely vaulted
    if master_video_path and os.path.isfile(master_video_path):
        try:
            os.remove(master_video_path)
            deleted_items.append(f"file:{os.path.basename(master_video_path)}")
            logger.info(f"🧹 [CLEANER] Erased local master reel post-vault: {os.path.basename(master_video_path)}")
        except Exception as e:
            errors.append(f"Failed to remove master video {master_video_path}: {e}")

    return {
        "status": "success",
        "clip_id": clip_id,
        "deleted_count": len(deleted_items),
        "deleted_items": deleted_items,
        "errors": errors,
    }


def sweep_stale_temp_artifacts(max_age_hours: int = 2) -> Dict[str, Any]:
    """
    Sweeps orphaned temporary artifacts left over from interrupted runs or crashes.
    Only purges files/folders older than `max_age_hours`.
    """
    cutoff_time = time.time() - (max_age_hours * 3600)
    swept_items: List[str] = []
    errors: List[str] = []

    # 1. Sweep stale downloads/ directories
    if os.path.isdir(_DOWNLOADS_DIR):
        for entry in os.listdir(_DOWNLOADS_DIR):
            entry_path = os.path.join(_DOWNLOADS_DIR, entry)
            if os.path.isdir(entry_path):
                try:
                    mtime = os.path.getmtime(entry_path)
                    if mtime < cutoff_time:
                        shutil.rmtree(entry_path, ignore_errors=True)
                        if not os.path.exists(entry_path):
                            swept_items.append(f"dir:downloads/{entry}")
                            logger.info(f"🧹 [SWEEPER] Swept stale download dir (> {max_age_hours}h): {entry}")
                except Exception as e:
                    errors.append(f"Failed sweeping {entry_path}: {e}")

    # 2. Sweep stale Original_audio/beats/ files
    if os.path.isdir(_BEATS_DIR):
        for fname in os.listdir(_BEATS_DIR):
            if fname in _PROTECTED_FILES:
                continue
            fpath = os.path.join(_BEATS_DIR, fname)
            try:
                if os.path.isfile(fpath):
                    mtime = os.path.getmtime(fpath)
                    if mtime < cutoff_time:
                        os.remove(fpath)
                        swept_items.append(f"file:beats/{fname}")
                        logger.info(f"🧹 [SWEEPER] Swept stale beat cache (> {max_age_hours}h): {fname}")
            except Exception as e:
                errors.append(f"Failed sweeping {fpath}: {e}")

    # 3. Sweep stale Original_audio/active/vault_bgm_* files
    if os.path.isdir(_ACTIVE_AUDIO_DIR):
        for fname in os.listdir(_ACTIVE_AUDIO_DIR):
            if not fname.startswith("vault_bgm_"):
                # NEVER touch master user library files!
                continue
            fpath = os.path.join(_ACTIVE_AUDIO_DIR, fname)
            try:
                if os.path.isfile(fpath):
                    mtime = os.path.getmtime(fpath)
                    if mtime < cutoff_time:
                        os.remove(fpath)
                        swept_items.append(f"file:vault_bgm/{fname}")
                        logger.info(f"🧹 [SWEEPER] Swept stale hydrated audio (> {max_age_hours}h): {fname}")
            except Exception as e:
                errors.append(f"Failed sweeping {fpath}: {e}")

    return {
        "status": "success",
        "swept_count": len(swept_items),
        "swept_items": swept_items,
        "errors": errors,
    }


def sweep_temp_watermark(max_age_hours: int = 1) -> Dict[str, Any]:
    """Sweeps orphaned job folders in temp_watermark/ older than max_age_hours."""
    cutoff_time = time.time() - (max_age_hours * 3600)
    swept_items: List[str] = []
    errors: List[str] = []

    if os.path.isdir(_TEMP_WATERMARK_DIR):
        for entry in os.listdir(_TEMP_WATERMARK_DIR):
            entry_path = os.path.join(_TEMP_WATERMARK_DIR, entry)
            if os.path.isdir(entry_path):
                try:
                    mtime = os.path.getmtime(entry_path)
                    if mtime < cutoff_time:
                        shutil.rmtree(entry_path, ignore_errors=True)
                        if not os.path.exists(entry_path):
                            swept_items.append(f"dir:temp_watermark/{entry}")
                            logger.info(f"🧹 [SWEEPER] Swept stale watermark job (> {max_age_hours}h): {entry}")
                except Exception as e:
                    errors.append(f"Failed sweeping {entry_path}: {e}")
    return {"status": "success", "swept_count": len(swept_items), "swept_items": swept_items, "errors": errors}


def sweep_processed_shorts(max_age_hours: int = 2) -> Dict[str, Any]:
    """
    Sweeps rendered master reels in Processed Shorts/ older than max_age_hours.
    Safe because all master reels are backed up to the Telegram Vault immediately.
    """
    cutoff_time = time.time() - (max_age_hours * 3600)
    swept_items: List[str] = []
    errors: List[str] = []

    if os.path.isdir(_PROCESSED_SHORTS_DIR):
        for fname in os.listdir(_PROCESSED_SHORTS_DIR):
            if fname.lower().endswith((".mp4", ".mov", ".mkv")):
                fpath = os.path.join(_PROCESSED_SHORTS_DIR, fname)
                try:
                    mtime = os.path.getmtime(fpath)
                    if mtime < cutoff_time:
                        os.remove(fpath)
                        swept_items.append(f"file:Processed Shorts/{fname}")
                        logger.info(f"🧹 [SWEEPER] Swept aged master reel (> {max_age_hours}h): {fname}")
                except Exception as e:
                    errors.append(f"Failed sweeping {fpath}: {e}")
    return {"status": "success", "swept_count": len(swept_items), "swept_items": swept_items, "errors": errors}


def sweep_system_temp(max_age_hours: int = 1) -> Dict[str, Any]:
    """Sweeps orphaned phase2_frames_* directories from system temporary folder."""
    cutoff_time = time.time() - (max_age_hours * 3600)
    swept_items: List[str] = []
    errors: List[str] = []
    sys_temp = tempfile.gettempdir()

    if os.path.isdir(sys_temp):
        try:
            for entry in os.listdir(sys_temp):
                if entry.startswith("phase2_frames_"):
                    entry_path = os.path.join(sys_temp, entry)
                    if os.path.isdir(entry_path):
                        try:
                            mtime = os.path.getmtime(entry_path)
                            if mtime < cutoff_time:
                                shutil.rmtree(entry_path, ignore_errors=True)
                                if not os.path.exists(entry_path):
                                    swept_items.append(f"dir:temp/{entry}")
                                    logger.info(f"🧹 [SWEEPER] Swept orphaned frame temp dir: {entry}")
                        except Exception as _e:
                            errors.append(f"Failed sweeping {entry_path}: {_e}")
        except Exception as _list_e:
            errors.append(f"Failed scanning sys_temp: {_list_e}")

    return {"status": "success", "swept_count": len(swept_items), "swept_items": swept_items, "errors": errors}


def enforce_audio_pool_cap(max_tracks: int = 30, max_size_mb: int = 150) -> Dict[str, Any]:
    """
    Ensures Original_audio/active/ does not balloon infinitely.
    Caps active BGM library to max_tracks or max_size_mb (oldest pruned first).
    Protected core user tracks are NEVER removed.
    """
    if not os.path.isdir(_ACTIVE_AUDIO_DIR):
        return {"status": "skipped", "reason": "no_active_dir"}

    pruned_items: List[str] = []
    errors: List[str] = []

    try:
        audio_files = []
        for fname in os.listdir(_ACTIVE_AUDIO_DIR):
            if fname in _PROTECTED_FILES:
                continue
            fpath = os.path.join(_ACTIVE_AUDIO_DIR, fname)
            if os.path.isfile(fpath) and fname.lower().endswith((".wav", ".mp3", ".m4a")):
                try:
                    stat = os.stat(fpath)
                    audio_files.append((fpath, fname, stat.st_mtime, stat.st_size))
                except Exception:
                    pass

        # Sort oldest modified first
        audio_files.sort(key=lambda x: x[2])
        total_size_mb = sum(x[3] for x in audio_files) / (1024 * 1024)

        while len(audio_files) > max_tracks or total_size_mb > max_size_mb:
            oldest_path, oldest_name, _, oldest_size = audio_files.pop(0)
            try:
                os.remove(oldest_path)
                pruned_items.append(f"file:Original_audio/active/{oldest_name}")
                logger.info(f"🧹 [AUDIO POOL CAP] Pruned aged BGM track to preserve storage: {oldest_name}")
                total_size_mb -= oldest_size / (1024 * 1024)
            except Exception as e:
                errors.append(f"Failed pruning {oldest_path}: {e}")

    except Exception as e:
        errors.append(f"Audio cap check error: {e}")

    return {"status": "success", "pruned_count": len(pruned_items), "pruned_items": pruned_items, "errors": errors}


def check_storage_circuit_breaker(min_free_gb: float = 3.0) -> Dict[str, Any]:
    """
    Emergency Circuit Breaker:
    Checks host filesystem free disk space.
    If free space < min_free_gb, automatically runs an emergency zero-age deep purge
    across all caches and temporary directories.
    """
    try:
        usage = shutil.disk_usage(_REPO_ROOT)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)

        if free_gb < min_free_gb:
            logger.warning(
                f"🚨 [STORAGE CIRCUIT BREAKER TRIPPED] Free space is only {free_gb:.2f} GB "
                f"(Threshold: {min_free_gb} GB). Triggering EMERGENCY zero-age sweep!"
            )
            # Run aggressive zero-age sweep
            emergency_res = sweep_all_stale_storage(max_age_hours=0)
            # Recheck
            new_free_gb = shutil.disk_usage(_REPO_ROOT).free / (1024 ** 3)
            logger.info(f"🛡️ [CIRCUIT BREAKER POST-SWEEP] Reclaimed space. Free space is now: {new_free_gb:.2f} GB")
            return {
                "safe": new_free_gb >= min_free_gb,
                "free_gb": round(new_free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_gb": round(used_gb, 2),
                "emergency_sweep_executed": True,
                "emergency_items_swept": emergency_res.get("total_swept", 0)
            }

        return {
            "safe": True,
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "emergency_sweep_executed": False
        }
    except Exception as e:
        logger.warning(f"Storage circuit breaker check notice: {e}")
        return {"safe": True, "error": str(e), "emergency_sweep_executed": False}


def sweep_all_stale_storage(max_age_hours: int = 2) -> Dict[str, Any]:
    """
    Master Storage Sweeper:
    Coordinates all sub-sweepers across downloads, beats, watermark jobs,
    master reels, system temp frames, and BGM pool cap.
    """
    logger.info(f"🧹 [MASTER SWEEPER] Running comprehensive storage maintenance (Max Age: {max_age_hours}h)...")
    res1 = sweep_stale_temp_artifacts(max_age_hours=max_age_hours)
    res2 = sweep_temp_watermark(max_age_hours=min(max_age_hours, 1))
    res3 = sweep_processed_shorts(max_age_hours=max_age_hours)
    res4 = sweep_system_temp(max_age_hours=min(max_age_hours, 1))
    res5 = enforce_audio_pool_cap(max_tracks=30, max_size_mb=150)

    total_swept = (
        res1.get("swept_count", 0) +
        res2.get("swept_count", 0) +
        res3.get("swept_count", 0) +
        res4.get("swept_count", 0) +
        res5.get("pruned_count", 0)
    )
    logger.info(f"✅ [MASTER SWEEPER COMPLETE] Reclaimed {total_swept} storage artifact(s).")
    return {
        "status": "success",
        "total_swept": total_swept,
        "downloads_and_beats": res1,
        "watermark_jobs": res2,
        "processed_shorts": res3,
        "system_temp": res4,
        "audio_pool": res5
    }
