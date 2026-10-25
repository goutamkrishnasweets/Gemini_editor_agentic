"""
Core_Modules / purger.py
=======================
Complete Clip & Asset Purger.
When a user rejects or deletes a clip:
1. Deletes rendered master video file from Processed Shorts/ & output directories
2. Deletes all intermediate FFmpeg step files (*.mp4) related to the clip
3. Deletes source download folder (downloads/<clip_id>/ containing video.mp4, proxies, extracted audio WAVs, beat analysis)
4. Purges clip metadata index from pool_metadata.json
"""

import os
import shutil
import logging
import glob
from typing import Dict, Any, Optional
from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore

logger = logging.getLogger("core.purger")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _force_delete_file(file_path: str) -> bool:
    """Helper to remove a file cleanly, unlocking write permissions if locked."""
    if not os.path.exists(file_path):
        return False
    try:
        os.chmod(file_path, 0o777)
    except Exception:
        pass
    try:
        os.remove(file_path)
        return True
    except Exception as exc:
        logger.warning(f"⚠️ Purger: Failed to force delete file {file_path}: {exc}")
        return False


def _force_delete_dir(dir_path: str) -> bool:
    """Helper to remove a directory recursively with permission unlocks."""
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        return False
    try:
        for root, dirs, files in os.walk(dir_path):
            for f in files:
                try:
                    os.chmod(os.path.join(root, f), 0o777)
                except Exception:
                    pass
        shutil.rmtree(dir_path, ignore_errors=False)
        return True
    except Exception as exc:
        logger.warning(f"⚠️ Purger: shutil.rmtree error on {dir_path}: {exc}. Attempting force fallback...")
        try:
            shutil.rmtree(dir_path, ignore_errors=True)
            return True
        except Exception as _e2:
            logger.error(f"❌ Purger: Force fallback directory removal failed for {dir_path}: {_e2}")
            return False


def purge_full_clip_and_assets(
    clip_id: Optional[str] = None,
    video_path: Optional[str] = None,
    attempt_history: Optional[list] = None,
    selected_audio: Optional[str] = None
) -> Dict[str, Any]:
    """
    Completely purges a clip and all related assets:
      - Master rendered video (.mp4)
      - Intermediate FFmpeg step files
      - Source download directory (downloads/<clip_id>/)
      - Extracted audio files & beat analysis JSONs
      - Entire <url> object of the processed input from files["social_media_id"]
      - Entire <url> object of the selected audio's harvest source from files["social_media_id"]
      - Audio pool entry from files[audio_name] and physical audio files in Original_audio/
      - Vault source entries from master_vault_index.json
      - Full cloud sync to Telegram Storage Group
    """
    purged_items = []

    # 1. Normalize clip_id and video_path
    if not clip_id and video_path:
        base_name = os.path.basename(video_path)
        clip_id = base_name.replace("_master.mp4", "").replace(".mp4", "")

    if not clip_id:
        logger.warning("⚠️ Purger: No clip_id or video_path provided for purge.")
        return {"status": "error", "error": "No clip_id provided", "purged_count": 0, "purged_items": []}

    logger.info(f"🗑️ Purger: Initiating full dual-input deep purge for clip '{clip_id}' (selected_audio={selected_audio})...")

    # 2. Recover selected_audio if not provided
    if not selected_audio:
        try:
            store = ClipIntelligenceStore()
            c_intel = store.load(clip_id) or {}
            selected_audio = (
                c_intel.get("audio_data", {}).get("selected_bgm_track")
                or c_intel.get("audio_data", {}).get("selected_audio_track")
                or c_intel.get("selected_audio")
            )
        except Exception:
            pass

    if not selected_audio:
        try:
            from Telegram_Storage_Modules.session_manager import SessionManager
            sm = SessionManager()
            sess = sm.get_session(clip_id)
            if not sess:
                for s in sm.sessions.values():
                    if s.get("clip_id") == clip_id or clip_id in s.get("clip_id", "") or s.get("session_id") == clip_id:
                        sess = s
                        break
            if sess:
                selected_audio = (
                    sess.get("selected_audio")
                    or sess.get("audio_candidate")
                    or sess.get("audio_track")
                    or sess.get("bgm_path")
                )
        except Exception:
            pass

    if not selected_audio:
        try:
            from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
            indexer = TelegramVaultIndexer()
            v_rec = indexer.lookup_processed_reel(session_id=clip_id) or indexer.find_entry_by_shortcode(clip_id)
            if v_rec:
                selected_audio = (
                    v_rec.get("selected_audio")
                    or v_rec.get("selected_audio_track")
                    or v_rec.get("audio_file_id")
                )
        except Exception:
            pass

    if not selected_audio:
        try:
            from Audio_Modules.audio_pool_manager import get_pool_manager
            apm = get_pool_manager()
            pool_entry = apm.find_entry_by_shortcode(clip_id)
            if pool_entry:
                selected_audio = (
                    pool_entry.get("audio_data", {}).get("selected_audio")
                    or pool_entry.get("media_file_ids", {}).get("selected_audio_file_id")
                    or pool_entry.get("media_file_ids", {}).get("extracted_audio_file_id")
                )
        except Exception:
            pass

    if not selected_audio:
        clean_cid = clip_id.replace("manual_", "").strip()
        selected_audio = f"extracted_{clean_cid}.wav"

    # 3. Gather paths to delete (ensuring absolute resolution via _REPO_ROOT)
    paths_to_delete = set()

    if video_path:
        abs_v = video_path if os.path.isabs(video_path) else os.path.join(_REPO_ROOT, video_path)
        paths_to_delete.add(abs_v)

    if attempt_history:
        for p in attempt_history:
            if p:
                abs_p = p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)
                # Defensive check: only delete attempt history files that belong to this clip
                base_name = os.path.basename(abs_p)
                if clip_id in base_name or clean_cid in base_name:
                    paths_to_delete.add(abs_p)
                else:
                    logger.warning(
                        f"🛡️ [PURGER GUARD] Skipping attempt history file '{base_name}' — does not match clip '{clip_id}'"
                    )

    for p in paths_to_delete:
        if _force_delete_file(p):
            purged_items.append(f"Video file: {os.path.basename(p)}")
            logger.info(f"🗑️ Purger: Deleted video file {p}")

    # 4. Delete intermediate FFmpeg step files across output directories
    search_dirs = [
        os.path.join(_REPO_ROOT, "Processed Shorts"),
        os.path.join(_REPO_ROOT, "output"),
        os.path.join(_REPO_ROOT, "downloads"),
        os.path.join(_REPO_ROOT, "cache"),
        os.path.join(_REPO_ROOT, "sessions"),
    ]

    for s_dir in search_dirs:
        if os.path.exists(s_dir):
            step_pattern = os.path.join(s_dir, f"*{clip_id}*.mp4")
            for step_file in glob.glob(step_pattern):
                if _force_delete_file(step_file):
                    purged_items.append(f"Intermediate file: {os.path.basename(step_file)}")
                    logger.info(f"🗑️ Purger: Deleted intermediate file {step_file}")

    # 5. Delete source download folder (downloads/<clip_id>/)
    downloads_dir = os.path.join(_REPO_ROOT, "downloads")
    source_folder = os.path.join(downloads_dir, clip_id)

    if not os.path.exists(source_folder) and os.path.exists(downloads_dir):
        # Search for matching folder name in downloads
        try:
            for entry in os.listdir(downloads_dir):
                if clip_id in entry or entry in clip_id:
                    source_folder = os.path.join(downloads_dir, entry)
                    break
        except Exception as _ls_err:
            logger.warning(f"⚠️ Purger: Error scanning downloads directory: {_ls_err}")

    if _force_delete_dir(source_folder):
        purged_items.append(f"Download directory: downloads/{os.path.basename(source_folder)}")
        logger.info(f"🗑️ Purger: Deleted download directory {source_folder}")

    # 6. Deep Dual-Input Purge in pool_metadata.json (deletes entire <url> objects)
    try:
        from Audio_Modules.audio_pool_manager import get_pool_manager
        apm = get_pool_manager()
        pool_res = apm.purge_media_and_audio_entries(reel_identifier=clip_id, selected_audio_name=selected_audio)
        if pool_res.get("purged_items"):
            purged_items.extend(pool_res["purged_items"])
    except Exception as _apm_err:
        logger.warning(f"⚠️ Purger: AudioPoolManager deep purge notice: {_apm_err}")

    # 7. Deep Purge in master_vault_index.json (removes sources & audio candidates)
    try:
        from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
        vault = TelegramVaultIndexer()
        v_res = vault.purge_source_and_audio(reel_shortcode_or_url=clip_id, audio_shortcode_or_track=selected_audio)
        if v_res.get("purged_items"):
            purged_items.extend(v_res["purged_items"])
    except Exception as _v_err:
        logger.warning(f"⚠️ Purger: TelegramVaultIndexer purge notice: {_v_err}")

    # 8. Clean up local clip intelligence file
    try:
        store = ClipIntelligenceStore()
        store.purge_clip(clip_id, clip_folder=source_folder)
    except Exception as e:
        logger.warning(f"⚠️ Purger: Notice purging local clip intelligence file: {e}")

    return {
        "status": "success",
        "clip_id": clip_id,
        "selected_audio": selected_audio,
        "purged_count": len(purged_items),
        "purged_items": purged_items
    }
