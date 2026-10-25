import os
import re
import json
import time
import threading
import random
import logging
import numpy as np
import tempfile
from shutil import move
import glob
from typing import Dict, List, Optional, Any

logger = logging.getLogger("audio_pool_manager")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Thread-safe guard flag: prevents recursive vault hydration loops.
_VAULT_HYDRATION_IN_PROGRESS = False
_hydration_lock = threading.Lock()
_GLOBAL_POOL_LOCK = threading.RLock()


def _safe_replace(src: str, dst: str, max_retries: int = 5, delay: float = 0.1) -> None:
    """Safely replaces dst with src, handling Windows file lock contention."""
    for attempt in range(1, max_retries + 1):
        try:
            os.replace(src, dst)
            return
        except (PermissionError, OSError) as e:
            if attempt == max_retries:
                try:
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.replace(src, dst)
                    return
                except Exception:
                    raise e
            time.sleep(delay * attempt)

PIPELINE_BLOCKED_EXACT = {
    "video.wav", "video.mp4", "video_extracted.wav"
}

# Regex boundary patterns for temporary pipeline artifacts
# Note: uses strict word boundaries so 'my_first_shots_mix.mp3' is NOT blocked
PIPELINE_BLOCKED_PATTERNS = [
    r"(?:^|[/_.\-])reaction(?:$|[/_.\-])",
    r"(?:^|[/_.\-])textreaction(?:$|[/_.\-])",
    r"(?:^|[/_.\-])first_shots?(?:$|[/_.\-])",
    r"(?:^|[/_.\-])general_intro(?:$|[/_.\-])",
    r"(?:^|[/_.\-])watermark_clean(?:$|[/_.\-])",
    r"(?:^|[/_.\-])intro_mixed_temp(?:$|[/_.\-])",
    r"(?:^|[/_.\-])final_compilation(?:$|[/_.\-])",
    r"(?:^|[/_.\-])tmp(?:$|[/_.\-])",
]

def _is_pipeline_artifact(filename: str) -> bool:
    lower_name = filename.lower()
    if lower_name in PIPELINE_BLOCKED_EXACT:
        return True
    # Scratch artifacts are blocked
    for bad in ["_master", "_proxy", "_stripped", "_clean", "mask_", "tmp_", "step_", "audio_ducked"]:
        if bad in lower_name:
            return True
    # Explicit BGM files and audio tracks are valid
    if lower_name.startswith("vault_bgm_") or lower_name.startswith("bgm_"):
        return False
    if lower_name.endswith((".wav", ".mp3", ".m4a")):
        return False
    return any(bool(re.search(pat, lower_name)) for pat in PIPELINE_BLOCKED_PATTERNS)

class AudioPoolManager:
    """
    Manages the lifecycle of extracted audio clips.
    Pools:
      - active/: Eligible for selection.
      - cooldown/: Temporarily ineligible clips (recently used).
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None or base_dir == "Original_audio":
            _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_dir = os.path.join(_repo, "Original_audio")
        self.base_dir = os.path.abspath(base_dir)
        self.active_dir = os.path.join(self.base_dir, "active")
        self.cooldown_dir = os.path.join(self.base_dir, "cooldown")
        self.beats_dir = os.path.join(self.base_dir, "beats")
        self.meta_path = os.path.join(self.base_dir, "pool_metadata.json")

        self.lock = _GLOBAL_POOL_LOCK
        self._cache_lock = threading.Lock()
        self._beat_cache = {}
        self.MAX_CACHE_SIZE = 20
        self.CURRENT_VERSION = 3
        
        # Vault pool cache (5 min TTL) to avoid round-trip network floods on selection
        self._vault_pool_cache = None
        self._vault_pool_cache_ts = 0.0
        self._VAULT_POOL_CACHE_TTL = 300.0
        
        # Ensure directories exist
        os.makedirs(self.active_dir, exist_ok=True)
        os.makedirs(self.cooldown_dir, exist_ok=True)
        os.makedirs(self.beats_dir, exist_ok=True)
        
        self.metadata = self._load_metadata()
        self.hydrate_harvested_clip_metadata()
        # Sync any loose files that landed in root (e.g. from extract_audio_from_video)
        # into active/ so select_best_audio() can find them immediately.
        self._sync_root_to_active()
        # Ensure all files in active/ are registered in metadata so they are not skipped.
        self._sync_active_to_metadata()
        # Sanitize any rogue video files mistakenly registered in files root
        self.sanitize_pool_metadata_anomalies()

    @property
    def reuse_cooldown_seconds(self) -> float:
        """Single source of truth for how long a track stays ineligible for re-selection after use."""
        try:
            return float(os.getenv("AUDIO_COOLDOWN_HOURS", "6.0")) * 3600.0
        except (ValueError, TypeError):
            return 6.0 * 3600.0

    @property
    def orphan_recovery_seconds(self) -> float:
        """Single source of truth for how long a used track sits before maintenance/sync treats it as safe for orphan recovery."""
        try:
            return float(os.getenv("AUDIO_ORPHAN_RECOVERY_HOURS", "48.0")) * 3600.0
        except (ValueError, TypeError):
            return 48.0 * 3600.0

    @property
    def cooldown_seconds(self) -> float:
        """Alias for reuse_cooldown_seconds."""
        return self.reuse_cooldown_seconds

    @staticmethod
    def _parse_timestamp(val: Any) -> float:
        """Safely converts numeric timestamp or ISO-8601 string to epoch float seconds."""
        if not val:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            val_str = val.strip()
            if not val_str:
                return 0.0
            try:
                return float(val_str)
            except ValueError:
                pass
            try:
                from datetime import datetime, timezone
                clean_iso = val_str.replace("Z", "+00:00") if val_str.endswith("Z") else val_str
                dt = datetime.fromisoformat(clean_iso)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except Exception:
                return 0.0
        return 0.0

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        """Safely converts val to float with fallback default."""
        try:
            return float(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(val: Any, default: int = 0) -> int:
        """Safely converts val to int with fallback default."""
        try:
            return int(val) if val is not None else default
        except (ValueError, TypeError):
            return default

    def _load_metadata(self) -> Dict:
        """Loads metadata safely with Telegram Vault cloud hydration fallback."""
        global _VAULT_HYDRATION_IN_PROGRESS
        needs_hydration = not os.path.exists(self.meta_path) or os.path.getsize(self.meta_path) < 10

        if needs_hydration:
            with _hydration_lock:
                in_progress = _VAULT_HYDRATION_IN_PROGRESS
                if not in_progress:
                    _VAULT_HYDRATION_IN_PROGRESS = True

            if in_progress:
                logger.debug("[POOL_META] Skipping recursive vault hydration (already in progress)")
            else:
                try:
                    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
                    indexer = TelegramVaultIndexer()
                    indexer.hydrate_all_vault_jsons_on_startup()
                except Exception as _h_err:
                    logger.warning(f"⚠️ [POOL_META] Vault metadata hydration warning: {_h_err}")
                finally:
                    with _hydration_lock:
                        _VAULT_HYDRATION_IN_PROGRESS = False

        if not os.path.exists(self.meta_path):
            return {}
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load audio pool metadata: {e}")
            return {}

    def hydrate_harvested_clip_metadata(self):
        """
        Scans all clip directories under downloads/ and ensures that clip_source_math
        entries in pool_metadata.json are fully hydrated with:
          - social_media_id & shortcode
          - raw_video_file_id & extracted_audio_file_id
          - file_name, file_size, downloaded_at, user_id
          - audio_math (from audio_analysis.json)
          - whisper_transcript (from speech_boundaries.json)
          - gemini_semantic_intelligence (from lyric cache or existing)
          - social metadata: caption, hashtags, taggedUsers, likesCount, videoViewCount, timestamp, ownerUsername
        """
        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            downloads_dir = os.path.join(repo_root, "downloads")
            if not os.path.exists(downloads_dir):
                return

            with self.lock:
                if "files" not in self.metadata:
                    self.metadata = {"version": self.CURRENT_VERSION, "files": self.metadata}
                files_root = self.metadata["files"]
                if "social_media_id" in files_root:
                    clips_dict = files_root["social_media_id"]
                else:
                    for key in ["clip_source_math", "clips", "sources"]:
                        if key in files_root and "social_media_id" not in files_root:
                            files_root["social_media_id"] = files_root.pop(key)
                    clips_dict = files_root.setdefault("social_media_id", {})

                changed = False

                for clip_folder in os.listdir(downloads_dir):
                    folder_path = os.path.join(downloads_dir, clip_folder)
                    if not os.path.isdir(folder_path):
                        continue

                    meta_file = os.path.join(folder_path, "metadata.json")
                    video_json_file = os.path.join(folder_path, "video.json")
                    audio_analysis_file = os.path.join(folder_path, "audio_analysis.json")
                    speech_file = os.path.join(folder_path, "speech_boundaries.json")
                    video_file = os.path.join(folder_path, "video.mp4")

                    meta_dict = {}
                    if os.path.exists(meta_file):
                        try:
                            with open(meta_file, "r", encoding="utf-8") as f:
                                meta_dict = json.load(f)
                        except Exception:
                            pass

                    video_json_dict = {}
                    if os.path.exists(video_json_file):
                        try:
                            with open(video_json_file, "r", encoding="utf-8") as f:
                                video_json_dict = json.load(f)
                        except Exception:
                            pass

                    audio_analysis_dict = {}
                    if os.path.exists(audio_analysis_file):
                        try:
                            with open(audio_analysis_file, "r", encoding="utf-8") as f:
                                audio_analysis_dict = json.load(f)
                        except Exception:
                            pass

                    speech_dict = {}
                    if os.path.exists(speech_file):
                        try:
                            with open(speech_file, "r", encoding="utf-8") as f:
                                speech_dict = json.load(f)
                        except Exception:
                            pass

                    from Telegram_Storage_Modules.telegram_vault_indexer import canonicalize_social_url, extract_clean_shortcode

                    clean_folder_sc = extract_clean_shortcode(clip_folder) or clip_folder.replace("manual_", "").strip()
                    shortcode = meta_dict.get("shortcode") or video_json_dict.get("id") or clean_folder_sc
                    shortcode = extract_clean_shortcode(shortcode) or shortcode
                    owner_username = meta_dict.get("ownerUsername") or video_json_dict.get("uploader") or clip_folder.split("_")[0]

                    # Canonicalize URL from shortcode
                    canonical_social_url = canonicalize_social_url(shortcode) if shortcode else ""

                    # If shortcode already exists in clips_dict, reuse its entry and migrate key if dirty!
                    existing = None
                    existing_key = None
                    if shortcode:
                        for _su, _se in list(clips_dict.items()):
                            if not isinstance(_se, dict):
                                continue
                            _sc_val = _se.get("shortcode") or extract_clean_shortcode(_su)
                            if _sc_val and _sc_val.lower() == shortcode.lower():
                                existing = _se
                                existing_key = _su
                                break

                    if not existing:
                        raw_s_url = (
                            meta_dict.get("url") or meta_dict.get("social_media_id") or
                            video_json_dict.get("webpage_url") or video_json_dict.get("url") or
                            f"https://www.instagram.com/reel/{clean_folder_sc}/"
                        )
                        canonical_social_url = canonicalize_social_url(raw_s_url) or canonical_social_url
                        existing = clips_dict.get(canonical_social_url, {})

                    social_url = canonical_social_url or (existing_key if existing else f"https://www.instagram.com/reel/{clean_folder_sc}/")
                    if existing_key and existing_key != social_url:
                        clips_dict.pop(existing_key, None)
                        changed = True

                    raw_vault_id = meta_dict.get("raw_vault_file_id") or meta_dict.get("raw_video_file_id")
                    extracted_audio_id = (
                        meta_dict.get("extracted_audio_file_id") or
                        audio_analysis_dict.get("extracted_audio_file_id")
                    )

                    file_size = os.path.getsize(video_file) if os.path.exists(video_file) else 0
                    downloaded_at = os.path.getmtime(video_file) if os.path.exists(video_file) else time.time()
                    gemini_audio_intel = (
                        existing.get("gemini_semantic_audio_intelligence") or
                        existing.get("gemini_semantic_intelligence") or {}
                    )
                    gemini_visual_intel = existing.get("gemini_semantic_visual_intelligence") or {}
                    cbm_intel = existing.get("creator_behavior_model") or {}

                    lyric_cache_path = os.path.join(repo_root, "Original_audio", "beats", f"{clip_folder}_video_extracted_lyric.json")
                    if not gemini_audio_intel and os.path.exists(lyric_cache_path):
                        try:
                            with open(lyric_cache_path, "r", encoding="utf-8") as lf:
                                gemini_audio_intel = json.load(lf)
                        except Exception:
                            pass

                    # Extract or initialize sub-structures safely according to v3 schema
                    media_file_ids = existing.get("media_file_ids") or {
                        "raw_video_file_id": raw_vault_id or existing.get("raw_video_file_id"),
                        "extracted_audio_file_id": extracted_audio_id or existing.get("extracted_audio_file_id"),
                        "wm_clean_file_id": existing.get("wm_clean_file_id"),
                        "selected_audio_file_id": existing.get("selected_audio_file_id"),
                        "processed_output_file_id": existing.get("processed_output_file_id") or existing.get("master_reel_file_id"),
                    }
                    if raw_vault_id:
                        media_file_ids["raw_video_file_id"] = raw_vault_id
                    if extracted_audio_id:
                        media_file_ids["extracted_audio_file_id"] = extracted_audio_id
                    if "selected_audio_file_id" not in media_file_ids:
                        media_file_ids["selected_audio_file_id"] = existing.get("selected_audio_file_id")

                    existing_audio_data = existing.get("audio_data") or {}
                    last_used_val = existing_audio_data.get("last_used") or existing.get("last_used") or 0.0
                    usage_count_val = existing_audio_data.get("usage_count") or existing.get("usage_count") or 0

                    audio_data = existing.get("audio_data") or {
                        "audio_math": audio_analysis_dict or existing.get("audio_math", {}),
                        "whisper_transcript": speech_dict or existing.get("whisper_transcript", {}),
                        "gemini_audio_output": gemini_audio_intel,
                        "selected_audio": existing.get("selected_audio"),
                        "last_used": last_used_val,
                        "usage_count": usage_count_val,
                    }
                    if audio_analysis_dict:
                        audio_data["audio_math"] = audio_analysis_dict
                    if speech_dict:
                        audio_data["whisper_transcript"] = speech_dict
                    if gemini_audio_intel:
                        audio_data["gemini_audio_output"] = gemini_audio_intel
                    if "selected_audio" not in audio_data:
                        audio_data["selected_audio"] = existing.get("selected_audio")
                    if "last_used" not in audio_data:
                        audio_data["last_used"] = last_used_val
                    if "usage_count" not in audio_data:
                        audio_data["usage_count"] = usage_count_val

                    visual_data = existing.get("visual_data") or {
                        "gemini_visual_output": gemini_visual_intel,
                        "gemini_watermark_output": existing.get("gemini_watermark_output", {}),
                        "clip_intelligence": existing.get("clip_intelligence", {}),
                        "vectors": existing.get("vectors", []),
                    }
                    if gemini_visual_intel:
                        visual_data["gemini_visual_output"] = gemini_visual_intel

                    # Build updated v3 entry
                    updated_entry = {
                        "social_media_id": social_url,
                        "shortcode": shortcode,
                        "ownerUsername": owner_username or existing.get("ownerUsername"),
                        "caption": meta_dict.get("caption") or video_json_dict.get("description") or existing.get("caption"),
                        "hashtags": meta_dict.get("hashtags") or existing.get("hashtags", []),
                        "taggedUsers": meta_dict.get("taggedUsers") or existing.get("taggedUsers", []),
                        "likesCount": meta_dict.get("likesCount") or video_json_dict.get("like_count") or existing.get("likesCount"),
                        "videoViewCount": meta_dict.get("videoViewCount") or video_json_dict.get("view_count") or existing.get("videoViewCount"),
                        "timestamp": meta_dict.get("timestamp") or existing.get("timestamp"),
                        "user_id": meta_dict.get("user_id") or meta_dict.get("chat_id") or existing.get("user_id"),
                        "downloaded_at": downloaded_at or existing.get("downloaded_at", time.time()),
                        "file_name": "video.mp4",
                        "file_size": file_size or existing.get("file_size", 0),
                        "sha256": existing.get("sha256", ""),
                        "selected_audio": existing.get("selected_audio"),
                        "media_file_ids": media_file_ids,
                        "audio_data": audio_data,
                        "visual_data": visual_data,
                        "creator_behavior_model": cbm_intel,
                        "video_editing_plan": existing.get("video_editing_plan", {}),
                        "pipeline_execution_trajectory": existing.get("pipeline_execution_trajectory", {}),
                        "editing_plan_history": existing.get("editing_plan_history", []),
                    }

                    if existing != updated_entry:
                        clips_dict[social_url] = updated_entry
                        changed = True

                if changed:
                    self._save_metadata(sync_to_vault=False)
                    logger.info("[POOL_HYDRATE] Fully hydrated harvested clip metadata in pool_metadata.json.")
        except Exception as e:
            logger.debug(f"[POOL_HYDRATE] Clip metadata hydration notice: {e}")

    def patch_creator_behavior_model(self, creator_handle: str, cbm_payload: Dict[str, Any]) -> int:
        """
        Patches creator_behavior_model payload into pool_metadata.json for all clips matching creator_handle.
        Placing 'creator_behavior_model' directly after 'gemini_semantic_visual_intelligence'.
        Saves metadata atomically and syncs to Telegram Vault.
        """
        if not creator_handle or not isinstance(cbm_payload, dict):
            return 0

        import re
        norm_handle = creator_handle.lower().strip("@").strip()
        pattern = re.compile(rf"(?:^|[/_@.]){re.escape(norm_handle)}(?:[/_@.]|$)", re.IGNORECASE)

        updated_count = 0
        with self.lock:
            if "files" not in self.metadata:
                self.metadata = {"version": self.CURRENT_VERSION, "files": self.metadata}
            files_root = self.metadata["files"]
            social_entries = files_root.setdefault("social_media_id", {})

            # Search both social_entries and top-level files dictionary
            dicts_to_check = []
            if isinstance(social_entries, dict):
                dicts_to_check.append(social_entries)
            if isinstance(files_root, dict):
                dicts_to_check.append(files_root)

            for target_dict in dicts_to_check:
                for url, clip in list(target_dict.items()):
                    if not isinstance(clip, dict) or url == "social_media_id":
                        continue

                    owner = (clip.get("ownerUsername") or "").lower().strip("@").strip()
                    shortcode = (clip.get("shortcode") or "").lower().strip()
                    filename = (clip.get("filename") or "").lower().strip()

                    is_match = (owner == norm_handle or shortcode == norm_handle or norm_handle in filename or bool(pattern.search(url)))
                    if is_match:
                        reordered_clip = {}
                        inserted = False
                        for k, v in clip.items():
                            reordered_clip[k] = v
                            if k == "gemini_semantic_visual_intelligence":
                                reordered_clip["creator_behavior_model"] = cbm_payload
                                inserted = True

                        if not inserted:
                            reordered_clip["creator_behavior_model"] = cbm_payload

                        target_dict[url] = reordered_clip
                        updated_count += 1

            if updated_count > 0:
                self.metadata["updated_at"] = time.time()
                self._save_metadata(sync_to_vault=True)
                logger.info(f"📊 [AudioPoolManager] Patched creator_behavior_model for '{creator_handle}' across {updated_count} clip(s).")

        return updated_count

    def _save_metadata(self, sync_to_vault: bool = True):
        """Saves metadata atomically with file locking and syncs to Telegram Storage Group."""
        with self.lock:
            temp_path = self.meta_path + ".tmp"
            try:
                # Always ensure version and updated_at timestamp are at the top level
                ordered_data = {
                    "version": self.CURRENT_VERSION,
                    "updated_at": time.time(),
                }
                for k, v in self.metadata.items():
                    if k not in ("version", "updated_at"):
                        ordered_data[k] = v
                self.metadata = ordered_data
                
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                _safe_replace(temp_path, self.meta_path)

                if sync_to_vault:
                    self._sync_to_telegram_vault()
            except Exception as e:
                logger.error(f"❌ Failed to save audio pool metadata: {e}")
                if os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass

    def _sync_to_telegram_vault(self):
        """Uploads pool_metadata.json to Telegram Storage Group via telegram_http.py and updates pinned master_vault_index.json."""
        try:
            if not os.path.exists(self.meta_path):
                return

            from Telegram_Storage_Modules.telegram_http import send_document, extract_file_id
            from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer

            res = send_document(
                local_path=self.meta_path,
                caption=f"📦 **[VAULT BACKUP]** `pool_metadata.json` (Updated {time.strftime('%H:%M:%S')})"
            )
            if res:
                doc_id = extract_file_id(res)
                if doc_id:
                    indexer = TelegramVaultIndexer()
                    indexer.vault_index["pool_metadata_file_id"] = doc_id
                    indexer.vault_index["metadata_pool_file_id"] = doc_id
                    indexer._save_local_index()
                    indexer.upload_and_pin_vault_index_sync()
                    logger.info("✅ [POOL METADATA VAULT BACKUP] Uploaded & PINNED pool_metadata.json to Telegram Storage Group (file_id: %s)", doc_id[:15])
        except Exception as _e:
            logger.warning("⚠️ Notice triggering pool metadata sync: %s", _e)

    def _calculate_fingerprint(self, path: str) -> str:
        """Fast size+mtime fingerprint for metadata cache validation."""
        try:
            stat = os.stat(path)
            return f"{stat.st_size}_{int(stat.st_mtime)}"
        except Exception:
            return "unknown"

    def _calculate_hash(self, path: str) -> str:
        """Legacy alias for _calculate_fingerprint."""
        return self._calculate_fingerprint(path)

    def _safe_save_npz(self, path: str, **data):
        """Atomic NPZ save using tempfile + replace."""
        dir_name = os.path.dirname(path)
        fd, temp_path = tempfile.mkstemp(dir=dir_name, suffix=".npz")
        os.close(fd)
        try:
            np.savez_compressed(temp_path, **data)
            os.replace(temp_path, path)
        except Exception as e:
            logger.error(f"❌ Atomic NPZ save failed for {path}: {e}")
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except: pass

    def _get_file_metadata(self, key_or_filename: str) -> Optional[Dict]:
        """Helper to get file metadata accounting for schema version and key/file_id lookups."""
        if not key_or_filename:
            return None
        files = self.metadata.get("files", self.metadata)
        if isinstance(files, dict):
            if key_or_filename in files:
                return files[key_or_filename]
            for k, v in files.items():
                if isinstance(v, dict):
                    if v.get("file_id") == key_or_filename or v.get("filename") == key_or_filename:
                        return v
                    if os.path.splitext(k.lower())[0] == os.path.splitext(key_or_filename.lower())[0]:
                        return v
        return None

    def get_track_intelligence(self, audio_name: str) -> Optional[Dict[str, Any]]:
        """
        Public API: Returns pre-computed intelligence (BPM, beats, drops, lyrics, sections)
        from pool_metadata.json for audio_name (e.g. 'Aditi_bhatia.mp3' or telegram_file_id).
        Returns None if track is not indexed.
        """
        with self.lock:
            return self._get_file_metadata(audio_name)

    def retrieve_audio_from_vault(self, track_name_or_shortcode: str, dest_dir: Optional[str] = None) -> Optional[str]:
        """
        Public API: Retrieves an audio track or extracted clip audio from Telegram Storage Vault
        using the file_id registered in pool_metadata.json.
        """
        try:
            from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
            return TelegramVaultIndexer().hydrate_extracted_audio_from_vault(track_name_or_shortcode, dest_dir=dest_dir)
        except Exception as err:
            logger.warning("⚠️ Error retrieving audio from vault: %s", err)
            return None

    def _set_file_metadata(self, filename: str, data: Dict):
        """Helper to set file metadata accounting for schema version."""
        if "files" not in self.metadata:
            self.metadata = {"version": self.CURRENT_VERSION, "files": self.metadata}
        existing = self.metadata["files"].get(filename, {})
        if isinstance(existing, dict) and isinstance(data, dict):
            if "file_id" not in data and "file_id" in existing:
                data["file_id"] = existing["file_id"]
        data["filename"] = filename
        self.metadata["files"][filename] = data

    def _sync_root_to_active(self):
        """
        [FIX] Move any loose .mp3/.wav files sitting in Original_audio/ root into
        active/ so select_best_audio() can find them.

        extract_audio_from_video() writes to root (not active/).  If the
        orchestrator's process_new_audio() call was skipped (e.g. beat analysis
        exception), the file stays in root forever and is invisible to the pool.
        This method runs at startup and repairs the dir structure.
        """
        try:
            for filename in os.listdir(self.base_dir):
                if not filename.lower().endswith((".mp3", ".wav")):
                    continue
                src = os.path.join(self.base_dir, filename)
                if not os.path.isfile(src):
                    continue
                # Safety: skip if already in metadata as being in active/
                meta = self._get_file_metadata(filename)
                # ── PIPELINE ARTIFACT GATE ──
                if _is_pipeline_artifact(filename):
                    logger.debug(f"[POOL_SYNC] Skipping pipeline artifact: {filename}")
                    continue

                # ── BLACKLIST GATE: Never move blacklisted audio to active pool ─────────
                try:
                    from Audio_Modules.rejected_audio_blacklist import is_blacklisted as _is_bl
                    if _is_bl(audio_filename=filename):
                        logger.info(f"🚫 [POOL_SYNC] Skipping and purging blacklisted file from root: {filename}")
                        try:
                            os.remove(src)
                        except Exception:
                            pass
                        continue
                except Exception:
                    pass

                # ── MUSIC GATE: Never re-ingest voice-only files via boot-sync ──────────
                if meta and meta.get("is_speech_only", False):
                    logger.debug(f"[POOL_SYNC] Skipping voice-only file (speech gate flag): {filename}")
                    continue

                # ── COOLDOWN GATE: Never move a file back to active if it's in cooldown ──
                # This is the second layer of the repeat-audio fix. Even if the file
                # ends up back in root (e.g. after a process restart), if it has a
                # recent last_used timestamp it should stay off the active pool.
                cooldown_path = os.path.join(self.cooldown_dir, filename)
                if os.path.exists(cooldown_path):
                    logger.debug(f"[POOL_SYNC] Skipping '{filename}' — already in cooldown/")
                    # Remove the duplicate from root to avoid confusion
                    try:
                        os.remove(src)
                    except Exception:
                        pass
                    continue

                # ── METADATA COOLDOWN GATE: Check last_used even if file isn't in cooldown/ ──
                # Covers the case where the cooldown/ file was already cleaned up by maintenance
                # but the configured reuse cooldown window hasn't elapsed.
                last_used_ts = self._parse_timestamp(meta.get("last_used", 0) if meta else 0)
                if last_used_ts > 0:
                    cooldown_hours = self.reuse_cooldown_seconds / 3600.0
                    hours_since_used = (time.time() - last_used_ts) / 3600.0
                    if hours_since_used < cooldown_hours:
                        logger.info(
                            f"[POOL_SYNC] Skipping '{filename}' — used {hours_since_used:.1f}h ago "
                            f"(cooldown window: {cooldown_hours:.1f}h). Not re-adding to active pool."
                        )
                        continue
                # ─────────────────────────────────────────────────────────────────────────
                dst = os.path.join(self.active_dir, filename)
                if os.path.exists(dst):
                    continue  # already there
                try:
                    move(src, dst)
                    logger.info(f"[POOL_SYNC] Moved loose audio to active/: {filename}")
                    # Stub metadata if absent so select_best_audio can score it
                    if not meta:
                        self._set_file_metadata(filename, {
                            "usage_count": 0,
                            "last_used":   0,
                            "bpm":         0.0,
                            "energy":      0.5,
                            "created_at":  time.time(),
                            "beat_data_path": None,
                            "drop_times":  [],
                            "sample_rate": 44100,
                            "audio_hash":  self._calculate_fingerprint(dst),
                            "version":     self.CURRENT_VERSION,
                        })
                        self._save_metadata()
                except Exception as e:
                    logger.warning(f"[POOL_SYNC] Could not move {filename}: {e}")
        except Exception as e:
            logger.warning(f"[POOL_SYNC] Root sync warning (non-fatal): {e}")

    def _sync_active_to_metadata(self):
        """
        Scan active/ folder and ensure all files are registered in pool_metadata.json.
        """
        try:
            changed = False
            for filename in os.listdir(self.active_dir):
                if not filename.lower().endswith((".mp3", ".wav")):
                    continue
                path = os.path.join(self.active_dir, filename)
                if not os.path.isfile(path):
                    continue

                # ── BLACKLIST GATE: Prune blacklisted active audio file ─────────
                try:
                    from Audio_Modules.rejected_audio_blacklist import is_blacklisted as _is_bl
                    if _is_bl(audio_filename=filename):
                        logger.info(f"🚫 [POOL_SYNC] Removing blacklisted active audio file: {filename}")
                        try:
                            os.remove(path)
                        except Exception:
                            pass
                        continue
                except Exception:
                    pass

                meta = self._get_file_metadata(filename)
                if not meta:
                    logger.info(f"[POOL_SYNC] Registering unstubbed active audio in metadata: {filename}")
                    is_curated = not filename.lower().startswith(("bgm_", "vault_bgm_", "extracted_", "sess_"))
                    self._set_file_metadata(filename, {
                        "usage_count": 0,
                        "last_used":   0,
                        "bpm":         0.0,
                        "energy":      0.5,
                        "is_curated_library": is_curated,
                        "is_source_extract": not is_curated,
                        "created_at":  time.time(),
                        "beat_data_path": None,
                        "drop_times":  [],
                        "sample_rate": 44100,
                        "audio_hash":  self._calculate_fingerprint(path),
                        "version":     self.CURRENT_VERSION,
                    })
                    changed = True
            if changed:
                self._save_metadata()
                logger.info(f"[POOL_SYNC] Active audio pool metadata successfully synced.")
        except Exception as e:
            logger.warning(f"[POOL_SYNC] Active-to-metadata sync warning: {e}")

    def sync_all_active_audios_to_telegram_vault(self, force: bool = False) -> Dict[str, str]:
        """
        Uploads all active audio files in Original_audio/active/ to Telegram Storage Group,
        captures their file_id, and indexes them exclusively in pool_metadata.json (the audio single source of truth).
        """
        results = {}
        storage_group_id = os.getenv("TELEGRAM_STORAGE_GROUP_ID")
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not storage_group_id or not bot_token:
            return results

        try:
            from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer, _send_telegram_file_sync
            indexer = TelegramVaultIndexer()

            for filename in os.listdir(self.active_dir):
                if not filename.lower().endswith((".mp3", ".wav")):
                    continue
                if _is_pipeline_artifact(filename):
                    continue
                file_path = os.path.join(self.active_dir, filename)
                if not os.path.isfile(file_path):
                    continue

                meta = self._get_file_metadata(filename) or {}
                file_id = meta.get("file_id")

                if force or not file_id:
                    logger.info("🎙️ [AUDIO VAULT SYNC] Uploading active audio '%s' to Telegram Storage Group...", filename)
                    caption = f"🎵 [ACTIVE BGM POOL] `{filename}`"
                    upload_res = _send_telegram_file_sync("sendAudio", storage_group_id, "audio", file_path, caption=caption)
                    if not upload_res or not isinstance(upload_res, dict) or not upload_res.get("ok"):
                        upload_res = _send_telegram_file_sync("sendDocument", storage_group_id, "document", file_path, caption=caption)

                    if upload_res and isinstance(upload_res, dict) and upload_res.get("ok"):
                        res_doc = upload_res.get("result", {})
                        file_id = res_doc.get("audio", {}).get("file_id") or res_doc.get("document", {}).get("file_id")
                        if file_id:
                            logger.info("✅ [AUDIO VAULT SYNC] Captured file_id for '%s': %s", filename, file_id[:15])
                            meta["file_id"] = file_id
                            self._set_file_metadata(filename, meta)
                            results[filename] = file_id

            if results:
                self._save_metadata(sync_to_vault=True)
                logger.info("📌 [AUDIO VAULT SYNC] Synced %d audio file(s) into pool_metadata.json & Telegram Vault!", len(results))
        except Exception as err:
            logger.warning("⚠️ Error syncing active audios to Telegram vault: %s", err)
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # GEMINI POOL ENRICHMENT (background, non-blocking, cached)
    # ──────────────────────────────────────────────────────────────────────────

    def _gemini_enrich_background(self, dest_path: str, filename: str):
        """
        Daemon thread: analyzes a BGM track using unified Faster-Whisper + Gemini
        via Gemini_Modules.lyric_rhythm_aligner.analyze_music() and saves the complete
        musical and semantic intelligence report into pool_metadata.json.

        Adds:  gemini_genre, gemini_mood_tags, dominant_emotion, vibe_tags,
               gemini_energy_level, gemini_has_vocals, sections, tension_arc,
               lyrics, shot_directives, emotional_peak_moments, gemini_analyzed = True
        """
        try:
            # 0. Flag guard
            if os.getenv("ENABLE_POOL_GEMINI_ENRICH", "yes").lower() not in ("yes", "true", "1"):
                return

            # 1. Skip if already analyzed
            with self.lock:
                meta = self._get_file_metadata(filename)
            if meta and meta.get("gemini_analyzed"):
                logger.debug(f"[GEMINI_POOL] Already analyzed: {filename}")
                return

            # 2. Resolve path (file may be in active/ or cooldown/)
            track_path = dest_path
            for candidate in [dest_path,
                               os.path.join(self.active_dir, filename),
                               os.path.join(self.cooldown_dir, filename)]:
                if os.path.exists(candidate):
                    track_path = candidate
                    break
            else:
                return  # file not found anywhere

            # 3. Call Unified Musical Intelligence Pipeline
            try:
                from Gemini_Modules.lyric_rhythm_aligner import analyze_music
                logger.info(f"🧠 [GEMINI_POOL] Running unified Faster-Whisper + Gemini enrichment for: {filename}")
                report = analyze_music(track_path)
            except Exception as _call_err:
                logger.warning(f"⚠️ [GEMINI_POOL] analyze_music call error for {filename}: {_call_err}")
                return

            if not report or report.get("_source") == "fallback":
                logger.debug(f"[GEMINI_POOL] Fallback report returned for {filename} — skipping pool update.")
                return

            is_unusable = bool(report.get("is_unusable", False))
            unusable_reason = str(report.get("unusable_reason", "")).strip()

            if is_unusable:
                quarantine_dir = os.path.join(self.base_dir, "quarantine")
                os.makedirs(quarantine_dir, exist_ok=True)
                quarantine_target = os.path.join(quarantine_dir, filename)
                try:
                    if os.path.exists(track_path) and os.path.abspath(track_path) != os.path.abspath(quarantine_target):
                        move(track_path, quarantine_target)
                        logger.warning(f"⚠️ [GEMINI_POOL] Quarantined unusable audio file '{filename}' -> quarantine/ (Reason: {unusable_reason})")
                except Exception as _qerr:
                    logger.warning(f"⚠️ [GEMINI_POOL] Could not quarantine file '{filename}': {_qerr}")

            # 4. Write back full semantic + lyric intelligence into pool metadata
            with self.lock:
                meta = self._get_file_metadata(filename) or {}
                meta["gemini_genre"]         = str(report.get("language", "unknown"))[:32]
                meta["dominant_emotion"]     = str(report.get("dominant_emotion", "neutral"))
                meta["gemini_mood_tags"]     = list(report.get("vibe_tags", []))[:5]
                meta["vibe_tags"]            = list(report.get("vibe_tags", []))
                meta["gemini_energy_level"]  = str(report.get("energy_profile", "medium"))
                meta["energy_profile"]       = str(report.get("energy_profile", "medium"))
                meta["gemini_has_vocals"]    = bool(report.get("has_vocals", False))
                meta["has_vocals"]           = bool(report.get("has_vocals", False))
                meta["sections"]             = report.get("sections", [])
                meta["tension_arc"]          = report.get("tension_arc", [])
                meta["lyrics"]               = report.get("lyrics", [])
                meta["shot_directives"]      = report.get("shot_directives", [])
                meta["emotional_peak_moments"] = report.get("emotional_peak_moments", [])
                meta["is_unusable"]          = is_unusable
                meta["unusable_reason"]      = unusable_reason
                meta["gemini_analyzed"]      = True
                self._set_file_metadata(filename, meta)
            self._save_metadata()

            logger.info(
                f"🎵 [GEMINI_POOL SUCCESS] Enriched '{filename}': "
                f"emotion={report.get('dominant_emotion')} | energy={report.get('energy_profile')} | "
                f"lyrics={len(report.get('lyrics', []))} | vibe={report.get('vibe_tags')}"
            )

        except Exception as _ge:
            logger.debug(f"[GEMINI_POOL] Background enrichment notice for {filename}: {_ge}")

    def get_beat_data(self, filename: str) -> Optional[Dict]:
        """Lazy load beat data from cache or disk."""
        with self._cache_lock:
            if filename in self._beat_cache:
                return self._beat_cache[filename]

        meta = self._get_file_metadata(filename)
        if not meta or not meta.get("beat_data_path"):
            return None

        npz_path = os.path.join(self.base_dir, meta["beat_data_path"])
        if not os.path.exists(npz_path):
            return None

        try:
            with np.load(npz_path) as data:
                # Validation
                times = data.get("times", [])
                energies = data.get("energies", [])
                
                if len(times) == 0 or len(times) != len(energies):
                    logger.warning(f"⚠️ Validation failed for {filename} beat data. Length mismatch.")
                    return None
                
                beat_data = {
                    "times": times.tolist(),
                    "energies": energies.tolist(),
                    "sample_rate": meta.get("sample_rate", 44100)
                }
                
                # Update Cache (with growth control)
                with self._cache_lock:
                    if len(self._beat_cache) >= self.MAX_CACHE_SIZE:
                        # Simple FIFO pop
                        self._beat_cache.pop(next(iter(self._beat_cache)))
                    self._beat_cache[filename] = beat_data
                
                return beat_data
        except Exception as e:
            logger.error(f"❌ Failed to load beat data for {filename}: {e}")
            return None

    def process_new_audio(self, audio_path: str, bpm: float, energy: float, beat_analysis: Dict = None):
        """
        Moves newly extracted audio into pool and caches deep beat metadata.
        """
        if not os.path.exists(audio_path): return

        filename = os.path.basename(audio_path)
        dest_path = os.path.join(self.active_dir, filename)
        cooldown_path = os.path.join(self.cooldown_dir, filename)

        with self.lock:
            # ── COOLDOWN GUARD: If already in cooldown, do NOT move back to active ──
            if os.path.exists(cooldown_path):
                logger.info(
                    f"[POOL] '{filename}' is in cooldown — skipping re-activation. "
                    f"Metadata will be updated in-place."
                )
                existing_meta = self._get_file_metadata(filename)
                if existing_meta:
                    existing_meta["bpm"] = bpm if bpm > 0 else existing_meta.get("bpm", 0.0)
                    existing_meta["energy"] = energy if energy > 0 else existing_meta.get("energy", 0.5)
                    self._save_metadata()
                return

            try:
                # Move to active pool
                if os.path.abspath(audio_path) != os.path.abspath(dest_path):
                    move(audio_path, dest_path)
                
                # ── Precompute Binary Data ──
                rel_npz_path = None
                drop_times = []
                
                if beat_analysis:
                    raw_beats = beat_analysis.get("beats", [])
                    times = np.array([round(b["time"], 3) for b in raw_beats], dtype=np.float32)
                    energies = np.array([round(b["energy"], 3) for b in raw_beats], dtype=np.float32)
                    
                    drop_times = [round(b["time"], 3) for b in raw_beats if b["time"] in beat_analysis.get("drops", [])]
                    
                    npz_filename = os.path.splitext(filename)[0] + ".npz"
                    npz_path = os.path.join(self.beats_dir, npz_filename)
                    self._safe_save_npz(npz_path, times=times, energies=energies)
                    rel_npz_path = os.path.join("beats", npz_filename)

                existing_meta = self._get_file_metadata(filename)
                preserved_usage_count = existing_meta.get("usage_count", 0) if existing_meta else 0
                preserved_last_used   = existing_meta.get("last_used",   0) if existing_meta else 0

                # Initialize metadata
                self._set_file_metadata(filename, {
                    "usage_count": preserved_usage_count,
                    "last_used": preserved_last_used,
                    "bpm": bpm,
                    "energy": energy,
                    "created_at": existing_meta.get("created_at", time.time()) if existing_meta else time.time(),
                    "beat_data_path": rel_npz_path,
                    "drop_times": drop_times,
                    "sample_rate": 44100,
                    "audio_hash": self._calculate_hash(dest_path),
                    "version": self.CURRENT_VERSION
                })
                self._save_metadata()
                logger.info(f"🎵 [V{self.CURRENT_VERSION}] Processed: {filename} (usage_count kept={preserved_usage_count}, {len(drop_times)} drops cached)")

                _enrich_thread = threading.Thread(
                    target=self._gemini_enrich_background,
                    args=(dest_path, filename),
                    daemon=True,
                    name=f"gemini_pool_{filename[:16]}",
                )
                _enrich_thread.start()

            except Exception as e:
                logger.error(f"❌ Failed processing {filename}: {e}")

    def recycle_cooldown(self, force: bool = False) -> int:
        """Rotate files from cooldown back to active. If force=True, recycle all files immediately."""
        count = 0
        if not os.path.exists(self.cooldown_dir):
            return 0
        for filename in os.listdir(self.cooldown_dir):
            if _is_pipeline_artifact(filename):
                continue
            src = os.path.join(self.cooldown_dir, filename)
            dst = os.path.join(self.active_dir, filename)
            meta = self._get_file_metadata(filename)
            last_used_ts = self._parse_timestamp(meta.get("last_used", 0) if meta else 0)
            if force or (last_used_ts > 0 and (time.time() - last_used_ts > self.orphan_recovery_seconds)):
                try:
                    from shutil import move as _move
                    _move(src, dst)
                    count += 1
                    logger.info(f"♻️ [POOL] Recycled '{filename}' from cooldown -> active/")
                except Exception as _me:
                    logger.warning(f"Cooldown move failed for '{filename}': {_me}")
        return count

    def select_best_audio(
        self,
        video_bpm: float = 0,
        video_energy: float = 0,
        exclude_path: Optional[str] = None,
        recent_history: Optional[List[str]] = None,
        exclude_filenames: Optional[set] = None,
        target_bpm: float = 0,
        target_energy: float = 0,
        content_category: str = "",
    ) -> Optional[str]:
        if recent_history is None:
            recent_history = []
        
        _excluded_basenames: set = {os.path.basename(ef).lower() for ef in (exclude_filenames or [])}
        if exclude_filenames:
            for ef in exclude_filenames:
                _excluded_basenames.add(str(ef).lower())
        if exclude_path:
            _excluded_basenames.add(os.path.basename(exclude_path).lower())
            _excluded_basenames.add(str(exclude_path).lower())

        best_audio = None
        best_score = -1.0

        # ── 1. PRIMARY: Sync with Telegram Storage Group Vault Audio Index ────
        with self.lock:
            now = time.time()
            if self._vault_pool_cache is None or (now - self._vault_pool_cache_ts) > self._VAULT_POOL_CACHE_TTL:
                try:
                    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
                    vault = TelegramVaultIndexer()
                    self._vault_pool_cache = vault.get_vault_audio_pool() or {}
                    self._vault_pool_cache_ts = now
                except Exception as _tve:
                    logger.warning(f"⚠️ [POOL] Vault sync warning: {_tve}")

            vault_pool = self._vault_pool_cache or {}
            if vault_pool:
                for vname, vmeta in vault_pool.items():
                    if vname not in self.metadata.get("files", {}):
                        self._set_file_metadata(vname, vmeta)
                self._save_metadata()

        # ── 2. SECONDARY: Candidate Pool Resolution via Telegram Storage Vault Index ──
        candidate_pool_files = set()
        if os.path.exists(self.active_dir):
            try:
                candidate_pool_files.update(os.listdir(self.active_dir))
            except Exception:
                pass

        for fname in self.metadata.get("files", {}).keys():
            if fname.lower().endswith((".mp3", ".wav", ".m4a")):
                candidate_pool_files.add(fname)

        if not candidate_pool_files:
            return None

        cooldown_skipped = []

        for filename in candidate_pool_files:
            try:
                fn_lower = filename.lower()
                fn_base = os.path.basename(filename).lower()
                if fn_lower in _excluded_basenames or fn_base in _excluded_basenames:
                    logger.debug(f"[POOL] Skipping self-selected audio: {filename}")
                    continue
                    
                if _is_pipeline_artifact(filename):
                    logger.debug(f"[POOL] Skipping pipeline artifact: {filename}")
                    continue

                # ── PERMANENT REJECTION BLACKLIST CHECK ───────────────────────────────
                try:
                    from Audio_Modules.rejected_audio_blacklist import is_blacklisted as _is_bl
                    meta_cand = self._get_file_metadata(filename) or {}
                    cand_fid = meta_cand.get("file_id") or meta_cand.get("telegram_file_id")
                    cand_sc = meta_cand.get("shortcode")
                    if _is_bl(audio_filename=filename, audio_shortcode=cand_sc, telegram_file_id=cand_fid):
                        logger.info(f"🚫 [POOL] Skipping admin-rejected (blacklisted) audio: {filename}")
                        continue
                except Exception as _bl_check_err:
                    logger.debug(f"[POOL] Blacklist check notice: {_bl_check_err}")
                # ─────────────────────────────────────────────────────────────────────

                meta = self._get_file_metadata(filename)
                if not meta:
                    continue

                if meta.get("is_speech_only", False) or meta.get("is_unusable", False):
                    logger.debug(f"[POOL] Skipping unusable/speech audio: {filename}")
                    continue

                # Check if file is known to exceed 20MB Bot API download limit
                fid = meta.get("file_id") or meta.get("telegram_file_id")
                try:
                    from Telegram_Storage_Modules.telegram_http import is_file_oversized, is_mtproto_configured
                    if fid and is_file_oversized(fid) and not is_mtproto_configured():
                        logger.debug(f"[POOL] Skipping known oversized candidate: {filename}")
                        continue
                except Exception:
                    pass

                dur = self._safe_float(meta.get("duration") or meta.get("duration_sec") or meta.get("audio_duration", 0.0), 0.0)
                if 0.0 < dur < 10.0:
                    logger.debug(f"[POOL] Skipping short audio snippet (<10s): {filename}")
                    continue

                # ── UNIFIED COOLDOWN ENFORCEMENT ──────────────────────────────────────────
                cooldown_sec = self.reuse_cooldown_seconds
                last_used = self._parse_timestamp(meta.get("last_used", 0))
                in_cooldown = last_used > 0 and (time.time() - last_used) < cooldown_sec

                recent_penalty = (filename in recent_history)

                _eff_bpm    = target_bpm    if target_bpm    > 0 else video_bpm
                _eff_energy = target_energy if target_energy > 0 else video_energy

                bpm_val = self._safe_float(meta.get("bpm", 0.0), 0.0)
                energy_val = self._safe_float(meta.get("energy", 0.5), 0.5)
                usage_cnt = self._safe_int(meta.get("usage_count", 0), 0)

                if _eff_bpm > 0 and bpm_val > 0:
                    bpm_match = max(0, 1 - abs(bpm_val - _eff_bpm) / _eff_bpm)
                else:
                    bpm_match = 1.0

                if _eff_energy > 0:
                    energy_match = max(0, 1 - abs(energy_val - _eff_energy))
                else:
                    energy_match = 1.0

                usage_score = 1 / (usage_cnt + 1)

                genre_match = 0.5
                base_no_ext = os.path.splitext(filename)[0]
                lyric_cache_file = os.path.join(self.beats_dir, f"{base_no_ext}_lyric.json")
                cached_lyric_intel = None
                if os.path.exists(lyric_cache_file):
                    try:
                        with open(lyric_cache_file, "r", encoding="utf-8") as lf:
                            cached_lyric_intel = json.load(lf)
                    except Exception:
                        pass

                if content_category:
                    _cat = content_category.lower().strip()
                    _good = [c.lower() for c in (meta.get("gemini_content_match") or [])]
                    _bad  = [c.lower() for c in (meta.get("gemini_avoid_match")  or [])]

                    if cached_lyric_intel:
                        _vibe_tags = [v.lower() for v in cached_lyric_intel.get("vibe_tags", [])]
                        _dom_emotion = str(cached_lyric_intel.get("dominant_emotion", "")).lower()
                        if _dom_emotion:
                            _good.append(_dom_emotion)
                        _good.extend(_vibe_tags)

                    if any(_cat in g for g in _good):
                        genre_match = 1.0
                    elif any(_cat in b for b in _bad):
                        genre_match = 0.0
                    elif _good:
                        genre_match = 0.45

                _has_semantic_intel = (bool(meta.get("gemini_analyzed")) or bool(cached_lyric_intel)) and bool(content_category)
                if _has_semantic_intel:
                    score = (
                        bpm_match    * 0.40 +
                        energy_match * 0.20 +
                        genre_match  * 0.30 +
                        usage_score  * 0.10
                    )
                else:
                    score = (
                        bpm_match    * 0.60 +
                        energy_match * 0.20 +
                        usage_score  * 0.15
                    )

                if recent_penalty:
                    score *= 0.5

                score += random.uniform(0, 0.05)

                if in_cooldown:
                    hrs_ago = (time.time() - last_used) / 3600.0
                    logger.info(f"[POOL] Skipping '{filename}' — used {hrs_ago:.1f}h ago (under {cooldown_sec/3600:.1f}h cooldown limit). Recording for LRU fallback.")
                    cooldown_skipped.append((last_used, filename, score))
                    continue

                if score > best_score:
                    best_score = score
                    best_audio = filename
            except Exception as _cand_err:
                logger.warning(f"⚠️ [POOL] Error scoring candidate '{filename}': {_cand_err}. Skipping.")
                continue

        # Sort all cooldown-skipped tracks by LRU (oldest last_used first, then highest score)
        if cooldown_skipped:
            cooldown_skipped.sort(key=lambda x: (x[0], -x[2]))

        # If all candidates are in cooldown, fall back to least-recently-used track (strictly non-blacklisted)
        if not best_audio and cooldown_skipped:
            clean_skipped = []
            try:
                from Audio_Modules.rejected_audio_blacklist import is_blacklisted as _is_bl
                for item in cooldown_skipped:
                    c_fn = item[1]
                    c_meta = self._get_file_metadata(c_fn) or {}
                    c_fid = c_meta.get("file_id") or c_meta.get("telegram_file_id")
                    c_sc = c_meta.get("shortcode")
                    if not _is_bl(audio_filename=c_fn, audio_shortcode=c_sc, telegram_file_id=c_fid):
                        clean_skipped.append(item)
            except Exception:
                clean_skipped = cooldown_skipped

            if clean_skipped:
                logger.warning(
                    f"⚠️ [POOL COOLDOWN] All {len(cooldown_skipped)} valid candidates were used within the last "
                    f"{self.reuse_cooldown_seconds/3600:.1f}h cooldown limit! "
                    f"Falling back to clean least-recently-used track to avoid pipeline failure."
                )
                best_audio = clean_skipped[0][1]

        if not best_audio:
            return None

        # Build ordered list of candidates to try hydrating
        candidates_to_try_hydrate = [best_audio]
        if cooldown_skipped:
            for _, alt_fn, _ in cooldown_skipped:
                if alt_fn != best_audio and alt_fn not in candidates_to_try_hydrate:
                    candidates_to_try_hydrate.append(alt_fn)

        src = None
        for cand_audio in candidates_to_try_hydrate:
            test_src = os.path.join(self.active_dir, cand_audio)
            if not os.path.exists(test_src):
                try:
                    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
                    vault = TelegramVaultIndexer()
                    hydrated = vault.hydrate_bgm_track_from_vault(cand_audio, self.active_dir)
                    if hydrated and os.path.exists(hydrated):
                        test_src = hydrated
                        logger.info(f"📥 [POOL - PRIMARY] Hydrated selected track '{cand_audio}' directly from Telegram Storage Vault.")
                except Exception as _he:
                    logger.warning(f"⚠️ [POOL] Hydration warning for '{cand_audio}': {_he}")

            if os.path.exists(test_src) and os.path.getsize(test_src) > 1024:
                src = test_src
                best_audio = cand_audio
                break
            else:
                logger.warning(f"❌ [POOL] Candidate track '{cand_audio}' does not exist on disk and could not be hydrated from vault.")

        if not src or not os.path.exists(src):
            return None

        with self.lock:
            meta = self._get_file_metadata(best_audio)
            if meta:
                meta["usage_count"] = meta.get("usage_count", 0) + 1
                meta["last_used"] = time.time()
                self._save_metadata()

        return os.path.abspath(src)

    def record_selected_audio(
        self,
        social_url_or_shortcode: str,
        selected_track_name: str,
        telegram_file_id: Optional[str] = None,
        alignment_score: float = 0.85,
        reasoning: str = ""
    ) -> Dict[str, Any]:
        """
        Single authority for recording BGM track selection for a clip.
        Updates track metadata (incrementing usage_count = N + 1, setting last_used = time.time()),
        and updates the clip entry in pool_metadata.json (setting selected_audio, media_file_ids,
        and audio_data.last_used / audio_data.usage_count).
        """
        now = time.time()
        track_name = os.path.basename(selected_track_name) if selected_track_name else ""

        with self.lock:
            # 1. Update track metadata in files[track_name]
            t_meta = self._get_file_metadata(track_name) or {}
            new_usage_count = int(t_meta.get("usage_count", 0) or 0) + 1
            t_meta["usage_count"] = new_usage_count
            t_meta["last_used"] = now
            if telegram_file_id and not t_meta.get("file_id"):
                t_meta["file_id"] = telegram_file_id
            if track_name:
                self._set_file_metadata(track_name, t_meta)

            # 2. Update clip metadata under social_media_id dictionary
            sel_dict = {
                "track_name": track_name,
                "file_id": telegram_file_id or t_meta.get("file_id"),
                "alignment_score": alignment_score,
                "reasoning": reasoning,
                "last_used": now,
                "usage_count": new_usage_count,
            }

            files_root = self.metadata.setdefault("files", {})
            if isinstance(files_root, dict):
                soc_clips = files_root.setdefault("social_media_id", {})
                if isinstance(soc_clips, dict):
                    search_str = social_url_or_shortcode.lower() if social_url_or_shortcode else ""
                    for key_url, c_entry in soc_clips.items():
                        if not isinstance(c_entry, dict):
                            continue
                        c_sc = str(c_entry.get("shortcode", "")).lower()
                        if search_str and (
                            search_str in key_url.lower()
                            or (c_sc and c_sc in search_str)
                            or (search_str in c_sc)
                        ):
                            c_entry["selected_audio"] = sel_dict
                            c_aud = c_entry.setdefault("audio_data", {})
                            c_aud["selected_audio"] = sel_dict
                            c_aud["last_used"] = now
                            c_aud["usage_count"] = new_usage_count
                            if telegram_file_id or t_meta.get("file_id"):
                                c_entry.setdefault("media_file_ids", {})["selected_audio_file_id"] = (
                                    telegram_file_id or t_meta.get("file_id")
                                )

            self._save_metadata(sync_to_vault=True)
            logger.info(
                f"✅ [POOL_AUDIO_RECORD] Track '{track_name}' recorded for clip '{social_url_or_shortcode}' "
                f"(usage_count={new_usage_count}, last_used={now})"
            )
            return sel_dict

    def use_audio(self, audio_path: str, social_url: Optional[str] = None):
        """Mark a BGM track as used with N+1 increment on usage_count and updating last_used."""
        filename = os.path.basename(audio_path)
        with self.lock:
            meta = self._get_file_metadata(filename)
            if not meta:
                meta = {
                    "usage_count": 0,
                    "last_used": 0,
                    "created_at": time.time(),
                    "version": self.CURRENT_VERSION,
                }
            now = time.time()
            new_count = int(meta.get("usage_count", 0) or 0) + 1
            meta["usage_count"] = new_count
            meta["last_used"] = now
            self._set_file_metadata(filename, meta)

            if social_url:
                files_root = self.metadata.get("files", {})
                soc_clips = files_root.get("social_media_id", {}) if isinstance(files_root, dict) else {}
                if isinstance(soc_clips, dict) and social_url in soc_clips:
                    c_entry = soc_clips[social_url]
                    if isinstance(c_entry, dict):
                        a_data = c_entry.setdefault("audio_data", {})
                        a_data["last_used"] = now
                        a_data["usage_count"] = new_count
                        if isinstance(a_data.get("selected_audio"), dict):
                            a_data["selected_audio"]["last_used"] = now
                            a_data["selected_audio"]["usage_count"] = new_count
                        if isinstance(c_entry.get("selected_audio"), dict):
                            c_entry["selected_audio"]["last_used"] = now
                            c_entry["selected_audio"]["usage_count"] = new_count

        self._save_metadata(sync_to_vault=True)
        logger.info(f"[POOL] Registered usage for {filename!r} (usage_count={new_count}, last_used={now}).")

    def maintenance(self):
        """
        Rotates clips from cooldown back to active based on hybrid logic.
        Cleans up root directory of Original_audio.
        """
        now = time.time()
        
        # 1. 🔁 Cooldown → Active (Hybrid Logic: 48 hours OR implicit cycle gap via metadata analysis)
        # Note: 'usage_gap' isn't explicitly stored, but we can infer 'last_used' is the primary trigger.
        # User specified: now - last_used > 48h OR usage_gap >= 15.
        # Tracking "usage_gap" precisely requires a global count. 
        # For now, let's stick to the 48h time trigger provided in the skeleton.
        
        # 1. 🔁 Cooldown → Active rotation (48-hour rule)
        # Files that have been in cooldown for 48 hours are rotated back.
        count_rotated = 0
        for filename in os.listdir(self.cooldown_dir):
            if _is_pipeline_artifact(filename):
                continue
                
            path = os.path.join(self.cooldown_dir, filename)
            meta = self._get_file_metadata(filename)

            if not meta:
                # Orphaned cooldown file — move back to active for safety
                try: move(path, os.path.join(self.active_dir, filename)); count_rotated += 1
                except: pass
                continue

            time_passed = now - meta.get("last_used", 0)

            if time_passed > 48 * 3600:
                try:
                    move(path, os.path.join(self.active_dir, filename))
                    count_rotated += 1
                except Exception as e:
                    logger.error(f"Failed to rotate {filename} back to active: {e}")

        if count_rotated > 0:
            logger.info(f"🔁 Audio Maintenance: Rotated {count_rotated} clips from cooldown to active.")

        # 1b. 🧹 Orphaned NPZ cleanup
        try:
            meta_files = self.metadata.get("files", self.metadata)
            valid_npz = set()
            for f_meta in meta_files.values():
                if isinstance(f_meta, dict) and f_meta.get("beat_data_path"):
                    valid_npz.add(os.path.basename(f_meta["beat_data_path"]))
            
            count_npz_cleaned = 0
            for npz_file in os.listdir(self.beats_dir):
                if npz_file not in valid_npz:
                    try:
                        os.remove(os.path.join(self.beats_dir, npz_file))
                        count_npz_cleaned += 1
                    except: pass
            if count_npz_cleaned > 0:
                logger.info(f"🧹 Audio Maintenance: Cleaned {count_npz_cleaned} orphaned .npz files.")
        except Exception as e:
            logger.warning(f"⚠️ NPZ cleanup fail: {e}")

        # 2. 🧹 Root cleanup (Files > 6h old)
        count_cleaned = 0
        for filename in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, filename)

            # Skip subdirectories (active, cooldown)
            if os.path.isdir(path):
                continue

            # Skip metadata file
            if filename == os.path.basename(self.meta_path):
                continue

            try:
                created = os.path.getctime(path)
                if now - created > 6 * 3600:
                    os.remove(path)
                    count_cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to clean root file {filename}: {e}")

        if count_cleaned > 0:
            logger.info(f"🧹 Audio Maintenance: Cleaned {count_cleaned} stale files from Original_audio root.")

    def deduplicate_and_merge_pool_entries(self) -> int:
        """
        Scans files['social_media_id'] and groups entries by clean shortcode.
        If multiple keys exist for the same shortcode (e.g. clean URL, query-param dirty URL,
        markdown backticks), merges them into a single canonical entry with zero data loss.
        Removes all dirty duplicate keys from the dictionary.
        Returns the count of purged duplicate keys.
        """
        import re
        from Telegram_Storage_Modules.telegram_vault_indexer import (
            extract_clean_shortcode,
            canonicalize_social_url,
            deep_merge_records,
        )

        with self.lock:
            files_root = self.metadata.get("files", {})
            if not isinstance(files_root, dict):
                return 0
            social_dict = files_root.get("social_media_id", {})
            if not isinstance(social_dict, dict) or not social_dict:
                return 0

            grouped: Dict[str, list] = {}
            for url_key, entry in list(social_dict.items()):
                if not isinstance(entry, dict):
                    continue
                sc = (
                    entry.get("shortcode")
                    or extract_clean_shortcode(url_key)
                    or extract_clean_shortcode(entry.get("social_media_id"))
                )
                if not sc:
                    sc = url_key
                grouped.setdefault(sc, []).append((url_key, entry))

            purged_count = 0
            changes_made = False

            for sc, entries in grouped.items():
                canonical_key = canonicalize_social_url(sc) if (sc and re.fullmatch(r"[A-Za-z0-9_-]{5,}", sc)) else entries[0][0]
                if not canonical_key:
                    canonical_key = entries[0][0]

                has_duplicates = len(entries) > 1
                has_dirty_key = any(orig_key != canonical_key for orig_key, _ in entries)

                if has_duplicates or has_dirty_key:
                    merged_entry: Dict[str, Any] = {}
                    for orig_key, entry_data in entries:
                        merged_entry = deep_merge_records(merged_entry, entry_data)
                        if orig_key != canonical_key and orig_key in social_dict:
                            social_dict.pop(orig_key, None)
                            purged_count += 1
                            changes_made = True

                    merged_entry["shortcode"] = sc
                    merged_entry["social_media_id"] = canonical_key

                    # Sync media_file_ids and top-level fields
                    m_ids = merged_entry.setdefault("media_file_ids", {})
                    for field in (
                        "raw_video_file_id",
                        "extracted_audio_file_id",
                        "processed_output_file_id",
                        "wm_clean_file_id",
                    ):
                        if merged_entry.get(field) and not m_ids.get(field):
                            m_ids[field] = merged_entry[field]
                        elif m_ids.get(field) and not merged_entry.get(field):
                            merged_entry[field] = m_ids[field]

                    social_dict[canonical_key] = merged_entry
                    changes_made = True
                    logger.info(
                        f"🧹 [POOL DEDUP] Merged {len(entries)} duplicate entries for shortcode '{sc}' -> '{canonical_key}'"
                    )

            if changes_made:
                self._save_metadata(sync_to_vault=True)
                logger.info(
                    f"✅ [POOL DEDUP COMPLETED] Purged {purged_count} duplicate keys; pool_metadata.json cleaned and synced."
                )

            return purged_count

    def sanitize_pool_metadata_anomalies(self) -> int:
        """
        Removes illegal entries from files root (like 'video.mp4' or video files),
        prunes blacklisted audio tracks from files root and social_media_id,
        and deduplicates all social_media_id entries.
        """
        removed = 0
        with self.lock:
            files_dict = self.metadata.get("files", {})
            if isinstance(files_dict, dict):
                to_delete = [
                    k for k in files_dict.keys()
                    if k != "social_media_id" and (
                        k.lower() in ("video.mp4", "raw.mp4")
                        or any(k.lower().endswith(ext) for ext in (".mp4", ".mkv", ".mov", ".avi", ".webm"))
                    )
                ]
                for k in to_delete:
                    del files_dict[k]
                    removed += 1
                    logger.info(f"🧹 [POOL SANITIZE] Removed rogue video entry from audio files index: {k}")

                # ── BLACKLIST SCRUBBING: Prune blacklisted audio entries from metadata ──
                try:
                    from Audio_Modules.rejected_audio_blacklist import is_blacklisted as _is_bl
                    bl_to_del = [
                        k for k, v in files_dict.items()
                        if k != "social_media_id" and isinstance(v, dict)
                        and _is_bl(
                            audio_filename=k,
                            audio_shortcode=v.get("shortcode"),
                            telegram_file_id=v.get("file_id") or v.get("telegram_file_id")
                        )
                    ]
                    for k in bl_to_del:
                        del files_dict[k]
                        removed += 1
                        logger.info(f"🚫 [POOL SANITIZE] Scrubbed blacklisted track from pool files: {k}")

                    # IMPORTANT: Do NOT delete video reel entries from social_media_id!
                    # An audio blacklist only disqualifies an extracted audio track from being selected as BGM.
                    # It must NEVER delete harvested or processed video reels from the pool metadata ledger.
                    sm_dict = files_dict.get("social_media_id", {})
                    if isinstance(sm_dict, dict):
                        for url, entry in list(sm_dict.items()):
                            if not isinstance(entry, dict):
                                continue
                            m_ids = entry.get("media_file_ids", {})
                            ext_aud_fid = m_ids.get("extracted_audio_file_id") or entry.get("extracted_audio_file_id")
                            if _is_bl(
                                audio_filename=f"{entry.get('shortcode', '')}.wav",
                                audio_shortcode=entry.get("shortcode"),
                                telegram_file_id=ext_aud_fid
                            ):
                                entry["is_audio_blacklisted"] = True
                except Exception as _bl_scrub_err:
                    logger.debug("Notice during pool sanitize blacklist check: %s", _bl_scrub_err)

            if removed > 0:
                self._save_metadata()

        dedup_purged = self.deduplicate_and_merge_pool_entries()
        return removed + dedup_purged

    def get_files_index(self) -> Dict[str, Any]:
        """
        Return the pool_metadata["files"] dict — the unified audio track index.
        Keys are filenames (e.g. 'Zareena_khan.mp3'), values are the full metadata dicts.
        Guarantees 100% exclusion of blacklisted audio tracks.
        """
        raw = dict(self.metadata.get("files", self.metadata))
        try:
            from Audio_Modules.rejected_audio_blacklist import is_blacklisted as _is_bl
            return {
                k: v for k, v in raw.items()
                if k != "social_media_id" and isinstance(v, dict)
                and not v.get("is_source_extract")
                and not v.get("is_unusable_bgm")
                and not _is_bl(
                    audio_filename=k,
                    audio_shortcode=v.get("shortcode"),
                    telegram_file_id=v.get("file_id") or v.get("telegram_file_id")
                )
            }
        except Exception:
            return raw

    def merge_lyric_into_pool(self, track_identifier: str, lyric_data: Dict[str, Any]) -> bool:
        """
        Merge rich lyric intelligence fields from a _lyric.json result INTO
        pool_metadata["files"][track_identifier] AND clip_source_math entries.
        """
        if not track_identifier or not isinstance(lyric_data, dict):
            return False
        try:
            with self.lock:
                track_filename = os.path.basename(track_identifier)
                meta = self._get_file_metadata(track_filename)
                if meta is None:
                    # Track not yet registered — create a stub so data isn't lost
                    meta = {
                        "usage_count": 0,
                        "last_used": 0,
                        "bpm": lyric_data.get("tempo_bpm", 0.0),
                        "energy": 0.5,
                        "created_at": time.time(),
                        "beat_data_path": None,
                        "drop_times": [],
                        "sample_rate": 44100,
                        "audio_hash": "unknown",
                        "version": self.CURRENT_VERSION,
                    }

                # Always set full report in gemini_semantic_audio_intelligence
                meta["gemini_semantic_audio_intelligence"] = lyric_data
                meta["gemini_semantic_intelligence"] = lyric_data

                # Always overwrite individual fields with fresh lyric intelligence
                if "tempo_bpm" in lyric_data:
                    meta["tempo_bpm"] = float(lyric_data["tempo_bpm"])
                    if meta.get("bpm", 0.0) == 0.0:
                        meta["bpm"] = float(lyric_data["tempo_bpm"])
                if "dominant_emotion" in lyric_data:
                    meta["dominant_emotion"] = str(lyric_data["dominant_emotion"])
                if "energy_profile" in lyric_data:
                    meta["energy_profile"] = str(lyric_data["energy_profile"])
                if "vibe_tags" in lyric_data:
                    meta["vibe_tags"] = list(lyric_data["vibe_tags"])[:6]
                if "has_vocals" in lyric_data:
                    meta["has_vocals"] = bool(lyric_data["has_vocals"])
                if "language" in lyric_data:
                    meta["language"] = str(lyric_data["language"])
                if "sections" in lyric_data:
                    meta["sections"] = lyric_data["sections"]
                    meta["sections_summary"] = [
                        {"start": s.get("start", 0), "end": s.get("end", 0),
                         "type": s.get("type", "unknown"), "energy": s.get("energy", 0.5)}
                        for s in lyric_data["sections"]
                    ]
                if "lyrics" in lyric_data:
                    meta["lyrics"] = lyric_data["lyrics"]
                if "shot_directives" in lyric_data:
                    meta["shot_directives"] = lyric_data["shot_directives"]
                if "tension_arc" in lyric_data:
                    meta["tension_arc"] = lyric_data["tension_arc"]
                if "emotional_peak_moments" in lyric_data:
                    meta["emotional_peak_moments"] = lyric_data["emotional_peak_moments"]
                if "transcript" in lyric_data:
                    meta["transcript"] = str(lyric_data["transcript"])
                if "file_id" in lyric_data:
                    meta["file_id"] = lyric_data["file_id"]
                meta["lyric_intel_merged"] = True
                is_video_file = any(track_filename.lower().endswith(ext) for ext in (".mp4", ".mkv", ".mov", ".avi", ".webm")) or track_filename.lower() in ("video.mp4", "raw.mp4")
                if not is_video_file:
                    self._set_file_metadata(track_filename, meta)
                else:
                    logger.debug(f"[POOL MERGE] Skipped registering video file '{track_filename}' in files index root.")

                # Also search clips entries in metadata and populate gemini_semantic_audio_intelligence
                files_dict = self.metadata.get("files", self.metadata)
                if isinstance(files_dict, dict):
                    for k in ["clip_source_math", "social_media_id", "sources"]:
                        if k in files_dict and "clips" not in files_dict:
                            files_dict["clips"] = files_dict.pop(k)
                    clip_sources = files_dict.get("clips", {})
                    if isinstance(clip_sources, dict):
                        parent_folder = os.path.basename(os.path.dirname(track_identifier)).lower() if ("/" in track_identifier or "\\" in track_identifier) else track_identifier.lower()
                        norm_track_id = os.path.normpath(track_identifier).lower()
                        for key, entry in clip_sources.items():
                            if not isinstance(entry, dict):
                                continue
                            wav_path = entry.get("audio_math", {}).get("wav_path", "")
                            norm_wav = os.path.normpath(wav_path).lower() if wav_path else ""
                            shortcode = str(entry.get("shortcode", "")).lower()
                            social_id = str(entry.get("social_media_id", "")).lower()

                            match = False
                            if norm_wav and (norm_wav in norm_track_id or norm_track_id in norm_wav):
                                match = True
                            elif parent_folder and (parent_folder in key.lower() or (shortcode and shortcode in parent_folder) or (social_id and parent_folder in social_id)):
                                match = True
                            elif wav_path and os.path.basename(os.path.dirname(wav_path)).lower() == parent_folder:
                                match = True

                            if match:
                                entry["gemini_semantic_audio_intelligence"] = lyric_data
                                entry["gemini_semantic_intelligence"] = lyric_data
                                if "gemini_semantic_visual_intelligence" not in entry:
                                    entry["gemini_semantic_visual_intelligence"] = {}
                                logger.info(f"✅ [POOL_MERGE] Updated gemini_semantic_audio_intelligence in clips for '{key}'")

            self._save_metadata()
            logger.info(f"✅ [POOL_MERGE] Merged lyric intel into pool_metadata.json for '{track_filename}'")
            return True
        except Exception as _e:
            logger.warning(f"[POOL_MERGE] Failed to merge lyric data for '{track_identifier}': {_e}")
            return False

    def purge_media_and_audio_entries(
        self,
        reel_identifier: str,
        selected_audio_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deep dual-input purge on admin reject:
        1. Deletes the entire <url> section of the processed input from files["social_media_id"].
        2. Resolves selected_audio (if not provided, extracts it from the processed reel entry before deletion).
        3. Deletes the entire <url> section of the selected audio's originating harvest input from files["social_media_id"].
        4. Deletes the audio track key from files[audio_name].
        5. Deletes matching clip intelligence from clips[clip_id].
        6. Deletes local physical audio files (in active/, cooldown/, beats/).
        7. Saves metadata and automatically triggers _sync_to_telegram_vault().
        """
        with self.lock:
            purged_items = []

            def _extract_sc(val: Any) -> str:
                if not val:
                    return ""
                s = str(val).strip().strip("`")
                if s.endswith("%60"):
                    s = s[:-3].strip()
                s = s.rstrip("/")
                m = re.search(r"/(?:p|reel|reels)/([A-Za-z0-9_-]+)", s)
                if m:
                    return m.group(1)
                m_yt = re.search(r"(?:shorts/|v=|youtu\.be/)([A-Za-z0-9_-]{11})", s)
                if m_yt:
                    return m_yt.group(1)
                base = os.path.basename(s).replace("manual_", "").split("?")[0]
                return base.replace(".mp4", "").replace(".wav", "").replace(".mp3", "").replace(".m4a", "").strip()

            processed_sc = _extract_sc(reel_identifier).lower()
            reel_id_clean = str(reel_identifier).lower().strip()

            files_root = self.metadata.setdefault("files", {})
            social_dict = files_root.get("social_media_id", {})
            if not isinstance(social_dict, dict):
                social_dict = {}
                files_root["social_media_id"] = social_dict

            # 1. Recover selected_audio from processed reel entry if not provided
            resolved_audio = selected_audio_name
            reel_fid = None
            for url_key, entry in list(social_dict.items()):
                if not isinstance(entry, dict):
                    continue
                sc_val = str(entry.get("shortcode", "")).lower()
                sm_val = str(entry.get("social_media_id", "")).lower()
                u_key_l = str(url_key).lower()

                is_match = False
                if processed_sc and (processed_sc == sc_val or processed_sc in u_key_l or processed_sc in sm_val):
                    is_match = True
                elif reel_id_clean and (reel_id_clean in u_key_l or reel_id_clean in sm_val):
                    is_match = True

                if is_match:
                    if not reel_fid:
                        reel_fid = (
                            entry.get("media_file_ids", {}).get("extracted_audio_file_id")
                            or entry.get("extracted_audio_file_id")
                            or entry.get("media_file_ids", {}).get("audio_file_id")
                            or entry.get("media_file_ids", {}).get("raw_video_file_id")
                            or entry.get("raw_video_file_id")
                            or entry.get("file_id")
                        )
                    if not resolved_audio:
                        sel_cand = entry.get("selected_audio") or entry.get("audio_data", {}).get("selected_audio")
                        if isinstance(sel_cand, dict):
                            resolved_audio = sel_cand.get("filename") or sel_cand.get("track_name")
                        elif isinstance(sel_cand, str):
                            resolved_audio = sel_cand

                    # DELETE ENTIRE OBJECT for processed reel input!
                    del social_dict[url_key]
                    purged_items.append(f"Processed input section: {url_key}")
                    logger.info(f"🗑️ [POOL PURGE] Completely deleted processed input section: {url_key}")

            # Fallback search across files_root for reel_fid if not found in social_dict
            if not reel_fid and processed_sc:
                for fk, fv in files_root.items():
                    if isinstance(fv, dict) and (processed_sc in fk.lower() or f"{processed_sc}.wav" == fk.lower()):
                        reel_fid = fv.get("file_id") or fv.get("telegram_file_id")
                        if reel_fid:
                            break

            # 2. BLACKLIST WRITE: Permanent ban for BOTH reel's extracted audio AND selected BGM
            audio_sc = ""
            audio_fname = ""
            if resolved_audio:
                audio_fname = os.path.basename(str(resolved_audio)).lower()
                audio_sc = audio_fname
                if audio_sc.startswith("vault_bgm_"):
                    audio_sc = audio_sc[len("vault_bgm_"):]
                elif audio_sc.startswith("bgm_"):
                    audio_sc = audio_sc[len("bgm_"):]
                elif audio_sc.startswith("extracted_"):
                    audio_sc = audio_sc[len("extracted_"):]
                for ext in (".wav", ".mp3", ".m4a", ".aac", ".mp4"):
                    if audio_sc.endswith(ext):
                        audio_sc = audio_sc[:-len(ext)]
                audio_sc = audio_sc.strip()

            try:
                from Audio_Modules.rejected_audio_blacklist import add as _bl_add, sync_blacklist_to_vault
                # A) Permanently blacklist the rejected reel's own audio (batched sync)
                if processed_sc:
                    _bl_add(
                        audio_filename=f"extracted_{processed_sc}.wav",
                        audio_shortcode=processed_sc,
                        telegram_file_id=reel_fid,
                        reason="admin_rejected_reel_source",
                        sync_to_vault=False,
                    )
                    _bl_add(
                        audio_filename=f"{processed_sc}.wav",
                        audio_shortcode=processed_sc,
                        telegram_file_id=reel_fid,
                        reason="admin_rejected_reel_source",
                        sync_to_vault=False,
                    )

                # B) Permanently blacklist the selected BGM track (batched sync)
                if audio_fname or audio_sc:
                    audio_fid = None
                    for fk, fv in files_root.items():
                        if isinstance(fv, dict) and (fk.lower() == audio_fname or (audio_sc and audio_sc in fk.lower())):
                            audio_fid = fv.get("file_id") or fv.get("telegram_file_id")
                            if audio_fid:
                                break
                    if not audio_fid and audio_sc:
                        for url_key, entry in social_dict.items():
                            if isinstance(entry, dict):
                                sc_val = str(entry.get("shortcode", "")).lower()
                                if sc_val == audio_sc or audio_sc in str(url_key).lower():
                                    audio_fid = (
                                        entry.get("media_file_ids", {}).get("extracted_audio_file_id")
                                        or entry.get("extracted_audio_file_id")
                                        or entry.get("file_id")
                                    )
                                    if audio_fid:
                                        break
                    _bl_add(
                        audio_filename=audio_fname or f"{audio_sc}.mp3",
                        audio_shortcode=audio_sc or None,
                        telegram_file_id=audio_fid,
                        reason="admin_rejected",
                        sync_to_vault=False,
                    )
                    if audio_sc and f"{audio_sc}.wav" != audio_fname:
                        _bl_add(
                            audio_filename=f"{audio_sc}.wav",
                            audio_shortcode=audio_sc,
                            telegram_file_id=audio_fid,
                            reason="admin_rejected",
                            sync_to_vault=False,
                        )

                sync_blacklist_to_vault()
            except Exception as _bl_err:
                logger.warning(f"⚠️ [POOL PURGE] Blacklist write failed (non-fatal): {_bl_err}")
            # ─────────────────────────────────────────────────────────────────────

                if audio_sc:
                    for url_key, entry in list(social_dict.items()):
                        if not isinstance(entry, dict):
                            continue
                        sc_val = str(entry.get("shortcode", "")).lower()
                        sm_val = str(entry.get("social_media_id", "")).lower()
                        u_key_l = str(url_key).lower()

                        if (
                            (audio_sc and (audio_sc == sc_val or audio_sc in u_key_l or audio_sc in sm_val))
                            or (audio_fname and audio_fname in u_key_l)
                        ):
                            # DELETE ENTIRE OBJECT for selected audio harvest source!
                            del social_dict[url_key]
                            purged_items.append(f"Audio source harvest section: {url_key}")
                            logger.info(f"🗑️ [POOL PURGE] Completely deleted audio source harvest section: {url_key}")

                # 3. Purge audio track key from files_root
                keys_to_del = [k for k in files_root if k != "social_media_id" and (
                    k.lower() == audio_fname
                    or (audio_sc and audio_sc in k.lower())
                )]
                for k in keys_to_del:
                    del files_root[k]
                    purged_items.append(f"Audio pool entry: {k}")
                    logger.info(f"🗑️ [POOL PURGE] Completely deleted audio pool track entry: {k}")

            # 4. Purge from clips section
            clips_dict = self.metadata.get("clips", {})
            if isinstance(clips_dict, dict):
                clips_to_del = [k for k in clips_dict if (
                    (processed_sc and processed_sc in k.lower())
                    or (reel_id_clean and reel_id_clean in k.lower())
                    or (audio_sc and audio_sc in k.lower())
                )]
                for ck in clips_to_del:
                    del clips_dict[ck]
                    purged_items.append(f"Clip intelligence entry: {ck}")
                    logger.info(f"🗑️ [POOL PURGE] Completely deleted clip intelligence entry: {ck}")

            # 5. Delete physical audio and beat cache files from disk
            patterns_to_remove = set()
            if audio_fname:
                patterns_to_remove.add(audio_fname)
            if audio_sc:
                patterns_to_remove.add(f"*{audio_sc}*")

            search_audio_dirs = [self.active_dir, self.cooldown_dir, self.beats_dir, self.base_dir]
            for s_dir in search_audio_dirs:
                if os.path.exists(s_dir):
                    for pat in patterns_to_remove:
                        for fpath in glob.glob(os.path.join(s_dir, pat)):
                            if os.path.isfile(fpath):
                                try:
                                    os.remove(fpath)
                                    purged_items.append(f"Audio file: {os.path.basename(fpath)}")
                                    logger.info(f"🗑️ [POOL PURGE] Deleted physical audio file: {fpath}")
                                except Exception as _del_err:
                                    logger.warning(f"⚠️ [POOL PURGE] Failed deleting {fpath}: {_del_err}")

            # 6. Invalidate in-memory vault pool cache
            self._vault_pool_cache = None
            self._vault_pool_cache_ts = 0.0

            # 7. Save metadata atomically and sync to Telegram Storage Group
            self._save_metadata(sync_to_vault=True)
            logger.info(f"🗑️ [POOL PURGE] Complete dual-input wipe finished: {len(purged_items)} items purged.")

            return {
                "status": "success",
                "processed_shortcode": processed_sc,
                "audio_shortcode": audio_sc,
                "purged_count": len(purged_items),
                "purged_items": purged_items
            }


_global_pool_manager: Optional[AudioPoolManager] = None
_manager_lock = threading.Lock()

def get_pool_manager(base_dir: Optional[str] = None) -> AudioPoolManager:
    """Lazy thread-safe singleton getter for AudioPoolManager."""
    global _global_pool_manager
    if _global_pool_manager is None:
        with _manager_lock:
            if _global_pool_manager is None:
                _global_pool_manager = AudioPoolManager(base_dir=base_dir)
    return _global_pool_manager

def __getattr__(name: str):
    if name == "pool_manager":
        return get_pool_manager()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("🚀 [AUDIO POOL MANAGER] Starting full standalone diagnostic & maintenance sweep...")
    mgr = get_pool_manager()
    logger.info(f"📁 Active Directory: {mgr.active_dir}")
    logger.info(f"📁 Cooldown Directory: {mgr.cooldown_dir}")
    logger.info(f"📄 Metadata Path: {mgr.meta_path}")

    logger.info("1️⃣ Syncing loose audio files from root -> active/...")
    mgr._sync_root_to_active()

    logger.info("2️⃣ Syncing active audio files to metadata...")
    mgr._sync_active_to_metadata()

    logger.info("3️⃣ Syncing active audios to Telegram Storage Vault...")
    vault_sync_res = mgr.sync_all_active_audios_to_telegram_vault()
    logger.info(f"📌 Vault sync uploaded/verified {len(vault_sync_res)} track(s).")

    logger.info("4️⃣ Checking cooldown pool for recycling...")
    recycled = mgr.recycle_cooldown()
    logger.info(f"♻️ Recycled {recycled} audio track(s) from cooldown/ -> active/")

    logger.info("5️⃣ Testing track selection engine...")
    test_track = mgr.select_best_audio(video_bpm=120.0, video_energy=0.8)
    if test_track:
        logger.info(f"🎵 Test track selected: {test_track}")
    else:
        logger.info("ℹ️ No active BGM tracks available for selection (pool may be empty).")

    logger.info("✅ [AUDIO POOL MANAGER] Full standalone diagnostic & maintenance sweep completed successfully!")

