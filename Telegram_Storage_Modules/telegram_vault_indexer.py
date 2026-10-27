"""
Telegram_Storage_Modules / telegram_vault_indexer.py
================================================================
Telegram Storage Group Unified Master Vault Indexer.

Turns Telegram into an unlimited, zero-cost cloud data lake for ephemeral runners
(GitHub Actions / Docker). Stores permanent file_id references and full visual/lyric
intelligence in a single pinned master_vault_index.json document inside the storage group.

Columns:
  Column 1 (processed_reels):
    Indexed by session_id, social_media_id, and custom_title.
    Stores master_video_file_id, audio_data (pool_metadata + lyric_intel),
    and visual_data (.clip_intelligence.json).

  Column 2 (downloaded_sources):
    Indexed by social_media_id (Instagram/YouTube URL) and session_id.
    Stores raw_video_file_id, extracted_audio_file_id, and audio_math.
    Enables 1.5s cache hits on duplicate URL requests without re-downloading.

Author: AMTCE Serverless Vault Architecture v1.0
"""

import os
import sys
import re
import json
import time
import uuid
import threading
import urllib.request
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("telegram_vault_indexer")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
MASTER_INDEX_FILE = os.path.join(DATA_DIR, "master_vault_index.json")

# Cooldown guard: prevents duplicate vault hydration within 60 seconds
_LAST_HYDRATION_TIMESTAMP = 0.0

# Pin rate limit guard: prevents Telegram 429 Too Many Requests on pinChatMessage
_LAST_VAULT_PIN_TIMESTAMP = 0.0
_MIN_VAULT_PIN_INTERVAL_SEC = 15.0


def _empty_vault_index() -> Dict[str, Any]:
    return {
        "version": 3.0,
        "updated_at": time.time(),
        "pinned_message_id": None,
        "lock": None,
        "pool_metadata_file_id": None,
        "telegram_users_file_id": None,
        "source_accounts_file_id": None,
        "scraper_rotation_pointer_file_id": None,
        "telegram_sessions_file_id": None,
        "rejected_audio_blacklist_file_id": None,
    }


def extract_clean_shortcode(val: Optional[str]) -> str:
    """Extract clean shortcode from URL or string, stripping query params, fragments, and markdown artifacts."""
    if not val or not isinstance(val, str):
        return ""
    import urllib.parse
    cleaned = urllib.parse.unquote(val).strip("`'\"\t\r\n ").split("?")[0].split("#")[0].rstrip("/")
    m = re.search(r"/(?:reel|reels|p|tv|shorts|v)/([A-Za-z0-9_-]{5,})", cleaned, re.IGNORECASE)
    if m:
        return m.group(1).strip("`'\"\t\r\n ")
    m_yt = re.search(r"youtu\.be/([A-Za-z0-9_-]{5,})", cleaned, re.IGNORECASE)
    if m_yt:
        return m_yt.group(1).strip("`'\"\t\r\n ")
    sc_clean = cleaned.replace("manual_", "").strip("`'\"\t\r\n ")
    if re.fullmatch(r"[A-Za-z0-9_-]{5,}", sc_clean):
        return sc_clean
    return ""


def canonicalize_social_url(url: Optional[str]) -> str:
    """
    Normalizes any social media URL or shortcode into a clean, canonical format:
    e.g. 'https://www.instagram.com/reel/<shortcode>/'
    Strips query parameters, tracking tokens, markdown formatting backticks, etc.
    """
    if not url or not isinstance(url, str):
        return ""
    import urllib.parse
    sc = extract_clean_shortcode(url)
    if sc:
        url_lower = url.lower()
        if "youtube" in url_lower or "youtu.be" in url_lower:
            return f"https://www.youtube.com/shorts/{sc}/"
        return f"https://www.instagram.com/reel/{sc}/"
    cleaned = urllib.parse.unquote(url).strip("`'\"\t\r\n ").split("?")[0].split("#")[0].rstrip("/")
    return (cleaned + "/") if cleaned else ""


def is_empty_value(val: Any) -> bool:
    if val is None or val == "" or val == "null":
        return True
    if isinstance(val, (dict, list)) and len(val) == 0:
        return True
    return False


def deep_merge_records(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges source into target, preserving non-empty data and preferring non-null."""
    for k, v in source.items():
        if k not in target or is_empty_value(target[k]):
            target[k] = v
        elif isinstance(target[k], dict) and isinstance(v, dict):
            target[k] = deep_merge_records(target[k], v)
        elif is_empty_value(v):
            continue
        else:
            if isinstance(target[k], list) and isinstance(v, list):
                for item in v:
                    if item not in target[k]:
                        target[k].append(item)
            else:
                target[k] = v
    return target


def _send_telegram_file_sync(
    method: str,
    chat_id: str,
    file_key: str,
    file_path: str,
    caption: Optional[str] = None,
    custom_filename: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token or not chat_id or not os.path.exists(file_path):
        return None

    filename = custom_filename or os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    # Telegram Bot API HTTP limit is 50MB. Proactively route files >= 45MB via Pyrogram MTProto
    if file_size >= 45 * 1024 * 1024:
        logger.info(f"🚀 File '{filename}' is {file_size / (1024*1024):.1f}MB (>=45MB). Using Pyrogram MTProto upload directly...")
        try:
            from Telegram_Storage_Modules.telegram_http import upload_file_with_pyrogram
            res = upload_file_with_pyrogram(
                local_path=file_path,
                chat_id=chat_id,
                caption=caption or "",
                file_name=filename,
                as_video=(method == "sendVideo" or file_key == "video")
            )
            if res and res.get("ok"):
                return res
        except Exception as _pyro_err:
            logger.warning(f"⚠️ Pyrogram direct upload error for {filename}: {_pyro_err}")

    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    url = f"https://api.telegram.org/bot{bot_token}/{method}"
    
    body = bytearray()
    
    def add_field(name, value):
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{value}\r\n".encode("utf-8"))

    add_field("chat_id", chat_id)
    if caption:
        add_field("caption", caption)

    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="{file_key}"; filename="{filename}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
    with open(file_path, "rb") as f:
        body.extend(f.read())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as err:
        logger.warning(f"⚠️ Telegram HTTP file upload failed for {filename}: {err}")
        # Automatic fallback to Pyrogram MTProto upload (handles HTTP 413, socket drops, etc.)
        try:
            from Telegram_Storage_Modules.telegram_http import upload_file_with_pyrogram
            logger.info(f"🔄 Attempting Pyrogram MTProto upload fallback for {filename}...")
            res = upload_file_with_pyrogram(
                local_path=file_path,
                chat_id=chat_id,
                caption=caption or "",
                file_name=filename,
                as_video=(method == "sendVideo" or file_key == "video")
            )
            if res and res.get("ok"):
                return res
        except Exception as _pyro_fallback_err:
            logger.warning(f"⚠️ Pyrogram fallback also failed for {filename}: {_pyro_fallback_err}")
        return None


_SAVE_LOCK = threading.Lock()


def _safe_replace(src: str, dst: str, max_retries: int = 5, delay: float = 0.1) -> None:
    """Safely replaces dst with src, with retries to handle Windows [WinError 5] Access is denied locks."""
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


class TelegramVaultIndexer:
    """
    Manages reading, writing, uploading, and pinning the master_vault_index.json
    inside TELEGRAM_STORAGE_GROUP_ID.
    """

    def __init__(self, index_file: str = MASTER_INDEX_FILE):
        self.index_file = index_file
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        self.vault_index = self._load_local_index()

    def _load_local_index(self) -> Dict[str, Any]:
        default_index = _empty_vault_index()
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and ("version" in data or "pool_metadata_file_id" in data):
                        for k, v in default_index.items():
                            if k not in data:
                                data[k] = v
                        return data
            except Exception as e:
                logger.warning(f"⚠️ Could not load local vault index: {e}")
        return default_index

    def _save_local_index(self):
        with _SAVE_LOCK:
            try:
                self.vault_index["updated_at"] = time.time()
                temp_path = self.index_file + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(self.vault_index, f, indent=2, ensure_ascii=False)
                _safe_replace(temp_path, self.index_file)
            except Exception as e:
                logger.error(f"❌ Failed to save local vault index: {e}")

    # ── VAULT JSON HYDRATION & CLOUD SYNC APIs ───────────────────────────────

    def download_vault_file_by_id(self, file_id: str, dest_path: str) -> bool:
        """
        Downloads a document file (e.g. telegram_users.json or metadata_pool.json)
        from Telegram Storage Group into dest_path via Telegram Bot API getFile or Pyrogram MTProto.
        """
        if not file_id:
            return False
        from Telegram_Storage_Modules.telegram_http import download_file_by_id
        return download_file_by_id(file_id, dest_path)

    def sync_pinned_index_from_telegram_sync(self) -> bool:
        """
        Synchronously fetches TELEGRAM_STORAGE_GROUP_ID for pinned master_vault_index.json,
        downloads it, and updates local vault_index.
        """
        storage_group_id = os.getenv("TELEGRAM_STORAGE_GROUP_ID")
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not storage_group_id or not bot_token:
            return False

        try:
            import urllib.request
            import json as _json
            url = f"https://api.telegram.org/bot{bot_token}/getChat?chat_id={storage_group_id}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
            
            if data.get("ok"):
                pinned = data.get("result", {}).get("pinned_message", {})
                doc = pinned.get("document", {})
                if doc.get("file_name") in ["telegram_media_index.json", "master_vault_index.json"] and doc.get("file_id"):
                    file_id = doc["file_id"]
                    if self.download_vault_file_by_id(file_id, self.index_file):
                        self.vault_index = self._load_local_index()
                        logger.info("📌 [VAULT SYNC SUCCESS] Downloaded and reloaded pinned master_vault_index.json from Telegram Storage Group!")
                        return True
        except Exception as err:
            logger.warning("⚠️ Sync pinned master index notice: %s", err)
        return False

    def hydrate_all_vault_jsons_on_startup(self, force: bool = False) -> Dict[str, bool]:
        """
        1. Downloads pinned master_vault_index.json from Telegram Storage Group.
        2. Downloads latest telegram_users.json, metadata_pool.json, and source_accounts.json using file_ids in index.
        """
        global _LAST_HYDRATION_TIMESTAMP
        now = time.time()
        if not force and (now - _LAST_HYDRATION_TIMESTAMP) < 60.0:
            logger.debug("⚡ [VAULT HYDRATION] Skipped duplicate hydration (completed %.1fs ago)", now - _LAST_HYDRATION_TIMESTAMP)
            return {"pinned_index": True, "cached": True}
        _LAST_HYDRATION_TIMESTAMP = now

        results = {
            "pinned_index": False,
            "telegram_users": False,
            "metadata_pool": False,
            "source_accounts": False,
            "rejected_audio_blacklist": False
        }
        try:
            # Step 1: Download pinned index from Telegram Storage Group first!
            results["pinned_index"] = self.sync_pinned_index_from_telegram_sync()

            # Step 2: Download telegram_users.json and merge with local records
            users_file_id = self.vault_index.get("telegram_users_file_id")
            if users_file_id:
                from Telegram_Storage_Modules.telegram_user_manager import USERS_JSON_PATH, load_all_users, save_all_users
                local_users = load_all_users()
                temp_download_path = USERS_JSON_PATH + ".download.tmp"
                if self.download_vault_file_by_id(users_file_id, temp_download_path):
                    try:
                        with open(temp_download_path, "r", encoding="utf-8") as f:
                            downloaded_users = json.load(f)
                        for uid, udata in downloaded_users.items():
                            if uid not in local_users:
                                local_users[uid] = udata
                            else:
                                for k, v in udata.items():
                                    if v:
                                        local_users[uid][k] = v
                        save_all_users(local_users, sync_to_vault=False)
                        results["telegram_users"] = True
                    except Exception as _m_err:
                        logger.warning("Notice merging hydrated users: %s", _m_err)
                    finally:
                        if os.path.exists(temp_download_path):
                            try:
                                os.remove(temp_download_path)
                            except Exception:
                                pass

            # Step 3: Download pool_metadata.json (backward compatible key check)
            pool_file_id = self.vault_index.get("pool_metadata_file_id") or self.vault_index.get("metadata_pool_file_id")
            if pool_file_id:
                from Audio_Modules.audio_pool_manager import AudioPoolManager
                pm = AudioPoolManager()
                results["pool_metadata"] = self.download_vault_file_by_id(pool_file_id, pm.meta_path)
                results["metadata_pool"] = results["pool_metadata"]  # Alias for backward compatibility

            # Step 4: Download source_accounts.json (Auto Input Source Accounts)
            sa_file_id = self.vault_index.get("auto_input_source_account_file_id") or self.vault_index.get("source_accounts_file_id")
            if sa_file_id:
                sa_path = os.path.join(_REPO_ROOT, "Content_Scraper_Modules", "source_accounts.json")
                results["source_accounts"] = self.download_vault_file_by_id(sa_file_id, sa_path)

            # Step 5: Download telegram_sessions.json (Master Active Sessions Index)
            ts_file_id = self.vault_index.get("telegram_sessions_file_id")
            if ts_file_id:
                ts_path = os.path.join(DATA_DIR, "telegram_sessions.json")
                results["telegram_sessions"] = self.download_vault_file_by_id(ts_file_id, ts_path)

            # Step 6: Download scraper_rotation_pointer.json (Master Scraper Rotation Pointer)
            srp_file_id = self.vault_index.get("scraper_rotation_pointer_file_id")
            if srp_file_id:
                srp_path = os.path.join(DATA_DIR, "scraper_rotation_pointer.json")
                results["scraper_rotation_pointer"] = self.download_vault_file_by_id(srp_file_id, srp_path)

            # Step 7: Download rejected_audio_blacklist.json (Master Rejection Blacklist)
            bl_file_id = self.vault_index.get("rejected_audio_blacklist_file_id")
            if bl_file_id:
                try:
                    from Audio_Modules.rejected_audio_blacklist import _BLACKLIST_FILE, merge_blacklist_from_file
                    temp_bl_path = _BLACKLIST_FILE + ".download.tmp"
                    if self.download_vault_file_by_id(bl_file_id, temp_bl_path):
                        results["rejected_audio_blacklist"] = merge_blacklist_from_file(temp_bl_path)
                    if os.path.exists(temp_bl_path):
                        try:
                            os.remove(temp_bl_path)
                        except Exception:
                            pass
                    # Scrub any newly-blacklisted items from local pool metadata
                    try:
                        from Audio_Modules.audio_pool_manager import AudioPoolManager
                        AudioPoolManager().sanitize_pool_metadata_anomalies()
                    except Exception as _san_err:
                        logger.debug("Notice sanitizing pool after blacklist merge: %s", _san_err)
                except Exception as _bl_err:
                    logger.warning("⚠️ Notice hydrating rejected_audio_blacklist: %s", _bl_err)
        except Exception as _h_err:
            logger.warning("⚠️ Vault JSON hydration notice: %s", _h_err)
        return results

    def hydrate_bgm_track_from_vault(
        self,
        track_name_or_file_id: str,
        dest_dir: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Synchronously hydrates a BGM track from Telegram Storage Group using
        file_id stored in pool_metadata.json or direct telegram_file_id string.
        """
        if not track_name_or_file_id:
            return None

        resolved_file_id = file_id
        filename = os.path.basename(track_name_or_file_id)

        if not resolved_file_id and len(track_name_or_file_id) > 20 and not track_name_or_file_id.endswith((".mp3", ".wav", ".m4a")):
            resolved_file_id = track_name_or_file_id
            filename = f"bgm_{resolved_file_id[:10]}.wav"

        if not dest_dir:
            dest_dir = os.path.join(_REPO_ROOT, "Original_audio", "active")
        os.makedirs(dest_dir, exist_ok=True)

        pm_path = os.path.join(_REPO_ROOT, "Original_audio", "pool_metadata.json")
        meta = {}
        if os.path.exists(pm_path):
            try:
                with open(pm_path, "r", encoding="utf-8") as f:
                    pm_data = json.load(f)
                files = pm_data.get("files", pm_data)
                meta = files.get(filename) or {}
                if not meta and resolved_file_id:
                    for k, v in files.items():
                        if isinstance(v, dict) and v.get("file_id") == resolved_file_id:
                            meta = v
                            filename = k
                            break
                if not meta and os.path.splitext(filename)[0]:
                    stem = os.path.splitext(filename.lower())[0]
                    for k, v in files.items():
                        if stem in k.lower() or k.lower() in filename.lower():
                            meta = v
                            filename = k
                            break
                if not resolved_file_id:
                    resolved_file_id = meta.get("file_id")

                # Also search harvested clips in social_media_id dictionary
                if not resolved_file_id and isinstance(files, dict) and "social_media_id" in files:
                    sm_dict = files.get("social_media_id") or {}
                    stem = os.path.splitext(filename.lower())[0] if filename else ""
                    for sm_url, sm_entry in sm_dict.items():
                        if not isinstance(sm_entry, dict):
                            continue
                        sc = (sm_entry.get("shortcode") or "").lower()
                        if (sc and (sc == stem or sc in stem)) or (stem and stem in sm_url.lower()):
                            meta = sm_entry
                            resolved_file_id = (
                                sm_entry.get("media_file_ids", {}).get("extracted_audio_file_id") or
                                sm_entry.get("extracted_audio_file_id") or
                                sm_entry.get("file_id")
                            )
                            if resolved_file_id:
                                break
            except Exception as _pe:
                logger.debug("Notice on pool_metadata BGM lookup: %s", _pe)

        local_path = os.path.join(dest_dir, filename)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
            logger.info("⚡ [LOCAL DISK CACHE HIT] BGM track '%s' already on local disk — skipping Telegram download.", filename)
            return local_path
        active_path = os.path.join(_REPO_ROOT, "Original_audio", "active", filename)
        if os.path.exists(active_path) and os.path.getsize(active_path) > 1024:
            logger.info("⚡ [LOCAL POOL CACHE HIT] BGM track '%s' found in active pool — copying to destination.", filename)
            try:
                import shutil
                if os.path.abspath(active_path) != os.path.abspath(local_path):
                    shutil.copy2(active_path, local_path)
                    return local_path
                return active_path
            except Exception as _cp_err:
                logger.debug("Failed copying from active pool: %s", _cp_err)
                return active_path

        if resolved_file_id:
            try:
                from Telegram_Storage_Modules.telegram_http import is_file_oversized, is_mtproto_configured
                if is_file_oversized(resolved_file_id):
                    if not is_mtproto_configured():
                        logger.warning(
                            "⚠️ [VAULT BGM HYDRATION] File '%s' (file_id: %s) exceeds Telegram 20MB Bot API limit without MTProto. Skipping hydration.",
                            filename, resolved_file_id[:15]
                        )
                        return None
                    else:
                        logger.info(
                            "🚀 [VAULT BGM HYDRATION] File '%s' (file_id: %s) exceeds 20MB Bot API limit. Hydrating via Pyrogram MTProto stream...",
                            filename, resolved_file_id[:15]
                        )
            except Exception:
                pass

            logger.info("📥 [VAULT BGM HYDRATION] Fetching BGM '%s' from Telegram Storage Group (file_id: %s)...", filename, resolved_file_id[:15])
            if self.download_vault_file_by_id(resolved_file_id, local_path):
                logger.info("✅ [VAULT BGM HYDRATION SUCCESS] Downloaded BGM '%s' from Telegram Storage Group!", filename)
                return local_path

        return None


    def get_vault_audio_pool(self, current_clip_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns audio track metadata from pool_metadata.json.
        Includes both active curated BGM files and valid harvested audio tracks from social_media_id.
        """
        pool = {}
        clip_stem = current_clip_id.lower().replace("manual_", "").strip() if current_clip_id else ""

        pm_path = os.path.join(_REPO_ROOT, "Original_audio", "pool_metadata.json")
        if os.path.exists(pm_path):
            try:
                with open(pm_path, "r", encoding="utf-8") as f:
                    pm_data = json.load(f)
                    files_dict = pm_data.get("files", pm_data) if isinstance(pm_data, dict) else {}
                    if isinstance(files_dict, dict):
                        try:
                            from Audio_Modules.rejected_audio_blacklist import is_blacklisted as _is_bl
                        except ImportError:
                            def _is_bl(*args, **kwargs): return False

                        for k, v in files_dict.items():
                            if isinstance(v, dict) and k != "social_media_id":
                                if v.get("is_source_extract") or v.get("is_unusable_bgm"):
                                    continue
                                if clip_stem and clip_stem in k.lower():
                                    continue
                                fid = v.get("file_id") or v.get("telegram_file_id")
                                sc_val = v.get("shortcode")
                                if _is_bl(audio_filename=k, audio_shortcode=sc_val, telegram_file_id=fid):
                                    continue
                                entry = dict(v)
                                g_intel = entry.get("gemini_semantic_audio_intelligence") or entry.get("gemini_semantic_intelligence") or {}
                                if isinstance(g_intel, dict):
                                    for gk, gv in g_intel.items():
                                        if (gk not in entry or not entry[gk]) and gv:
                                            entry[gk] = gv
                                pool[k] = entry

                        # ALSO include valid audio tracks from social_media_id indexed by shortcode!
                        sm_dict = files_dict.get("social_media_id", {})
                        if isinstance(sm_dict, dict):
                            for sm_url, sm_entry in sm_dict.items():
                                if not isinstance(sm_entry, dict):
                                    continue
                                sc = sm_entry.get("shortcode") or ""
                                if not sc or (clip_stem and clip_stem in sc.lower()):
                                    continue
                                ext_fid = (
                                    sm_entry.get("media_file_ids", {}).get("extracted_audio_file_id") or
                                    sm_entry.get("extracted_audio_file_id")
                                )
                                a_data = sm_entry.get("audio_data") or {}
                                if not ext_fid:
                                    continue
                                if _is_bl(audio_filename=f"{sc}.wav", audio_shortcode=sc, telegram_file_id=ext_fid):
                                    continue

                                # ── STRICT CURATED BGM GATE ───────────────────────────
                                # Harvested tracks from other clips can only enter the candidate
                                # BGM pool if they are confirmed musical tracks or explicitly curated.
                                # Raw source extracts, spoken dialogue, and uncurated noise must
                                # NEVER masquerade as background music for other clips.
                                is_src = (
                                    sm_entry.get("is_source_extract") or
                                    a_data.get("is_source_extract") or
                                    sm_entry.get("is_unusable_bgm") or
                                    a_data.get("is_unusable_bgm")
                                )
                                if is_src and not (sm_entry.get("is_curated_bgm") or a_data.get("is_curated_bgm")):
                                    continue
                                if sm_entry.get("is_unusable") or a_data.get("is_unusable"):
                                    continue
                                if sm_entry.get("is_speech_only") or a_data.get("is_speech_only"):
                                    continue

                                sp_intel = sm_entry.get("speech_intelligence") or a_data.get("speech_intelligence") or {}
                                sp_mode = sp_intel.get("speech_mode", "")
                                if sp_mode in ("on_camera_dialogue", "voiceover_narration") and not a_data.get("has_music", False):
                                    continue

                                # Check noise keywords in reason or vibes
                                reason = str(sm_entry.get("unusable_reason", "")).lower()
                                vibe_str = str(a_data.get("vibe", "")).lower()
                                vibe_tags = [str(x).lower() for x in (a_data.get("vibe_tags") or [])]
                                noise_kws = ("crowd", "babble", "screaming", "traffic", "street_noise", "horn", "chatter", "podcast", "interview", "static")
                                if any(kw in reason for kw in noise_kws) or any(kw in vibe_str for kw in noise_kws) or any(any(kw in vt for kw in noise_kws) for vt in vibe_tags):
                                    continue

                                bpm_val = float(a_data.get("tempo_bpm") or a_data.get("bpm") or 120.0)
                                energy_val = float(a_data.get("avg_energy") or a_data.get("energy") or 0.7)

                                # Reject if flagged as spoken vocal without musical tempo
                                if a_data.get("is_speech_vocal") and bpm_val <= 0:
                                    continue

                                # Reject if file is known to exceed 20MB Bot API limit without MTProto
                                try:
                                    from Telegram_Storage_Modules.telegram_http import is_file_oversized, is_mtproto_configured
                                    if is_file_oversized(ext_fid) and not is_mtproto_configured():
                                        continue
                                except Exception:
                                    pass

                                audio_fsize = float(
                                    sm_entry.get("media_file_ids", {}).get("extracted_audio_file_size") or
                                    sm_entry.get("extracted_audio_file_size") or
                                    a_data.get("file_size") or 0.0
                                )
                                try:
                                    from Telegram_Storage_Modules.telegram_http import is_mtproto_configured
                                    if audio_fsize > 20 * 1024 * 1024 and not is_mtproto_configured():
                                        continue
                                except Exception:
                                    pass

                                track_key = f"{sc}.wav"

                                # Extract rich semantic intelligence (lyrics, transcript, caption, language)
                                w_trans = a_data.get("whisper_transcript") or sm_entry.get("whisper_transcript") or {}
                                transcript_str = ""
                                if isinstance(w_trans, dict):
                                    transcript_str = str(w_trans.get("text") or "").strip()
                                    if not transcript_str and "sentences" in w_trans:
                                        transcript_str = " ".join(str(s.get("text", "")) for s in w_trans["sentences"] if isinstance(s, dict)).strip()
                                elif isinstance(w_trans, str):
                                    transcript_str = w_trans.strip()

                                lyrics_str = str(a_data.get("lyrics") or sm_entry.get("lyrics") or transcript_str).strip()
                                caption_str = str(sm_entry.get("caption") or a_data.get("caption") or "").strip()
                                tags_list = sm_entry.get("hashtags") or a_data.get("hashtags") or []
                                lang_val = str(a_data.get("language") or (w_trans.get("language") if isinstance(w_trans, dict) else "") or "").strip()
                                dom_emotion = str(a_data.get("dominant_emotion") or a_data.get("emotion") or "").strip()
                                hook_t = float(a_data.get("hook_time") or a_data.get("audio_math", {}).get("hook_time") or 0.0)

                                pool[track_key] = {
                                    "shortcode": sc,
                                    "file_id": ext_fid,
                                    "telegram_file_id": ext_fid,
                                    "bpm": bpm_val,
                                    "energy": energy_val,
                                    "vibe": a_data.get("vibe") or "energetic",
                                    "vibe_tags": a_data.get("vibe_tags") or [a_data.get("vibe", "energetic")],
                                    "genre": a_data.get("genre") or a_data.get("gemini_genre") or "music",
                                    "duration": float(a_data.get("duration") or a_data.get("audio_duration") or 15.0),
                                    "file_size": audio_fsize,
                                    "is_source_extract": True,
                                    "usage_count": sm_entry.get("usage_count", 0),
                                    "last_used": sm_entry.get("last_used", 0),
                                    "transcript": transcript_str,
                                    "lyrics": lyrics_str,
                                    "caption": caption_str,
                                    "hashtags": tags_list if isinstance(tags_list, list) else [str(tags_list)],
                                    "language": lang_val,
                                    "dominant_emotion": dom_emotion,
                                    "hook_time": hook_t,
                                }
            except Exception as _pme:
                logger.debug("Local pool metadata read notice: %s", _pme)

        return pool

    def upload_and_pin_vault_index_sync(self, upload_fn=None, force: bool = False):
        """Uploads master_vault_index.json to TELEGRAM_STORAGE_GROUP_ID and pins it (debounced)."""
        global _LAST_VAULT_PIN_TIMESTAMP
        storage_group_id = os.getenv("TELEGRAM_STORAGE_GROUP_ID")
        if not storage_group_id or not os.path.exists(self.index_file):
            return

        now = time.time()
        if not force and (now - _LAST_VAULT_PIN_TIMESTAMP) < _MIN_VAULT_PIN_INTERVAL_SEC:
            logger.debug(
                "⏳ [VAULT PIN DEBOUNCE] Skipping master_vault_index pin (last pinned %.1fs ago < %.0fs limit).",
                now - _LAST_VAULT_PIN_TIMESTAMP, _MIN_VAULT_PIN_INTERVAL_SEC
            )
            return

        try:
            caption = f"📌 **[VAULT MASTER INDEX]** Auto-Synced\n🕒 `{time.strftime('%Y-%m-%d %H:%M:%S')}`"
            res = None
            if upload_fn:
                res = upload_fn("sendDocument", storage_group_id, "document", self.index_file, caption=caption)
            else:
                from Telegram_Storage_Modules.telegram_http import send_document
                res = send_document(local_path=self.index_file, caption=caption, chat_id=str(storage_group_id))

            if res and isinstance(res, dict):
                msg_id = res.get("message_id")
                if msg_id:
                    self.vault_index["pinned_message_id"] = msg_id
                    self._save_local_index()
                    pin_ok = False
                    try:
                        from Telegram_Storage_Modules.telegram_http import pin_message
                        pin_ok = pin_message(message_id=int(msg_id), chat_id=str(storage_group_id))
                    except Exception as _p_call_err:
                        logger.warning("Notice on pinChatMessage call: %s", _p_call_err)
                    if pin_ok:
                        _LAST_VAULT_PIN_TIMESTAMP = time.time()
                        logger.info("📌 [VAULT PIN SUCCESS] Uploaded & PINNED master_vault_index.json in Storage Group (Message ID: %s)", msg_id)
                    else:
                        logger.warning("⚠️ [VAULT PIN NOTICE] Uploaded master_vault_index.json (Message ID: %s) but pinChatMessage did not confirm.", msg_id)
        except Exception as _pin_err:
            logger.warning("⚠️ Vault index upload/pin notice: %s", _pin_err)

    def _normalize_search_keys(self, target: Any) -> List[str]:
        """Generates all normalized search keys (stripping manual_, URL parsing, unquoting, markdown links, etc.)."""
        import urllib.parse
        raw_target = str(target).strip()
        # Strip markdown link formatting e.g. [https://...](https://...)
        m_md = re.match(r"\[(.*?)\]\((.*?)\)", raw_target)
        if m_md:
            raw_target = m_md.group(2).strip()

        # Repeatedly strip trailing URL-encoded backticks (%60), backticks, quotes, whitespace
        raw_target = re.sub(r"(?:%60)+$", "", raw_target).strip()
        unquoted = urllib.parse.unquote(raw_target)
        clean_target = re.sub(r"`+$", "", unquoted).strip("`\"' \r\n\t")
        target_nomanual = clean_target.replace("manual_", "").strip()

        extracted = None
        m_ig = re.search(r"/(?:p|reel|reels)/([A-Za-z0-9_-]+)", clean_target)
        if m_ig:
            extracted = m_ig.group(1)
        if not extracted:
            m_yt = re.search(r"(?:shorts/|v=|youtu\.be/)([A-Za-z0-9_-]{11})", clean_target)
            if m_yt:
                extracted = m_yt.group(1)
        if not extracted:
            m_tt = re.search(r"/video/(\d+)", clean_target)
            if m_tt:
                extracted = m_tt.group(1)

        search_keys = []
        canon = canonicalize_social_url(raw_target)
        if canon:
            search_keys.append(canon)
        for k in [raw_target, unquoted, clean_target, target_nomanual, extracted]:
            if not k:
                continue
            for cand in (k, k.lower(), k.split("?")[0].rstrip("/"), k.split("?")[0].rstrip("/") + "/"):
                if cand and cand not in search_keys:
                    search_keys.append(cand)
            k_nomanual = k.replace("manual_", "").strip()
            if k_nomanual and k_nomanual not in search_keys:
                search_keys.append(k_nomanual)
                search_keys.append(k_nomanual.lower())
                search_keys.append(f"manual_{k_nomanual}")
        return search_keys

    def find_entry_by_shortcode(self, shortcode: str) -> Optional[Dict[str, Any]]:
        """
        Finds video metadata entry in pool_metadata.json by shortcode or URL.
        Robustly extracts shortcode from URLs, handles 'manual_' prefixes, and merges
        multi-candidate entries so no media_file_ids (raw, clean, audio, processed) are lost.
        """
        if not shortcode:
            return None
        search_keys = self._normalize_search_keys(shortcode)

        matching_entries: List[Dict[str, Any]] = []
        # 1. Search pool_metadata.json -> files -> social_media_id AND top-level clips
        try:
            from Audio_Modules.audio_pool_manager import AudioPoolManager
            pm = AudioPoolManager()
            clips = pm.metadata.get("files", {}).get("social_media_id", {})
            top_clips = pm.metadata.get("clips", {})

            # Exact key lookups first
            for k in search_keys:
                if k in clips and clips[k] not in matching_entries:
                    matching_entries.append(clips[k])
                if k in top_clips and top_clips[k] not in matching_entries:
                    matching_entries.append(top_clips[k])

            # Fuzzy/shortcode lookups across all stored entries
            for pool_source in [clips, top_clips]:
                for stored_url, entry in pool_source.items():
                    if not isinstance(entry, dict) or entry in matching_entries:
                        continue
                    entry_sc = str(entry.get("shortcode", "")).strip().lower()
                    stored_clean = stored_url.split("?")[0].rstrip("/").lower()
                    for k in search_keys:
                        kl = k.lower()
                        if (
                            (entry_sc and (entry_sc == kl or entry_sc == kl.replace("manual_", ""))) or
                            (kl and (kl in stored_url.lower() or kl.replace("manual_", "") in stored_url.lower())) or
                            (stored_clean and (stored_clean in kl or kl in stored_clean)) or
                            (kl and kl in str(entry.get("file_name", "")).lower())
                        ):
                            matching_entries.append(entry)
                            break
        except Exception as e:
            logger.debug("Notice on find_entry_by_shortcode pool_metadata: %s", e)

        # 2. Check TelegramSessionManager
        sess_entry = None
        try:
            from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
            sm = TelegramSessionManager()
            for k in search_keys:
                sess = sm.get_session(k)
                if sess:
                    sess_entry = {
                        "shortcode": sess.get("clip_id", "").replace("manual_", ""),
                        "session_id": sess.get("session_id"),
                        "raw_video_file_id": sess.get("raw_video_file_id"),
                        "raw_video_path": sess.get("raw_video_path"),
                        "media_file_ids": {
                            "raw_video_file_id": sess.get("raw_video_file_id")
                        }
                    }
                    break
        except Exception:
            pass

        # 3. Consolidate and merge all matching candidates
        found_entry = None
        if matching_entries:
            def _score_entry(e: Dict[str, Any]) -> int:
                m = e.get("media_file_ids", {})
                s = 0
                if m.get("raw_video_file_id") or e.get("raw_video_file_id") or e.get("raw_file_id"):
                    s += 10
                if m.get("processed_output_file_id") or e.get("processed_output_file_id") or e.get("master_reel_file_id"):
                    s += 5
                if m.get("wm_clean_file_id") or e.get("wm_clean_file_id"):
                    s += 5
                if m.get("extracted_audio_file_id") or e.get("extracted_audio_file_id"):
                    s += 5
                if e.get("shortcode"):
                    s += 2
                return s

            matching_entries.sort(key=_score_entry, reverse=True)
            # Base candidate is the most complete one
            found_entry = dict(matching_entries[0])
            for other in matching_entries[1:]:
                # Merge media_file_ids
                other_m = other.get("media_file_ids", {}) if isinstance(other.get("media_file_ids"), dict) else {}
                base_m = found_entry.setdefault("media_file_ids", {})
                for mk, mv in other_m.items():
                    if mv and not base_m.get(mk):
                        base_m[mk] = mv
                # Merge top-level fields
                for ok, ov in other.items():
                    if ov and not found_entry.get(ok):
                        found_entry[ok] = ov

        # Resolve best raw_file_id across pool candidates and sessions
        entry_raw_fid = None
        if found_entry:
            entry_raw_fid = (
                found_entry.get("media_file_ids", {}).get("raw_video_file_id") or
                found_entry.get("raw_video_file_id") or
                found_entry.get("raw_file_id")
            )

        best_raw_fid = (
            entry_raw_fid
            or (sess_entry.get("raw_video_file_id") if sess_entry else None)
        )

        if found_entry:
            if best_raw_fid:
                found_entry.setdefault("media_file_ids", {})["raw_video_file_id"] = best_raw_fid
                found_entry["raw_video_file_id"] = best_raw_fid
            # Also backfill wm_clean_file_id, processed_output_file_id, extracted_audio_file_id to top level
            m_ids = found_entry.get("media_file_ids", {})
            for fid_key in ["wm_clean_file_id", "processed_output_file_id", "extracted_audio_file_id"]:
                if m_ids.get(fid_key) and not found_entry.get(fid_key):
                    found_entry[fid_key] = m_ids[fid_key]
            return found_entry

        return sess_entry

    def hydrate_raw_video_from_vault(self, identifier: str, dest_dir: Optional[str] = None, check_clean_first: bool = True) -> Optional[str]:
        """
        Downloads already-stored source video from Telegram Storage Group.
        Checks for watermark-cleaned clip (video_inpainted_clean.mp4 / wm_clean_file_id) FIRST.
        If watermark-cleaned clip is not found, falls back to raw source video (raw_video_file_id).
        Accepts social_url, shortcode, clip_id, or session_id.
        """
        if not identifier:
            return None

        # 0. Check for watermark-cleaned clip first if enabled
        if check_clean_first:
            try:
                clean_vid = self.hydrate_clean_video_from_vault(identifier, dest_dir=dest_dir)
                if clean_vid and os.path.exists(clean_vid) and os.path.getsize(clean_vid) > 1024:
                    logger.info(f"✨ [VAULT HYDRATE] Watermark-cleaned clip found first -> {clean_vid}")
                    return clean_vid
            except Exception as _ce:
                logger.debug(f"[VAULT HYDRATE] Clean video check notice: {_ce}")

        search_keys = self._normalize_search_keys(identifier)
        entry = self.find_entry_by_shortcode(identifier) or self.lookup_downloaded_source(identifier)

        # Safe shortcode extraction
        shortcode = None
        if entry and entry.get("shortcode"):
            shortcode = str(entry["shortcode"]).strip()
        if not shortcode:
            for k in search_keys:
                if k and not k.startswith("http") and "/" not in k and "\\" not in k and ":" not in k and "?" not in k:
                    shortcode = k.strip()
                    break
        if shortcode:
            shortcode = shortcode.replace("manual_clean_", "").replace("auto_clean_", "").replace("manual_raw_", "").replace("auto_raw_", "").replace("manual_", "").replace("auto_", "").strip()
        if not shortcode:
            shortcode = "clip"

        is_auto = "auto" in str(identifier).lower() or (dest_dir and "auto" in str(dest_dir).lower())
        folder_prefix = "auto_" if is_auto else "manual_"
        prefix = "auto_raw_" if is_auto else "manual_raw_"

        # 1. Check if already exists on local disk (prioritize cleaned over raw)
        candidate_dirs = [
            dest_dir,
            os.path.join(_REPO_ROOT, "downloads", f"{folder_prefix}{shortcode}"),
            os.path.join(_REPO_ROOT, "downloads", f"manual_{shortcode}"),
            os.path.join(_REPO_ROOT, "downloads", f"auto_{shortcode}"),
            os.path.join(_REPO_ROOT, "downloads", shortcode),
        ]
        if check_clean_first:
            candidate_filenames = [
                f"{prefix}{shortcode}.mp4",
                f"manual_raw_{shortcode}.mp4",
                f"auto_raw_{shortcode}.mp4",
                f"{folder_prefix}{shortcode}.mp4",
                f"manual_{shortcode}.mp4",
                f"auto_{shortcode}.mp4",
                "video_inpainted_clean.mp4",
                "video.mp4",
                f"{shortcode}.mp4",
            ]
            if entry and entry.get("file_name"):
                candidate_filenames.insert(1, entry["file_name"])
        else:
            candidate_filenames = [
                f"{prefix}{shortcode}.mp4",
                f"manual_raw_{shortcode}.mp4",
                f"auto_raw_{shortcode}.mp4",
                "video.mp4",
                f"{shortcode}.mp4",
            ]

        for c_dir in candidate_dirs:
            if c_dir and os.path.exists(c_dir):
                for c_fn in candidate_filenames:
                    if "_master.mp4" in c_fn.lower():
                        continue
                    if not check_clean_first and ("_clean" in c_fn.lower() or "inpainted" in c_fn.lower()):
                        continue
                    c_out = os.path.join(c_dir, c_fn)
                    if os.path.exists(c_out) and os.path.getsize(c_out) > 1024:
                        return c_out

        # Also check session manager for raw_video_path
        sess = None
        try:
            from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
            sm = TelegramSessionManager()
            for k in search_keys:
                sess = sm.get_session(k)
                if sess:
                    break
            if sess and sess.get("raw_video_path") and os.path.exists(sess["raw_video_path"]):
                if "_master.mp4" not in sess["raw_video_path"].lower() and os.path.getsize(sess["raw_video_path"]) > 1024:
                    return sess["raw_video_path"]
        except Exception:
            pass

        # 2. Resolve raw_video_file_id from entry or session
        raw_file_id = None
        if entry:
            raw_file_id = (
                entry.get("media_file_ids", {}).get("raw_video_file_id") or
                entry.get("raw_video_file_id") or
                entry.get("raw_file_id")
            )
        if not raw_file_id and sess:
            raw_file_id = sess.get("raw_video_file_id")

        if not raw_file_id:
            logger.warning(f"⚠️ [VAULT HYDRATE] No raw_video_file_id found for '{identifier}'")
            return None

        # 3. Download via vault file downloader
        if not dest_dir:
            dest_dir = os.path.join(_REPO_ROOT, "downloads", f"{folder_prefix}{shortcode}")
        os.makedirs(dest_dir, exist_ok=True)
        raw_filename = f"{prefix}{shortcode}.mp4"
        out_path = os.path.join(dest_dir, raw_filename)

        logger.info(f"📥 [VAULT HYDRATE] Downloading raw source for '{shortcode}' from Telegram Storage Group -> {raw_filename}...")
        if self.download_vault_file_by_id(raw_file_id, out_path):
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                # Maintain legacy video.mp4 for backward compatibility
                legacy_path = os.path.join(dest_dir, "video.mp4")
                if not os.path.exists(legacy_path):
                    try:
                        import shutil
                        shutil.copyfile(out_path, legacy_path)
                    except Exception:
                        pass
                logger.info(f"✅ [VAULT HYDRATE] Raw video recovered successfully -> {out_path}")
                return out_path

        return None

    def hydrate_processed_reel_from_vault(self, shortcode_or_session_id: str, dest_dir: Optional[str] = None) -> Optional[str]:
        """
        Retrieves processed master reel from Telegram Storage Group.
        Accepts shortcode, session_id, or social_url.
        Resolves processed_output_file_id from pool_metadata.json v3 (media_file_ids.processed_output_file_id).
        Returns local filesystem path to the recovered video.
        """
        if not shortcode_or_session_id:
            return None

        # 1. Check local session manager first
        sess = None
        try:
            from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
            sess = TelegramSessionManager().get_session(shortcode_or_session_id)
            if sess and sess.get("video_path") and os.path.exists(sess["video_path"]):
                if os.path.getsize(sess["video_path"]) > 1024:
                    return sess["video_path"]
        except Exception:
            pass

        # 2. Look up entry in pool_metadata.json
        entry = self.find_entry_by_shortcode(shortcode_or_session_id) or self.lookup_processed_reel(session_id=shortcode_or_session_id)
        shortcode = (entry.get("shortcode") if entry else None) or (sess.get("clip_id") if sess else None) or shortcode_or_session_id

        # Check default local processed reels path
        if not dest_dir:
            dest_dir = os.path.join(_REPO_ROOT, "downloads", "Processed_Shorts")
        os.makedirs(dest_dir, exist_ok=True)
        out_path = os.path.join(dest_dir, f"{shortcode}_master.mp4")

        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            return out_path

        # 3. Resolve processed_output_file_id (pool_metadata first, then session fallback)
        processed_file_id = None
        if entry:
            processed_file_id = (
                entry.get("media_file_ids", {}).get("processed_output_file_id") or
                entry.get("processed_output_file_id") or
                entry.get("master_file_id")
            )

        if not processed_file_id and sess:
            processed_file_id = (
                sess.get("processed_file_id") or
                sess.get("master_file_id") or
                sess.get("media_file_ids", {}).get("processed_output_file_id")
            )

        if not processed_file_id:
            logger.warning(f"⚠️ [VAULT PROCESSED HYDRATE] No processed_output_file_id found for '{shortcode_or_session_id}'")
            return None

        # 4. Download from vault
        logger.info(f"📥 [VAULT PROCESSED HYDRATE] Downloading processed reel for '{shortcode}' from Telegram Storage Group...")
        if self.download_vault_file_by_id(processed_file_id, out_path):
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                logger.info(f"✅ [VAULT PROCESSED HYDRATE] Processed reel recovered successfully -> {out_path}")
                return out_path

        return None

    def hydrate_clean_video_from_vault(self, identifier: str, dest_dir: Optional[str] = None) -> Optional[str]:
        """
        Downloads or locates the already watermark-cleaned video (video_inpainted_clean.mp4)
        from Telegram Storage Group or local disk using wm_clean_file_id.
        Allows Phase 2 to completely bypass OpenCV frame-by-frame watermark inpainting!
        """
        if not identifier:
            return None

        search_keys = self._normalize_search_keys(identifier)
        entry = self.find_entry_by_shortcode(identifier) or self.lookup_downloaded_source(identifier)

        # Safe shortcode extraction
        shortcode = None
        if entry and entry.get("shortcode"):
            shortcode = str(entry["shortcode"]).strip()
        if not shortcode:
            for k in search_keys:
                if k and not k.startswith("http") and "/" not in k and "\\" not in k and ":" not in k and "?" not in k:
                    shortcode = k.strip()
                    break
        if shortcode:
            shortcode = shortcode.replace("manual_clean_", "").replace("auto_clean_", "").replace("manual_raw_", "").replace("auto_raw_", "").replace("manual_", "").replace("auto_", "").strip()
        if not shortcode:
            shortcode = "clip"

        is_auto = "auto" in str(identifier).lower() or (dest_dir and "auto" in str(dest_dir).lower())
        folder_prefix = "auto_" if is_auto else "manual_"
        prefix = "auto_clean_" if is_auto else "manual_clean_"
        clean_filename = f"{prefix}{shortcode}.mp4"

        # 1. Local disk check across candidate directories — ONLY genuine clean/inpainted videos allowed
        candidate_dirs = [
            dest_dir,
            os.path.join(_REPO_ROOT, "downloads", f"{folder_prefix}{shortcode}"),
            os.path.join(_REPO_ROOT, "downloads", f"manual_{shortcode}"),
            os.path.join(_REPO_ROOT, "downloads", f"auto_{shortcode}"),
            os.path.join(_REPO_ROOT, "downloads", shortcode),
        ]
        clean_candidates = [
            clean_filename,
            f"manual_clean_{shortcode}.mp4",
            f"auto_clean_{shortcode}.mp4",
            "video_inpainted_clean.mp4",
        ]
        for c_dir in candidate_dirs:
            if c_dir and os.path.exists(c_dir):
                for c_fn in clean_candidates:
                    if "_master.mp4" in c_fn.lower():
                        continue
                    c_clean = os.path.join(c_dir, c_fn)
                    if os.path.exists(c_clean) and os.path.getsize(c_clean) > 1024:
                        logger.info(f"⚡ [VAULT CLEAN HYDRATE] Local clean video found -> {c_clean}")
                        return c_clean

        # Check TelegramSessionManager for clean video path (NEVER use video_path which points to master reels)
        sess = None
        try:
            from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
            sm = TelegramSessionManager()
            for k in search_keys:
                sess = sm.get_session(k)
                if sess:
                    for s_k in ["clean_video_path", "clean_raw_path"]:
                        sp = sess.get(s_k)
                        if not sp:
                            continue
                        bname = os.path.basename(sp).lower()
                        if "_master.mp4" in bname or "processed shorts" in sp.lower():
                            continue
                        if ("clean" in bname or "inpainted" in bname) and os.path.exists(sp) and os.path.getsize(sp) > 1024:
                            return sp
                    break
        except Exception:
            pass

        # 2. Vault download via wm_clean_file_id
        clean_file_id = None
        if entry:
            clean_file_id = (
                entry.get("media_file_ids", {}).get("wm_clean_file_id") or
                entry.get("wm_clean_file_id")
            )
        if not clean_file_id and sess:
            clean_file_id = (
                sess.get("media_file_ids", {}).get("wm_clean_file_id") if isinstance(sess.get("media_file_ids"), dict) else None
            ) or sess.get("wm_clean_file_id")

        if clean_file_id:
            if not dest_dir:
                dest_dir = os.path.join(_REPO_ROOT, "downloads", f"{folder_prefix}{shortcode}")
            os.makedirs(dest_dir, exist_ok=True)
            clean_path = os.path.join(dest_dir, clean_filename)
            logger.info(f"⚡ [VAULT CLEAN HYDRATE] Downloading watermark-cleaned video for '{shortcode}' from Telegram Vault -> {clean_filename}...")
            if self.download_vault_file_by_id(clean_file_id, clean_path):
                if os.path.exists(clean_path) and os.path.getsize(clean_path) > 1024:
                    # Maintain legacy video_inpainted_clean.mp4 for backward compatibility
                    legacy_path = os.path.join(dest_dir, "video_inpainted_clean.mp4")
                    if not os.path.exists(legacy_path):
                        try:
                            import shutil
                            shutil.copyfile(clean_path, legacy_path)
                        except Exception:
                            pass
                    logger.info(f"✅ [VAULT CLEAN HYDRATE] Clean video recovered successfully -> {clean_path}")
                    return clean_path

        return None

    def hydrate_selected_audio_from_vault(self, identifier: str, dest_dir: Optional[str] = None) -> Optional[str]:
        """
        Downloads or locates the chosen BGM audio track from Telegram Storage Group or local disk
        using selected_audio_file_id or track name in pool_metadata.json.
        """
        if not identifier:
            return None

        entry = self.find_entry_by_shortcode(identifier) or self.lookup_downloaded_source(identifier)
        track_name = None
        selected_fid = None
        if entry:
            selected_fid = (
                entry.get("media_file_ids", {}).get("selected_audio_file_id")
                or entry.get("selected_audio_file_id")
            )
            sel_dict = entry.get("audio_data", {}).get("selected_audio")
            if isinstance(sel_dict, dict):
                track_name = sel_dict.get("track_name")
                selected_fid = selected_fid or sel_dict.get("file_id")
            elif isinstance(entry.get("selected_audio"), str):
                track_name = entry.get("selected_audio")

        if not track_name and not selected_fid:
            track_name = os.path.basename(identifier)

        if not dest_dir:
            dest_dir = os.path.join(_REPO_ROOT, "Original_audio", "active")
        os.makedirs(dest_dir, exist_ok=True)

        if track_name:
            out_path = os.path.join(dest_dir, track_name)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                return out_path
            return self.hydrate_bgm_track_from_vault(track_name, dest_dir, file_id=selected_fid)
        elif selected_fid:
            out_path = os.path.join(dest_dir, f"vault_bgm_{selected_fid[:10]}.mp3")
            if self.download_vault_file_by_id(selected_fid, out_path):
                return out_path

        return None

    def hydrate_extracted_audio_from_vault(self, identifier: str, dest_dir: Optional[str] = None) -> Optional[str]:
        """
        Downloads or locates the clip's extracted audio ({shortcode}.wav / video_extracted.wav)
        from Telegram Storage Group using extracted_audio_file_id.
        """
        if not identifier:
            return None

        search_keys = self._normalize_search_keys(identifier)
        entry = self.find_entry_by_shortcode(identifier) or self.lookup_downloaded_source(identifier)

        shortcode = None
        if entry and entry.get("shortcode"):
            shortcode = str(entry["shortcode"]).strip()
        if not shortcode:
            for k in search_keys:
                if k and not k.startswith("http") and "/" not in k and "\\" not in k and ":" not in k and "?" not in k:
                    shortcode = k.replace("manual_", "").strip()
                    break
        if not shortcode:
            shortcode = "clip"
        else:
            shortcode = os.path.splitext(shortcode)[0]

        if not dest_dir:
            dest_dir = os.path.join(_REPO_ROOT, "downloads", f"manual_{shortcode}")
        os.makedirs(dest_dir, exist_ok=True)

        # Check local candidate files
        cand_files = [
            os.path.join(dest_dir, f"{shortcode}.wav"),
            os.path.join(dest_dir, "video_extracted.wav"),
        ]
        for cf in cand_files:
            if os.path.exists(cf) and os.path.getsize(cf) > 1024:
                return cf

        # Check local active pool cache (already harvested)
        active_candidates = [
            os.path.join(_REPO_ROOT, "Original_audio", "active", f"{shortcode}.wav"),
            os.path.join(_REPO_ROOT, "Original_audio", "active", f"bgm_{shortcode}.wav"),
        ]
        for ac in active_candidates:
            if os.path.exists(ac) and os.path.getsize(ac) > 1024:
                out_path = os.path.join(dest_dir, f"{shortcode}.wav")
                try:
                    import shutil
                    shutil.copy2(ac, out_path)
                    legacy_wav = os.path.join(dest_dir, "video_extracted.wav")
                    if not os.path.exists(legacy_wav):
                        shutil.copy2(ac, legacy_wav)
                    logger.info(f"⚡ [LOCAL POOL CACHE HIT] Extracted audio found in active pool: {ac} -> {out_path}")
                    return out_path
                except Exception:
                    return ac

        # Resolve extracted_audio_file_id from entry or session
        ext_fid = None
        if entry:
            ext_fid = (
                entry.get("media_file_ids", {}).get("extracted_audio_file_id") or
                entry.get("extracted_audio_file_id")
            )
        if not ext_fid:
            try:
                from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
                sm = TelegramSessionManager()
                for k in search_keys:
                    sess = sm.get_session(k)
                    if sess:
                        ext_fid = (
                            sess.get("media_file_ids", {}).get("extracted_audio_file_id") if isinstance(sess.get("media_file_ids"), dict) else None
                        ) or sess.get("extracted_audio_file_id")
                        if ext_fid:
                            break
            except Exception:
                pass

        # Resolve from AudioPoolManager pool_metadata.json (track intelligence & files index)
        if not ext_fid:
            try:
                from Audio_Modules.audio_pool_manager import AudioPoolManager
                pm = AudioPoolManager()
                for t_name in [f"{shortcode}.wav", f"bgm_{shortcode}.wav", f"{shortcode}.mp3", identifier]:
                    t_intel = pm.get_track_intelligence(t_name)
                    if t_intel and t_intel.get("file_id"):
                        ext_fid = t_intel["file_id"]
                        break
                if not ext_fid:
                    f_dict = pm.metadata.get("files", {})
                    for fk, fv in f_dict.items():
                        if isinstance(fv, dict) and fv.get("file_id"):
                            if shortcode.lower() in fk.lower():
                                ext_fid = fv["file_id"]
                                break
            except Exception as _pme:
                logger.debug("Notice on AudioPoolManager file_id lookup: %s", _pme)

        # Resolve from master_vault_index.json Column 2
        if not ext_fid:
            try:
                c2_sess = self.vault_index.get("column_2_downloaded_sources", {}).get("by_session_id", {})
                for c2_k, c2_v in c2_sess.items():
                    if shortcode.lower() in c2_k.lower() and isinstance(c2_v, dict) and c2_v.get("extracted_audio_file_id"):
                        ext_fid = c2_v["extracted_audio_file_id"]
                        break
            except Exception:
                pass

        if ext_fid:
            out_path = os.path.join(dest_dir, f"{shortcode}.wav")
            logger.info(f"⚡ [VAULT AUDIO HYDRATE] Downloading extracted audio for '{shortcode}' from Telegram Vault (file_id: {ext_fid[:15]})...")
            if self.download_vault_file_by_id(ext_fid, out_path):
                if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                    legacy_wav = os.path.join(dest_dir, "video_extracted.wav")
                    if not os.path.exists(legacy_wav):
                        try:
                            import shutil
                            shutil.copy2(out_path, legacy_wav)
                        except Exception:
                            pass
                    logger.info(f"✅ [VAULT AUDIO HYDRATE] Extracted audio recovered successfully -> {out_path}")
                    return out_path

        return None

    async def send_raw_video_to_user_chat(self, bot, chat_id: int | str, identifier: str) -> bool:
        """
        Sends raw downloaded video directly to the user's chat.
        Never leaks internal file_id values in the user-visible caption.
        Prioritizes:
          1. Existing local file
          2. Direct file_id delivery via Telegram Bot API (instant, 0 bandwidth)
          3. On-demand vault hydration fallback
        """
        if not bot or not chat_id or not identifier:
            return False

        caption = "📥 **Raw Downloaded Video**\n📁 `video.mp4`"
        search_keys = self._normalize_search_keys(identifier)
        shortcode = identifier.replace("manual_", "")

        # 1. Check if raw video file already exists on local disk
        candidate_dirs = [
            os.path.join(_REPO_ROOT, "downloads", identifier),
            os.path.join(_REPO_ROOT, "downloads", shortcode),
            os.path.join(_REPO_ROOT, "downloads", f"manual_{shortcode}"),
        ]
        local_raw = None
        for c_dir in candidate_dirs:
            if c_dir and os.path.exists(c_dir):
                for c_fn in ["video.mp4", f"{shortcode}.mp4", f"manual_{shortcode}.mp4"]:
                    c_out = os.path.join(c_dir, c_fn)
                    if os.path.exists(c_out) and os.path.getsize(c_out) > 1024:
                        local_raw = c_out
                        break
            if local_raw:
                break

        if not local_raw:
            try:
                from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
                sm = TelegramSessionManager()
                for k in search_keys:
                    s = sm.get_session(k)
                    if s and s.get("raw_video_path") and os.path.exists(s["raw_video_path"]):
                        if os.path.getsize(s["raw_video_path"]) > 1024:
                            local_raw = s["raw_video_path"]
                            break
            except Exception:
                pass

        if local_raw and os.path.exists(local_raw) and os.path.getsize(local_raw) > 1024:
            try:
                with open(local_raw, "rb") as vf:
                    await bot.send_video(
                        chat_id=int(chat_id),
                        video=vf,
                        caption=caption,
                        supports_streaming=True,
                        read_timeout=600,
                        write_timeout=600
                    )
                logger.info(f"✅ [RAW VIDEO DISPATCH] Sent local raw video to chat {chat_id} for '{identifier}'")
                return True
            except Exception as _send_err:
                logger.warning(f"⚠️ Failed to send local raw video file: {_send_err}")

        # 2. Try sending directly via Telegram file_id (instant cloud delivery, zero bandwidth)
        entry = self.find_entry_by_shortcode(identifier) or self.lookup_downloaded_source(identifier)
        raw_file_id = None
        if entry:
            raw_file_id = (
                entry.get("media_file_ids", {}).get("raw_video_file_id") or
                entry.get("raw_video_file_id") or
                entry.get("raw_file_id")
            )

        if not raw_file_id:
            try:
                from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
                sm = TelegramSessionManager()
                for k in search_keys:
                    s = sm.get_session(k)
                    if s and s.get("raw_video_file_id"):
                        raw_file_id = s["raw_video_file_id"]
                        break
            except Exception:
                pass

        if raw_file_id:
            try:
                await bot.send_video(
                    chat_id=int(chat_id),
                    video=raw_file_id,
                    caption=caption,
                    supports_streaming=True,
                    read_timeout=600,
                    write_timeout=600
                )
                logger.info(f"✅ [RAW VIDEO DISPATCH] Sent vault raw video by file_id to chat {chat_id} for '{identifier}'")
                return True
            except Exception as _fid_err:
                logger.warning(f"⚠️ Direct file_id send failed, attempting hydration fallback: {_fid_err}")

        # 3. Fallback: on-demand vault hydration and stream
        hydrated_path = self.hydrate_raw_video_from_vault(identifier)
        if hydrated_path and os.path.exists(hydrated_path) and os.path.getsize(hydrated_path) > 1024:
            try:
                with open(hydrated_path, "rb") as vf:
                    await bot.send_video(
                        chat_id=int(chat_id),
                        video=vf,
                        caption=caption,
                        supports_streaming=True,
                        read_timeout=600,
                        write_timeout=600
                    )
                logger.info(f"✅ [RAW VIDEO DISPATCH] Sent hydrated raw video to chat {chat_id} for '{identifier}'")
                return True
            except Exception as _h_err:
                logger.error(f"❌ Failed to send hydrated raw video: {_h_err}")

        return False

    async def send_clean_video_to_user_chat(self, bot, chat_id: int | str, identifier: str) -> bool:
        """
        Sends raw watermark-cleaned video directly to the user's chat.
        Never leaks internal file_id values in the user-visible caption.
        Prioritizes:
          1. Existing local clean file (manual_<shortcode>.mp4 / auto_<shortcode>.mp4 / video_inpainted_clean.mp4)
          2. Direct file_id delivery via Telegram Bot API (instant, 0 bandwidth)
          3. On-demand vault hydration fallback via hydrate_clean_video_from_vault
        """
        if not bot or not chat_id or not identifier:
            return False

        shortcode = identifier.replace("manual_clean_", "").replace("auto_clean_", "").replace("manual_raw_", "").replace("auto_raw_", "").replace("manual_", "").replace("auto_", "").strip()
        is_auto = "auto" in str(identifier).lower()
        folder_prefix = "auto_" if is_auto else "manual_"
        clean_prefix = "auto_clean_" if is_auto else "manual_clean_"
        clean_filename = f"{clean_prefix}{shortcode}.mp4"
        caption = f"🧼 **Raw Watermark Cleaned Video**\n📁 `{clean_filename}`"
        search_keys = self._normalize_search_keys(identifier)

        # 1. Check if genuine clean video file already exists on local disk
        candidate_dirs = [
            os.path.join(_REPO_ROOT, "downloads", identifier),
            os.path.join(_REPO_ROOT, "downloads", f"{folder_prefix}{shortcode}"),
            os.path.join(_REPO_ROOT, "downloads", f"manual_{shortcode}"),
            os.path.join(_REPO_ROOT, "downloads", f"auto_{shortcode}"),
            os.path.join(_REPO_ROOT, "downloads", shortcode),
        ]
        clean_candidates = [
            clean_filename,
            f"manual_clean_{shortcode}.mp4",
            f"auto_clean_{shortcode}.mp4",
            "video_inpainted_clean.mp4",
        ]
        local_clean = None
        for c_dir in candidate_dirs:
            if c_dir and os.path.exists(c_dir):
                for c_fn in clean_candidates:
                    c_out = os.path.join(c_dir, c_fn)
                    if os.path.exists(c_out) and os.path.getsize(c_out) > 1024:
                        local_clean = c_out
                        break
            if local_clean:
                break

        if not local_clean:
            try:
                from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
                sm = TelegramSessionManager()
                for k in search_keys:
                    s = sm.get_session(k)
                    if s:
                        for s_k in ["clean_video_path", "clean_raw_path"]:
                            sp = s.get(s_k)
                            if sp and os.path.exists(sp) and os.path.getsize(sp) > 1024:
                                bname = os.path.basename(sp).lower()
                                if "_master.mp4" not in bname and ("clean" in bname or "inpainted" in bname):
                                    local_clean = sp
                                    break
                    if local_clean:
                        break
            except Exception:
                pass

        if local_clean and os.path.exists(local_clean) and os.path.getsize(local_clean) > 1024:
            try:
                real_caption = f"🧼 **Raw Watermark Cleaned Video**\n📁 `{os.path.basename(local_clean)}`"
                with open(local_clean, "rb") as vf:
                    await bot.send_video(
                        chat_id=int(chat_id),
                        video=vf,
                        caption=real_caption,
                        supports_streaming=True,
                        read_timeout=600,
                        write_timeout=600
                    )
                logger.info(f"✅ [CLEAN VIDEO DISPATCH] Sent local clean video to chat {chat_id} for '{identifier}'")
                return True
            except Exception as _send_err:
                logger.warning(f"⚠️ Failed to send local clean video file: {_send_err}")

        # 2. Try sending directly via Telegram file_id (instant cloud delivery, zero bandwidth)
        entry = self.find_entry_by_shortcode(identifier) or self.lookup_downloaded_source(identifier)
        clean_file_id = None
        if entry:
            clean_file_id = (
                entry.get("media_file_ids", {}).get("wm_clean_file_id") or
                entry.get("wm_clean_file_id")
            )

        if not clean_file_id:
            try:
                from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
                sm = TelegramSessionManager()
                for k in search_keys:
                    s = sm.get_session(k)
                    if s:
                        m_ids = s.get("media_file_ids") if isinstance(s.get("media_file_ids"), dict) else {}
                        clean_file_id = m_ids.get("wm_clean_file_id") or s.get("wm_clean_file_id")
                        if clean_file_id:
                            break
            except Exception:
                pass

        if clean_file_id:
            try:
                await bot.send_video(
                    chat_id=int(chat_id),
                    video=clean_file_id,
                    caption=caption,
                    supports_streaming=True,
                    read_timeout=600,
                    write_timeout=600
                )
                logger.info(f"✅ [CLEAN VIDEO DISPATCH] Sent vault clean video by file_id to chat {chat_id} for '{identifier}'")
                return True
            except Exception as _fid_err:
                logger.warning(f"⚠️ Direct clean file_id send failed, attempting hydration fallback: {_fid_err}")

        # 3. Fallback: on-demand vault hydration and stream
        hydrated_path = self.hydrate_clean_video_from_vault(identifier)
        if hydrated_path and os.path.exists(hydrated_path) and os.path.getsize(hydrated_path) > 1024:
            try:
                real_caption = f"🧼 **Raw Watermark Cleaned Video**\n📁 `{os.path.basename(hydrated_path)}`"
                with open(hydrated_path, "rb") as vf:
                    await bot.send_video(
                        chat_id=int(chat_id),
                        video=vf,
                        caption=real_caption,
                        supports_streaming=True,
                        read_timeout=600,
                        write_timeout=600
                    )
                logger.info(f"✅ [CLEAN VIDEO DISPATCH] Sent hydrated clean video to chat {chat_id} for '{identifier}'")
                return True
            except Exception as _h_err:
                logger.error(f"❌ Failed to send hydrated clean video: {_h_err}")

        # 4. Fallback: On-demand upfront inpainting if raw video exists on disk
        try:
            raw_cands = [
                os.path.join(_REPO_ROOT, "downloads", identifier, "video.mp4"),
                os.path.join(_REPO_ROOT, "downloads", f"{folder_prefix}{shortcode}", "video.mp4"),
                os.path.join(_REPO_ROOT, "downloads", f"manual_{shortcode}", "video.mp4"),
                os.path.join(_REPO_ROOT, "downloads", f"auto_{shortcode}", "video.mp4"),
                os.path.join(_REPO_ROOT, "downloads", shortcode, "video.mp4"),
            ]
            raw_target = next((rc for rc in raw_cands if os.path.exists(rc) and os.path.getsize(rc) > 1024), None)
            if raw_target:
                from Watermark_and_Inpainting.watermark_main import run_watermark_removal
                target_dir = os.path.dirname(raw_target)
                target_out = os.path.join(target_dir, clean_filename)
                logger.info(f"🧼 [ON-DEMAND INPAINT] Generating clean video for '{identifier}' from raw source: {raw_target}")
                inp_res, _ = run_watermark_removal(input_path=raw_target, output_path=target_out, retry_level=1)
                if inp_res and os.path.exists(inp_res) and os.path.getsize(inp_res) > 1024:
                    real_caption = f"🧼 **Raw Watermark Cleaned Video**\n📁 `{os.path.basename(inp_res)}`"
                    with open(inp_res, "rb") as vf:
                        await bot.send_video(
                            chat_id=int(chat_id),
                            video=vf,
                            caption=real_caption,
                            supports_streaming=True,
                            read_timeout=600,
                            write_timeout=600
                        )
                    logger.info(f"✅ [CLEAN VIDEO DISPATCH] Sent on-demand cleaned video to chat {chat_id} for '{identifier}'")
                    return True
        except Exception as _ondemand_err:
            logger.warning(f"On-demand clean generation notice: {_ondemand_err}")

        return False


    def lookup_downloaded_source(self, social_url: str) -> Optional[Dict[str, Any]]:
        """Lookup harvested clip entry by URL or shortcode in pool_metadata.json."""
        if not social_url:
            return None
        return self.find_entry_by_shortcode(social_url)

    def lookup_processed_reel(self, session_id: Optional[str] = None, social_url: Optional[str] = None, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Lookup processed reel entry by session_id, social_url, or user_id in pool_metadata.json."""
        try:
            from Audio_Modules.audio_pool_manager import AudioPoolManager
            pm = AudioPoolManager()
            clips = pm.metadata.get("files", {}).get("social_media_id", {})
            for stored_url, entry in clips.items():
                if session_id and (
                    session_id in stored_url
                    or session_id == entry.get("session_id")
                    or session_id == entry.get("shortcode")
                ):
                    return entry
                if social_url and (social_url.strip() in stored_url or stored_url in social_url.strip()):
                    return entry
                if user_id and str(entry.get("user_id")) == str(user_id):
                    if session_id and session_id not in stored_url and session_id != entry.get("session_id"):
                        continue
                    return entry
        except Exception as e:
            logger.debug("Notice on lookup_processed_reel: %s", e)
        return None

    async def download_audio_track_from_vault(self, bot, track_name: str, dest_dir: Optional[str] = None) -> Optional[str]:
        """On-Demand Audio Vault Fetcher via pool_metadata.json."""
        if not track_name or not bot:
            return None

        filename = os.path.basename(track_name)
        if not dest_dir:
            dest_dir = os.path.join(_REPO_ROOT, "data", "runtime_audio")
        os.makedirs(dest_dir, exist_ok=True)
        local_path = os.path.join(dest_dir, filename)

        if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
            return local_path

        file_id = None
        try:
            from Audio_Modules.audio_pool_manager import AudioPoolManager
            pm = AudioPoolManager()
            clips = pm.metadata.get("files", {}).get("social_media_id", {})
            for _url, entry in clips.items():
                media_ids = entry.get("media_file_ids") or {}
                aud_id = media_ids.get("extracted_audio_file_id") or entry.get("extracted_audio_file_id")
                if aud_id and (filename in _url or filename in str(entry.get("shortcode", ""))):
                    file_id = aud_id
                    break
        except Exception as _fe:
            logger.debug("Notice on download_audio_track_from_vault lookup: %s", _fe)

        if file_id:
            try:
                t_file = await bot.get_file(file_id)
                await t_file.download_to_drive(custom_path=local_path)
                if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
                    return local_path
            except Exception as e:
                logger.warning(f"⚠️ Vault on-demand audio fetch failed for '{filename}': {e}")
        return None

    async def sync_vault_index_from_telegram(self, bot) -> bool:
        """Startup Sync from Telegram Storage Group."""
        storage_group_id = os.getenv("TELEGRAM_STORAGE_GROUP_ID")
        if not storage_group_id or not bot:
            return False

        try:
            chat = await bot.get_chat(chat_id=int(storage_group_id))
            pinned = chat.pinned_message if hasattr(chat, "pinned_message") else None

            if pinned and pinned.document and pinned.document.file_name == "master_vault_index.json":
                doc_file = await bot.get_file(pinned.document.file_id)
                temp_down = os.path.join(DATA_DIR, "pinned_vault_down.json")
                await doc_file.download_to_drive(custom_path=temp_down)

                if os.path.exists(temp_down) and os.path.getsize(temp_down) > 50:
                    with open(temp_down, "r", encoding="utf-8") as f:
                        remote_index = json.load(f)

                    if isinstance(remote_index, dict) and "column_1_processed_reels" in remote_index:
                        self.vault_index = remote_index
                        self.vault_index["pinned_message_id"] = pinned.message_id
                        self._save_local_index()

                        # Immediately sync rejected audio blacklist if present
                        bl_file_id = remote_index.get("rejected_audio_blacklist_file_id")
                        if bl_file_id:
                            try:
                                from Audio_Modules.rejected_audio_blacklist import _BLACKLIST_FILE, merge_blacklist_from_file
                                bl_file = await bot.get_file(bl_file_id)
                                temp_bl = _BLACKLIST_FILE + ".sync.tmp"
                                await bl_file.download_to_drive(custom_path=temp_bl)
                                if os.path.exists(temp_bl):
                                    merge_blacklist_from_file(temp_bl)
                                    try:
                                        os.remove(temp_bl)
                                    except Exception:
                                        pass
                            except Exception as _sync_bl_err:
                                logger.warning(f"⚠️ [VAULT SYNC] Could not sync rejected_audio_blacklist: {_sync_bl_err}")

                        self._hydrate_local_caches()
                        return True
        except Exception as e:
            logger.warning(f"⚠️ [VAULT SYNC] Could not fetch pinned vault index: {e}")
        return False

    def _hydrate_local_caches(self):
        try:
            c1_reels = self.vault_index.get("column_1_processed_reels", {}).get("by_session_id", {})
            if c1_reels:
                from Audio_Modules.audio_pool_manager import AudioPoolManager
                import Audio_Modules.audio_pool_manager as _apm_mod
                from Audio_Modules.rejected_audio_blacklist import is_blacklisted as _is_bl
                _apm_mod._VAULT_HYDRATION_IN_PROGRESS = True
                try:
                    pm = AudioPoolManager()
                    for sess_id, rdata in c1_reels.items():
                        adata = rdata.get("audio_data") or {}
                        pool_meta = adata.get("pool_metadata") or {}
                        track_name = pool_meta.get("selected_audio_track") or pool_meta.get("selected_bgm_track")
                        if track_name and pool_meta:
                            base_track = os.path.basename(track_name)
                            fid = pool_meta.get("file_id") or pool_meta.get("telegram_file_id") or adata.get("file_id")
                            sc = pool_meta.get("shortcode") or adata.get("shortcode")
                            if _is_bl(audio_filename=base_track, audio_shortcode=sc, telegram_file_id=fid):
                                logger.info(f"🚫 [VAULT HYDRATE] Skipping blacklisted track '{base_track}' from reel session '{sess_id}'")
                                continue
                            pm._set_file_metadata(base_track, pool_meta)
                    pm.sanitize_pool_metadata_anomalies()
                    pm._save_metadata(sync_to_vault=False)
                finally:
                    _apm_mod._VAULT_HYDRATION_IN_PROGRESS = False
        except Exception as e:
            logger.debug(f"[VAULT HYDRATE] Cache hydration notice: {e}")

    def record_ingested_clip_source(
        self,
        social_url: str,
        raw_video_path: str,
        upload_fn,
        existing_raw_file_id: Optional[str] = None,
        extracted_audio_path: Optional[str] = None,
        audio_math: Optional[Dict[str, Any]] = None,
        whisper_transcript: Optional[Dict[str, Any]] = None,
        gemini_semantic: Optional[Dict[str, Any]] = None,
        file_size: Optional[int] = None,
        sha256: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Unified Storage Authority Entry Point."""
        storage_group_id = os.getenv("TELEGRAM_STORAGE_GROUP_ID")
        filename = os.path.basename(raw_video_path)
        ext = os.path.splitext(filename)[1].lower()
        
        method = "sendVideo"
        file_param = "video"
        if ext in [".mp3", ".m4a", ".aac", ".flac", ".wav"]:
            method = "sendAudio"
            file_param = "audio"
        elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
            method = "sendPhoto"
            file_param = "photo"
        elif ext not in [".mp4", ".mkv", ".mov", ".webm", ".avi"]:
            method = "sendDocument"
            file_param = "document"

        caption = f"🎬 {filename}\n🔗 `{social_url}`" + (f"\n👤 User: `{user_id}`" if user_id else "")
        
        raw_file_id = existing_raw_file_id
        extracted_audio_file_id = None

        if storage_group_id and upload_fn:
            try:
                if not raw_file_id and os.path.exists(raw_video_path):
                    upload_res = upload_fn(method, storage_group_id, file_param, raw_video_path, caption=caption)
                    if upload_res and isinstance(upload_res, dict):
                        raw_file_id = upload_res.get(file_param, {}).get("file_id") or (upload_res.get("document", {}).get("file_id") if upload_res.get("document") else None)
                
                if extracted_audio_path and os.path.exists(extracted_audio_path):
                    audio_filename = os.path.basename(extracted_audio_path)
                    audio_caption = f"🎵 [EXTRACTED AUDIO] `{audio_filename}`\n🔗 `{social_url}`"
                    audio_upload_res = upload_fn("sendAudio", storage_group_id, "audio", extracted_audio_path, caption=audio_caption)
                    if audio_upload_res and isinstance(audio_upload_res, dict):
                        extracted_audio_file_id = audio_upload_res.get("audio", {}).get("file_id") or (audio_upload_res.get("document", {}).get("file_id") if audio_upload_res.get("document") else None)
            except Exception as _up_err:
                logger.warning("⚠️ Storage Group upload warning: %s", _up_err)

        canonical_url = canonicalize_social_url(social_url)
        shortcode_val = extract_clean_shortcode(social_url)
        target_key = canonical_url or social_url

        clip_entry = {
            "social_media_id": target_key,
            "shortcode": shortcode_val,
            "raw_video_file_id": raw_file_id,
            "extracted_audio_file_id": extracted_audio_file_id,
            "file_name": filename,
            "file_size": file_size or (os.path.getsize(raw_video_path) if os.path.exists(raw_video_path) else 0),
            "sha256": sha256 or "",
            "downloaded_at": time.time(),
            "user_id": user_id,
            "audio_math": audio_math or {},
            "whisper_transcript": whisper_transcript or {},
            "gemini_semantic_intelligence": gemini_semantic or {}
        }

        try:
            from Audio_Modules.audio_pool_manager import AudioPoolManager
            pm = AudioPoolManager()

            # 1. Unify storage: save into authoritative files.social_media_id
            social_dict = pm.metadata.setdefault("files", {}).setdefault("social_media_id", {})
            existing_key = None
            if shortcode_val:
                for k, v in list(social_dict.items()):
                    if isinstance(v, dict):
                        v_sc = v.get("shortcode") or extract_clean_shortcode(k) or extract_clean_shortcode(v.get("social_media_id"))
                        if v_sc and v_sc.lower() == shortcode_val.lower():
                            existing_key = k
                            break

            target_entry = social_dict.get(existing_key) if existing_key else social_dict.get(target_key, {})
            target_entry.update(clip_entry)
            target_entry["media_file_ids"] = {
                "raw_video_file_id": raw_file_id,
                "extracted_audio_file_id": extracted_audio_file_id,
            }
            if existing_key and existing_key != target_key:
                social_dict.pop(existing_key, None)
            social_dict[target_key] = target_entry

            # 2. Register extracted audio into candidate audio pool
            if extracted_audio_file_id and shortcode_val:
                audio_track_name = f"{shortcode_val}.wav"
                files_dict = pm.metadata.setdefault("files", {})
                if audio_track_name not in files_dict or not isinstance(files_dict[audio_track_name], dict):
                    files_dict[audio_track_name] = {}
                track_entry = files_dict[audio_track_name]
                track_entry["shortcode"] = shortcode_val
                track_entry["file_id"] = extracted_audio_file_id
                track_entry["telegram_file_id"] = extracted_audio_file_id
                track_entry["filename"] = audio_track_name
                if audio_math:
                    track_entry["bpm"] = float(audio_math.get("tempo_bpm", 120.0) or 120.0)
                    track_entry["energy"] = float(audio_math.get("avg_energy", 0.7) or 0.7)
                    track_entry["duration"] = float(audio_math.get("duration", audio_math.get("duration_sec", 15.0)) or 15.0)
                    track_entry["vibe"] = audio_math.get("vibe", "energetic")
                track_entry.setdefault("usage_count", 0)
                track_entry.setdefault("last_used", 0.0)

            csm = pm.metadata.setdefault("clip_source_math", {})
            csm[target_key] = clip_entry
            pm._save_metadata(sync_to_vault=True)

            if storage_group_id and upload_fn and os.path.exists(pm.meta_path):
                pool_upload_res = upload_fn("sendDocument", storage_group_id, "document", pm.meta_path, caption=f"📦 **[VAULT BACKUP]** `pool_metadata.json` (Updated {time.strftime('%H:%M:%S')})")
                if pool_upload_res and isinstance(pool_upload_res, dict):
                    pool_doc_id = pool_upload_res.get("document", {}).get("file_id")
                    self.vault_index["pool_metadata_file_id"] = pool_doc_id
                    self.vault_index["metadata_pool_file_id"] = pool_doc_id
                    self.upload_and_pin_vault_index_sync(upload_fn)

            try:
                from Telegram_Storage_Modules.telegram_user_manager import USERS_JSON_PATH
                if storage_group_id and upload_fn and os.path.exists(USERS_JSON_PATH):
                    users_upload_res = upload_fn("sendDocument", storage_group_id, "document", USERS_JSON_PATH, caption=f"👤 **[VAULT BACKUP]** `telegram_users.json` (Updated {time.strftime('%H:%M:%S')})")
                    if users_upload_res and isinstance(users_upload_res, dict):
                        users_doc_id = users_upload_res.get("document", {}).get("file_id")
                        self.vault_index["telegram_users_file_id"] = users_doc_id
            except Exception as _users_err:
                logger.warning("⚠️ Could not upload telegram_users.json: %s", _users_err)
        except Exception as _pool_err:
            logger.warning("⚠️ Could not save/upload metadata_pool.json: %s", _pool_err)

        self._save_local_index()

        return {
            "raw_file_id": raw_file_id,
            "method": method,
            "file_param": file_param,
            "clip_entry": clip_entry,
        }

    def update_inpainted_clean_source_in_vault(self, clean_video_path: str, clip_folder_name: str) -> Optional[str]:
        """Uploads clean video_inpainted_clean.mp4 to Telegram Storage Group and updates wm_clean_file_id in pool_metadata.json."""
        if not clean_video_path or not os.path.exists(clean_video_path):
            return None

        storage_group_id = os.getenv("TELEGRAM_STORAGE_GROUP_ID")
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not storage_group_id or not bot_token:
            return None

        try:
            is_auto = "auto" in clip_folder_name.lower()
            clean_sc = clip_folder_name.replace("manual_clean_", "").replace("auto_clean_", "").replace("manual_", "").replace("auto_", "").rstrip(".mp4")
            prefix = "auto_clean_" if is_auto else "manual_clean_"
            clean_display_name = f"{prefix}{clean_sc}.mp4"
            caption = f"🎬 [CLEAN INPAINTED SOURCE] `{clean_display_name}`\n🆔 `{clip_folder_name}`"

            upload_res = _send_telegram_file_sync(
                "sendVideo",
                storage_group_id,
                "video",
                clean_video_path,
                caption=caption,
                custom_filename=clean_display_name
            )
            if not upload_res or not isinstance(upload_res, dict) or not upload_res.get("ok"):
                upload_res = _send_telegram_file_sync(
                    "sendDocument",
                    storage_group_id,
                    "document",
                    clean_video_path,
                    caption=caption,
                    custom_filename=clean_display_name
                )

            clean_file_id = None
            if upload_res and isinstance(upload_res, dict) and upload_res.get("ok"):
                res_doc = upload_res.get("result", {})
                clean_file_id = res_doc.get("video", {}).get("file_id") or res_doc.get("document", {}).get("file_id")

            if clean_file_id:
                try:
                    from Audio_Modules.audio_pool_manager import AudioPoolManager
                    pm = AudioPoolManager()
                    clips = pm.metadata.setdefault("files", {}).setdefault("social_media_id", {})
                    top_clips = pm.metadata.setdefault("clips", {})
                    for pool_dict in [clips, top_clips]:
                        for url_key, entry in pool_dict.items():
                            if not isinstance(entry, dict):
                                continue
                            entry_sc = str(entry.get("shortcode", "")).strip().lower()
                            if (
                                clip_folder_name.lower() in url_key.lower() or
                                clean_sc.lower() in url_key.lower() or
                                (clean_sc and entry_sc == clean_sc.lower())
                            ):
                                m_ids = entry.setdefault("media_file_ids", {})
                                m_ids["wm_clean_file_id"] = clean_file_id
                                entry["wm_clean_file_id"] = clean_file_id
                    pm._save_metadata(sync_to_vault=True)
                    logger.info(f"✅ [VAULT CLEAN INDEX] Saved wm_clean_file_id to pool_metadata.json: {clean_file_id[:15]}...")
                except Exception as _pe:
                    logger.debug("Notice updating clean_file_id in pool_metadata: %s", _pe)

                return clean_file_id
        except Exception as _e:
            logger.warning("⚠️ Could not upload/update clean video in Telegram vault: %s", _e)

        return None

    def sync_visual_pool_metadata(self, clip_id: str, clip_data: Dict[str, Any]) -> bool:
        """Visual intelligence is merged directly into pool_metadata.json."""
        return True

    def upload_visual_pool_metadata_backup(self) -> bool:
        """Visual intelligence is backed up with pool_metadata.json."""
        return True

    def enrich_and_update_pool_metadata(self, social_url: str = "", clip_dir: str = "") -> bool:
        """
        Syncs extracted audio file_ids and harvested clip metadata into pool_metadata.json.
        """
        try:
            from Audio_Modules.audio_pool_manager import AudioPoolManager
            pm = AudioPoolManager()
            pm.hydrate_harvested_clip_metadata()
            return True
        except Exception as e:
            logger.warning("⚠️ [VAULT] Error enriching pool metadata: %s", e)
            return False

    async def record_downloaded_source(
        self,
        bot=None,
        social_url: str = "",
        session_id: str = "",
        raw_video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        beat_math: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        pin_now: bool = True,
        raw_file_id: Optional[str] = None,
        audio_file_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record downloaded raw video and audio into Telegram Storage Group & pool_metadata.json."""
        storage_group_id = os.getenv("TELEGRAM_STORAGE_GROUP_ID")

        canonical_url = canonicalize_social_url(social_url)
        sc_val = extract_clean_shortcode(social_url)
        if not sc_val and raw_video_path:
            parent_name = os.path.basename(os.path.dirname(raw_video_path))
            sc_val = extract_clean_shortcode(parent_name)
        if not canonical_url and sc_val:
            canonical_url = canonicalize_social_url(sc_val)
        target_key = canonical_url or social_url

        is_auto = ("auto" in (session_id or "").lower()) or ("auto" in (raw_video_path or "").lower()) or (user_id == "auto")
        prefix = "auto_raw_" if is_auto else "manual_raw_"
        raw_display_name = f"{prefix}{sc_val}.mp4" if sc_val else f"{prefix}{os.path.basename(raw_video_path)}"

        if storage_group_id and bot:
            try:
                if raw_video_path and os.path.exists(raw_video_path):
                    with open(raw_video_path, "rb") as rf:
                        caption_text = f"📥 **[VAULT RAW SOURCE]** `{raw_display_name}`\n🔗 `{social_url}`\n🆔 `{session_id}`" + (f"\n👤 User: `{user_id}`" if user_id else "")
                        try:
                            rmsg = await bot.send_video(
                                chat_id=int(storage_group_id),
                                video=rf,
                                filename=raw_display_name,
                                caption=caption_text,
                                read_timeout=600,
                                write_timeout=600
                            )
                        except TypeError:
                            rf.seek(0)
                            rmsg = await bot.send_video(
                                chat_id=int(storage_group_id),
                                video=rf,
                                caption=caption_text,
                                read_timeout=600,
                                write_timeout=600
                            )
                        if rmsg and rmsg.video:
                            raw_file_id = rmsg.video.file_id

                if audio_path and os.path.exists(audio_path):
                    try:
                        with open(audio_path, "rb") as af:
                            amsg = await bot.send_document(
                                chat_id=int(storage_group_id),
                                document=af,
                                filename=os.path.basename(audio_path),
                                caption=f"🎵 **[VAULT AUDIO EXTRACT]** `{os.path.basename(audio_path)}`\n🆔 `{session_id}`" + (f"\n👤 User: `{user_id}`" if user_id else ""),
                                read_timeout=600,
                                write_timeout=600
                            )
                            if amsg:
                                audio_file_id = amsg.document.file_id if amsg.document else (amsg.audio.file_id if amsg.audio else None)
                    except Exception as _aud_err:
                        logger.warning(f"❌ [VAULT AUDIO ERROR] Audio upload failed: {_aud_err}")
            except Exception as e:
                logger.warning(f"⚠️ Vault raw source upload warning: {e}")

        try:
            from Audio_Modules.audio_pool_manager import AudioPoolManager
            pm = AudioPoolManager()
            clips = pm.metadata.setdefault("files", {}).setdefault("social_media_id", {})
            
            # Shortcode-first lookup across existing keys to prevent split duplicate entries
            existing_key = None
            entry = None
            if sc_val:
                for k, v in list(clips.items()):
                    if not isinstance(v, dict):
                        continue
                    v_sc = v.get("shortcode") or extract_clean_shortcode(k) or extract_clean_shortcode(v.get("social_media_id"))
                    if (v_sc and v_sc.lower() == sc_val.lower()) or (sc_val.lower() in k.lower()):
                        existing_key = k
                        entry = v
                        break

            if entry is None:
                entry = clips.get(target_key) or clips.get(social_url) or {"social_media_id": target_key}

            if sc_val:
                entry["shortcode"] = sc_val
            entry["social_media_id"] = target_key
            if raw_video_path:
                entry["file_name"] = raw_display_name
            m_ids = entry.setdefault("media_file_ids", {})
            if raw_file_id:
                m_ids["raw_video_file_id"] = raw_file_id
                entry["raw_video_file_id"] = raw_file_id
            if audio_file_id:
                m_ids["extracted_audio_file_id"] = audio_file_id
                entry["extracted_audio_file_id"] = audio_file_id
            if beat_math:
                entry.setdefault("audio_data", {})["audio_math"] = beat_math
            if user_id:
                entry["user_id"] = user_id

            if audio_file_id and sc_val:
                audio_track_name = f"{sc_val}.wav"
                files_dict = pm.metadata.setdefault("files", {})
                if audio_track_name not in files_dict or not isinstance(files_dict[audio_track_name], dict):
                    files_dict[audio_track_name] = {}
                track_entry = files_dict[audio_track_name]
                track_entry["shortcode"] = sc_val
                track_entry["file_id"] = audio_file_id
                track_entry["telegram_file_id"] = audio_file_id
                track_entry["filename"] = audio_track_name
                if beat_math:
                    track_entry["bpm"] = float(beat_math.get("tempo_bpm", 120.0) or 120.0)
                    track_entry["energy"] = float(beat_math.get("avg_energy", 0.7) or 0.7)
                    track_entry["duration"] = float(beat_math.get("duration", beat_math.get("duration_sec", 15.0)) or 15.0)
                    track_entry["vibe"] = beat_math.get("vibe", "groove")
                track_entry.setdefault("usage_count", 0)
                track_entry.setdefault("last_used", 0.0)

            # Remove old dirty/duplicate key if it differed from canonical key
            if existing_key and existing_key != target_key:
                clips.pop(existing_key, None)
            if social_url and social_url != target_key:
                clips.pop(social_url, None)

            clips[target_key] = entry
            pm._save_metadata(sync_to_vault=True)
        except Exception as _pe:
            logger.debug("Notice on record_downloaded_source pool_metadata save: %s", _pe)

        # NOTE: Clip data (raw_video_file_id, extracted_audio_file_id, etc.) is stored ONLY in
        # pool_metadata.json and telegram_sessions.json.
        # master_vault_index.json stores ONLY JSON file_id pointers (pool_metadata_file_id, etc.).

        # Update TelegramSessionManager with raw_video_file_id
        if raw_file_id:
            try:
                from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
                sm = TelegramSessionManager()
                if session_id:
                    sm.update_raw_video_file_id(session_id, raw_file_id)
                if sc_val:
                    sm.update_raw_video_file_id(sc_val, raw_file_id)
                    sm.update_raw_video_file_id(f"manual_{sc_val}", raw_file_id)
            except Exception as _sm_err:
                logger.debug("Notice on update_raw_video_file_id in session_manager: %s", _sm_err)

        return {
            "social_media_id": target_key,
            "session_id": session_id,
            "raw_video_file_id": raw_file_id,
            "extracted_audio_file_id": audio_file_id,
            "user_id": user_id,
        }

    async def record_processed_reel(
        self,
        bot,
        session_id: str,
        social_url: Optional[str],
        custom_title: Optional[str],
        master_video_path: str,
        clip_intel: Optional[Dict[str, Any]] = None,
        lyric_intel: Optional[Dict[str, Any]] = None,
        master_file_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record processed video reel into Telegram Storage Group & pool_metadata.json."""
        storage_group_id = os.getenv("TELEGRAM_STORAGE_GROUP_ID")

        if not master_file_id and storage_group_id and bot and master_video_path and os.path.exists(master_video_path):
            try:
                filename = os.path.basename(master_video_path)
                with open(master_video_path, "rb") as vf:
                    vmsg = await bot.send_video(
                        chat_id=int(storage_group_id),
                        video=vf,
                        caption=f"🎬 **[VAULT PROCESSED REEL]** `{filename}`\n🆔 `{session_id}`" + (f"\n👤 User: `{user_id}`" if user_id else ""),
                        read_timeout=600,
                        write_timeout=600
                    )
                    if vmsg and vmsg.video:
                        master_file_id = vmsg.video.file_id
            except Exception as _mv_err:
                logger.warning("⚠️ Could not upload processed video reel to Telegram Storage Group: %s", _mv_err)

        canonical_url = canonicalize_social_url(social_url)
        sc_val = extract_clean_shortcode(social_url) or extract_clean_shortcode(session_id)
        if not canonical_url and sc_val:
            canonical_url = canonicalize_social_url(sc_val)
        target_key = canonical_url or social_url or session_id or f"direct_upload_{int(time.time())}"

        try:
            from Audio_Modules.audio_pool_manager import AudioPoolManager
            pm = AudioPoolManager()
            clips = pm.metadata.setdefault("files", {}).setdefault("social_media_id", {})
            
            # Shortcode / session_id first lookup to prevent duplicate split entries
            existing_key = None
            entry = None
            for k, v in list(clips.items()):
                if not isinstance(v, dict):
                    continue
                v_sc = v.get("shortcode") or extract_clean_shortcode(k) or extract_clean_shortcode(v.get("social_media_id"))
                if (sc_val and v_sc and v_sc.lower() == sc_val.lower()) or (session_id and session_id in k):
                    existing_key = k
                    entry = v
                    break

            if entry is None:
                entry = clips.get(target_key) or {"social_media_id": target_key}

            if sc_val:
                entry["shortcode"] = sc_val
            if session_id:
                entry["session_id"] = session_id
            entry["social_media_id"] = target_key
            m_ids = entry.setdefault("media_file_ids", {})
            if master_file_id:
                m_ids["processed_output_file_id"] = master_file_id
            if clip_intel:
                v_data = entry.setdefault("visual_data", {})
                v_data["clip_intelligence"] = clip_intel
                sel_audio = (
                    clip_intel.get("audio_data", {}).get("selected_bgm_track")
                    or clip_intel.get("audio_data", {}).get("selected_audio_track")
                    or clip_intel.get("selected_audio_track")
                )
                if sel_audio:
                    entry["selected_audio"] = sel_audio
                    entry.setdefault("audio_data", {})["selected_audio"] = sel_audio
            if lyric_intel:
                a_data = entry.setdefault("audio_data", {})
                a_data["gemini_audio_output"] = lyric_intel
            if user_id:
                entry["user_id"] = user_id

            # Remove old dirty/duplicate key if it differed from canonical key
            if existing_key and existing_key != target_key:
                clips.pop(existing_key, None)
            if social_url and social_url != target_key:
                clips.pop(social_url, None)

            clips[target_key] = entry
            pm._save_metadata()

            # Synchronize master_file_id into active Telegram session as a resilient secondary store
            if master_file_id and session_id:
                try:
                    from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
                    TelegramSessionManager().update_processed_file_id(session_id, master_file_id)
                except Exception as _sm_rec_err:
                    logger.debug("Notice linking session processed_file_id in record_processed_reel: %s", _sm_rec_err)
        except Exception as _pe:
            logger.debug("Notice on record_processed_reel pool_metadata save: %s", _pe)

        return {
            "session_id": session_id,
            "social_media_id": target_key,
            "processed_output_file_id": master_file_id,
            "user_id": user_id,
        }

    async def update_pipeline_trajectory(
        self,
        bot,
        session_id: str,
        stage_name: str,
        stage_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """AI Trajectory Store in pool_metadata.json."""
        try:
            from Audio_Modules.audio_pool_manager import AudioPoolManager
            pm = AudioPoolManager()
            clips = pm.metadata.setdefault("files", {}).setdefault("social_media_id", {})
            target_entry = None
            for url_key, entry in clips.items():
                if session_id in url_key or entry.get("shortcode") == session_id:
                    target_entry = entry
                    break
            if not target_entry:
                target_entry = clips.setdefault(session_id, {"social_media_id": session_id})

            trajectory = target_entry.setdefault("pipeline_execution_trajectory", {
                "stage_0_intent_classification": {},
                "stage_1_visual_forensics": {},
                "stage_2_audio_intelligence": {},
                "stage_3_attempts_and_re_edits": [],
                "stage_4_final_verdict": {},
            })

            stage_key = {
                "stage_0_intent": "stage_0_intent_classification",
                "stage_1_visual": "stage_1_visual_forensics",
                "stage_2_audio": "stage_2_audio_intelligence",
                "stage_3_attempts": "stage_3_attempts_and_re_edits",
                "stage_4_verdict": "stage_4_final_verdict",
            }.get(stage_name, stage_name)

            if stage_key == "stage_3_attempts_and_re_edits":
                if not isinstance(trajectory.get("stage_3_attempts_and_re_edits"), list):
                    trajectory["stage_3_attempts_and_re_edits"] = []
                trajectory["stage_3_attempts_and_re_edits"].append(stage_data)
            else:
                trajectory[stage_key] = stage_data

            pm._save_metadata()
            return trajectory
        except Exception as e:
            logger.warning("Notice updating pipeline trajectory: %s", e)
            return None

    async def record_plan_attempt(
        self,
        bot,
        session_id: str,
        attempt_number: int,
        editing_plan: Dict[str, Any],
        user_approved: bool = False,
        user_feedback: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """RAG Creator Behavior Store in pool_metadata.json."""
        try:
            from Audio_Modules.audio_pool_manager import AudioPoolManager
            pm = AudioPoolManager()
            clips = pm.metadata.setdefault("files", {}).setdefault("social_media_id", {})
            target_entry = None
            for url_key, entry in clips.items():
                if session_id in url_key or entry.get("shortcode") == session_id:
                    target_entry = entry
                    break
            if not target_entry:
                return None

            history = target_entry.setdefault("editing_plan_history", [])
            attempt_record = {
                "attempt_number": attempt_number,
                "timestamp": time.time(),
                "user_approved": user_approved,
                "user_feedback": user_feedback or ("Approved by user" if user_approved else "Rejected/Re-edit requested"),
                "editing_plan": editing_plan or {},
            }
            history.append(attempt_record)
            target_entry["editing_plan_history"] = sorted(history, key=lambda x: int(x.get("attempt_number", 0)))
            pm._save_metadata()
            return attempt_record
        except Exception as e:
            logger.warning("Notice recording plan attempt: %s", e)
            return None

    async def _upload_and_pin_index(self, bot, storage_group_id: Optional[str]):
        """Uploads updated master_vault_index.json to TELEGRAM_STORAGE_GROUP_ID and pins it."""
        if not storage_group_id or not bot or not os.path.exists(self.index_file):
            return

        try:
            with open(self.index_file, "rb") as idf:
                doc_msg = await bot.send_document(
                    chat_id=int(storage_group_id),
                    document=idf,
                    filename="master_vault_index.json",
                    caption=f"📌 **[VAULT MASTER INDEX]** Auto-Synced\n🕒 `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
                    read_timeout=600,
                    write_timeout=600
                )
                if doc_msg and doc_msg.message_id:
                    await bot.pin_chat_message(
                        chat_id=int(storage_group_id),
                        message_id=doc_msg.message_id,
                        disable_notification=True
                    )
                    self.vault_index["pinned_message_id"] = doc_msg.message_id
        except Exception as e:
            logger.warning(f"⚠️ Vault index upload/pin notice: {e}")

    def purge_source_and_audio(
        self,
        reel_shortcode_or_url: str,
        audio_shortcode_or_track: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Purges both the processed reel entry and the selected audio track entry
        from master_vault_index.json (column_2_downloaded_sources), saves, and pins to Telegram.
        """
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

        proc_sc = _extract_sc(reel_shortcode_or_url).lower()
        reel_id_clean = str(reel_shortcode_or_url).lower().strip()

        audio_sc = ""
        if audio_shortcode_or_track:
            ab = os.path.basename(str(audio_shortcode_or_track)).lower()
            audio_sc = ab
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

        purged_vault_items = []
        # Clip data is stored ONLY in pool_metadata.json and telegram_sessions.json.
        # master_vault_index.json stores ONLY JSON file_id pointers.
        self._save_local_index()

        # Cloud sync / pin to Telegram Storage Group
        try:
            from Telegram_Storage_Modules.telegram_http import send_document
            storage_group_id = os.getenv("TELEGRAM_STORAGE_GROUP_ID")
            if storage_group_id and os.path.exists(self.index_file):
                res = send_document(
                    local_path=self.index_file,
                    caption=f"📌 **[VAULT MASTER INDEX]** Purged & Auto-Synced\n🕒 `{time.strftime('%Y-%m-%d %H:%M:%S')}`"
                )
                if res and isinstance(res, dict):
                    msg_id = res.get("message_id")
                    if msg_id:
                        self.vault_index["pinned_message_id"] = msg_id
                        self._save_local_index()
                        pin_ok = False
                        try:
                            from Telegram_Storage_Modules.telegram_http import pin_message
                            pin_ok = pin_message(message_id=int(msg_id), chat_id=str(storage_group_id))
                        except Exception as _p_err:
                            logger.warning("Notice on purge pinChatMessage call: %s", _p_err)
                        if pin_ok:
                            _LAST_VAULT_PIN_TIMESTAMP = time.time()
                            logger.info("📌 [VAULT PURGE PIN SUCCESS] Uploaded & PINNED updated master_vault_index.json")
                        else:
                            logger.warning("⚠️ [VAULT PURGE PIN NOTICE] Uploaded updated master_vault_index.json but pinChatMessage was not confirmed.")
        except Exception as _sync_e:
            logger.warning(f"⚠️ Vault purge sync notice: {_sync_e}")

        return {
            "status": "success",
            "purged_count": len(purged_vault_items),
            "purged_items": purged_vault_items
        }


# ── ADVISORY LOCK ─────────────────────────────────────────────────────────────

_LOCK_TTL_SEC = float(os.getenv("VAULT_LOCK_TTL_SEC", "45"))
_LOCK_MAX_WAIT_SEC = float(os.getenv("VAULT_LOCK_MAX_WAIT_SEC", "60"))
_LOCK_POLL_SEC = 2.0

_LOCK_HOLDER_ID: Optional[str] = None
_LOCK_HOLDER_LOCK = threading.Lock()


def _get_holder_id() -> str:
    global _LOCK_HOLDER_ID
    with _LOCK_HOLDER_LOCK:
        if _LOCK_HOLDER_ID is None:
            node = os.getenv("COMPUTERNAME") or os.getenv("HOSTNAME") or "host"
            _LOCK_HOLDER_ID = f"{node}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        return _LOCK_HOLDER_ID


def _peek_pinned_lock_state(last_known_msg_id: Optional[int]) -> Dict[str, Any]:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    storage_group_id = os.getenv("TELEGRAM_STORAGE_GROUP_ID", "").strip()
    if not bot_token or not storage_group_id:
        return {"lock": None, "msg_id": None}

    try:
        url = f"https://api.telegram.org/bot{bot_token}/getChat?chat_id={storage_group_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "AMTCE-VaultLock/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        if not data.get("ok"):
            return {"lock": None, "msg_id": last_known_msg_id}

        pinned = data.get("result", {}).get("pinned_message") or {}
        current_msg_id = pinned.get("message_id")

        if current_msg_id and current_msg_id == last_known_msg_id:
            indexer = TelegramVaultIndexer()
            return {"lock": indexer.vault_index.get("lock"), "msg_id": current_msg_id}

        doc = pinned.get("document", {})
        file_id = doc.get("file_id") if doc.get("file_name") in (
            "master_vault_index.json", "telegram_media_index.json") else None

        if file_id:
            indexer = TelegramVaultIndexer()
            if indexer.download_vault_file_by_id(file_id, indexer.index_file):
                indexer.vault_index = indexer._load_local_index()
                return {"lock": indexer.vault_index.get("lock"), "msg_id": current_msg_id}

        return {"lock": None, "msg_id": current_msg_id}

    except Exception as e:
        logger.debug("[vault_lock] _peek_pinned_lock_state error: %s", e)
        return {"lock": None, "msg_id": last_known_msg_id}


def acquire_lock(purpose: str = "", ttl_sec: float = _LOCK_TTL_SEC,
                 max_wait_sec: float = _LOCK_MAX_WAIT_SEC) -> Optional[str]:
    holder = _get_holder_id()
    deadline = time.time() + max_wait_sec

    indexer = TelegramVaultIndexer()
    last_msg_id = indexer.vault_index.get("pinned_message_id")

    while time.time() < deadline:
        peek = _peek_pinned_lock_state(last_msg_id)
        last_msg_id = peek.get("msg_id") or last_msg_id
        lock = peek.get("lock")
        now = time.time()

        lock_free = (
            lock is None
            or not lock.get("held_by")
            or float(lock.get("expires_at", 0)) < now
        )

        if lock_free:
            indexer2 = TelegramVaultIndexer()
            indexer2.sync_pinned_index_from_telegram_sync()
            indexer2.vault_index["lock"] = {
                "held_by": holder,
                "purpose": purpose,
                "acquired_at": now,
                "expires_at": now + ttl_sec,
            }
            indexer2._save_local_index()
            try:
                indexer2.upload_and_pin_vault_index_sync(
                    upload_fn=lambda method, chat_id, file_key, file_path, caption=None:
                        _send_telegram_file_sync(method, chat_id, file_key, file_path, caption)
                )
            except Exception as _push_err:
                logger.debug("[vault_lock] push during acquire: %s", _push_err)

            confirm = _peek_pinned_lock_state(None)
            confirm_lock = confirm.get("lock") or {}
            if confirm_lock.get("held_by") == holder:
                return holder

        time.sleep(_LOCK_POLL_SEC)

    return None


def release_lock(holder: str) -> bool:
    indexer = TelegramVaultIndexer()
    indexer.sync_pinned_index_from_telegram_sync()
    lock = indexer.vault_index.get("lock") or {}

    if lock.get("held_by") != holder:
        return False

    indexer.vault_index["lock"] = None
    indexer._save_local_index()
    try:
        indexer.upload_and_pin_vault_index_sync(
            upload_fn=lambda method, chat_id, file_key, file_path, caption=None:
                _send_telegram_file_sync(method, chat_id, file_key, file_path, caption)
        )
    except Exception as _rel_err:
        logger.debug("[vault_lock] push during release: %s", _rel_err)

    return True


class vault_session:
    def __init__(self, purpose: str = "", ttl_sec: float = _LOCK_TTL_SEC,
                 max_wait_sec: float = _LOCK_MAX_WAIT_SEC):
        self.purpose = purpose
        self.ttl_sec = ttl_sec
        self.max_wait_sec = max_wait_sec
        self._holder: Optional[str] = None

    def __enter__(self) -> bool:
        self._holder = acquire_lock(self.purpose, self.ttl_sec, self.max_wait_sec)
        return self._holder is not None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._holder:
            release_lock(self._holder)
        return False


