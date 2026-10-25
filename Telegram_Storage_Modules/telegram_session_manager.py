"""
Telegram_Storage_Modules / telegram_session_manager.py
========================================================================
Tracks reel review sessions, message IDs, approval states, and custom titles.
Saves session state persistently to `data/telegram_sessions.json` and syncs to Storage Group.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("telegram_session_manager")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
SESSIONS_FILE = os.path.join(DATA_DIR, "telegram_sessions.json")


class TelegramSessionManager:
    """
    Manages persistent Telegram approval sessions and title capture states.
    """

    def __init__(self, sessions_file: str = SESSIONS_FILE):
        self.sessions_file = sessions_file
        self._last_vault_sync_time = 0.0
        os.makedirs(os.path.dirname(self.sessions_file), exist_ok=True)
        self.sessions: Dict[str, Dict[str, Any]] = self._load_sessions()

    def _load_sessions(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Could not load telegram sessions file: {e}")
        return {}

    def _save_sessions(self):
        try:
            with open(self.sessions_file, "w", encoding="utf-8") as f:
                json.dump(self.sessions, f, indent=2, ensure_ascii=False)
            
            # Automatic sync to Telegram Storage Group Vault (debounced: max once per 15s to prevent pin flood 429)
            try:
                now = time.time()
                if (now - self._last_vault_sync_time) >= 15.0:
                    from Telegram_Storage_Modules.telegram_user_manager import _upload_file_to_telegram_storage
                    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
                    storage_group_id = os.getenv("TELEGRAM_STORAGE_GROUP_ID")
                    if storage_group_id and os.path.exists(self.sessions_file):
                        caption = f"📝 **[VAULT BACKUP]** `telegram_sessions.json` (Updated {time.strftime('%H:%M:%S')})"
                        ts_doc_id = _upload_file_to_telegram_storage(self.sessions_file, caption=caption)
                        if ts_doc_id:
                            self._last_vault_sync_time = now
                            indexer = TelegramVaultIndexer()
                            indexer.vault_index["telegram_sessions_file_id"] = ts_doc_id
                            indexer._save_local_index()
                            indexer.upload_and_pin_vault_index_sync()
                            logger.info("✅ [SESSION VAULT BACKUP] Uploaded & PINNED updated telegram_sessions.json to Storage Group (file_id: %s)", ts_doc_id[:15])
            except Exception as _ts_err:
                logger.debug("Notice uploading telegram_sessions.json to vault: %s", _ts_err)
        except Exception as e:
            logger.error(f"❌ Failed to save telegram sessions file: {e}")

    def create_session(
        self,
        video_path: str,
        creator: str = "General",
        clip_id: Optional[str] = None,
        raw_video_path: Optional[str] = None,
        requestor_chat_id: Optional[int] = None,
        selected_audio: Optional[str] = None,
        raw_video_file_id: Optional[str] = None,
        social_url: Optional[str] = None,
        processed_file_id: Optional[str] = None
    ) -> str:
        """
        Creates a new review session for a rendered video reel.
        Robustly resolves clip_id, raw_video_path, selected_audio, and raw_video_file_id.
        """
        bname = os.path.basename(video_path)
        if not clip_id or clip_id == "Processed Shorts":
            if "_master.mp4" in bname:
                clip_id = bname.replace("_master.mp4", "")
            else:
                clip_id = os.path.basename(os.path.dirname(video_path))

        if not raw_video_path and clip_id:
            clean_cid = clip_id.replace("manual_", "").strip()
            for cand_path in [
                os.path.join(_REPO_ROOT, "downloads", clip_id, "video.mp4"),
                os.path.join(_REPO_ROOT, "downloads", clean_cid, "video.mp4"),
                os.path.join(_REPO_ROOT, "downloads", f"manual_{clean_cid}", "video.mp4"),
            ]:
                if os.path.exists(cand_path) and os.path.getsize(cand_path) > 1024:
                    raw_video_path = cand_path
                    break

        if not raw_video_file_id and clip_id:
            try:
                from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
                vi = TelegramVaultIndexer()
                clean_cid = clip_id.replace("manual_", "").strip()
                entry = vi.find_entry_by_shortcode(clean_cid) or vi.find_entry_by_shortcode(clip_id)
                if entry:
                    raw_video_file_id = (
                        entry.get("media_file_ids", {}).get("wm_clean_file_id") or
                        entry.get("wm_clean_file_id") or
                        entry.get("media_file_ids", {}).get("raw_video_file_id") or
                        entry.get("raw_video_file_id") or
                        entry.get("raw_file_id")
                    )
            except Exception:
                pass

        if not selected_audio and clip_id:
            try:
                from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore
                c_intel = ClipIntelligenceStore().load(clip_id)
                if c_intel:
                    a_data = c_intel.get("audio_data", {})
                    selected_audio = (
                        a_data.get("selected_bgm_track")
                        or a_data.get("selected_audio_track")
                        or c_intel.get("selected_audio")
                    )
            except Exception:
                pass

        sess_id = f"sess_{int(time.time())}_{abs(hash(video_path)) % 10000}"
        session_data = {
            "session_id": sess_id,
            "telegram_message_id": None,
            "requestor_chat_id": requestor_chat_id,
            "video_path": os.path.abspath(video_path),
            "raw_video_path": os.path.abspath(raw_video_path) if raw_video_path and os.path.exists(raw_video_path) else None,
            "raw_video_file_id": raw_video_file_id,
            "processed_file_id": processed_file_id,
            "master_file_id": processed_file_id,
            "social_url": social_url,
            "clip_id": clip_id,
            "creator": creator,
            "status": "AWAITING_REVIEW",
            "selected_audio": selected_audio,
            "custom_title": None,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        self.sessions[sess_id] = session_data
        logger.info(f"📝 [SESSION CREATED] {sess_id} for clip '{clip_id}' (raw_file_id={bool(raw_video_file_id)}, processed_file_id={bool(processed_file_id)}, social_url={social_url})")
        self._save_sessions()
        logger.info(f"📝 Created Telegram review session: {sess_id} for '{os.path.basename(video_path)}' (clip_id='{clip_id}', selected_audio='{selected_audio}')")
        return sess_id

    def update_raw_video_file_id(self, session_id_or_clip_id: str, raw_file_id: str):
        """Links raw_video_file_id to active session. Skips save if file_id is unchanged (prevents Telegram 429)."""
        if not raw_file_id:
            return
        sess = self.get_session(session_id_or_clip_id)
        if sess:
            if sess.get("raw_video_file_id") == raw_file_id:
                logger.debug(f"[SESSION] raw_video_file_id unchanged for {sess.get('session_id')} — skipping redundant cloud save.")
                return
            sess["raw_video_file_id"] = raw_file_id
            sess["updated_at"] = time.time()
            self._save_sessions()
            logger.info(f"💾 [SESSION] Linked raw_video_file_id to session {sess.get('session_id')}")

    def update_processed_file_id(self, session_id_or_clip_id: str, processed_file_id: str):
        """Links rendered master reel processed_file_id to active session. Skips save if file_id is unchanged."""
        if not processed_file_id:
            return
        sess = self.get_session(session_id_or_clip_id)
        if sess:
            if sess.get("processed_file_id") == processed_file_id and sess.get("master_file_id") == processed_file_id:
                logger.debug(f"[SESSION] processed_file_id unchanged for {sess.get('session_id')} — skipping redundant save.")
                return
            sess["processed_file_id"] = processed_file_id
            sess["master_file_id"] = processed_file_id
            sess["updated_at"] = time.time()
            self._save_sessions()
            logger.info(f"💾 [SESSION] Linked processed_file_id to session {sess.get('session_id')}")


    def update_message_id(self, session_id: str, message_id: int):
        """Links Telegram message ID to session."""
        if session_id in self.sessions:
            self.sessions[session_id]["telegram_message_id"] = message_id
            self.sessions[session_id]["updated_at"] = time.time()
            self._save_sessions()

    def set_awaiting_title(self, session_id: str) -> bool:
        """Marks session as awaiting custom title from user."""
        if session_id in self.sessions:
            self.sessions[session_id]["status"] = "AWAITING_TITLE"
            self.sessions[session_id]["updated_at"] = time.time()
            self._save_sessions()
            return True
        return False

    def revert_awaiting_title(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Reverts session status from AWAITING_TITLE back to AWAITING_REVIEW."""
        if session_id in self.sessions:
            self.sessions[session_id]["status"] = "AWAITING_REVIEW"
            self.sessions[session_id]["updated_at"] = time.time()
            self._save_sessions()
            logger.info(f"↩️ Session {session_id} reverted from AWAITING_TITLE back to AWAITING_REVIEW.")
            return self.sessions[session_id]
        return None

    def set_approved_title(self, session_id: str, custom_title: str) -> Optional[Dict[str, Any]]:
        """Sets custom title, marks session APPROVED, and returns session data."""
        if session_id in self.sessions:
            self.sessions[session_id]["custom_title"] = custom_title
            self.sessions[session_id]["status"] = "APPROVED"
            self.sessions[session_id]["updated_at"] = time.time()
            self._save_sessions()
            logger.info(f"✅ Session {session_id} approved with title: '{custom_title}'")
            return self.sessions[session_id]
        return None

    def set_rejected(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Marks session REJECTED and returns session data."""
        if session_id in self.sessions:
            self.sessions[session_id]["status"] = "REJECTED"
            self.sessions[session_id]["updated_at"] = time.time()
            self._save_sessions()
            logger.info(f"🗑️ Session {session_id} marked REJECTED.")
            return self.sessions[session_id]
        return None

    def record_rendered_attempt(self, chat_id_or_sess_id: Any, video_path: str, clip_id: Optional[str] = None):
        """Records rendered attempt path into active session attempt history matching the specific clip."""
        str_key = str(chat_id_or_sess_id)
        target_sess = None
        if str_key in self.sessions:
            target_sess = self.sessions[str_key]
        else:
            cid = clip_id or os.path.basename(video_path).replace("_master.mp4", "")
            clean_cid = cid.replace("manual_", "").replace("auto_", "").strip()
            matching = [s for s in self.sessions.values() if str(s.get("requestor_chat_id")) == str_key]
            # Only match sessions belonging to this clip to prevent cross-session contamination
            clip_matching = [
                s for s in matching
                if s.get("clip_id") in (cid, f"manual_{clean_cid}", clean_cid)
                or (s.get("video_path") and (clean_cid in s.get("video_path") or cid in s.get("video_path")))
            ]
            if clip_matching:
                clip_matching.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
                target_sess = clip_matching[0]

        if target_sess:
            history = target_sess.setdefault("attempt_history", [])
            if video_path not in history:
                history.append(video_path)
            target_sess["updated_at"] = time.time()
            self._save_sessions()

    def register_retry(self, chat_id_or_user_id: Any) -> Tuple[bool, int]:
        """Registers a retry attempt for the active session. Returns (should_retry, new_count)."""
        str_key = str(chat_id_or_user_id)
        target_sess = None
        if str_key in self.sessions:
            target_sess = self.sessions[str_key]
        else:
            matching = [s for s in self.sessions.values() if str(s.get("requestor_chat_id")) == str_key]
            if matching:
                matching.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
                target_sess = matching[0]

        if target_sess:
            count = target_sess.get("retry_count", 0) + 1
            target_sess["retry_count"] = count
            target_sess["updated_at"] = time.time()
            self._save_sessions()
            from Core_Modules import MAX_RETRIES
            return (count <= MAX_RETRIES, count)

        return (True, 1)

    def get_pending_title_session(self, chat_id: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Returns the most recent session waiting for a custom title, optionally filtered by chat_id."""
        awaiting = [
            s for s in self.sessions.values()
            if s.get("status") == "AWAITING_TITLE"
        ]
        if awaiting:
            awaiting.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
            if chat_id is not None:
                chat_str = str(chat_id).strip()
                for s in awaiting:
                    if str(s.get("requestor_chat_id", "")).strip() == chat_str:
                        return s
            return awaiting[0]
        return None

    def get_session(self, session_id_or_clip_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves session by session_id, or by matching clip_id (most recent active session first)."""
        if not session_id_or_clip_id:
            return None
        sid = str(session_id_or_clip_id).strip()
        if sid in self.sessions:
            return self.sessions[sid]

        sid_clean = sid.replace("manual_", "").strip()
        matches = []
        for s in self.sessions.values():
            c_id = str(s.get("clip_id", "")).strip()
            c_clean = c_id.replace("manual_", "").strip()
            if sid == c_id or (sid_clean and sid_clean == c_clean):
                matches.append((3, s))
            elif sid in c_id or (c_clean and c_clean in sid):
                matches.append((2, s))

        if matches:
            matches.sort(
                key=lambda item: (
                    item[0],
                    bool(item[1].get("raw_video_file_id") or (item[1].get("raw_video_path") and os.path.exists(item[1]["raw_video_path"]))),
                    item[1].get("status") != "REJECTED",
                    item[1].get("updated_at", 0)
                ),
                reverse=True
            )
            return matches[0][1]
        return None


# Global singleton instance
session_manager = TelegramSessionManager()
