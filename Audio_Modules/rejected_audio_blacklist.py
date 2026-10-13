"""
Audio_Modules/rejected_audio_blacklist.py
==========================================
Persistent blacklist of audio tracks that were admin-rejected.

Any audio track that survives a purge attempt with blacklist=True
is recorded here by filename, audio_shortcode, AND Telegram file_id.

select_best_audio_for_clip() and get_vault_audio_pool() both call
is_blacklisted() before adding any candidate to the selection set.

This makes rejection IRREVERSIBLE regardless of:
  - Pool rebuilds
  - Vault resync from Telegram
  - Scheduler re-indexing
  - Apify re-scraping the same source
"""

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("rejected_audio_blacklist")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BLACKLIST_FILE = os.path.join(_REPO_ROOT, "data", "rejected_audio_blacklist.json")
_lock = threading.RLock()

# Module-level in-memory cache (loaded once per process, reloaded on file mtime change)
_cache: Optional[Dict[str, Any]] = None
_cache_mtime: float = 0.0


def clean_track_identifier(val: Optional[str]) -> str:
    """Normalizes an audio track filename, shortcode, URL, or path into a canonical clean key.
    Strips prefixes (vault_bgm_, bgm_, extracted_, audio_, manual_) and extensions (.wav, .mp3, etc.).
    Safely extracts shortcodes from Instagram/YouTube/TikTok URLs.
    """
    if not val:
        return ""
    val_str = str(val).strip().strip("`").rstrip("/")
    if val_str.endswith("%60"):
        val_str = val_str[:-3].strip()

    # URL extraction for reels/shorts/posts
    m_ig = re.search(r"/(?:p|reel|reels)/([A-Za-z0-9_-]+)", val_str, re.IGNORECASE)
    if m_ig:
        val_str = m_ig.group(1)
    else:
        m_yt = re.search(r"(?:shorts/|v=|youtu\.be/)([A-Za-z0-9_-]{11})", val_str, re.IGNORECASE)
        if m_yt:
            val_str = m_yt.group(1)

    s = os.path.basename(val_str).lower().strip().split("?")[0]
    for pfx in ("vault_bgm_", "bgm_", "extracted_", "audio_", "manual_"):
        if s.startswith(pfx):
            s = s[len(pfx):]
    for ext in (".wav", ".mp3", ".m4a", ".aac", ".mp4", ".ogg", ".flac"):
        if s.endswith(ext):
            s = s[:-len(ext)]
    return s.strip()


def _load() -> Dict[str, Any]:
    global _cache, _cache_mtime
    if os.path.isfile(_BLACKLIST_FILE):
        try:
            mtime = os.path.getmtime(_BLACKLIST_FILE)
            if _cache is not None and mtime <= _cache_mtime:
                return _cache
            with open(_BLACKLIST_FILE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
                _cache_mtime = mtime
                return _cache
        except Exception as e:
            logger.warning(f"[Blacklist] Failed to load blacklist file: {e}")
            if _cache is not None:
                return _cache
    if _cache is not None:
        return _cache
    _cache = {"version": 1, "entries": {}}
    _cache_mtime = 0.0
    return _cache


def _save(data: Dict[str, Any]) -> None:
    global _cache, _cache_mtime
    _cache = data
    try:
        os.makedirs(os.path.dirname(_BLACKLIST_FILE), exist_ok=True)
        with open(_BLACKLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        _cache_mtime = os.path.getmtime(_BLACKLIST_FILE)
    except Exception as e:
        logger.warning(f"[Blacklist] Failed to save blacklist file: {e}")


def add(
    audio_filename: Optional[str],
    audio_shortcode: Optional[str] = None,
    telegram_file_id: Optional[str] = None,
    reason: str = "admin_rejected",
) -> None:
    """
    Permanently blacklist an audio track.
    Called from purge_media_and_audio_entries() on every admin rejection.

    Keyed by audio_filename (canonical). Also indexes by shortcode and file_id
    so is_blacklisted() can catch vault_bgm_*.wav mismatches.
    """
    with _lock:
        data = _load()
        entries: Dict[str, Any] = data.setdefault("entries", {})

        fname_key = os.path.basename(str(audio_filename)).lower().strip() if audio_filename else ""
        sc_key = str(audio_shortcode).lower().strip() if audio_shortcode else ""

        # Derive clean shortcode from filename if not explicitly given
        derived_sc = clean_track_identifier(fname_key)
        if not sc_key and derived_sc:
            sc_key = derived_sc

        if not fname_key and not sc_key:
            logger.warning("[Blacklist] add() called with no identifiable audio — skipping.")
            return

        key = fname_key or sc_key
        record = {
            "filename": fname_key,
            "shortcode": sc_key,
            "telegram_file_id": telegram_file_id or "",
            "reason": reason,
            "blacklisted_at": time.time(),
        }
        entries[key] = record

        # Also index by derived/explicit clean shortcode so reverse lookups always work
        if sc_key and sc_key != key:
            entries[sc_key] = record
        if derived_sc and derived_sc != key and derived_sc != sc_key:
            entries[derived_sc] = record

        _save(data)
        logger.info(
            f"🚫 [Blacklist] Permanently blacklisted audio: filename='{fname_key}' "
            f"shortcode='{sc_key}' file_id='{telegram_file_id}' reason='{reason}'"
        )

        # Cloud sync to Telegram Storage Group Vault
        try:
            sync_blacklist_to_vault()
        except Exception as _sync_err:
            logger.debug("[Blacklist] Vault sync notice: %s", _sync_err)


def sync_blacklist_to_vault() -> Optional[str]:
    """
    Uploads rejected_audio_blacklist.json to Telegram Storage Group via telegram_http.py
    and updates rejected_audio_blacklist_file_id in master_vault_index.json.
    """
    try:
        if not os.path.isfile(_BLACKLIST_FILE):
            return None

        from Telegram_Storage_Modules.telegram_http import send_document, extract_file_id
        from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer

        res = send_document(
            local_path=_BLACKLIST_FILE,
            caption=f"🚫 **[VAULT BACKUP]** `rejected_audio_blacklist.json` (Updated {time.strftime('%H:%M:%S')})"
        )
        if res:
            doc_id = extract_file_id(res)
            if doc_id:
                indexer = TelegramVaultIndexer()
                indexer.vault_index["rejected_audio_blacklist_file_id"] = doc_id
                indexer._save_local_index()
                indexer.upload_and_pin_vault_index_sync()
                logger.info("✅ [BLACKLIST VAULT BACKUP] Uploaded & PINNED rejected_audio_blacklist.json in Telegram Storage Group (file_id: %s)", doc_id[:15])
                return doc_id
    except Exception as _e:
        logger.warning("⚠️ Notice triggering blacklist vault sync: %s", _e)
    return None


def merge_blacklist_from_file(remote_file_path: str) -> bool:
    """
    Merges an incoming rejected_audio_blacklist.json (e.g. downloaded from Telegram vault)
    into the local blacklist store.
    """
    if not os.path.isfile(remote_file_path):
        return False
    try:
        with open(remote_file_path, "r", encoding="utf-8") as f:
            incoming = json.load(f)
        if not isinstance(incoming, dict):
            return False

        incoming_entries = incoming.get("entries", {})
        if not incoming_entries:
            return True

        with _lock:
            data = _load()
            local_entries = data.setdefault("entries", {})
            for k, v in incoming_entries.items():
                if k not in local_entries:
                    local_entries[k] = v
                elif isinstance(v, dict):
                    for subk, subv in v.items():
                        if subv and not local_entries[k].get(subk):
                            local_entries[k][subk] = subv
            _save(data)
            logger.info("🔄 [Blacklist] Merged %d entries from remote vault file into local blacklist.", len(incoming_entries))
            return True
    except Exception as e:
        logger.warning("⚠️ [Blacklist] Failed to merge remote blacklist file: %s", e)
        return False


def is_blacklisted(
    audio_filename: Optional[str],
    audio_shortcode: Optional[str] = None,
    telegram_file_id: Optional[str] = None,
) -> bool:
    """
    Returns True if any of the given identifiers matches a blacklisted entry.
    Called before any audio candidate is added to the selection set.
    """
    with _lock:
        data = _load()
        entries: Dict[str, Any] = data.get("entries", {})

        if not entries:
            return False

        fname_raw = str(audio_filename).strip() if audio_filename else ""
        fname_base = os.path.basename(fname_raw).lower().strip() if fname_raw else ""
        sc_raw = str(audio_shortcode).lower().strip() if audio_shortcode else ""
        fid_key = str(telegram_file_id).strip() if telegram_file_id else ""

        norm_fname = clean_track_identifier(fname_base)
        norm_sc = clean_track_identifier(sc_raw) if sc_raw else norm_fname

        # 0. Fast O(1) set membership check
        b_set = get_blacklisted_shortcodes()
        for k_test in (norm_fname, norm_sc, fname_base, sc_raw):
            if k_test and k_test in b_set:
                return True

        # 1. Direct key hits
        for test_key in (fname_base, sc_raw, norm_fname, norm_sc):
            if test_key and test_key in entries:
                return True

        # 2. Telegram file_id hit (catches vault resync of same audio with new filename)
        if fid_key:
            for entry in entries.values():
                if isinstance(entry, dict) and entry.get("telegram_file_id"):
                    if str(entry.get("telegram_file_id")).strip() == fid_key:
                        return True

        # 3. Canonical normalized comparison across all blacklisted records
        for blk_key, entry in entries.items():
            if not isinstance(entry, dict):
                # Legacy scalar entry
                bl_clean = clean_track_identifier(str(blk_key))
                if norm_fname and norm_fname == bl_clean:
                    return True
                if norm_sc and norm_sc == bl_clean:
                    return True
                continue

            bl_fn = clean_track_identifier(entry.get("filename"))
            bl_sc = clean_track_identifier(entry.get("shortcode"))
            bl_k = clean_track_identifier(blk_key)

            if norm_fname and (norm_fname == bl_fn or norm_fname == bl_sc or norm_fname == bl_k):
                return True
            if norm_sc and (norm_sc == bl_fn or norm_sc == bl_sc or norm_sc == bl_k):
                return True

            # Substring safety scan (catches embedded shortcodes >= 6 chars)
            target_sc = norm_fname or norm_sc
            if target_sc and len(target_sc) >= 6:
                if bl_sc and len(bl_sc) >= 6 and (target_sc in bl_sc or bl_sc in target_sc):
                    return True
                if bl_k and len(bl_k) >= 6 and (target_sc in bl_k or bl_k in target_sc):
                    return True

        return False


def get_all() -> Dict[str, Any]:
    """Returns the full blacklist entries dict (read-only reference)."""
    with _lock:
        return dict(_load().get("entries", {}))


def get_blacklisted_shortcodes() -> Set[str]:
    """Returns a set of all blacklisted shortcodes and normalized filenames for fast O(1) membership testing."""
    with _lock:
        data = _load()
        result: Set[str] = set()
        for k, entry in data.get("entries", {}).items():
            k_clean = clean_track_identifier(str(k))
            if k_clean:
                result.add(k_clean)
            if isinstance(entry, dict):
                sc = clean_track_identifier(entry.get("shortcode"))
                fn = clean_track_identifier(entry.get("filename"))
                raw_fn = str(entry.get("filename", "")).lower().strip()
                if sc:
                    result.add(sc)
                if fn:
                    result.add(fn)
                if raw_fn:
                    result.add(raw_fn)
                    result.add(os.path.basename(raw_fn))
        return result
