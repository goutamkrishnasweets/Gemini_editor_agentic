"""
Telegram_Storage_Modules / telegram_user_manager.py
===========================================================================
Manages user registration, nickname setup, password creation, hashing,
session verification, OTP recovery, and persistent storage in data/telegram_users.json.
"""

import os
import re
import time
import json
import secrets
import logging
import hashlib
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger("telegram_user_manager")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_REPO_ROOT, "data")
USERS_JSON_PATH = os.path.join(DATA_DIR, "telegram_users.json")


def _hash_password(password: str) -> str:
    """Computes SHA-256 hash for secure password storage."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_all_users() -> Dict[str, Dict]:
    """Loads all registered user records from data/telegram_users.json, and hydrates from USER_<id>_* environment variables."""
    users = {}
    if os.path.exists(USERS_JSON_PATH):
        try:
            with open(USERS_JSON_PATH, "r", encoding="utf-8") as f:
                users = json.load(f)
        except Exception as err:
            logger.error("Failed to load %s: %s", USERS_JSON_PATH, err)
            users = {}

    # Dynamically hydrate from any USER_<chat_id>_<var> variables in os.environ
    for k, v in os.environ.items():
        if k.startswith("USER_") and v and v.strip():
            m = re.match(r"^USER_(\d+)_(.+)$", k)
            if m:
                uid = m.group(1)
                suffix = m.group(2).lower()
                val = v.strip()
                u_rec = users.setdefault(uid, {})
                if suffix in ("ig_business_token", "meta_page_token", "meta_token"):
                    if not u_rec.get("ig_business_token"):
                        u_rec["ig_business_token"] = val
                    if not u_rec.get("meta_page_token"):
                        u_rec["meta_page_token"] = val
                elif suffix in ("ig_business_id", "meta_page_id"):
                    if not u_rec.get("ig_business_id"):
                        u_rec["ig_business_id"] = val
                    if not u_rec.get("meta_page_id"):
                        u_rec["meta_page_id"] = val
                elif suffix in ("client_secret_json", "youtube_client_secret"):
                    if not u_rec.get("youtube_client_secret"):
                        u_rec["youtube_client_secret"] = val
                elif suffix in ("token_json", "youtube_token_json"):
                    if not u_rec.get("youtube_token_json"):
                        u_rec["youtube_token_json"] = val
                elif suffix in ("gemini_api_key",):
                    if not u_rec.get("gemini_api_key"):
                        u_rec["gemini_api_key"] = val
                elif suffix in ("apify_api_token", "apify_token"):
                    if not u_rec.get("apify_api_token"):
                        u_rec["apify_api_token"] = val
                elif suffix in ("brand_watermark_text", "brand_watermark"):
                    if not u_rec.get("brand_watermark"):
                        u_rec["brand_watermark"] = val
                elif suffix in ("telegram_public_group_id",):
                    if not u_rec.get("telegram_public_group_id"):
                        u_rec["telegram_public_group_id"] = val
                elif suffix in ("auto_input_schedule_times",):
                    if not u_rec.get("auto_input_schedule_times"):
                        u_rec["auto_input_schedule_times"] = val
                elif suffix in ("auto_scrape_limit", "autoscraplimit", "scrape_limit"):
                    if not u_rec.get("auto_scrape_limit"):
                        try:
                            u_rec["auto_scrape_limit"] = int(val)
                        except (ValueError, TypeError):
                            pass

    return users


def _upload_file_to_telegram_storage(file_path: str, caption: str = "") -> Optional[str]:
    """Uploads a file to TELEGRAM_STORAGE_GROUP_ID using telegram_http sender (with fallback)."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    storage_group_id = os.getenv("TELEGRAM_STORAGE_GROUP_ID")
    if not bot_token or not storage_group_id or not os.path.exists(file_path):
        return None
    
    # Primary sender: Telegram_Storage_Modules.telegram_http
    try:
        from Telegram_Storage_Modules.telegram_http import send_document, extract_file_id
        res = send_document(local_path=file_path, chat_id=str(storage_group_id), caption=caption)
        if res:
            doc_id = extract_file_id(res) if isinstance(res, dict) else None
            if doc_id:
                return doc_id
    except Exception as e:
        logger.warning("Notice uploading %s via telegram_http: %s", file_path, e)

    # Fallback to direct urllib multipart upload
    try:
        import uuid
        import urllib.request
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body = bytearray()
        
        # Add chat_id
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="chat_id"\r\n\r\n{storage_group_id}\r\n'.encode("utf-8"))
        
        if caption:
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="caption"\r\n\r\n{caption}\r\n'.encode("utf-8"))
        
        # Add document file
        filename = os.path.basename(file_path)
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="document"; filename="{filename}"\r\n'.encode("utf-8"))
        body.extend(b'Content-Type: application/json\r\n\r\n')
        with open(file_path, "rb") as f:
            body.extend(f.read())
        body.extend(b'\r\n')
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))
        
        url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
        req = urllib.request.Request(
            url,
            data=bytes(body),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "AMTCE-Vault-Uploader"
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("ok"):
                return data.get("result", {}).get("document", {}).get("file_id")
    except Exception as e:
        logger.warning("Notice uploading %s to Telegram Vault fallback: %s", file_path, e)
    return None


def sync_users_json_to_telegram_vault(upload_fn=None):
    """Uploads updated telegram_users.json to Storage Group & updates pinned master_vault_index.json."""
    try:
        from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
        if os.path.exists(USERS_JSON_PATH):
            caption = f"👤 **[VAULT BACKUP]** `telegram_users.json` (Updated {time.strftime('%H:%M:%S')})"
            users_doc_id = _upload_file_to_telegram_storage(USERS_JSON_PATH, caption=caption)
            if users_doc_id:
                indexer = TelegramVaultIndexer()
                indexer.vault_index["telegram_users_file_id"] = users_doc_id
                indexer._save_local_index()
                indexer.upload_and_pin_vault_index_sync(upload_fn)
                logger.info("✅ [USER VAULT BACKUP] Uploaded & PINNED updated telegram_users.json to Storage Group (file_id: %s)", users_doc_id[:15])
    except Exception as err:
        logger.warning("Notice uploading telegram_users.json to vault: %s", err)


def save_all_users(users: Dict[str, Dict], upload_fn=None, sync_to_vault: bool = True) -> bool:
    """Saves user records dictionary to telegram_users.json and syncs to Telegram Storage Group."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(USERS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False)
        
        if sync_to_vault:
            sync_users_json_to_telegram_vault(upload_fn=upload_fn)
        return True
    except Exception as err:
        logger.error("Failed to save %s: %s", USERS_JSON_PATH, err)
        return False


SESSION_TIMEOUT_HOURS = float(os.getenv("TELEGRAM_SESSION_TIMEOUT_HOURS", "168.0"))  # Default: 7 days (168 hours)


def _escape_md(text: str) -> str:
    """Escapes Markdown special characters (like underscores in usernames) to prevent parser errors."""
    return text.replace("_", "\\_").replace("*", "\\*").replace("`", "\\`").replace("[", "\\[")


def check_and_expire_session(user_record: Dict) -> bool:
    """
    Checks if authenticated user session has expired due to inactivity (>7 days / 168 hours).
    Returns True if session was expired, False if still active.
    """
    if not user_record.get("authenticated", False):
        return False
    last_act_str = user_record.get("last_active", "")
    if last_act_str:
        try:
            last_act_dt = datetime.fromisoformat(last_act_str)
            inactive_hours = (datetime.now() - last_act_dt).total_seconds() / 3600.0
            if inactive_hours > SESSION_TIMEOUT_HOURS:
                user_record["authenticated"] = False
                user_id_str = str(user_record.get("user_id") or user_record.get("chat_id"))
                users = load_all_users()
                if user_id_str in users:
                    users[user_id_str]["authenticated"] = False
                    save_all_users(users)
                logger.info("🔒 [TELEGRAM USER MANAGER] Session expired due to inactivity (%.1f hrs) for User ID %s", inactive_hours, user_id_str)
                return True
        except Exception as _ex:
            logger.debug("Session parse error: %s", _ex)
    return False


def get_or_register_user(from_user: Dict, chat_id: str, admin_chat_id: Optional[str] = None) -> Dict:
    """
    Registers a new Telegram user record if not present, updates last active timestamp,
    and checks for 7-day session inactivity expiration.
    """
    users = load_all_users()
    user_id_str = str(from_user.get("id") or chat_id)

    if user_id_str not in users:
        is_admin = bool(admin_chat_id and (user_id_str == str(admin_chat_id) or chat_id == str(admin_chat_id)))
        users[user_id_str] = {
            "chat_id": str(chat_id),
            "user_id": from_user.get("id"),
            "first_name": from_user.get("first_name", "User"),
            "username": from_user.get("username", ""),
            "nickname": "",  # Set via /setnickname <nickname>
            "role": "admin" if is_admin else "user",
            "password_hash": "",
            "recovery_otp": "",
            "authenticated": False,  # Requires 1-time Nickname & Password setup
            "scrape_count": 0,
            "daily_scrape_count": 0,
            "quota_lock_timestamp": "",
            "apify_api_token": "",
            "joined_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat()
        }
        save_all_users(users)
        logger.info("✨ [TELEGRAM USER MANAGER] Registered user: %s (@%s) [ID: %s]", from_user.get("first_name"), from_user.get("username"), user_id_str)
    else:
        # Check if session expired due to inactivity
        if users[user_id_str].get("password_hash") and users[user_id_str].get("nickname"):
            check_and_expire_session(users[user_id_str])
        else:
            users[user_id_str]["authenticated"] = False
        users[user_id_str].setdefault("scrape_count", 0)
        users[user_id_str].setdefault("daily_scrape_count", 0)
        users[user_id_str].setdefault("quota_lock_timestamp", "")
        users[user_id_str].setdefault("apify_api_token", "")
        users[user_id_str]["last_active"] = datetime.now().isoformat()
        if admin_chat_id and (user_id_str == str(admin_chat_id) or chat_id == str(admin_chat_id)):
            users[user_id_str]["role"] = "admin"
        save_all_users(users)

    return users[user_id_str]


def check_and_reset_daily_quota(user_record: Dict) -> Tuple[int, Optional[str], Optional[str]]:
    """
    Checks if 24 hours have elapsed since the 5th clip download timestamp.
    If 24h passed, auto-resets daily_scrape_count = 0.
    Returns (current_daily_count, countdown_str, reset_time_str).
    """
    lock_ts_str = user_record.get("quota_lock_timestamp", "")
    current_count = user_record.get("daily_scrape_count", 0)

    if not lock_ts_str:
        return current_count, None, None

    try:
        from datetime import datetime, timedelta
        lock_dt = datetime.fromisoformat(lock_ts_str)
        elapsed_sec = (datetime.now() - lock_dt).total_seconds()

        if elapsed_sec >= 86400.0:  # 24 hours = 86,400 seconds
            user_record["daily_scrape_count"] = 0
            user_record["quota_lock_timestamp"] = ""
            logger.info("⏳ [QUOTA RESET] 24 hours elapsed since 5th clip lock for user. Quota reset to 0/5!")
            return 0, None, None
        else:
            rem_sec = 86400.0 - elapsed_sec
            hours = int(rem_sec // 3600)
            mins = int((rem_sec % 3600) // 60)
            countdown_str = f"{hours}h {mins}m"
            reset_time_str = (lock_dt + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
            return current_count, countdown_str, reset_time_str
    except Exception as _e:
        logger.warning("Error computing daily quota reset: %s", _e)
        return current_count, None, None


def increment_user_scrape_count(user_id_str: str) -> int:
    """Increments the daily scrape count for a registered user, logging 5th clip lock timestamp."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    if user_id_str in users:
        u_rec = users[user_id_str]
        check_and_reset_daily_quota(u_rec)
        new_cnt = u_rec.get("daily_scrape_count", 0) + 1
        u_rec["daily_scrape_count"] = new_cnt
        u_rec["scrape_count"] = u_rec.get("scrape_count", 0) + 1
        if new_cnt >= 5 and not u_rec.get("quota_lock_timestamp"):
            u_rec["quota_lock_timestamp"] = datetime.now().isoformat()
            logger.info("⏱️ [QUOTA LOCK 5th CLIP] User ID %s reached 5th clip at %s. 24h timer started!", user_id_str, u_rec["quota_lock_timestamp"])
        save_all_users(users)
        logger.info("📊 [TELEGRAM USER MANAGER] User ID %s daily scrape count incremented to %d/5", user_id_str, new_cnt)
        return new_cnt
    return 0


def _is_admin_user(user_id_str: str) -> bool:
    """
    Returns True if user_id_str matches any admin ID in TELEGRAM_ADMIN_ID.
    Strictly checks TELEGRAM_ADMIN_ID (supports comma-separated admin IDs).
    Also returns True if user has role='admin' in telegram_users.json.
    """
    user_id_clean = str(user_id_str).strip()
    admin_env = str(os.getenv("TELEGRAM_ADMIN_ID", "")).strip()

    # Check comma-separated admin IDs strictly from TELEGRAM_ADMIN_ID
    if admin_env:
        admin_env_clean = admin_env.strip('"').strip("'")
        admin_ids = [x.strip().strip('"').strip("'") for x in admin_env_clean.split(",") if x.strip()]
        if user_id_clean in admin_ids:
            return True

    # Check role in telegram_users.json
    try:
        users = load_all_users()
        u_rec = users.get(user_id_clean, {})
        if u_rec.get("role") == "admin":
            return True
    except Exception:
        pass

    return False


# Public alias
is_admin_user = _is_admin_user


def sync_user_secret_to_github(user_id_str: str, secret_suffix: str, secret_value: str) -> bool:
    """
    Syncs credentials to GitHub Secrets:
    - Admin users: Updates the root canonical secret (e.g. IG_BUSINESS_ID) in-place.
      Does NOT create USER_<id>_ prefixed variables for admins.
    - Non-admin users: Creates/updates the user-prefixed secret (e.g. USER_7822881619_IG_BUSINESS_ID).
      Does NOT touch root secrets — preserving admin isolation.

    Supports comma-separated TELEGRAM_ADMIN_ID for multiple admin accounts.
    """
    try:
        from Utilities.github_secret_updater import sync_custom_secret_to_github
        user_id_clean = str(user_id_str).strip()
        is_admin = _is_admin_user(user_id_clean)

        if is_admin:
            # Admin: update root canonical secret directly (replace in-place, no prefix)
            root_secret_name = secret_suffix.upper()
            ok = sync_custom_secret_to_github(root_secret_name, secret_value)
            logger.info("🔒 [GITHUB SYNC] Admin root secret '%s' updated → %s", root_secret_name, "OK" if ok else "FAILED")
            return ok
        else:
            # Non-admin: update user-prefixed secret only
            user_secret_name = f"USER_{user_id_clean}_{secret_suffix.upper()}"
            ok = sync_custom_secret_to_github(user_secret_name, secret_value)
            logger.info("🔒 [GITHUB SYNC] User secret '%s' updated → %s", user_secret_name, "OK" if ok else "FAILED")
            return ok
    except Exception as _ex:
        logger.warning("⚠️ Notice on user GitHub Secret sync: %s", _ex)
        return False



def set_user_apify_token(user_id_str: str, apify_token: str) -> bool:
    """Saves user personal Apify API token and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_token = apify_token.strip()
    if clean_token:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["apify_api_token"] = clean_token
        save_all_users(users)
        sync_user_secret_to_github(user_id_str, "APIFY_API_TOKEN", clean_token)
        logger.info("🔑 [TELEGRAM USER MANAGER] Personal Apify token saved for User ID %s", user_id_str)
        return True
    return False


def get_user_apify_token(user_id_str: str) -> Optional[str]:
    """Returns user personal Apify API token if present."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    if user_id_str in users:
        return users[user_id_str].get("apify_api_token", "").strip() or None
    return None


def set_user_gemini_key(user_id_str: str, gemini_key: str) -> bool:
    """Saves user personal Gemini API key and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_key = gemini_key.strip()
    if clean_key:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["gemini_api_key"] = clean_key
        save_all_users(users)
        sync_user_secret_to_github(user_id_str, "GEMINI_API_KEY", clean_key)
        logger.info("🤖 [TELEGRAM USER MANAGER] Personal Gemini API key saved for User ID %s", user_id_str)
        return True
    return False


def set_user_meta_token(user_id_str: str, meta_token: str) -> bool:
    """Saves user personal Meta/Instagram Access Token and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_token = meta_token.strip()
    if clean_token:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["meta_page_token"] = clean_token
        save_all_users(users)
        sync_user_secret_to_github(user_id_str, "META_PAGE_TOKEN", clean_token)
        logger.info("📸 [TELEGRAM USER MANAGER] Personal Meta Access Token saved for User ID %s", user_id_str)
        return True
    return False


def set_user_youtube_token(user_id_str: str, token_json_str: str) -> bool:
    """Saves user personal YouTube OAuth token JSON, writes token files locally, and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_json = token_json_str.strip()
    if clean_json:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["youtube_token_json"] = clean_json
        save_all_users(users)

        # Write local disk files for immediate auth use (shared disk path, only do this for admin)
        if _is_admin_user(user_id_str):
            for fpath in ["Credentials/youtube/token.json", "Credentials/token.json"]:
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(clean_json)

        sync_user_secret_to_github(user_id_str, "TOKEN_JSON", clean_json)
        logger.info("🔴 [TELEGRAM USER MANAGER] Personal YouTube OAuth Token saved & synced for User ID %s", user_id_str)
        return True
    return False


def set_user_branding(user_id_str: str, brand_text: str) -> bool:
    """Saves user personal brand watermark text and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_brand = brand_text.strip()
    if clean_brand:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["brand_watermark"] = clean_brand
        save_all_users(users)
        sync_user_secret_to_github(user_id_str, "BRAND_WATERMARK_TEXT", clean_brand)
        logger.info("🏷️ [TELEGRAM USER MANAGER] Personal brand watermark saved for User ID %s: %s", user_id_str, clean_brand)
        return True
    return False


def set_user_public_group_id(user_id_str: str, group_id: str) -> bool:
    """Saves user personal Telegram Public Group ID and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_group = group_id.strip()
    if clean_group:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["telegram_public_group_id"] = clean_group
        save_all_users(users)
        synced = sync_user_secret_to_github(user_id_str, "TELEGRAM_PUBLIC_GROUP_ID", clean_group)
        logger.info("📢 [TELEGRAM USER MANAGER] Telegram Public Group ID saved for User ID %s: %s (GitHub Synced: %s)", user_id_str, clean_group, synced)
        return synced
    return False


def set_user_schedule_times(user_id_str: str, schedule_times: str) -> bool:
    """Saves user personal auto-input schedule times and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_sched = schedule_times.strip()
    if clean_sched:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["auto_input_schedule_times"] = clean_sched
        save_all_users(users)
        synced = sync_user_secret_to_github(user_id_str, "AUTO_INPUT_SCHEDULE_TIMES", clean_sched)
        logger.info("⏰ [TELEGRAM USER MANAGER] Schedule times saved for User ID %s: %s (GitHub Synced: %s)", user_id_str, clean_sched, synced)
        return synced
    return False


def set_user_youtube_client_secret(user_id_str: str, client_secret_str: str) -> bool:
    """Saves user personal YouTube client_secret.json content, writes files locally, and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_secret = client_secret_str.strip()
    if clean_secret:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["youtube_client_secret"] = clean_secret
        save_all_users(users)

        # Write local disk files only for admin to avoid overwriting shared credentials
        if _is_admin_user(user_id_str):
            for fpath in ["Credentials/youtube/client_secret.json", "Credentials/client_secret.json"]:
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(clean_secret)

        sync_user_secret_to_github(user_id_str, "CLIENT_SECRET_JSON", clean_secret)
        logger.info("🔐 [TELEGRAM USER MANAGER] YouTube Client Secret saved & synced for User ID %s", user_id_str)
        return True
    return False


def set_user_auto_scrape_limit(user_id_str: str, limit_val: int) -> bool:
    """Saves user personal auto-scrape limit (default 8) and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    try:
        val = int(limit_val)
        if val <= 0:
            val = 8
    except (ValueError, TypeError):
        val = 8

    u_rec = users.setdefault(user_id_str, {})
    u_rec["auto_scrape_limit"] = val
    save_all_users(users)
    synced = sync_user_secret_to_github(user_id_str, "AUTO_SCRAPE_LIMIT", str(val))
    logger.info("🔢 [TELEGRAM USER MANAGER] Auto-scrape limit saved for User ID %s: %d (GitHub Synced: %s)", user_id_str, val, synced)
    return True


def get_user_auto_scrape_limit(user_id_str: str) -> int:
    """Returns user personal auto-scrape limit, falling back to env AUTO_SCRAPE_LIMIT or default 8."""
    users = load_all_users()
    u_rec = users.get(str(user_id_str), {})
    lim = u_rec.get("auto_scrape_limit")
    if lim:
        try:
            return int(lim)
        except (ValueError, TypeError):
            pass
    env_user = os.getenv(f"USER_{user_id_str}_AUTO_SCRAPE_LIMIT")
    if env_user:
        try:
            return int(env_user)
        except (ValueError, TypeError):
            pass
    env_root = os.getenv("AUTO_SCRAPE_LIMIT")
    if env_root:
        try:
            return int(env_root)
        except (ValueError, TypeError):
            pass
    return 8


def set_user_instagram_token(user_id_str: str, token: str) -> bool:
    """Saves user Instagram/Meta Access Token and syncs to IG_BUSINESS_TOKEN and META_PAGE_TOKEN in GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_tok = token.strip()
    if clean_tok:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["ig_business_token"] = clean_tok
        u_rec["meta_page_token"] = clean_tok
        save_all_users(users)
        sync_user_secret_to_github(user_id_str, "IG_BUSINESS_TOKEN", clean_tok)
        sync_user_secret_to_github(user_id_str, "META_PAGE_TOKEN", clean_tok)
        logger.info("📸 [TELEGRAM USER MANAGER] Instagram/Meta Access Token saved for User ID %s", user_id_str)
        return True
    return False


def set_user_instagram_id(user_id_str: str, ig_id: str) -> bool:
    """Saves user Instagram Business Account ID and syncs to IG_BUSINESS_ID in GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_id = ig_id.strip()
    if clean_id:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["ig_business_id"] = clean_id
        save_all_users(users)
        sync_user_secret_to_github(user_id_str, "IG_BUSINESS_ID", clean_id)
        logger.info("🆔 [TELEGRAM USER MANAGER] Instagram Business ID saved for User ID %s: %s", user_id_str, clean_id)
        return True
    return False


def set_user_facebook_id(user_id_str: str, fb_page_id: str) -> bool:
    """Saves user Facebook Page ID and syncs to META_PAGE_ID in GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_id = fb_page_id.strip()
    if clean_id:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["meta_page_id"] = clean_id
        save_all_users(users)
        sync_user_secret_to_github(user_id_str, "META_PAGE_ID", clean_id)
        logger.info("📘 [TELEGRAM USER MANAGER] Facebook Page ID saved for User ID %s: %s", user_id_str, clean_id)
        return True
    return False


def set_user_facebook_token(user_id_str: str, fb_token: str) -> bool:
    """Saves user Facebook Page Access Token and syncs to META_PAGE_TOKEN in GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_tok = fb_token.strip()
    if clean_tok:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["meta_page_token"] = clean_tok
        save_all_users(users)
        sync_user_secret_to_github(user_id_str, "META_PAGE_TOKEN", clean_tok)
        logger.info("🔑 [TELEGRAM USER MANAGER] Facebook Page Token saved for User ID %s", user_id_str)
        return True
    return False


def set_user_tiktok_token(user_id_str: str, tiktok_token: str) -> bool:
    """Saves user TikTok Access Token and syncs to TIKTOK_ACCESS_TOKEN in GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_tok = tiktok_token.strip()
    if clean_tok:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["tiktok_access_token"] = clean_tok
        save_all_users(users)
        sync_user_secret_to_github(user_id_str, "TIKTOK_ACCESS_TOKEN", clean_tok)
        logger.info("🎵 [TELEGRAM USER MANAGER] TikTok Access Token saved for User ID %s", user_id_str)
        return True
    return False


def get_user_gemini_key(user_id_str: str) -> Optional[str]:
    """Returns user personal Gemini API key if present."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    if user_id_str in users:
        return users[user_id_str].get("gemini_api_key", "").strip() or None
    return None


def format_user_credentials_summary(user_id_str: str) -> str:
    """Formats a status checklist of all configured user credentials with exact command syntax."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    u_rec = users.get(user_id_str, {})
    is_admin = _is_admin_user(user_id_str)
    
    def _chk(key, env_var):
        if u_rec.get(key, "").strip():
            return "✅ Configured"
        user_env = os.getenv(f"USER_{user_id_str}_{env_var}", "").strip()
        if user_env:
            return "✅ Configured (Secret)"
        if is_admin and os.getenv(env_var, "").strip():
            return "✅ Active (Admin Root)"
        return "❌ Not Set"

    role_badge = "👑 **Role:** `Admin` (Root Credentials)" if is_admin else "👤 **Role:** `Standard User` (Personal Credentials)"

    def _val(key, env_var):
        return u_rec.get(key) or os.getenv(f"USER_{user_id_str}_{env_var}") or (os.getenv(env_var) if is_admin else None) or "Not Set"

    lines = [
        "⚙️ **Your SaaS Configuration Checklist & Command Guide**",
        role_badge + "\n",
        f"🤖 **Gemini API Key:** `{_chk('gemini_api_key', 'GEMINI_API_KEY')}`",
        "   👉 Command: `/setgemini YOUR_GEMINI_KEY`\n",
        f"🕷️ **Apify Scraper Token:** `{_chk('apify_api_token', 'APIFY_API_TOKEN')}`",
        "   👉 Command: `/setapify YOUR_APIFY_TOKEN`\n",
        f"🏷️ **Brand Watermark:** `{_val('brand_watermark', 'BRAND_WATERMARK_TEXT')}`",
        "   👉 Command: `/setbranding YOUR_WATERMARK_TEXT`\n",
        f"📢 **Public Group ID:** `{_val('telegram_public_group_id', 'TELEGRAM_PUBLIC_GROUP_ID')}`",
        "   👉 Command: `/setgroup YOUR_GROUP_ID`\n",
        f"⏰ **Schedule Times:** `{_val('auto_input_schedule_times', 'AUTO_INPUT_SCHEDULE_TIMES')}`",
        "   👉 Command: `/setschedule HH:MM,HH:MM`\n",
        f"🔢 **Auto-Scrape Limit:** `{_val('auto_scrape_limit', 'AUTO_SCRAPE_LIMIT') or '8'} reels/account`",
        "   👉 Command: `/setautoscraplimit <number>` (e.g. `/setautoscraplimit 8`)\n",
        f"📸 **Instagram Token:** `{_chk('ig_business_token', 'IG_BUSINESS_TOKEN')}`",
        "   👉 Command: `/instagramtoken YOUR_ACCESS_TOKEN`\n",
        f"🆔 **Instagram Business ID:** `{_val('ig_business_id', 'IG_BUSINESS_ID')}`",
        "   👉 Command: `/instagramid YOUR_BUSINESS_ID`\n",
        f"📘 **Facebook Page ID:** `{_val('meta_page_id', 'META_PAGE_ID')}`",
        "   👉 Command: `/facebookid YOUR_PAGE_ID`\n",
        f"🔴 **YouTube OAuth Token:** `{_chk('youtube_token_json', 'TOKEN_JSON')}`",
        "   👉 Command: `/setytclient CLIENT_JSON` then `/ytcode AUTH_CODE`\n",
        f"🎵 **TikTok Access Token:** `{_chk('tiktok_access_token', 'TIKTOK_ACCESS_TOKEN')}`",
        "   👉 Command: `/tiktoktoken YOUR_ACCESS_TOKEN`\n",
        "💡 *Tip: Personal keys bypass queues and rate limits! Run `/autosetup` for the 6-step wizard or `/myconfig` anytime.*"
    ]
    return "\n".join(lines)


def get_user_branding(user_id_str: Optional[Union[str, int]] = None) -> str:
    """
    Returns personal brand watermark text isolated by user_id/chat_id:
    - If user has configured 'brand_watermark', returns their personal brand.
    - If user is an Admin and has not set a custom brand, defaults to root BRAND_WATERMARK_TEXT.
    - If user is a Non-Admin and has not set a brand, returns empty string ("") — NEVER leaks admin branding!
    - If user_id_str is None (headless CLI / background daemon runner), defaults to root BRAND_WATERMARK_TEXT.
    """
    if user_id_str is None or str(user_id_str).strip() == "":
        return ""

    users = load_all_users()
    clean_uid = str(user_id_str).strip()

    # 1. Look up user record by key or chat_id/user_id attribute
    u_rec = users.get(clean_uid)
    if not u_rec:
        for _uid_key, rec in users.items():
            if str(rec.get("chat_id", "")) == clean_uid or str(rec.get("user_id", "")) == clean_uid:
                u_rec = rec
                break

    # 2. Check if user set personal brand in users JSON
    if u_rec:
        brand = (u_rec.get("brand_watermark") or "").strip()
        if brand:
            return brand

    # 3. Check user-prefixed environment variable (e.g. USER_12345_BRAND_WATERMARK_TEXT)
    env_user_brand = os.getenv(f"USER_{clean_uid}_BRAND_WATERMARK_TEXT", "").strip()
    if env_user_brand:
        return env_user_brand

    # 4. Strict isolation: Only admins inherit the root BRAND_WATERMARK_TEXT from .env / GitHub Secrets
    if _is_admin_user(clean_uid):
        return os.getenv("BRAND_WATERMARK_TEXT", "").strip().strip('"').strip("'")

    # Non-admin without personal branding gets NO watermark (zero branding leak)
    return ""


def set_user_nickname(user_id_str: str, nickname_text: str) -> Tuple[bool, str]:
    """Sets personal unique nickname for a registered user, enforcing strict uniqueness across all accounts."""
    users = load_all_users()
    clean_nick = nickname_text.strip()
    clean_nick_lower = clean_nick.lower()
    
    if not clean_nick:
        return False, "⚠️ *Nickname Cannot Be Blank*\n\nUsage: `/setnickname your_nickname`"

    # Enforce strict unique nickname constraint across all registered users
    for uid, udata in users.items():
        if str(uid) != str(user_id_str):
            existing_nick = (udata.get("nickname") or "").strip().lower()
            if existing_nick and existing_nick == clean_nick_lower:
                logger.warning("⚠️ [TELEGRAM USER MANAGER] Duplicate nickname attempt '%s' rejected for User ID %s (owned by User ID %s)", clean_nick, user_id_str, uid)
                return False, f"❌ *Nickname Taken*\n\nNickname `'{clean_nick}'` is already registered to another account! Please choose a unique nickname."

    if str(user_id_str) in users:
        users[str(user_id_str)]["nickname"] = clean_nick
        save_all_users(users)
        logger.info("🏷️ [TELEGRAM USER MANAGER] Unique nickname set to '%s' for User ID %s", clean_nick, user_id_str)
        return True, f"✅ *Nickname Updated!* You will now be called *{_escape_md(clean_nick)}*."
    return False, "❌ User account not found."


def set_user_password(user_id_str: str, plain_password: str) -> bool:
    """Sets and hashes personal account password for a registered user."""
    users = load_all_users()
    if user_id_str in users and plain_password.strip():
        users[user_id_str]["password_hash"] = _hash_password(plain_password.strip())
        users[user_id_str]["authenticated"] = True
        save_all_users(users)
        logger.info("🔐 [TELEGRAM USER MANAGER] Password set and session authenticated for User ID %s", user_id_str)
        return True
    return False


def verify_and_login_user(user_id_str: str, plain_password: str) -> bool:
    """Verifies plain text password against stored hash and authenticates user session."""
    users = load_all_users()
    if user_id_str in users:
        stored_hash = users[user_id_str].get("password_hash", "")
        if stored_hash and stored_hash == _hash_password(plain_password.strip()):
            users[user_id_str]["authenticated"] = True
            save_all_users(users)
            logger.info("🔓 [TELEGRAM USER MANAGER] User ID %s logged in successfully", user_id_str)
            return True
    return False


def generate_recovery_otp(user_id_str: str, valid_minutes: int = 10) -> Optional[Tuple[str, int]]:
    """
    Generates a 6-digit secure single-use OTP with an expiration time window (default: 10 minutes).
    Returns (otp_code, valid_minutes) or None on failure.
    """
    users = load_all_users()
    if user_id_str in users:
        otp = f"OTP-{secrets.randbelow(900000) + 100000}"
        expires_dt = datetime.now() + timedelta(minutes=valid_minutes)
        users[user_id_str]["recovery_otp"] = otp
        users[user_id_str]["recovery_otp_expires"] = expires_dt.isoformat()
        save_all_users(users)
        logger.info("🔑 [TELEGRAM USER MANAGER] Generated recovery OTP for User ID %s (expires in %d min)", user_id_str, valid_minutes)
        return otp, valid_minutes
    return None


def reset_password_with_otp(user_id_str: str, otp_input: str, new_password: str) -> Tuple[bool, str]:
    """
    Verifies OTP code and expiration timestamp.
    Returns (success: bool, status_reason: str).
    """
    users = load_all_users()
    if user_id_str in users:
        stored_otp = users[user_id_str].get("recovery_otp", "")
        expires_str = users[user_id_str].get("recovery_otp_expires", "")

        if not stored_otp or stored_otp.strip() != otp_input.strip():
            return False, "invalid"

        if expires_str:
            try:
                expires_dt = datetime.fromisoformat(expires_str)
                if datetime.now() > expires_dt:
                    users[user_id_str]["recovery_otp"] = ""
                    users[user_id_str]["recovery_otp_expires"] = ""
                    save_all_users(users)
                    logger.warning("⏳ [TELEGRAM USER MANAGER] Expired OTP attempt for User ID %s", user_id_str)
                    return False, "expired"
            except Exception as _te:
                logger.debug("OTP expiry parse notice: %s", _te)

        if new_password.strip():
            users[user_id_str]["password_hash"] = _hash_password(new_password.strip())
            users[user_id_str]["authenticated"] = True
            users[user_id_str]["recovery_otp"] = ""
            users[user_id_str]["recovery_otp_expires"] = ""
            save_all_users(users)
            logger.info("🎉 [TELEGRAM USER MANAGER] Successfully reset password via OTP for User ID %s", user_id_str)
            return True, "success"

    return False, "invalid"


def logout_user(user_id_str: str) -> bool:
    """Logs out user session."""
    users = load_all_users()
    if user_id_str in users:
        users[user_id_str]["authenticated"] = False
        save_all_users(users)
        return True
    return False
