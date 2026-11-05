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
from typing import Dict, Optional, Tuple, Union, Any

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
            m = re.match(r"^USER_(-?\d+)_(.+)$", k)
            if m:
                uid = m.group(1)
                suffix = m.group(2).lower()
                val = v.strip()
                u_rec = users.setdefault(uid, {})
                if suffix in ("ig_business_id", "instagram_id"):
                    if not u_rec.get("ig_business_id"):
                        u_rec["ig_business_id"] = val
                elif suffix in ("meta_page_id", "facebook_page_id", "facebook_id"):
                    if not u_rec.get("meta_page_id"):
                        u_rec["meta_page_id"] = val
                elif suffix in ("ig_business_token", "instagram_token"):
                    if not u_rec.get("ig_business_token"):
                        u_rec["ig_business_token"] = val
                elif suffix in ("meta_page_token", "facebook_page_token", "facebook_token"):
                    if not u_rec.get("meta_page_token"):
                        u_rec["meta_page_token"] = val
                elif suffix in ("meta_token",):
                    if not u_rec.get("ig_business_token"):
                        u_rec["ig_business_token"] = val
                    if not u_rec.get("meta_page_token"):
                        u_rec["meta_page_token"] = val
                elif suffix in ("meta_app_id", "facebook_app_id"):
                    if not u_rec.get("meta_app_id"):
                        u_rec["meta_app_id"] = val
                elif suffix in ("meta_app_secret", "facebook_app_secret"):
                    if not u_rec.get("meta_app_secret"):
                        u_rec["meta_app_secret"] = val
                elif suffix in ("tiktok_access_token", "tiktok_token"):
                    if not u_rec.get("tiktok_access_token"):
                        u_rec["tiktok_access_token"] = val
                elif suffix in ("make_webhook_url", "webhook_url"):
                    if not u_rec.get("make_webhook_url"):
                        u_rec["make_webhook_url"] = val
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

    # Hydrate system-level configuration if present
    sys_cfg = users.get("__system_config__", {})
    if isinstance(sys_cfg, dict):
        if sys_cfg.get("META_GRAPH_VERSION") and not os.getenv("META_GRAPH_VERSION"):
            os.environ["META_GRAPH_VERSION"] = sys_cfg["META_GRAPH_VERSION"]
        if sys_cfg.get("YOUTUBE_API_VERSION") and not os.getenv("YOUTUBE_API_VERSION"):
            os.environ["YOUTUBE_API_VERSION"] = sys_cfg["YOUTUBE_API_VERSION"]

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
    user_id_str = str(user_id_str).strip()
    clean_token = apify_token.strip()
    if clean_token:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["apify_api_token"] = clean_token
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["APIFY_API_TOKEN"] = clean_token
            _persist_env_var_to_disk("APIFY_API_TOKEN", clean_token)
        else:
            os.environ[f"USER_{user_id_str}_APIFY_API_TOKEN"] = clean_token
        sync_user_secret_to_github(user_id_str, "APIFY_API_TOKEN", clean_token)
        logger.info("🔑 [TELEGRAM USER MANAGER] Personal Apify token saved for User ID %s", user_id_str)
        return True
    return False


def get_user_apify_token(user_id_str: Optional[Union[str, int]] = None) -> Optional[str]:
    """Returns user personal Apify API token if present. Non-admins NEVER inherit admin token."""
    if not user_id_str:
        return os.getenv("APIFY_API_TOKEN", "").strip() or None
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    if user_id_str in users:
        val = users[user_id_str].get("apify_api_token", "").strip()
        if val:
            return val
    env_val = os.getenv(f"USER_{user_id_str}_APIFY_API_TOKEN", "").strip()
    if env_val:
        return env_val
    if _is_admin_user(user_id_str):
        return os.getenv("APIFY_API_TOKEN", "").strip() or None
    return None


def set_user_gemini_key(user_id_str: str, gemini_key: str) -> bool:
    """Saves user personal Gemini API key and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    clean_key = gemini_key.strip()
    if clean_key:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["gemini_api_key"] = clean_key
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["GEMINI_API_KEY"] = clean_key
            _persist_env_var_to_disk("GEMINI_API_KEY", clean_key)
        else:
            os.environ[f"USER_{user_id_str}_GEMINI_API_KEY"] = clean_key
        sync_user_secret_to_github(user_id_str, "GEMINI_API_KEY", clean_key)
        logger.info("🤖 [TELEGRAM USER MANAGER] Personal Gemini API key saved for User ID %s", user_id_str)
        return True
    return False


def set_user_meta_token(user_id_str: str, meta_token: str) -> bool:
    """Saves user personal Meta/Instagram Access Token and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    clean_token = meta_token.strip()
    if clean_token:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["meta_page_token"] = clean_token
        u_rec["ig_business_token"] = clean_token
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["META_PAGE_TOKEN"] = clean_token
            os.environ["IG_BUSINESS_TOKEN"] = clean_token
            _persist_env_var_to_disk("META_PAGE_TOKEN", clean_token)
            _persist_env_var_to_disk("IG_BUSINESS_TOKEN", clean_token)
        else:
            os.environ[f"USER_{user_id_str}_META_PAGE_TOKEN"] = clean_token
            os.environ[f"USER_{user_id_str}_IG_BUSINESS_TOKEN"] = clean_token
        sync_user_secret_to_github(user_id_str, "META_PAGE_TOKEN", clean_token)
        sync_user_secret_to_github(user_id_str, "IG_BUSINESS_TOKEN", clean_token)
        logger.info("📸 [TELEGRAM USER MANAGER] Personal Meta Access Token saved for User ID %s", user_id_str)
        return True
    return False


def set_user_youtube_token(user_id_str: str, token_json_str: str) -> bool:
    """Saves user personal YouTube OAuth token JSON, writes token files locally, and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    clean_json = token_json_str.strip()
    if clean_json:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["youtube_token_json"] = clean_json
        save_all_users(users)

        # Write local disk files for immediate auth use (shared disk path, only do this for admin)
        if _is_admin_user(user_id_str):
            os.environ["TOKEN_JSON"] = clean_json
            os.environ["YOUTUBE_TOKEN_JSON"] = clean_json
            for fpath in ["Credentials/youtube/token.json", "Credentials/token.json"]:
                full_p = os.path.join(_REPO_ROOT, fpath)
                os.makedirs(os.path.dirname(full_p), exist_ok=True)
                with open(full_p, "w", encoding="utf-8") as f:
                    f.write(clean_json)
        else:
            os.environ[f"USER_{user_id_str}_TOKEN_JSON"] = clean_json
            os.environ[f"USER_{user_id_str}_YOUTUBE_TOKEN_JSON"] = clean_json

        sync_user_secret_to_github(user_id_str, "TOKEN_JSON", clean_json)
        logger.info("🔴 [TELEGRAM USER MANAGER] Personal YouTube OAuth Token saved & synced for User ID %s", user_id_str)
        return True
    return False


def set_user_branding(user_id_str: str, brand_text: str) -> bool:
    """Saves user personal brand watermark text and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    clean_brand = brand_text.strip()
    if clean_brand:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["brand_watermark"] = clean_brand
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["BRAND_WATERMARK_TEXT"] = clean_brand
            _persist_env_var_to_disk("BRAND_WATERMARK_TEXT", clean_brand)
        else:
            os.environ[f"USER_{user_id_str}_BRAND_WATERMARK_TEXT"] = clean_brand
        sync_user_secret_to_github(user_id_str, "BRAND_WATERMARK_TEXT", clean_brand)
        logger.info("🏷️ [TELEGRAM USER MANAGER] Personal brand watermark saved for User ID %s: %s", user_id_str, clean_brand)
        return True
    return False


def set_user_public_group_id(user_id_str: str, group_id: str) -> bool:
    """Saves user personal Telegram Public Group ID and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    clean_group = group_id.strip()
    if clean_group:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["telegram_public_group_id"] = clean_group
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["TELEGRAM_PUBLIC_GROUP_ID"] = clean_group
            _persist_env_var_to_disk("TELEGRAM_PUBLIC_GROUP_ID", clean_group)
        else:
            os.environ[f"USER_{user_id_str}_TELEGRAM_PUBLIC_GROUP_ID"] = clean_group
        synced = sync_user_secret_to_github(user_id_str, "TELEGRAM_PUBLIC_GROUP_ID", clean_group)
        logger.info("📢 [TELEGRAM USER MANAGER] Telegram Public Group ID saved for User ID %s: %s (GitHub Synced: %s)", user_id_str, clean_group, synced)
        return synced
    return False


def set_user_schedule_times(user_id_str: str, schedule_times: str) -> bool:
    """Saves user personal auto-input schedule times and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    clean_sched = schedule_times.strip()
    if clean_sched:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["auto_input_schedule_times"] = clean_sched
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["AUTO_INPUT_SCHEDULE_TIMES"] = clean_sched
            _persist_env_var_to_disk("AUTO_INPUT_SCHEDULE_TIMES", clean_sched)
        else:
            os.environ[f"USER_{user_id_str}_AUTO_INPUT_SCHEDULE_TIMES"] = clean_sched
        synced = sync_user_secret_to_github(user_id_str, "AUTO_INPUT_SCHEDULE_TIMES", clean_sched)
        logger.info("⏰ [TELEGRAM USER MANAGER] Schedule times saved for User ID %s: %s (GitHub Synced: %s)", user_id_str, clean_sched, synced)
        return synced
    return False


def set_user_youtube_client_secret(user_id_str: str, client_secret_str: str) -> bool:
    """Saves user personal YouTube client_secret.json content, writes files locally, and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    clean_secret = client_secret_str.strip()
    if clean_secret:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["youtube_client_secret"] = clean_secret
        save_all_users(users)

        # Write local disk files only for admin to avoid overwriting shared credentials
        if _is_admin_user(user_id_str):
            os.environ["CLIENT_SECRET_JSON"] = clean_secret
            os.environ["YOUTUBE_CLIENT_SECRET"] = clean_secret
            for fpath in ["Credentials/youtube/client_secret.json", "Credentials/client_secret.json"]:
                full_p = os.path.join(_REPO_ROOT, fpath)
                os.makedirs(os.path.dirname(full_p), exist_ok=True)
                with open(full_p, "w", encoding="utf-8") as f:
                    f.write(clean_secret)
        else:
            os.environ[f"USER_{user_id_str}_CLIENT_SECRET_JSON"] = clean_secret
            os.environ[f"USER_{user_id_str}_YOUTUBE_CLIENT_SECRET"] = clean_secret

        sync_user_secret_to_github(user_id_str, "CLIENT_SECRET_JSON", clean_secret)
        logger.info("🔐 [TELEGRAM USER MANAGER] YouTube Client Secret saved & synced for User ID %s", user_id_str)
        return True
    return False


def set_user_auto_scrape_limit(user_id_str: str, limit_val: int) -> bool:
    """Saves user personal auto-scrape limit (default 8) and syncs to GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    try:
        val = int(limit_val)
        if val <= 0:
            val = 8
    except (ValueError, TypeError):
        val = 8

    u_rec = users.setdefault(user_id_str, {})
    u_rec["auto_scrape_limit"] = val
    save_all_users(users)
    if _is_admin_user(user_id_str):
        os.environ["AUTO_SCRAPE_LIMIT"] = str(val)
        _persist_env_var_to_disk("AUTO_SCRAPE_LIMIT", str(val))
    else:
        os.environ[f"USER_{user_id_str}_AUTO_SCRAPE_LIMIT"] = str(val)
    synced = sync_user_secret_to_github(user_id_str, "AUTO_SCRAPE_LIMIT", str(val))
    logger.info("🔢 [TELEGRAM USER MANAGER] Auto-scrape limit saved for User ID %s: %d (GitHub Synced: %s)", user_id_str, val, synced)
    return True


def get_user_auto_scrape_limit(user_id_str: Optional[Union[str, int]] = None) -> int:
    """Returns user personal auto-scrape limit, falling back to env AUTO_SCRAPE_LIMIT or default 8."""
    if not user_id_str:
        env_root = os.getenv("AUTO_SCRAPE_LIMIT")
        if env_root:
            try:
                return int(env_root)
            except (ValueError, TypeError):
                pass
        return 8
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
    if _is_admin_user(str(user_id_str)):
        env_root = os.getenv("AUTO_SCRAPE_LIMIT")
        if env_root:
            try:
                return int(env_root)
            except (ValueError, TypeError):
                pass
    return 8


def set_user_instagram_token(user_id_str: str, token: str) -> bool:
    """Saves user Instagram Access Token and syncs to IG_BUSINESS_TOKEN in GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    clean_tok = token.strip()
    if clean_tok:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["ig_business_token"] = clean_tok
        if not u_rec.get("meta_page_token"):
            u_rec["meta_page_token"] = clean_tok
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["IG_BUSINESS_TOKEN"] = clean_tok
            _persist_env_var_to_disk("IG_BUSINESS_TOKEN", clean_tok)
        else:
            os.environ[f"USER_{user_id_str}_IG_BUSINESS_TOKEN"] = clean_tok
        sync_user_secret_to_github(user_id_str, "IG_BUSINESS_TOKEN", clean_tok)
        logger.info("📸 [TELEGRAM USER MANAGER] Instagram Access Token saved for User ID %s", user_id_str)
        return True
    return False


def set_user_instagram_id(user_id_str: str, ig_id: str) -> bool:
    """Saves user Instagram Business Account ID and syncs to IG_BUSINESS_ID in GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    clean_id = ig_id.strip()
    if clean_id:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["ig_business_id"] = clean_id
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["IG_BUSINESS_ID"] = clean_id
            _persist_env_var_to_disk("IG_BUSINESS_ID", clean_id)
        else:
            os.environ[f"USER_{user_id_str}_IG_BUSINESS_ID"] = clean_id
        sync_user_secret_to_github(user_id_str, "IG_BUSINESS_ID", clean_id)
        logger.info("🆔 [TELEGRAM USER MANAGER] Instagram Business ID saved for User ID %s: %s", user_id_str, clean_id)
        return True
    return False


def set_user_facebook_id(user_id_str: str, fb_page_id: str) -> bool:
    """Saves user Facebook Page ID and syncs to META_PAGE_ID in GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    clean_id = fb_page_id.strip()
    if clean_id:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["meta_page_id"] = clean_id
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["META_PAGE_ID"] = clean_id
            _persist_env_var_to_disk("META_PAGE_ID", clean_id)
        else:
            os.environ[f"USER_{user_id_str}_META_PAGE_ID"] = clean_id
        sync_user_secret_to_github(user_id_str, "META_PAGE_ID", clean_id)
        logger.info("📘 [TELEGRAM USER MANAGER] Facebook Page ID saved for User ID %s: %s", user_id_str, clean_id)
        return True
    return False


def set_user_facebook_token(user_id_str: str, fb_token: str) -> bool:
    """Saves user Facebook Page Access Token and syncs to META_PAGE_TOKEN in GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    clean_tok = fb_token.strip()
    if clean_tok:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["meta_page_token"] = clean_tok
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["META_PAGE_TOKEN"] = clean_tok
            _persist_env_var_to_disk("META_PAGE_TOKEN", clean_tok)
        else:
            os.environ[f"USER_{user_id_str}_META_PAGE_TOKEN"] = clean_tok
        sync_user_secret_to_github(user_id_str, "META_PAGE_TOKEN", clean_tok)
        logger.info("🔑 [TELEGRAM USER MANAGER] Facebook Page Token saved for User ID %s", user_id_str)
        return True
    return False
    if clean_tok:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["meta_page_token"] = clean_tok
        save_all_users(users)
        sync_user_secret_to_github(user_id_str, "META_PAGE_TOKEN", clean_tok)
        logger.info("🔑 [TELEGRAM USER MANAGER] Facebook Page Token saved for User ID %s", user_id_str)
        return True
    return False


def set_user_meta_app(user_id_str: str, app_id: str, app_secret: str) -> bool:
    """Saves Meta App ID and Secret. Admins update root .env; non-admins update user-prefixed env and json."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_id = app_id.strip()
    clean_sec = app_secret.strip()
    if clean_id and clean_sec:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["meta_app_id"] = clean_id
        u_rec["meta_app_secret"] = clean_sec
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["META_APP_ID"] = clean_id
            os.environ["FACEBOOK_APP_ID"] = clean_id
            os.environ["META_APP_SECRET"] = clean_sec
            os.environ["FACEBOOK_APP_SECRET"] = clean_sec
            _persist_env_var_to_disk("META_APP_ID", clean_id)
            _persist_env_var_to_disk("FACEBOOK_APP_ID", clean_id)
            _persist_env_var_to_disk("META_APP_SECRET", clean_sec)
            _persist_env_var_to_disk("FACEBOOK_APP_SECRET", clean_sec)
        else:
            os.environ[f"USER_{user_id_str}_META_APP_ID"] = clean_id
            os.environ[f"USER_{user_id_str}_META_APP_SECRET"] = clean_sec
        sync_user_secret_to_github(user_id_str, "META_APP_ID", clean_id)
        sync_user_secret_to_github(user_id_str, "META_APP_SECRET", clean_sec)
        logger.info("🔑 [TELEGRAM USER MANAGER] Meta App ID & Secret saved for User ID %s", user_id_str)
        return True
    return False


def set_user_meta_app_id(user_id_str: str, app_id: str) -> bool:
    """Saves Meta App ID. Admins update root .env; non-admins update isolated personal record."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_id = app_id.strip()
    if clean_id:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["meta_app_id"] = clean_id
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["META_APP_ID"] = clean_id
            os.environ["FACEBOOK_APP_ID"] = clean_id
            _persist_env_var_to_disk("META_APP_ID", clean_id)
            _persist_env_var_to_disk("FACEBOOK_APP_ID", clean_id)
        else:
            os.environ[f"USER_{user_id_str}_META_APP_ID"] = clean_id
        sync_user_secret_to_github(user_id_str, "META_APP_ID", clean_id)
        return True
    return False


def set_user_meta_app_secret(user_id_str: str, app_secret: str) -> bool:
    """Saves Meta App Secret. Admins update root .env; non-admins update isolated personal record."""
    users = load_all_users()
    user_id_str = str(user_id_str)
    clean_sec = app_secret.strip()
    if clean_sec:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["meta_app_secret"] = clean_sec
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["META_APP_SECRET"] = clean_sec
            os.environ["FACEBOOK_APP_SECRET"] = clean_sec
            _persist_env_var_to_disk("META_APP_SECRET", clean_sec)
            _persist_env_var_to_disk("FACEBOOK_APP_SECRET", clean_sec)
        else:
            os.environ[f"USER_{user_id_str}_META_APP_SECRET"] = clean_sec
        sync_user_secret_to_github(user_id_str, "META_APP_SECRET", clean_sec)
        return True
    return False


def get_user_meta_app_id(user_id_str: str) -> Optional[str]:
    """Returns user personal Meta App ID. Non-admins NEVER inherit admin root app ID."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    if user_id_str in users:
        val = users[user_id_str].get("meta_app_id", "").strip()
        if val:
            return val
    env_val = os.getenv(f"USER_{user_id_str}_META_APP_ID", "").strip()
    if env_val:
        return env_val
    if _is_admin_user(user_id_str):
        return os.getenv("META_APP_ID", "").strip() or os.getenv("FACEBOOK_APP_ID", "").strip() or None
    return None


def get_user_meta_app_secret(user_id_str: str) -> Optional[str]:
    """Returns user personal Meta App Secret. Non-admins NEVER inherit admin root app secret."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    if user_id_str in users:
        val = users[user_id_str].get("meta_app_secret", "").strip()
        if val:
            return val
    env_val = os.getenv(f"USER_{user_id_str}_META_APP_SECRET", "").strip()
    if env_val:
        return env_val
    if _is_admin_user(user_id_str):
        return os.getenv("META_APP_SECRET", "").strip() or os.getenv("FACEBOOK_APP_SECRET", "").strip() or None
    return None


def get_user_field(user_id_str: str, field_key: str, default: Any = None) -> Any:
    """
    Generic per-user field reader from telegram_users.json.
    Does NOT fall back to env vars — suitable for lightweight internal flags
    such as daily velocity counters that are local-only.
    """
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    return users.get(user_id_str, {}).get(field_key, default)


def set_user_field(user_id_str: str, field_key: str, value: Any) -> bool:
    """
    Generic per-user field writer to telegram_users.json (local only, no vault sync).
    Suitable for lightweight ephemeral flags such as daily velocity counters.
    """
    try:
        users = load_all_users()
        user_id_str = str(user_id_str).strip()
        u_rec = users.setdefault(user_id_str, {})
        u_rec[field_key] = value
        # sync_to_vault=False — velocity counters are ephemeral, skip vault upload overhead
        save_all_users(users, sync_to_vault=False)
        return True
    except Exception as _e:
        logger.warning("set_user_field(%s, %s): %s", user_id_str, field_key, _e)
        return False


def set_user_tiktok_token(user_id_str: str, tiktok_token: str) -> bool:
    """Saves user TikTok Access Token and syncs to TIKTOK_ACCESS_TOKEN in GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    clean_tok = tiktok_token.strip()
    if clean_tok:
        u_rec = users.setdefault(user_id_str, {})
        u_rec["tiktok_access_token"] = clean_tok
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["TIKTOK_ACCESS_TOKEN"] = clean_tok
            _persist_env_var_to_disk("TIKTOK_ACCESS_TOKEN", clean_tok)
        else:
            os.environ[f"USER_{user_id_str}_TIKTOK_ACCESS_TOKEN"] = clean_tok
        sync_user_secret_to_github(user_id_str, "TIKTOK_ACCESS_TOKEN", clean_tok)
        logger.info("🎵 [TELEGRAM USER MANAGER] TikTok Access Token saved for User ID %s", user_id_str)
        return True
    return False


def set_user_make_webhook_url(user_id_str: str, webhook_url: str) -> bool:
    """Saves user Make.com Webhook URL and syncs to MAKE_WEBHOOK_URL in GitHub Secrets."""
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    clean_url = webhook_url.strip().strip("`'\"")
    if clean_url and clean_url.startswith(("http://", "https://")):
        u_rec = users.setdefault(user_id_str, {})
        u_rec["make_webhook_url"] = clean_url
        save_all_users(users)
        if _is_admin_user(user_id_str):
            os.environ["MAKE_WEBHOOK_URL"] = clean_url
            _persist_env_var_to_disk("MAKE_WEBHOOK_URL", clean_url)
        else:
            os.environ[f"USER_{user_id_str}_MAKE_WEBHOOK_URL"] = clean_url
        sync_user_secret_to_github(user_id_str, "MAKE_WEBHOOK_URL", clean_url)
        logger.info("🌐 [TELEGRAM USER MANAGER] Make.com Webhook URL saved for User ID %s", user_id_str)
        return True
    return False


def get_user_make_webhook_url(user_id_str: Optional[Union[str, int]] = None) -> Optional[str]:
    """Returns user personal Make.com Webhook URL if present. Non-admins NEVER inherit admin webhook."""
    if not user_id_str:
        return os.getenv("MAKE_WEBHOOK_URL", "").strip() or None
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    if user_id_str in users:
        val = users[user_id_str].get("make_webhook_url", "").strip()
        if val:
            return val
    env_val = os.getenv(f"USER_{user_id_str}_MAKE_WEBHOOK_URL", "").strip()
    if env_val:
        return env_val
    if _is_admin_user(user_id_str):
        return os.getenv("MAKE_WEBHOOK_URL", "").strip() or None
    return None


def get_user_gemini_key(user_id_str: Optional[Union[str, int]] = None) -> Optional[str]:
    """Returns user personal Gemini API key if present. Non-admins NEVER inherit admin key."""
    if not user_id_str:
        return os.getenv("GEMINI_API_KEY", "").strip() or None
    users = load_all_users()
    user_id_str = str(user_id_str).strip()
    if user_id_str in users:
        val = users[user_id_str].get("gemini_api_key", "").strip()
        if val:
            return val
    env_val = os.getenv(f"USER_{user_id_str}_GEMINI_API_KEY", "").strip()
    if env_val:
        return env_val
    if _is_admin_user(user_id_str):
        return os.getenv("GEMINI_API_KEY", "").strip() or None
    return None


def get_user_instagram_token(user_id_str: Optional[Union[str, int]] = None) -> Optional[str]:
    """Returns user personal Instagram/Meta token. Non-admins NEVER inherit admin root token."""
    if not user_id_str:
        return os.getenv("IG_BUSINESS_TOKEN", "").strip() or os.getenv("META_PAGE_TOKEN", "").strip() or None
    users = load_all_users()
    clean_uid = str(user_id_str).strip()
    if clean_uid in users:
        val = (users[clean_uid].get("ig_business_token") or users[clean_uid].get("meta_page_token") or "").strip()
        if val:
            return val
    for var in [f"USER_{clean_uid}_IG_BUSINESS_TOKEN", f"USER_{clean_uid}_META_PAGE_TOKEN"]:
        env_val = os.getenv(var, "").strip()
        if env_val:
            return env_val
    if _is_admin_user(clean_uid):
        return os.getenv("IG_BUSINESS_TOKEN", "").strip() or os.getenv("META_PAGE_TOKEN", "").strip() or None
    return None


def get_user_instagram_id(user_id_str: Optional[Union[str, int]] = None) -> Optional[str]:
    """Returns user personal Instagram Business Account ID. Non-admins NEVER inherit admin root ID."""
    if not user_id_str:
        return os.getenv("IG_BUSINESS_ID", "").strip() or os.getenv("IG_BUSINESS_ACCOUNT_ID", "").strip() or None
    users = load_all_users()
    clean_uid = str(user_id_str).strip()
    if clean_uid in users:
        val = (users[clean_uid].get("ig_business_id") or "").strip()
        if val:
            return val
    env_val = os.getenv(f"USER_{clean_uid}_IG_BUSINESS_ID", "").strip()
    if env_val:
        return env_val
    if _is_admin_user(clean_uid):
        return os.getenv("IG_BUSINESS_ID", "").strip() or os.getenv("IG_BUSINESS_ACCOUNT_ID", "").strip() or None
    return None


def get_user_facebook_id(user_id_str: Optional[Union[str, int]] = None) -> Optional[str]:
    """Returns user personal Facebook Page ID. Non-admins NEVER inherit admin root ID."""
    if not user_id_str:
        return os.getenv("META_PAGE_ID", "").strip() or None
    users = load_all_users()
    clean_uid = str(user_id_str).strip()
    if clean_uid in users:
        val = (users[clean_uid].get("meta_page_id") or "").strip()
        if val:
            return val
    env_val = os.getenv(f"USER_{clean_uid}_META_PAGE_ID", "").strip()
    if env_val:
        return env_val
    if _is_admin_user(clean_uid):
        return os.getenv("META_PAGE_ID", "").strip() or None
    return None


def get_user_facebook_token(user_id_str: Optional[Union[str, int]] = None) -> Optional[str]:
    """Returns user personal Facebook Page Access Token. Non-admins NEVER inherit admin root token."""
    if not user_id_str:
        return os.getenv("META_PAGE_TOKEN", "").strip() or os.getenv("META_TOKEN", "").strip() or None
    users = load_all_users()
    clean_uid = str(user_id_str).strip()
    if clean_uid in users:
        val = (users[clean_uid].get("meta_page_token") or users[clean_uid].get("meta_token") or "").strip()
        if val:
            return val
    for var in [f"USER_{clean_uid}_META_PAGE_TOKEN", f"USER_{clean_uid}_META_TOKEN"]:
        env_val = os.getenv(var, "").strip()
        if env_val:
            return env_val
    if _is_admin_user(clean_uid):
        return os.getenv("META_PAGE_TOKEN", "").strip() or os.getenv("META_TOKEN", "").strip() or None
    return None


def get_user_tiktok_token(user_id_str: Optional[Union[str, int]] = None) -> Optional[str]:
    """Returns user personal TikTok Access Token. Non-admins NEVER inherit admin root token."""
    if not user_id_str:
        return os.getenv("TIKTOK_ACCESS_TOKEN", "").strip() or None
    users = load_all_users()
    clean_uid = str(user_id_str).strip()
    if clean_uid in users:
        val = (users[clean_uid].get("tiktok_access_token") or "").strip()
        if val:
            return val
    env_val = os.getenv(f"USER_{clean_uid}_TIKTOK_ACCESS_TOKEN", "").strip()
    if env_val:
        return env_val
    if _is_admin_user(clean_uid):
        return os.getenv("TIKTOK_ACCESS_TOKEN", "").strip() or None
    return None


def get_user_youtube_token_json(user_id_str: Optional[Union[str, int]] = None) -> Optional[str]:
    """Returns user personal YouTube token JSON string. Non-admins NEVER inherit admin root credentials."""
    if not user_id_str:
        tok = os.getenv("TOKEN_JSON", "").strip() or os.getenv("YOUTUBE_TOKEN_JSON", "").strip()
        if tok:
            return tok
        for fpath in ["Credentials/youtube/token.json", "Credentials/token.json"]:
            full_p = os.path.join(_REPO_ROOT, fpath)
            if os.path.exists(full_p):
                try:
                    with open(full_p, "r", encoding="utf-8") as f:
                        return f.read().strip()
                except Exception:
                    pass
        return None
    users = load_all_users()
    clean_uid = str(user_id_str).strip()
    if clean_uid in users:
        val = (users[clean_uid].get("youtube_token_json") or "").strip()
        if val:
            return val
    for var in [f"USER_{clean_uid}_TOKEN_JSON", f"USER_{clean_uid}_YOUTUBE_TOKEN_JSON"]:
        env_val = os.getenv(var, "").strip()
        if env_val:
            return env_val
    if _is_admin_user(clean_uid):
        tok = os.getenv("TOKEN_JSON", "").strip() or os.getenv("YOUTUBE_TOKEN_JSON", "").strip()
        if tok:
            return tok
        for fpath in ["Credentials/youtube/token.json", "Credentials/token.json"]:
            full_p = os.path.join(_REPO_ROOT, fpath)
            if os.path.exists(full_p):
                try:
                    with open(full_p, "r", encoding="utf-8") as f:
                        return f.read().strip()
                except Exception:
                    pass
    return None


def get_user_youtube_client_secret(user_id_str: Optional[Union[str, int]] = None) -> Optional[str]:
    """Returns user personal YouTube client secret JSON string. Non-admins NEVER inherit admin root credentials."""
    if not user_id_str:
        sec = os.getenv("CLIENT_SECRET_JSON", "").strip() or os.getenv("YOUTUBE_CLIENT_SECRET", "").strip()
        if sec:
            return sec
        for fpath in ["Credentials/youtube/client_secret.json", "Credentials/client_secret.json"]:
            full_p = os.path.join(_REPO_ROOT, fpath)
            if os.path.exists(full_p):
                try:
                    with open(full_p, "r", encoding="utf-8") as f:
                        return f.read().strip()
                except Exception:
                    pass
        return None
    users = load_all_users()
    clean_uid = str(user_id_str).strip()
    if clean_uid in users:
        val = (users[clean_uid].get("youtube_client_secret") or "").strip()
        if val:
            return val
    for var in [f"USER_{clean_uid}_CLIENT_SECRET_JSON", f"USER_{clean_uid}_YOUTUBE_CLIENT_SECRET"]:
        env_val = os.getenv(var, "").strip()
        if env_val:
            return env_val
    if _is_admin_user(clean_uid):
        sec = os.getenv("CLIENT_SECRET_JSON", "").strip() or os.getenv("YOUTUBE_CLIENT_SECRET", "").strip()
        if sec:
            return sec
        for fpath in ["Credentials/youtube/client_secret.json", "Credentials/client_secret.json"]:
            full_p = os.path.join(_REPO_ROOT, fpath)
            if os.path.exists(full_p):
                try:
                    with open(full_p, "r", encoding="utf-8") as f:
                        return f.read().strip()
                except Exception:
                    pass
    return None


def get_user_public_group(user_id_str: Optional[Union[str, int]] = None) -> Optional[str]:
    """Returns user personal Telegram Public Group/Channel ID. Non-admins NEVER inherit admin group, and NEVER falls back to storage vault."""
    if not user_id_str:
        return os.getenv("TELEGRAM_PUBLIC_GROUP_ID", "").strip() or None
    users = load_all_users()
    clean_uid = str(user_id_str).strip()
    if clean_uid in users:
        val = (users[clean_uid].get("telegram_public_group_id") or "").strip()
        if val:
            return val
    env_val = os.getenv(f"USER_{clean_uid}_TELEGRAM_PUBLIC_GROUP_ID", "").strip()
    if env_val:
        return env_val
    if _is_admin_user(clean_uid):
        return os.getenv("TELEGRAM_PUBLIC_GROUP_ID", "").strip() or None
    return None


def get_user_schedule_times(user_id_str: Optional[Union[str, int]] = None) -> Optional[str]:
    """Returns user personal auto-input schedule times. Non-admins NEVER inherit admin schedule."""
    if not user_id_str:
        return os.getenv("AUTO_INPUT_SCHEDULE_TIMES", "").strip() or os.getenv("SCRAPING_AUTO_INPUT_TIMES", "").strip() or os.getenv("APIFY_SCRAPE_SCHEDULE_TIMES", "").strip() or None
    users = load_all_users()
    clean_uid = str(user_id_str).strip()
    if clean_uid in users:
        val = (users[clean_uid].get("auto_input_schedule_times") or "").strip()
        if val:
            return val
    env_val = os.getenv(f"USER_{clean_uid}_AUTO_INPUT_SCHEDULE_TIMES", "").strip()
    if env_val:
        return env_val
    if _is_admin_user(clean_uid):
        return os.getenv("AUTO_INPUT_SCHEDULE_TIMES", "").strip() or os.getenv("SCRAPING_AUTO_INPUT_TIMES", "").strip() or os.getenv("APIFY_SCRAPE_SCHEDULE_TIMES", "").strip() or None
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
        f"📱 **Meta App ID:** `{_val('meta_app_id', 'META_APP_ID')}`",
        "   👉 Commands: `/setmetaappid <APP_ID>` and `/setmetaappsecret <APP_SECRET>`\n",
        f"⚡ **Meta 1-Click OAuth Status:** `{_chk('ig_business_token', 'IG_BUSINESS_TOKEN')}`",
        "   👉 Command: `/connectmeta`\n",
        f"🔴 **YouTube OAuth Token:** `{_chk('youtube_token_json', 'TOKEN_JSON')}`",
        "   👉 Command: `/setytclient CLIENT_JSON` then `/ytcode AUTH_CODE`\n",
        f"🎵 **TikTok Access Token:** `{_chk('tiktok_access_token', 'TIKTOK_ACCESS_TOKEN')}`",
        "   👉 Command: `/tiktoktoken YOUR_ACCESS_TOKEN`\n",
        f"🌐 **Make.com Webhook URL:** `{_chk('make_webhook_url', 'MAKE_WEBHOOK_URL')}`",
        "   👉 Command: `/setwebhook <URL>` (Guide: `/setwebhook guide` or `/guide make`)\n",
    ]
    if is_admin:
        meta_ver = get_meta_graph_api_version()
        yt_ver = get_youtube_api_version()
        lines.extend([
            "👑 **Admin API Global Versions:**",
            f"⚡ **Meta Graph API Version:** `{meta_ver}`",
            "   👉 Command: `/updatemetaendpoints <version>` (e.g. `/updatemetaendpoints v21.0` or `/updatemetaendpoints auto`)\n",
            f"🎬 **YouTube API Version:** `{yt_ver}`",
            "   👉 Command: `/updateyoutubeendpoints <version>` (e.g. `/updateyoutubeendpoints v3`)\n"
        ])
    lines.append("💡 *Tip: Personal keys bypass queues and rate limits! Run `/autosetup` for the 6-step wizard or `/myconfig` anytime.*")
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


# ── Dynamic API Endpoint Versioning & Admin Controls ─────────────────────────

def _persist_env_var_to_disk(key: str, value: str) -> bool:
    """Updates or appends a key=value pair to .env and Credentials/.env files if they exist."""
    success = False
    for rel_path in [".env", os.path.join("Credentials", ".env")]:
        full_path = os.path.join(_REPO_ROOT, rel_path)
        if os.path.exists(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                pattern = rf"^{re.escape(key)}\s*=.*$"
                if re.search(pattern, content, flags=re.MULTILINE):
                    new_content = re.sub(pattern, f"{key}={value}", content, flags=re.MULTILINE)
                else:
                    new_content = content.rstrip() + f"\n{key}={value}\n"
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                logger.info("💾 [TELEGRAM USER MANAGER] Persisted %s=%s in %s", key, value, rel_path)
                success = True
            except Exception as e:
                logger.warning("Could not persist %s in %s: %s", key, rel_path, e)
    return success


def get_meta_graph_api_version() -> str:
    """Returns the currently active Meta Graph API version."""
    try:
        from Publishing_Modules.meta_uploader import get_meta_graph_version
        return get_meta_graph_version()
    except Exception:
        return os.getenv("META_GRAPH_VERSION", "v21.0")


def update_meta_graph_api_version(version: str, admin_user_id: Optional[str] = None) -> Tuple[bool, str]:
    """
    Updates the Meta Graph API version:
      1. Normalizes formatting (e.g. '21.0' -> 'v21.0').
      2. Updates in-memory environment variable & meta_uploader module.
      3. Persists to data/telegram_users.json under __system_config__.
      4. Persists to .env and Credentials/.env.
      5. Syncs to GitHub Secrets as META_GRAPH_VERSION via PAT if available.
    """
    clean_ver = version.strip()
    if not clean_ver:
        return False, "Version string cannot be empty."
    if not clean_ver.startswith("v") and clean_ver and clean_ver[0].isdigit():
        clean_ver = f"v{clean_ver}"

    # 1. Update in-memory runtime
    os.environ["META_GRAPH_VERSION"] = clean_ver
    try:
        from Publishing_Modules.meta_uploader import set_meta_graph_version
        set_meta_graph_version(clean_ver)
    except Exception as _e:
        logger.debug("meta_uploader update notice: %s", _e)

    # 2. Persist in telegram_users.json
    try:
        users = load_all_users()
        sys_cfg = users.setdefault("__system_config__", {})
        sys_cfg["META_GRAPH_VERSION"] = clean_ver
        save_all_users(users)
    except Exception as _e:
        logger.warning("Could not save to users json: %s", _e)

    # 3. Persist to local .env files
    _persist_env_var_to_disk("META_GRAPH_VERSION", clean_ver)

    # 4. Sync to GitHub Secrets
    try:
        from Utilities.github_secret_updater import sync_custom_secret_to_github
        gh_ok = sync_custom_secret_to_github("META_GRAPH_VERSION", clean_ver)
        if gh_ok:
            logger.info("🔑 [TELEGRAM USER MANAGER] Synced META_GRAPH_VERSION to GitHub Secrets.")
    except Exception as _ge:
        logger.warning("GitHub Secret sync notice for META_GRAPH_VERSION: %s", _ge)

    return True, f"Meta Graph API version successfully updated to: {clean_ver}"


def get_youtube_api_version() -> str:
    """Returns the currently active YouTube Data API version."""
    try:
        from Publishing_Modules.uploader import get_youtube_api_version as _get_yt_ver
        return _get_yt_ver()
    except Exception:
        return os.getenv("YOUTUBE_API_VERSION", "v3")


def update_youtube_api_version(version: str, admin_user_id: Optional[str] = None) -> Tuple[bool, str]:
    """
    Updates the YouTube Data API version:
      1. Normalizes formatting (e.g. '3' -> 'v3').
      2. Updates in-memory runtime & uploader module.
      3. Persists to data/telegram_users.json under __system_config__.
      4. Persists to .env and Credentials/.env.
      5. Syncs to GitHub Secrets as YOUTUBE_API_VERSION via PAT if available.
    """
    clean_ver = version.strip()
    if not clean_ver:
        return False, "Version string cannot be empty."
    if not clean_ver.startswith("v") and clean_ver and clean_ver[0].isdigit():
        clean_ver = f"v{clean_ver}"

    # 1. Update in-memory runtime
    os.environ["YOUTUBE_API_VERSION"] = clean_ver
    try:
        from Publishing_Modules.uploader import set_youtube_api_version as _set_yt_ver
        _set_yt_ver(clean_ver)
    except Exception as _e:
        logger.debug("uploader update notice: %s", _e)

    # 2. Persist in telegram_users.json
    try:
        users = load_all_users()
        sys_cfg = users.setdefault("__system_config__", {})
        sys_cfg["YOUTUBE_API_VERSION"] = clean_ver
        save_all_users(users)
    except Exception as _e:
        logger.warning("Could not save to users json: %s", _e)

    # 3. Persist to local .env files
    _persist_env_var_to_disk("YOUTUBE_API_VERSION", clean_ver)

    # 4. Sync to GitHub Secrets
    try:
        from Utilities.github_secret_updater import sync_custom_secret_to_github
        gh_ok = sync_custom_secret_to_github("YOUTUBE_API_VERSION", clean_ver)
        if gh_ok:
            logger.info("🔑 [TELEGRAM USER MANAGER] Synced YOUTUBE_API_VERSION to GitHub Secrets.")
    except Exception as _ge:
        logger.warning("GitHub Secret sync notice for YOUTUBE_API_VERSION: %s", _ge)

    return True, f"YouTube Data API version successfully updated to: {clean_ver}"


def query_gemini_recommended_endpoint_version(platform: str) -> Dict[str, Any]:
    """
    Uses Gemini API to discover the latest official stable API version
    for Meta Graph API or YouTube Data API.
    """
    platform_clean = platform.strip().lower()
    api_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

    defaults = {
        "meta": {"version": "v21.0", "notes": "Meta Graph API v21.0 (Current official stable cycle)"},
        "youtube": {"version": "v3", "notes": "YouTube Data API v3 (Standard official video publishing endpoint)"}
    }
    fallback = defaults.get(platform_clean, {"version": "v21.0", "notes": "Default fallback endpoint"})

    if not api_key:
        return fallback

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        prompt = (
            f"What is the current latest active, stable version of the {platform.title()} API for video publishing / uploads?\n"
            f"If platform is 'meta', return the latest Facebook/Instagram Graph API version (e.g. 'v21.0' or 'v22.0').\n"
            f"If platform is 'youtube', return the YouTube Data API version (e.g. 'v3').\n"
            f"Respond ONLY with a JSON object in this format:\n"
            f'{{"version": "vXX.X", "notes": "Brief 1-sentence release status"}}\n'
            f"Do not include markdown code fences or any other text."
        )

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        ver = data.get("version", "").strip()
        if ver:
            if not ver.startswith("v") and ver and ver[0].isdigit():
                ver = f"v{ver}"
            return {"version": ver, "notes": data.get("notes", "")}
    except Exception as e:
        logger.warning("⚠️ [GEMINI VERSION CHECK] Could not query Gemini for %s: %s", platform, e)

    return fallback

