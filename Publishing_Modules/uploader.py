import os
import time
import argparse
import logging
from typing import Optional
import asyncio
import subprocess
import sys
import json
import shutil
import uuid
import ssl
import socket
import urllib.request
import urllib.parse
from urllib3.exceptions import ProtocolError, ReadTimeoutError
from requests.exceptions import ConnectionError, Timeout

FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
if not shutil.which(FFPROBE_BIN):
    FFPROBE_BIN = "ffprobe"

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
except ImportError:
    build = None
    MediaFileUpload = None
    HttpError = Exception
    Credentials = None
    InstalledAppFlow = None
    Request = None

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # REQUIRED for Community Promotion (Comments)
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"
]

# ── Repo Root ──
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# If we're in "simpler update", go up one more level to get to actual AMTCE root
if os.path.basename(_REPO_ROOT) == "simpler update":
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)

# ── Default YouTube credential paths ──────────────────────────────────────────
CLIENT_SECRET_FILE = os.environ.get("CLIENT_SECRET_FILE", os.path.join(_REPO_ROOT, "Credentials", "youtube", "client_secret.json"))
if not os.path.exists(CLIENT_SECRET_FILE) and os.path.exists(os.path.join(_REPO_ROOT, "Credentials", "client_secret.json")):
    CLIENT_SECRET_FILE = os.path.join(_REPO_ROOT, "Credentials", "client_secret.json")

TOKEN_FILE = os.environ.get("YOUTUBE_TOKEN_FILE", os.path.join(_REPO_ROOT, "Credentials", "youtube", "token.json"))
if not os.path.exists(TOKEN_FILE) and os.path.exists(os.path.join(_REPO_ROOT, "Credentials", "token.json")):
    TOKEN_FILE = os.path.join(_REPO_ROOT, "Credentials", "token.json")

# ── Niche credential root ──
SOCIAL_MEDIA_CREDS_ROOT = "Credentials/social_media"

logger = logging.getLogger("uploader")
logger.setLevel(logging.INFO)

# One-time warning for scope change
if os.path.exists(TOKEN_FILE):
    logger.info("ℹ️ NOTE: Community Promotion/Analytics requires re-authentication (delete token.json). Uploads will continue normally without it.")

def check_platform_lock() -> bool:
    """Checks if the 2h platform safety lock is active (Quota or Copyright)."""
    lock_file = "youtube_platform.lock"
    if os.path.exists(lock_file):
        try:
            with open(lock_file, 'r') as f:
                data = json.load(f)
                timestamp = data.get("timestamp", 0)
                reason = data.get("reason", "Unknown Enforcement")
            
            # Check 2h expiration
            if time.time() - timestamp < 7200:
                return True
            else:
                os.remove(lock_file)
                return False
        except Exception:
            return False
    return False

def set_platform_lock(reason: str):
    """Sets a 2h platform safety lock."""
    with open("youtube_platform.lock", "w") as f:
        json.dump({"timestamp": time.time(), "reason": reason}, f)
    logger.warning(f"🛑 PLATFORM LOCK SET: {reason}. Uploads paused for 2 hours.")
    send_telegram_notification(f"🛑 [MONETIZATION GUARDRAIL] YouTube Lock Active: {reason}\nUploads paused for 2h.")


def send_telegram_notification(message: str):
    """Sends a notification via Telegram if token and admin ID are present."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    admin_raw = os.getenv("TELEGRAM_ADMIN_ID") or os.getenv("TELEGRAM_OWNER_CHAT_ID") or os.getenv("ADMIN_IDS", "")
    if not token or not admin_raw:
        return

    admin_ids = [x.strip() for x in str(admin_raw).split(",") if x.strip()]
    for admin_id in admin_ids:
        # Prevent blasting auth/error messages to public groups/channels
        if admin_id and (str(admin_id).startswith("@") or str(admin_id).startswith("-")):
            logger.warning(f"⚠️ WARNING: admin_id='{admin_id}' looks like a public group/channel. Notification blocked.")
            continue

        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": admin_id, "text": message}).encode("utf-8")
            urllib.request.urlopen(url, data=data, timeout=10)
            logger.info("📡 Telegram notification sent to %s.", admin_id)
        except Exception as e:
            logger.warning(f"⚠️ Failed to send Telegram notification to {admin_id}: {e}")


def _resolve_credential_paths(niche: str = None):
    """
    Niche-Aware Credential Resolver.

    Resolution order:
      1. Niche folder:          Credentials/social_media/<niche>/token.json
      2. General_Fallback:      Credentials/social_media/General_Fallback/token.json
      3. Root default:          Credentials/token.json  (original behaviour)

    Returns (token_path, client_secret_path) — always a valid pair.
    Both files must exist for the tier to be accepted.
    """
    def _tier_valid(folder: str) -> bool:
        t = os.path.join(folder, "token.json")
        c = os.path.join(folder, "client_secret.json")
        if not (os.path.exists(t) and os.path.exists(c)):
            return False
            
        # --- DEMO DETECTION ---
        # Check if the secret file contains "DEMO_" placeholders
        try:
             with open(c, 'r', encoding='utf-8') as f:
                 raw = f.read()
                 if "DEMO_CLIENT_ID" in raw or "DEMO_CLIENT_SECRET" in raw:
                      logger.warning(f"⚠️ [UPLOADER] {c} contains placeholder 'DEMO' credentials. Skipping tier.")
                      return False
        except Exception:
             pass
        return True

    # Tier 1 — requested niche (skip if niche is None or already General_Fallback)
    if niche and niche != "General_Fallback":
        niche_folder = os.path.join(SOCIAL_MEDIA_CREDS_ROOT, niche)
        if _tier_valid(niche_folder):
            logger.info(f"🎯 [UPLOADER] Using niche credentials: {niche}")
            return (
                os.path.join(niche_folder, "token.json"),
                os.path.join(niche_folder, "client_secret.json"),
            )
        else:
            logger.info(f"📂 [UPLOADER] No credentials for niche '{niche}'. Falling back to General_Fallback.")

    # Tier 2 — General_Fallback
    fallback_folder = os.path.join(SOCIAL_MEDIA_CREDS_ROOT, "General_Fallback")
    if _tier_valid(fallback_folder):
        logger.info("🔀 [UPLOADER] Using General_Fallback credentials.")
        return (
            os.path.join(fallback_folder, "token.json"),
            os.path.join(fallback_folder, "client_secret.json"),
        )
    else:
        logger.info("📂 [UPLOADER] No General_Fallback credentials found. Falling back to root Credentials/.")

    # Tier 3 — root default (original TOKEN_FILE / CLIENT_SECRET_FILE)
    logger.info("🔑 [UPLOADER] Using root default credentials.")
    return TOKEN_FILE, CLIENT_SECRET_FILE


def get_valid_credentials(niche: str = None):
    """
    Retrieves and refreshes valid credentials.
    Accepts an optional niche to route to the correct credential folder.
    """
    token_file, client_secret_file = _resolve_credential_paths(niche)

    creds = None
    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
        except Exception:
            logger.warning("Failed to read token file, will run auth flow.")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                # SAVE THE REFRESHED TOKEN
                refresh_retry = 0
                while refresh_retry < 3:
                    try:
                        creds.refresh(Request())
                        with open(token_file, "w", encoding="utf-8") as f:
                            f.write(creds.to_json())
                        logger.info("✅ Token refreshed and saved.")
                        try:
                            from Utilities.github_secret_updater import sync_token_to_github_secret
                            sync_token_to_github_secret(token_file, creds.to_json())
                        except Exception as _se:
                            logger.debug(f"[SECRET_SYNC] Non-fatal sync error: {_se}")
                        break
                    except (ssl.SSLError, socket.error, ConnectionError, Timeout) as re:
                        refresh_retry += 1
                        logger.warning(f"🔄 Transient error during token refresh (attempt {refresh_retry}/3): {re}")
                        if refresh_retry >= 3:
                            raise
                        time.sleep(2 * refresh_retry)
            except Exception as e:
                logger.warning(f"Refresh failed: {e}")
                creds = None
        if not creds:
            logger.warning("🔄 YouTube Token expired/missing. Auto-launching scripts/auth_youtube.py...")
            try:
                script_path = os.path.join(_REPO_ROOT, "scripts", "auth_youtube.py")
                if not os.path.exists(script_path) and os.path.exists(os.path.join(_REPO_ROOT, "simpler update", "scripts", "auth_youtube.py")):
                    script_path = os.path.join(_REPO_ROOT, "simpler update", "scripts", "auth_youtube.py")
                auth_cmd = [sys.executable, script_path]
                if token_file:
                    auth_cmd.extend(["--token", token_file])
                if client_secret_file:
                    auth_cmd.extend(["--secret", client_secret_file])
                admin_id = os.getenv("TELEGRAM_ADMIN_ID") or os.getenv("TELEGRAM_OWNER_CHAT_ID") or os.getenv("TELEGRAM_STORAGE_GROUP_ID")
                if admin_id:
                    auth_cmd.extend(["--admin-id", str(admin_id)])
                
                logger.info(f"🚀 Auto-running YouTube authentication script: {' '.join(auth_cmd)}")
                subprocess.run(auth_cmd, timeout=360, cwd=_REPO_ROOT)
                
                # Reload credentials after script finishes
                if os.path.exists(token_file):
                    creds = Credentials.from_authorized_user_file(token_file, SCOPES)
                    if creds and creds.valid:
                        logger.info("✅ YouTube credentials successfully re-authenticated!")
            except Exception as e:
                logger.error(f"❌ Auto-auth launch failed: {e}")

            if not creds or not creds.valid:
                logger.error(f"❌ Authentication failed for niche {niche or 'Root'}: Token expired or missing.")
                raise Exception(f"YouTube Authentication Failed (Niche: {niche or 'Root'}). Please run 'python scripts/auth_youtube.py' locally to refresh credentials.")
    return creds

def _get_service_sync(niche: str = None):
    import socket
    creds = get_valid_credentials(niche=niche)
    # Force a 60-second default socket timeout so uploads don't hang infinitely on network drop
    socket.setdefaulttimeout(60)
    service = build("youtube", "v3", credentials=creds)
    return service


def _get_service_from_token_json(token_json_str: str, user_id_str: Optional[str] = None):
    """
    Builds a YouTube service directly from a token JSON string (in-memory).
    Used for per-user YouTube uploads without touching disk credential files.
    Persists refreshed token back to user record if refreshed.
    """
    import socket
    import json as _json
    token_data = _json.loads(token_json_str)
    from google.oauth2.credentials import Credentials as GoogleCreds
    from google.auth.transport.requests import Request
    creds = GoogleCreds.from_authorized_user_info(token_data, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            logger.info("[USER YT] Token refreshed in-memory for user %s.", user_id_str or "unspecified")
            if user_id_str:
                try:
                    from Telegram_Storage_Modules.telegram_user_manager import set_user_youtube_token
                    set_user_youtube_token(str(user_id_str), creds.to_json())
                    logger.info("[USER YT] Refreshed token saved to user %s record.", user_id_str)
                except Exception as _pe:
                    logger.warning("[USER YT] Could not persist refreshed token for user %s: %s", user_id_str, _pe)
        except Exception as re:
            logger.warning("[USER YT] Token refresh failed: %s", re)
    if not creds or not creds.valid:
        raise Exception("[USER YT] YouTube token is expired/invalid. User must re-authenticate.")
    socket.setdefaulttimeout(60)
    return build("youtube", "v3", credentials=creds)


def verify_metadata(file_path: str) -> bool:
    """
    Checks if the video file has fresh metadata (Unique ID, Creation Time).
    Returns True if fresh, False otherwise.
    """
    try:
        cmd = [
            FFPROBE_BIN, "-v", "quiet", 
            "-print_format", "json", 
            "-show_format", 
            file_path
        ]
        result = subprocess.check_output(cmd).decode().strip()  # shell=False (default) — cmd is a safe list
        data = json.loads(result)
        tags = data.get("format", {}).get("tags", {})
        
        comment = tags.get("comment", "")
        creation_time = tags.get("creation_time", "")
        
        is_fresh = False
        if "ID:" in comment or "Unique ID:" in comment:
            logger.info(f"✅ Metadata Verified: Found Unique ID in comments.")
            is_fresh = True
        else:
            logger.warning(f"⚠️ Metadata Warning: No 'Unique ID' found in file comments (Comment: {comment[:50]}...).")
            
        if creation_time:
            logger.info(f"✅ Metadata Verified: Creation Time = {creation_time}")
        else:
            # Fallback to filesystem timestamp
            try:
                fs_ctime = time.ctime(os.path.getctime(file_path))
                logger.info(f"✅ Metadata Verified: Filesystem Timestamp = {fs_ctime}")
            except Exception:
                logger.warning(f"⚠️ Metadata Warning: No 'creation_time' found.")
            
        return is_fresh
    except Exception as e:
        logger.warning(f"⚠️ Failed to verify metadata: {e}")
        return False


def refresh_metadata(file_path: str) -> bool:
    """
    Injects a fresh Unique ID into the video metadata without re-encoding.
    """
    try:
        new_id = str(uuid.uuid4())
        temp_path = file_path + ".temp.mp4"
        logger.info(f"🔄 Injecting Fresh Unique ID: {new_id}...")
        
        # ffmpeg -i input -map 0 -c copy -metadata comment="ID:<uuid>" temp
        cmd = [
            FFPROBE_BIN.replace("ffprobe", "ffmpeg"), "-y", 
            "-i", file_path,
            "-map", "0",
            "-c", "copy",
            "-metadata", f"comment=ID:{new_id}",
            temp_path
        ]
        
        # Run safely
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Verify output exists
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            # Atomic Move (with Retry)
            try:
                if os.path.exists(file_path):
                     os.remove(file_path) # Force delete original
                shutil.move(temp_path, file_path)
                logger.info("✅ Metadata refreshed successfully.")
                return True
            except PermissionError:
                 logger.error("⚠️ File is locked! Cannot inject Unique ID. Proceeding with original file (Filesystem Timestamp will be used).")
                 if os.path.exists(temp_path): os.remove(temp_path) # Clean temp
                 return True # Allow upload to proceed
        else:
            logger.error("❌ Metadata refresh failed: Output empty.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to refresh metadata: {e}")
        return True # Soft Fail: Allow upload even if metadata injection fails


def _upload_sync(
    file_path: str,
    hashtags: str = "",
    title: Optional[str] = None,
    description: Optional[str] = None,
    privacy: str = "public",
    publish_at: Optional[str] = None,
    niche: Optional[str] = None,
    user_yt_token_json: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[str]:
    # 0. Check Platform Lock First
    if check_platform_lock():
        logger.warning("Upload Skipped: YouTube Platform Safety Lock is Active.")
        return None

    # Enforce .mp4 extension
    if not file_path.lower().endswith(".mp4"):
        logger.error("Upload rejected: File must be .mp4")
        return None

    # Build service: prefer in-memory user token if provided, else use niche disk credentials
    if user_yt_token_json:
        logger.info("[YOUTUBE] Using in-memory user YouTube token (user_id=%s).", user_id or "unspecified")
        service = _get_service_from_token_json(user_yt_token_json, user_id_str=user_id)
    else:
        service = _get_service_sync(niche=niche)
    logger.info(f"DEBUG: _upload_sync called with title input: '{title}'")
    
    # Robust title logic: Ensure it's not None, not empty, and not just whitespace
    if title:
        # Sanitize: Remove newlines, tabs, and forbidden characters
        final_title = title.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        final_title = final_title.replace("<", "").replace(">", "") # No HTML-like tags
        final_title = final_title.strip()
    else:
        final_title = ""

    if not final_title:
        final_title = "Untitled Video"
        logger.warning("⚠️ Title was empty or whitespace. Defaulting to 'Untitled Video'.")
        
    # Enforce YouTube Length Limit (100 chars)
    if len(final_title) > 95:
        final_title = final_title[:95]
        
    logger.info(f"📋 Final Title for Upload: '{final_title}'")
        
    # --- OPTIONAL COPYRIGHT DISCLAIMER ---
    show_disclaimer = os.getenv("SHOW_COPYRIGHT_DISCLAIMER", "true").lower() == "true"
    if show_disclaimer:
        # Check for custom disclaimer in .env
        env_disclaimer = os.getenv("COPYRIGHT_DISCLAIMER")
        if env_disclaimer:
            DISCLAIMER = "\n\n---\n" + env_disclaimer
        else:
            # Fallback to standard Fair Use disclaimer
            DISCLAIMER = (
                "\n\n---"
                "\nCopyright Disclaimer Under Section 107 of the Copyright Act 1976, allowance is made for \"fair use\" "
                "for purposes such as criticism, commenting, news reporting, teaching, scholarship, and research. "
                "Fair use is a use permitted by copyright statute that might otherwise be infringing. "
                "Non-profit, educational or personal use tips the balance in favor of fair use."
            )
    else:
        DISCLAIMER = ""

    final_description = ((description or "").strip() + ("\n\n" + hashtags if hashtags else "")).strip() + DISCLAIMER

    status_dict = {
        "privacyStatus": privacy,
        "selfDeclaredMadeForKids": False,
    }

    # Handle Scheduling
    if publish_at:
        status_dict["publishAt"] = publish_at
        status_dict["privacyStatus"] = "private" # Must be private for scheduled upload
        logger.info(f"📅 Scheduled Upload: {publish_at}")

    body = {
        "snippet": {
            "title": final_title,
            "description": final_description,
            "categoryId": "22",  # People & Blogs
        },
        "status": status_dict,
    }

    logger.info("🚀 Starting upload request to YouTube API...")
    
    # Verify Metadata Freshness (Must be done BEFORE opening MediaFileUpload to avoid File Locking)
    if not verify_metadata(file_path):
        logger.warning("🔄 Stale/Missing Metadata detected. Engaging Auto-Refresh Safety Net...")
        refresh_metadata(file_path)
        # Verify again just to be sure
        verify_metadata(file_path)

    # 1. Start the Chunked Upload
    media = MediaFileUpload(
        file_path,
        chunksize=1024 * 1024 * 2,  # 2 MB chunks (forces granular progress tracking)
        resumable=True,
        mimetype="video/mp4"
    )
    
    request = service.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )
    logger.info("✅ Upload request created. Starting chunk upload loop...")

    logger.info("🚀 Starting upload: %s", file_path)
    retry = 0
    while True:
        try:
            status, response = request.next_chunk()
            
            # If we made progress, reset the retry counter!
            if status:
                retry = 0 
                if hasattr(status, "progress"):
                    try:
                        progress = int(status.progress() * 100)
                        logger.info(f"📤 Upload progress: {progress}%")
                    except Exception:
                        pass # Fallback if progress() isn't float compatible

            if response is not None:
                video_id = response.get("id")
                if video_id:
                    logger.info("✅ Upload complete: %s", video_id)
                    
                    # 4. Set Thumbnail (Optional)
                    thumb_path = file_path.replace(".mp4", "_thumb.jpg")
                    if os.path.exists(thumb_path):
                         set_youtube_thumbnail(video_id, thumb_path, niche=niche)
                    
                    return f"https://youtube.com/watch?v={video_id}"
                return None

        except HttpError as e:
            # Smart Error Handling
            error_reason = ""
            try:
                error_content = json.loads(e.content.decode('utf-8'))
                error_reason = str(error_content).lower()
            except: 
                error_reason = str(e).lower()

            if "uploadlimitexceeded" in error_reason or "quotaexceeded" in error_reason:
                logger.error("❌ CRITICAL: YouTube Upload Quota Exceeded.")
                set_platform_lock("Upload Quota Exceeded")
                return None
                
            if any(x in error_reason for x in ["copyright", "policy", "strike", "reused_content", "blocked"]):
                logger.error(f"❌ CRITICAL: YouTube Safety Violation detected: {error_reason}")
                set_platform_lock(f"Safety Violation: {error_reason}")
                return None

            logger.warning("⚠️ YouTube API HttpError on chunk (Retry %d/5): %s", retry + 1, e)
            retry += 1
            if retry > 5:
                logger.error("Max retries reached for upload due to HttpError.")
                return None
            time.sleep(2 ** retry)

        except (ssl.SSLError, socket.error, ConnectionError, ProtocolError, ReadTimeoutError, Timeout) as e:
            logger.warning("🌐 Transient Network Error (Retry %d/10): %s", retry + 1, e)
            retry += 1
            if retry > 10:
                logger.error("Max retries reached due to persistent network errors.")
                return None
            time.sleep(min(2 ** retry, 60))

        except Exception as e:
            logger.exception("⚠️ Unexpected Upload Error (Retry %d/5): %s", retry + 1, e)
            retry += 1
            if retry > 5:
                logger.error("Max retries reached for upload due to Exception.")
                return None
            time.sleep(2 ** retry)


def set_youtube_thumbnail(video_id: str, thumb_path: str, niche: Optional[str] = None):
    """Sets a custom thumbnail for a YouTube video."""
    try:
        service = _get_service_sync(niche=niche)
        logger.info(f"🖼️ Setting custom thumbnail for {video_id}: {os.path.basename(thumb_path)}")
        
        request = service.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(thumb_path, mimetype='image/jpeg')
        )
        response = request.execute()
        logger.info(f"✅ Thumbnail updated for {video_id}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Failed to set thumbnail: {e}")
        return False


async def upload_to_youtube(
    file_path: str,
    hashtags: str = "",
    title: Optional[str] = None,
    description: Optional[str] = None,
    privacy: str = "public",
    publish_at: Optional[str] = None,
    niche: Optional[str] = None,
) -> Optional[str]:
    print(f"DEBUG: uploader.upload_to_youtube called for {file_path}")
    return await asyncio.to_thread(_upload_sync, file_path, hashtags, title, description, privacy, publish_at, niche)

# Expose authentication for other modules (e.g., community_promoter)
get_authenticated_service = _get_service_sync


def main():
    """CLI Entry Point for standalone YouTube uploads."""
    parser = argparse.ArgumentParser(description="AMTCE Uploader Module: Standalone YouTube Uploader")
    parser.add_argument("--file", "-f", required=True, help="Path to the video file (.mp4)")
    parser.add_argument("--title", "-t", help="Video title")
    parser.add_argument("--description", "-d", help="Video description")
    parser.add_argument("--hashtags", help="Hashtags to append to description")
    parser.add_argument("--privacy", "-p", choices=["public", "private", "unlisted"], default="public", help="Video privacy status")
    parser.add_argument("--schedule", help="ISO 8601 timestamp for scheduled upload (e.g. 2024-12-31T23:59:59Z)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()

    # Configure Console Output
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    if not os.path.exists(args.file):
        print(f"\n❌ ERROR: File not found: {args.file}")
        sys.exit(1)

    print(f"\n🚀 AMTCE Uploader: Starting Standalone Upload...")
    print(f"   ├─ File:    {os.path.abspath(args.file)}")
    print(f"   ├─ Title:   {args.title or 'N/A'}")
    print(f"   └─ Privacy: {args.privacy}")

    try:
        # Note: _upload_sync is synchronous
        link = _upload_sync(
            args.file,
            hashtags=args.hashtags or "",
            title=args.title,
            description=args.description,
            privacy=args.privacy,
            publish_at=args.schedule
        )

        if link:
            print(f"\n✅ SUCCESS!")
            print(f"   └─ Link: {link}")
        else:
            print(f"\n❌ FAILED: Upload did not return a link. Check logs.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
