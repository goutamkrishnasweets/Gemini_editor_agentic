import os
import sys
import time
import json
import urllib.request
import urllib.parse
import argparse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Utilities.github_secret_updater import sync_token_to_github_secret
except ImportError:
    sync_token_to_github_secret = lambda path, content=None: False

try:
    from Telegram_Storage_Modules.telegram_user_manager import is_admin_user
except ImportError:
    try:
        from Telegram_Storage_Modules.telegram_user_manager import _is_admin_user as is_admin_user
    except ImportError:
        def is_admin_user(uid):
            adm = os.getenv("TELEGRAM_ADMIN_ID", "")
            return str(uid).strip() in [x.strip() for x in adm.split(",") if x.strip()]

# ==============================================================================
# AMTCE YouTube Authentication Script
# 
# To refresh or generate a token manually via CLI, run this script from the AMTCE root directory:
#
#   1. Default (Root credentials):
#      python scripts/auth_youtube.py
#
#   2. Niche-specific credentials (e.g., for 'fashion'):
#      python scripts/auth_youtube.py --secret "Credentials/social_media/fashion/client_secret.json" --token "Credentials/social_media/fashion/token.json"
#
# This will trigger the OAuth flow and save the new token.json to the specified path.
# ==============================================================================

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"
]

DEFAULT_CLIENT_SECRET_FILE = "Credentials/client_secret.json"
DEFAULT_TOKEN_FILE         = "Credentials/token.json"
AUTH_CODE_FILE             = "Credentials/yt_auth_code.txt"

DEVICE_CODE_URL  = "https://oauth2.googleapis.com/device/code"
TOKEN_URL        = "https://oauth2.googleapis.com/token"


# ── Telegram helpers ──────────────────────────────────────────────────────────

def _send_telegram(message: str, token: str, admin_id, button_url: str = None):
    if not token or not admin_id:
        return False

    if isinstance(admin_id, (list, tuple, set)):
        sent_any = False
        for cid in admin_id:
            if _send_telegram(message, token, str(cid), button_url=button_url):
                sent_any = True
        return sent_any

    if isinstance(admin_id, str) and "," in admin_id:
        sent_any = False
        for cid in [x.strip() for x in admin_id.split(",") if x.strip()]:
            if _send_telegram(message, token, cid, button_url=button_url):
                sent_any = True
        return sent_any

    chat_id = str(admin_id).strip()
    try:
        api_url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        if button_url:
            payload["reply_markup"] = json.dumps({
                "inline_keyboard": [[{"text": "🔗 Tap to Authorize", "url": button_url}]]
            })
        data = urllib.parse.urlencode(payload).encode("utf-8")
        urllib.request.urlopen(api_url, data=data, timeout=10)
        print(f"📡 Telegram notification sent to chat_id={chat_id}.")
        return True
    except Exception as e:
        print(f"⚠️ Telegram send with inline button failed for chat_id={chat_id} ({e}) — retrying without inline button...")
        try:
            payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
            data = urllib.parse.urlencode(payload).encode("utf-8")
            urllib.request.urlopen(api_url, data=data, timeout=10)
            print(f"📡 Telegram notification sent to chat_id={chat_id} (fallback text mode).")
            return True
        except Exception as e2:
            print(f"⚠️ Telegram fallback send to chat_id={chat_id} failed: {e2}")
            return False


def _get_telegram_creds(override_admin_id=None):
    """
    Returns (bot_token, admin_chat_id).
    Strictly reads TELEGRAM_ADMIN_ID set manually by the admin in GitHub Secrets/environment.
    """
    try:
        from dotenv import load_dotenv
        for p in ["Credentials/.env", ".env"]:
            if os.path.exists(p):
                load_dotenv(p, override=False)
                break
    except ImportError:
        pass

    token = os.getenv("TELEGRAM_BOT_TOKEN")

    admin_id = override_admin_id or os.getenv("TELEGRAM_ADMIN_ID") or os.getenv("TELEGRAM_OWNER_CHAT_ID")
    if not admin_id:
        admin_id = os.getenv("TELEGRAM_STORAGE_GROUP_ID")

    if not admin_id:
        print("⚠️ No TELEGRAM_ADMIN_ID found in environment secrets. Auth messages cannot be sent.")

    print(f"📡 Auth will notify admin chat_id: {admin_id}")
    return token, admin_id


def _load_client_secret(secret_path):
    with open(secret_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Supports both "installed" and "tv" / "web" top-level keys
    for key in ("installed", "tv", "web"):
        if key in data:
            return data[key]
    raise ValueError(f"Unrecognised client_secret.json format (top-level keys: {list(data.keys())})")


# ── Device Flow (fully automatic — user just goes to URL and signs in) ────────

def _try_device_flow(client_id, client_secret, tg_token, tg_admin, token_path):
    """
    Google Device Authorization Grant.
    Requires app type = 'TV and Limited Input devices' in Google Cloud Console.
    Returns True on success, False if device flow is unsupported.
    """
    print("📺 Trying Device Authorization Flow...")

    # Step 1 — request device + user code
    try:
        req_data = urllib.parse.urlencode({
            "client_id": client_id,
            "scope": " ".join(SCOPES)
        }).encode("utf-8")
        resp = urllib.request.urlopen(DEVICE_CODE_URL, data=req_data, timeout=15)
        device_resp = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"ℹ️ Device flow not available: {e}")
        return False

    if "error" in device_resp:
        print(f"ℹ️ Device flow rejected by Google: {device_resp.get('error')}")
        return False

    device_code      = device_resp["device_code"]
    user_code        = device_resp["user_code"]
    verification_url = device_resp.get("verification_url", "https://www.google.com/device")
    expires_in       = int(device_resp.get("expires_in", 1800))
    interval         = int(device_resp.get("interval", 5))

    print(f"\n📺 DEVICE FLOW ACTIVE")
    print(f"   Go to: {verification_url}")
    print(f"   Enter code: {user_code}\n")

    # Step 2 — tell user via Telegram
    if tg_token and tg_admin:
        msg = (
            f"🔐 <b>YouTube Auth Required</b>\n\n"
            f"1️⃣ Open this link on your phone:\n"
            f"<a href='{verification_url}'>{verification_url}</a>\n\n"
            f"2️⃣ Enter this code:\n"
            f"<code>  {user_code}  </code>\n\n"
            f"3️⃣ Sign in with Google\n\n"
            f"✅ Authorization will complete automatically!"
        )
        _send_telegram(msg, tg_token, tg_admin, button_url=verification_url)

    # Step 3 — poll for completion
    deadline = time.time() + expires_in
    print("⏳ Polling for authorization...")
    while time.time() < deadline:
        time.sleep(interval)
        try:
            poll_data = urllib.parse.urlencode({
                "client_id":     client_id,
                "client_secret": client_secret,
                "device_code":   device_code,
                "grant_type":    "urn:ietf:params:oauth:grant-type:device_code"
            }).encode("utf-8")
            poll_resp = urllib.request.urlopen(TOKEN_URL, data=poll_data, timeout=15)
            token_data = json.loads(poll_resp.read().decode("utf-8"))

            if "access_token" in token_data:
                # Build a token.json compatible with google-auth
                token_json = {
                    "token":         token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token"),
                    "token_uri":     TOKEN_URL,
                    "client_id":     client_id,
                    "client_secret": client_secret,
                    "scopes":        SCOPES,
                }
                token_json_str = json.dumps(token_json, indent=2)
                user_is_admin = is_admin_user(str(tg_admin)) if tg_admin else True

                os.makedirs(os.path.dirname(token_path) or ".", exist_ok=True)
                with open(token_path, "w", encoding="utf-8") as f:
                    f.write(token_json_str)

                if user_is_admin:
                    print(f"✅ Authorized! Admin token saved to {token_path}")
                    sync_token_to_github_secret(token_path, token_json_str)
                else:
                    print(f"✅ Authorized! User {tg_admin} isolated token saved to {token_path}")

                if tg_admin:
                    try:
                        from Telegram_Storage_Modules.telegram_user_manager import set_user_youtube_token
                        set_user_youtube_token(str(tg_admin), token_json_str)
                        print(f"✅ User {tg_admin} youtube_token_json updated in telegram_users.json & synced to Telegram Storage Group Vault")
                    except Exception as _ute:
                        print(f"⚠️ Could not update telegram_users.json: {_ute}")

                if tg_token and tg_admin:
                    msg_text = (
                        "✅ <b>YouTube Authorized!</b>\n\nAdmin token saved. Uploads will resume automatically."
                        if user_is_admin else
                        "✅ <b>YouTube Authorized!</b>\n\nPersonal YouTube token saved securely. Uploads will use your personal channel."
                    )
                    _send_telegram(msg_text, tg_token, tg_admin)
                return True

        except urllib.error.HTTPError as e:
            body = json.loads(e.read().decode("utf-8"))
            err  = body.get("error", "")
            if err == "authorization_pending":
                continue          # normal — user hasn't approved yet
            if err == "slow_down":
                interval += 5
                continue
            if err in ("access_denied", "expired_token"):
                print(f"❌ Device flow failed: {err}")
                if tg_token and tg_admin:
                    _send_telegram(f"❌ Auth failed: {err}. Send /ytcode to try again.", tg_token, tg_admin)
                return True       # Handled (even if denied)
            print(f"⚠️ Unexpected device flow error: {body}")
            return False
        except Exception as e:
            print(f"⚠️ Poll error: {e}")
            time.sleep(interval)

    print("❌ Device flow timed out.")
    if tg_token and tg_admin:
        _send_telegram("⏱️ Auth timed out. Send /ytcode to start again.", tg_token, tg_admin)
    return True  # Handled (just timed out)


# ── Fallback: URL + Telegram code exchange flow ───────────────────────────────

def _fallback_url_flow(secret_path, token_path, tg_token, tg_admin):
    """
    URL + Telegram code exchange flow.
    1. Generates authorization URL immediately.
    2. Sends it to Telegram chat with a direct 'Tap to Authorize' button and HTML link.
    3. Opens browser locally on the host machine.
    4. Waits for code either via:
       - Telegram bot (/ytcode command dropping code in Credentials/yt_auth_code.txt)
       - Direct URL paste into Telegram bot chat
    5. Exchanges code for token.json, saves it, and syncs to GitHub Secrets via GH PAT.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow
    import webbrowser

    flow = InstalledAppFlow.from_client_secrets_file(secret_path, SCOPES)
    flow.redirect_uri = "http://localhost"

    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
    print(f"\n🔗 AUTH URL:\n{auth_url}\n")

    # 1. Immediately notify user on Telegram with the authorization link & instructions
    if tg_token and tg_admin:
        msg = (
            f"🔐 <b>YouTube Authorization Required</b>\n\n"
            f"1️⃣ Tap <b>Tap to Authorize</b> below (or click the link) to sign in with Google:\n"
            f"<a href='{auth_url}'>👉 Click here to Sign in with Google</a>\n\n"
            f"2️⃣ Grant the requested YouTube permissions.\n\n"
            f"3️⃣ Google will redirect your browser to a page starting with <code>http://localhost/?code=...</code>.\n"
            f"<i>(This page may show 'site cannot be reached' — this is 100% normal!)</i>\n\n"
            f"4️⃣ Copy the entire URL from your browser's address bar and reply to this bot with:\n"
            f"<code>/ytcode YOUR_COPIED_URL</code>\n"
            f"<i>(or simply paste the URL directly into this chat)</i>"
        )
        _send_telegram(msg, tg_token, tg_admin, button_url=auth_url)
        print("📲 Dispatched YouTube Authorization link to Telegram admin chat.")

    # 2. Open browser locally only on desktop systems (skip in GitHub Actions / headless CI)
    is_headless = (
        os.getenv("GITHUB_ACTIONS") == "true"
        or os.getenv("CI") == "true"
        or (sys.platform != "win32" and not os.environ.get("DISPLAY"))
    )
    if not is_headless:
        try:
            webbrowser.open(auth_url)
        except Exception:
            pass

    # 3. Clean up any stale auth code file before starting polling
    user_code_file = f"Credentials/yt_auth_code_{tg_admin}.txt" if tg_admin else None
    for p in ([user_code_file, AUTH_CODE_FILE] if user_code_file else [AUTH_CODE_FILE]):
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    # 4. Poll for code dropped by /ytcode Telegram bot command
    print("⏳ Waiting for authorization code via Telegram /ytcode or localhost URL (5 min)...")
    deadline = time.time() + 300
    while time.time() < deadline:
        chosen_file = None
        if user_code_file and os.path.exists(user_code_file):
            chosen_file = user_code_file
        elif os.path.exists(AUTH_CODE_FILE):
            chosen_file = AUTH_CODE_FILE

        if chosen_file:
            try:
                with open(chosen_file, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                try:
                    os.remove(chosen_file)
                except Exception:
                    pass

                if raw:
                    # Auto-extract code parameter if full URL was pasted
                    if raw.startswith("http"):
                        parsed = urllib.parse.urlparse(raw)
                        qs = urllib.parse.parse_qs(parsed.query)
                        raw = qs.get("code", [raw])[0]

                    print(f"🔑 Exchanging authorization code for OAuth token...")
                    flow.fetch_token(code=raw)
                    creds = flow.credentials
                    token_json_str = creds.to_json()

                    user_is_admin = is_admin_user(str(tg_admin)) if tg_admin else True

                    os.makedirs(os.path.dirname(token_path) or ".", exist_ok=True)
                    with open(token_path, "w", encoding="utf-8") as f:
                        f.write(token_json_str)

                    if user_is_admin:
                        print(f"✅ Admin token saved to {token_path}")
                        sync_token_to_github_secret(token_path, token_json_str)
                    else:
                        print(f"✅ User {tg_admin} personal token saved to {token_path}")

                    # Update telegram_users.json and sync to Telegram Storage Group Vault
                    if tg_admin:
                        try:
                            from Telegram_Storage_Modules.telegram_user_manager import set_user_youtube_token
                            set_user_youtube_token(str(tg_admin), token_json_str)
                            print(f"✅ User {tg_admin} youtube_token_json updated in telegram_users.json & synced to Telegram Storage Group Vault")
                        except Exception as _ute:
                            print(f"⚠️ Could not update telegram_users.json: {_ute}")

                    if tg_token and tg_admin:
                        msg_text = (
                            f"✅ <b>YouTube Authorized Successfully!</b>\n\n"
                            f"YouTube OAuth Token Synced as your secret 🔒\n\n"
                            f"🚀 Uploads will resume automatically."
                            if user_is_admin else
                            f"✅ <b>YouTube Authorized Successfully!</b>\n\n"
                            f"YouTube OAuth Token Synced to your personal account 🔒\n\n"
                            f"🚀 Uploads will resume automatically using your personal channel."
                        )
                        _send_telegram(msg_text, tg_token, tg_admin)
                    return True
            except Exception as e:
                print(f"❌ Code exchange failed: {e}")
                if tg_token and tg_admin:
                    _send_telegram(f"❌ <b>Code exchange failed:</b>\n<code>{e}</code>\n\nSend /ytcode to try again.", tg_token, tg_admin)
                deadline = time.time() + 180

        time.sleep(2)

    print("❌ Timed out waiting for auth code.")
    if tg_token and tg_admin:
        _send_telegram("⏱️ <b>Auth timed out.</b> Send /ytcode to start again.", tg_token, tg_admin)
    return False


# ── Main entry point ──────────────────────────────────────────────────────────

def authenticate(client_secret_file=None, token_file=None, admin_id=None):
    tg_token, tg_admin = _get_telegram_creds(override_admin_id=admin_id)
    is_admin = is_admin_user(str(tg_admin)) if tg_admin else True

    # Check for personal client secret if caller is non-admin
    user_secret_data = None
    if tg_admin and not is_admin:
        try:
            from Telegram_Storage_Modules.telegram_user_manager import load_all_users
            users = load_all_users()
            u_rec = users.get(str(tg_admin), {})
            user_secret_data = u_rec.get("youtube_client_secret") or os.getenv(f"USER_{tg_admin}_CLIENT_SECRET_JSON")
        except Exception:
            pass

    targets = []
    if not is_admin:
        # Non-admin user flow: strictly isolate token and secret paths
        isolated_token_path = token_file or f"Credentials/user_{tg_admin}_token.json"
        if client_secret_file and os.path.exists(client_secret_file):
            targets.append((client_secret_file, isolated_token_path))
        elif user_secret_data:
            user_secret_path = f"Credentials/user_{tg_admin}_client_secret.json"
            os.makedirs(os.path.dirname(user_secret_path) or ".", exist_ok=True)
            with open(user_secret_path, "w", encoding="utf-8") as sf:
                sf.write(user_secret_data.strip())
            targets.append((user_secret_path, isolated_token_path))
        else:
            # Fall back to server's shared client secret for OAuth authorization
            shared_secret = None
            for s_cand in [DEFAULT_CLIENT_SECRET_FILE, "Credentials/youtube/client_secret.json"]:
                if os.path.exists(s_cand):
                    shared_secret = s_cand
                    break
            if shared_secret:
                targets.append((shared_secret, isolated_token_path))
    else:
        # Admin flow
        # Auto-hydrate client_secret.json from env or admin user state if missing on disk
        if not client_secret_file and not os.path.exists(DEFAULT_CLIENT_SECRET_FILE) and not os.path.exists("Credentials/youtube/client_secret.json"):
            env_secret = os.getenv("CLIENT_SECRET_JSON") or os.getenv("YOUTUBE_CLIENT_SECRET")
            if not env_secret and tg_admin:
                try:
                    from Telegram_Storage_Modules.telegram_user_manager import load_all_users
                    users = load_all_users()
                    u_rec = users.get(str(tg_admin), {})
                    if isinstance(u_rec, dict) and u_rec.get("youtube_client_secret"):
                        env_secret = u_rec["youtube_client_secret"]
                except Exception:
                    pass
            if env_secret:
                for _fpath in ["Credentials/youtube/client_secret.json", DEFAULT_CLIENT_SECRET_FILE]:
                    os.makedirs(os.path.dirname(_fpath), exist_ok=True)
                    with open(_fpath, "w", encoding="utf-8") as _sf:
                        _sf.write(env_secret.strip())
                print("🔑 Auto-hydrated client_secret.json from environment/state.")

        if client_secret_file:
            targets.append((client_secret_file, token_file or DEFAULT_TOKEN_FILE))
        else:
            # Scan all known credential directories
            possible_pairs = [
                ("Credentials/social_media/Fashion/client_secret.json", "Credentials/social_media/Fashion/token.json"),
                ("Credentials/social_media/NSFW/client_secret.json", "Credentials/social_media/NSFW/token.json"),
                ("Credentials/social_media/General_Fallback/client_secret.json", "Credentials/social_media/General_Fallback/token.json"),
                (DEFAULT_CLIENT_SECRET_FILE, DEFAULT_TOKEN_FILE),
                ("Credentials/youtube/client_secret.json", "Credentials/youtube/token.json"),
            ]
            for s_path, t_path in possible_pairs:
                if os.path.exists(s_path) and (s_path, t_path) not in targets:
                    targets.append((s_path, t_path))

    if not targets:
        msg = (
            f"❌ <b>YouTube Auth FAILED</b>\n\n"
            f"No valid <b>client_secret.json</b> found in any credential folder.\n\n"
            f"Download from Google Cloud Console → APIs &amp; Services → Credentials."
        )
        print("❌ No client_secret.json found!")
        if tg_token and tg_admin:
            _send_telegram(msg, tg_token, tg_admin)
        return

    print(f"🚀 Starting YouTube Authentication across {len(targets)} target(s)...")

    seen_client_ids = set()
    for secret_path, token_path in targets:
        print(f"\n🔑 Authenticating target: {secret_path}")
        try:
            secret = _load_client_secret(secret_path)
        except Exception as e:
            print(f"❌ Failed to read {secret_path}: {e}")
            continue

        client_id     = secret["client_id"]
        client_secret = secret["client_secret"]

        if client_id in seen_client_ids:
            print(f"⏩ Client ID '{client_id[:15]}...' already authenticated in this run. Skipping duplicate prompt.")
            if is_admin:
                for _src in [DEFAULT_TOKEN_FILE, "Credentials/youtube/token.json", token_path]:
                    if os.path.exists(_src):
                        try:
                            os.makedirs(os.path.dirname(token_path) or ".", exist_ok=True)
                            with open(_src, "r", encoding="utf-8") as _rf, open(token_path, "w", encoding="utf-8") as _wf:
                                _wf.write(_rf.read())
                            break
                        except Exception:
                            pass
            continue

        seen_client_ids.add(client_id)

        # Try Device Flow first (fully automatic — no code pasting)
        handled = _try_device_flow(client_id, client_secret, tg_token, tg_admin, token_path)

        if not handled:
            # Device flow unsupported → fallback to URL + /ytcode paste
            print("⬇️ Falling back to URL auth flow...")
            success = _fallback_url_flow(secret_path, token_path, tg_token, tg_admin)
            if success and is_admin:
                # Mirror token to both root and youtube/ paths for convenience (ADMIN ONLY)
                for _dst in [DEFAULT_TOKEN_FILE, "Credentials/youtube/token.json"]:
                    if _dst != token_path:
                        try:
                            os.makedirs(os.path.dirname(_dst) or ".", exist_ok=True)
                            with open(token_path, "r", encoding="utf-8") as _rf, open(_dst, "w", encoding="utf-8") as _wf:
                                _wf.write(_rf.read())
                        except Exception:
                            pass


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.basename(root_dir) == "simpler update":
        root_dir = os.path.dirname(root_dir)
    os.chdir(root_dir)

    parser = argparse.ArgumentParser(description="AMTCE YouTube Authentication")
    parser.add_argument("--secret",   help="Path to client_secret.json")
    parser.add_argument("--token",    help="Path to save token.json")
    parser.add_argument("--admin-id", help="Telegram Admin Chat ID to send auth links to")
    args = parser.parse_args()

    authenticate(client_secret_file=args.secret, token_file=args.token, admin_id=args.admin_id)
