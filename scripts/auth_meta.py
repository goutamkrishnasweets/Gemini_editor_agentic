"""
scripts/auth_meta.py — 1-Click Meta (Instagram & Facebook) OAuth & Token Manager
================================================================================
Automates 1-click authorization for Meta Graph API publishing:
  1. Builds the OAuth authorization URL with all required publishing scopes:
     (instagram_basic, instagram_content_publish, pages_show_list, pages_read_engagement, pages_manage_posts).
  2. Launches a local callback listener (localhost:8080) and opens your browser.
  3. Captures the authorization code automatically upon clicking 'Allow' in Facebook.
  4. Automatically upgrades to 60-day Long-Lived User Access Token.
  5. Auto-discovers all Facebook Pages and linked Instagram Business Accounts.
  6. Saves META_PAGE_ID, META_PAGE_TOKEN, IG_BUSINESS_ID, IG_BUSINESS_TOKEN to .env and syncs to GitHub Secrets.

Usage:
  python scripts/auth_meta.py
  python scripts/auth_meta.py --app-id YOUR_APP_ID --app-secret YOUR_APP_SECRET
"""

import os
import sys
import time
import json
import socket
import argparse
import webbrowser
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

# Set UTF-8 encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add repo root to path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from dotenv import load_dotenv
    for p in [os.path.join(_REPO_ROOT, "Credentials", ".env"), os.path.join(_REPO_ROOT, ".env")]:
        if os.path.exists(p):
            load_dotenv(p, override=False)
except ImportError:
    pass

from Publishing_Modules.meta_uploader import (
    build_meta_oauth_url,
    exchange_meta_oauth_code,
    save_and_sync_meta_credentials,
    get_meta_graph_version,
)


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Local HTTP handler that intercepts the OAuth callback redirect."""
    server_auth_code: Optional[str] = None
    server_error: Optional[str] = None

    def log_message(self, format, *args):
        pass  # Suppress default HTTP server noisy logs

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/callback":
            query = urllib.parse.parse_qs(parsed.query)
            if "code" in query:
                _OAuthCallbackHandler.server_auth_code = query["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                html = """
                <!DOCTYPE html>
                <html>
                <head><title>Meta Authorization Successful</title></head>
                <body style="font-family: Arial, sans-serif; text-align: center; padding-top: 60px; background-color: #f7f9fa;">
                  <div style="display: inline-block; padding: 40px 50px; background: white; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                    <h1 style="color: #10B981; margin-bottom: 12px;">✅ Authorization Successful!</h1>
                    <p style="color: #374151; font-size: 16px;">Meta permissions granted. Facebook Pages & Instagram Business accounts are now syncing...</p>
                    <p style="color: #9CA3AF; font-size: 14px; margin-top: 20px;">You can now safely close this tab and return to your terminal.</p>
                  </div>
                </body>
                </html>
                """
                self.wfile.write(html.encode("utf-8"))
            elif "error" in query:
                _OAuthCallbackHandler.server_error = query.get("error_description", query["error"])[0]
                self.send_response(400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                err_html = f"<html><body><h2>⚠️ Authorization Failed: {_OAuthCallbackHandler.server_error}</h2></body></html>"
                self.wfile.write(err_html.encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def _send_telegram_notification(message: str, auth_url: Optional[str] = None):
    """Sends 1-click authorization link to admin Telegram chat if configured."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    admin_id = os.getenv("TELEGRAM_ADMIN_ID") or os.getenv("TELEGRAM_OWNER_CHAT_ID")
    if not bot_token or not admin_id:
        return

    admin_ids = [x.strip() for x in str(admin_id).split(",") if x.strip()]
    for cid in admin_ids:
        try:
            import urllib.request
            api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {"chat_id": cid, "text": message, "parse_mode": "HTML"}
            if auth_url:
                payload["reply_markup"] = json.dumps({
                    "inline_keyboard": [[{"text": "🔗 Tap to Authorize Meta / Instagram", "url": auth_url}]]
                })
            data = urllib.parse.urlencode(payload).encode("utf-8")
            urllib.request.urlopen(api_url, data=data, timeout=10)
        except Exception:
            pass


def run_meta_auth(app_id: Optional[str] = None, app_secret: Optional[str] = None, port: int = 8080, no_browser: bool = False):
    print("=" * 70)
    print("🚀 AMTCE 1-Click Meta (Instagram & Facebook) OAuth & Token Manager")
    print(f"   API Version: {get_meta_graph_version()}")
    print("=" * 70)

    # 1. Resolve App ID & Secret
    resolved_id = (app_id or os.getenv("META_APP_ID") or os.getenv("FACEBOOK_APP_ID") or os.getenv("IG_APP_ID") or "").strip()
    resolved_secret = (app_secret or os.getenv("META_APP_SECRET") or os.getenv("FACEBOOK_APP_SECRET") or os.getenv("IG_APP_SECRET") or "").strip()

    if not resolved_id:
        print("\n📌 Meta App ID not found in environment.")
        resolved_id = input("👉 Enter your Meta App ID: ").strip()

    if not resolved_secret:
        print("\n📌 Meta App Secret not found in environment.")
        resolved_secret = input("👉 Enter your Meta App Secret: ").strip()

    if not resolved_id or not resolved_secret:
        print("❌ ERROR: Both Meta App ID and Meta App Secret are required.")
        sys.exit(1)

    redirect_uri = f"http://localhost:{port}/callback"

    # Build 1-click URL
    auth_url = build_meta_oauth_url(app_id=resolved_id, redirect_uri=redirect_uri)

    print("\n🔗 1-Click Authorization URL Generated:")
    print(f"   {auth_url}\n")

    # Send link to Telegram if admin is active
    _send_telegram_notification(
        "⚡ <b>Meta / Instagram 1-Click Authorization Ready</b>\n\nTap the button below to grant publishing permissions once:",
        auth_url=auth_url
    )

    auth_code = None

    # Check if local port is available
    if not _is_port_in_use(port):
        server = HTTPServer(("localhost", port), _OAuthCallbackHandler)
        server.timeout = 1.0
        print(f"🌐 Local callback listener started on http://localhost:{port}/callback")
        print("⏳ Waiting for your approval in the browser (Timeout: 180s)...")

        if not no_browser:
            try:
                webbrowser.open(auth_url)
            except Exception:
                pass

        start_time = time.time()
        while time.time() - start_time < 180:
            server.handle_request()
            if _OAuthCallbackHandler.server_auth_code:
                auth_code = _OAuthCallbackHandler.server_auth_code
                break
            if _OAuthCallbackHandler.server_error:
                print(f"\n❌ Authorization rejected by user: {_OAuthCallbackHandler.server_error}")
                sys.exit(1)

        server.server_close()
    else:
        print(f"⚠️ Port {port} is already in use. Switching to manual redirect paste mode.")

    # Fallback to manual paste if callback wasn't received
    if not auth_code:
        print("\n📌 If your browser didn't redirect automatically, paste the redirected URL or 'code=' parameter here:")
        manual_in = input("👉 Paste here: ").strip()
        auth_code = manual_in

    if not auth_code:
        print("\n❌ Authorization cancelled or timed out. No code received.")
        sys.exit(1)

    print("\n🔄 Exchanging code & discovering linked Facebook Pages & Instagram accounts...")
    discovery = exchange_meta_oauth_code(
        code=auth_code,
        app_id=resolved_id,
        app_secret=resolved_secret,
        redirect_uri=redirect_uri
    )

    if not discovery.get("success"):
        print(f"\n❌ Discovery failed: {discovery.get('error')}")
        sys.exit(1)

    print("\n✅ Discovered Accounts:")
    pages = discovery.get("pages", [])
    for idx, p in enumerate(pages, 1):
        print(f"   [{idx}] Facebook Page: {p.get('page_name')} (ID: {p.get('page_id')})")
        if p.get("instagram_id"):
            print(f"       └─ Linked Instagram: @{p.get('instagram_username')} (ID: {p.get('instagram_id')})")
        else:
            print("       └─ No Instagram Business Account linked to this page.")

    # Save credentials
    ok, save_msg = save_and_sync_meta_credentials(discovery)
    if ok:
        print("\n🎉 SUCCESS!")
        print(f"   {save_msg}")
        print("   • Saved to Credentials/.env and .env")
        print("   • Synced to GitHub Secrets (META_PAGE_ID, META_PAGE_TOKEN, IG_BUSINESS_ID, IG_BUSINESS_TOKEN)")
        print("\n🚀 You can now publish Instagram Reels and Facebook Page videos with zero manual setup!")
    else:
        print(f"\n⚠️ Notice: {save_msg}")


def main():
    parser = argparse.ArgumentParser(description="AMTCE 1-Click Meta OAuth & Token Manager")
    parser.add_argument("--app-id", help="Meta / Facebook App ID")
    parser.add_argument("--app-secret", help="Meta / Facebook App Secret")
    parser.add_argument("--port", type=int, default=8080, help="Local callback port (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    run_meta_auth(app_id=args.app_id, app_secret=args.app_secret, port=args.port, no_browser=args.no_browser)


if __name__ == "__main__":
    main()

