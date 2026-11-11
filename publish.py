#!/usr/bin/env python3
"""
publish.py — Standalone Multi-Platform Video Publisher CLI.
================================================================================
Directly publish any video file to your configured social channels without running
the full Telegram bot, scraper, or video editor.

Supported Platforms:
  • Buffer   : TikTok, Pinterest, X (Twitter)
  • YouTube  : Shorts / Videos
  • Meta     : Instagram Reels & Facebook Pages
  • Telegram : Channel / Public Group
  • Make.com : Auxiliary Webhook

Usage Examples:
  # Check status of all social publishing credentials:
  python publish.py --status

  # Test & list your connected Buffer channels (TikTok, Pinterest, X):
  python publish.py --buffer-status

  # Publish to Buffer only (TikTok, Pinterest, X):
  python publish.py --video my_reel.mp4 --title "Summer Style Tips" --platform buffer

  # Publish to specific Buffer channels:
  python publish.py --video my_reel.mp4 --title "Summer Style Tips" --platform buffer --channels tiktok,x

  # Broadcast across ALL configured platforms simultaneously:
  python publish.py --video my_reel.mp4 --title "Summer Style Tips" --platform all
"""

import os
import sys
import logging
import argparse

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)
    load_dotenv(os.path.join(_REPO_ROOT, "Credentials", ".env"), override=False)
except ImportError:
    pass

from Publishing_Modules.media_publisher_main import run_phase4_publishing
from Publishing_Modules.buffer_publisher import (
    publish_to_buffer,
    is_buffer_configured,
    get_buffer_profiles,
    _get_buffer_token,
)
from Publishing_Modules.make_publisher import is_make_configured

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("publish_cli")


def print_status():
    print("\n" + "=" * 65)
    print("📊 SOCIAL PUBLISHING CHANNELS STATUS CHECK")
    print("=" * 65)
    yt_cfg = os.path.exists("Credentials/youtube/token.json") or bool(os.getenv("TOKEN_JSON"))
    print(f"  • YouTube Shorts:        {'✅ Configured' if yt_cfg else '❌ Not Set'}")

    meta_cfg = bool(os.getenv("INSTAGRAM_ACCESS_TOKEN") or os.getenv("META_ACCESS_TOKEN"))
    print(f"  • Meta (Instagram & FB): {'✅ Configured' if meta_cfg else '❌ Not Set'}")

    tg_cfg = bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_PUBLIC_GROUP_ID"))
    print(f"  • Telegram Channel:      {'✅ Configured' if tg_cfg else '❌ Not Set'}")

    make_cfg = is_make_configured()
    print(f"  • Make.com Webhook:      {'✅ Configured' if make_cfg else '❌ Not Set'}")

    buf_cfg = is_buffer_configured()
    print(f"  • Buffer API (TikTok/Pin/X): {'✅ Configured' if buf_cfg else '❌ Not Set'}")
    if buf_cfg:
        tok = _get_buffer_token()
        profiles = get_buffer_profiles(tok) if tok else []
        if profiles:
            for p in profiles:
                svc = p.get('service', 'channel').capitalize()
                user = p.get('service_username') or p.get('id')
                pid = p.get('id')
                print(f"      ↳ [{svc}] @{user} (id: {pid})")
        else:
            print("      ↳ (Token set, but no channels connected yet in Buffer)")
    print("=" * 65)
    print("💡 To configure Buffer, set BUFFER_ACCESS_TOKEN in .env or run: python publish.py --set-buffer <token>\n")


def print_buffer_status():
    tok = _get_buffer_token()
    if not tok:
        print("\n❌ BUFFER_ACCESS_TOKEN is not set.")
        print("   Get token: https://publish.buffer.com/settings/api?tab=personal-access")
        print("   Set it via: python publish.py --set-buffer <your_token>\n")
        sys.exit(1)

    print(f"\n🔑 Buffer Token: {tok[:10]}... (active)")
    profiles = get_buffer_profiles(tok, force_refresh=True)
    if profiles:
        print(f"✅ Discovered {len(profiles)} connected channel(s):")
        for p in profiles:
            svc = p.get('service', 'channel').capitalize()
            user = p.get('service_username') or p.get('id')
            pid = p.get('id')
            print(f"   • [{svc}] @{user} (ID: {pid})")
    else:
        print("⚠️ No channels connected yet in your Buffer account.")
        print("   Connect accounts at: https://publish.buffer.com/channels")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Multi-Platform Video Publisher CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", "-v", type=str, default=None, help="Path to video file (.mp4) to publish")
    parser.add_argument("--title", "-t", type=str, default="", help="Video title / caption headline")
    parser.add_argument("--desc", "-d", type=str, default="", help="Full caption / description")
    parser.add_argument("--tags", type=str, default="#viral #shorts #trending", help="Hashtags string (e.g. '#fashion #style')")
    parser.add_argument(
        "--platform", "-p", type=str, default="all",
        help="Target platform(s): 'all', 'buffer', 'youtube', 'meta', 'instagram', 'facebook', 'telegram', 'make' (comma-separated)"
    )
    parser.add_argument(
        "--channels", "-c", type=str, default="tiktok,pinterest,x",
        help="Sub-channels for Buffer (comma-separated: tiktok,pinterest,x)"
    )
    parser.add_argument("--affiliate", "-a", type=str, default=None, help="Optional affiliate or product destination URL")
    parser.add_argument("--queue", action="store_true", help="Queue to Buffer calendar instead of posting immediately")
    parser.add_argument("--status", action="store_true", help="Show configuration status for all publishing platforms")
    parser.add_argument("--buffer-status", action="store_true", help="Test Buffer token and list connected channels")
    parser.add_argument("--set-buffer", type=str, default=None, metavar="TOKEN", help="Save a Buffer Access Token to your environment")

    args = parser.parse_args()

    if args.set_buffer:
        from Publishing_Modules.buffer_publisher import set_user_buffer_token
        ok = set_user_buffer_token("admin", args.set_buffer)
        if ok:
            print("✅ Buffer Access Token successfully saved!")
            print_buffer_status()
        else:
            print("❌ Failed to save Buffer Access Token.")
        sys.exit(0)

    if args.status:
        print_status()
        sys.exit(0)

    if args.buffer_status:
        print_buffer_status()
        sys.exit(0)

    if not args.video:
        parser.print_help()
        print("\n❌ Error: Please specify a video file using --video <path_to_video.mp4>")
        sys.exit(1)

    if not os.path.isfile(args.video):
        print(f"\n❌ Error: File not found: {args.video}")
        sys.exit(1)

    target_platforms = [p.strip().lower() for p in args.platform.split(",") if p.strip()]
    vid_title = args.title or os.path.splitext(os.path.basename(args.video))[0]

    print("\n" + "=" * 65)
    print("🚀 STANDALONE PUBLISH JOB")
    print("=" * 65)
    print(f"📁 Video:     {os.path.abspath(args.video)}")
    print(f"📝 Title:     {vid_title}")
    print(f"🌐 Platforms: {', '.join(target_platforms)}")
    if "buffer" in target_platforms or "all" in target_platforms:
        print(f"📱 Buffer:    {args.channels}")
    print("=" * 65 + "\n")

    # If publishing exclusively to Buffer:
    if target_platforms == ["buffer"]:
        ch_list = [c.strip().lower() for c in args.channels.split(",") if c.strip()]
        res = publish_to_buffer(
            video_path=args.video,
            title=vid_title,
            description=args.desc,
            tags=args.tags,
            affiliate_link=args.affiliate,
            channels=ch_list,
            now=not args.queue,
        )
        print("\n" + "=" * 65)
        print(f"🎯 BUFFER RESULT: {res.get('status', 'unknown').upper()}")
        print(f"Published to: {res.get('published_to') or 'none'}")
        print(f"Failed on:    {res.get('failed_on') or 'none'}")
        for ch, details in (res.get("results") or {}).items():
            st = details.get("status", "unknown")
            msg = details.get("update_id") or details.get("error") or details.get("reason") or "ok"
            print(f"  • {ch.upper()}: {st} ({msg})")
        print("=" * 65 + "\n")
        return

    # Multi-platform or All
    res = run_phase4_publishing(
        video_path=args.video,
        title=vid_title,
        description=args.desc,
        tags=args.tags,
        target_platforms=target_platforms,
    )

    print("\n" + "=" * 65)
    print(f"🎉 PUBLISH SUMMARY — Success: {res.get('success')} (Published count: {res.get('published_count', 0)})")
    for plat, pdata in res.get("platforms", {}).items():
        st = pdata.get("status", "unknown")
        info = pdata.get("link") or pdata.get("url") or pdata.get("update_id") or pdata.get("error") or pdata.get("message") or ""
        print(f"  • {plat.upper()}: {st} ({info})")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
