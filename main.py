"""
main.py — Master Orchestration & Telegram Bot Entrypoint
=========================================================
Kingdom: Master System Controller
Purpose: Connect Phase 1 (Ingestion & Scraping), Phase 2 & 3 (AI Perception & Rendering),
         Phase 4 (Multi-Platform Publishing), and Telegram Admin Review Buttons.

Telegram 4-Button Approval Suite:
  - [✅ Clean (Yes)]      callback_data="wm_clean"        -> Confirm clip quality & watermark clean
  - [❌ Bad (No)]         callback_data="wm_bad"          -> Flag quality issue & trigger re-edit
  - [🚀 Approve & Post]   callback_data="approve_post"    -> Publish to YouTube, Meta, TikTok
  - [🗑️ Reject]           callback_data="reject_discard"  -> Discard ticket & delete local .mp4
"""

import os
import sys
import time
import json
import re
import asyncio
import threading
import logging
import argparse
import signal
from typing import Optional, Dict, Any, List, Tuple, Union, Set
# ── Windows Job Object: Auto-Kill Subprocess Tree on Exit ─────────────────────
_GLOBAL_WINDOWS_JOB_HANDLE = None

def _setup_windows_job_object():
    global _GLOBAL_WINDOWS_JOB_HANDLE
    if sys.platform != "win32":
        return
    try:
        import ctypes
        import ctypes.wintypes
        kernel32 = ctypes.windll.kernel32

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('PerProcessUserTimeLimit', ctypes.c_int64),
                ('PerJobUserTimeLimit', ctypes.c_int64),
                ('LimitFlags', ctypes.c_uint32),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', ctypes.c_uint32),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', ctypes.c_uint32),
                ('SchedulingClass', ctypes.c_uint32),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('ReadOperationCount', ctypes.c_uint64),
                ('WriteOperationCount', ctypes.c_uint64),
                ('OtherOperationCount', ctypes.c_uint64),
                ('ReadTransferCount', ctypes.c_uint64),
                ('WriteTransferCount', ctypes.c_uint64),
                ('OtherTransferCount', ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ('IoInfo', IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryLimit', ctypes.c_size_t),
                ('PeakJobMemoryLimit', ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.restype = ctypes.wintypes.HANDLE
        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        kernel32.AssignProcessToJobObject.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.HANDLE]

        _GLOBAL_WINDOWS_JOB_HANDLE = kernel32.CreateJobObjectW(None, None)
        if _GLOBAL_WINDOWS_JOB_HANDLE:
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = 0x2000 # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            res = kernel32.SetInformationJobObject(
                _GLOBAL_WINDOWS_JOB_HANDLE,
                9, # JobObjectExtendedLimitInformation
                ctypes.byref(info),
                ctypes.sizeof(info)
            )
            if res:
                process_handle = kernel32.GetCurrentProcess()
                kernel32.AssignProcessToJobObject(_GLOBAL_WINDOWS_JOB_HANDLE, process_handle)
    except Exception:
        pass

_setup_windows_job_object()

# Instant Ctrl+C Termination Handler
def _instant_sigint_handler(signum, frame):
    print("\n🛑 [INSTANT EXIT] Ctrl+C detected. Terminating process immediately...", flush=True)
    os._exit(130)

try:
    signal.signal(signal.SIGINT, _instant_sigint_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _instant_sigint_handler)
except Exception:
    pass

# Ensure UTF-8 output and immediate line buffering for CI / Docker / Windows consoles
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

# Ensure the canonical workspace root is on sys.path.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ── Logging & Polling Filter Setup ───────────────────────────────────────────
class UnbufferedStreamHandler(logging.StreamHandler):
    """Ensures every log record is instantly written and flushed to stdout for live CI console streaming."""
    def emit(self, record):
        try:
            super().emit(record)
            self.flush()
        except Exception:
            self.handleError(record)

_log_handler = UnbufferedStreamHandler(sys.stdout)
_log_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | [%(name)s] %(message)s", datefmt="%H:%M:%S"))

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_root.handlers = [_log_handler]

logger = logging.getLogger("main_orchestrator")

class PollingFilter(logging.Filter):
    def filter(self, record):
        return "getUpdates" not in record.getMessage()

for lib in ["httpx", "telegram", "telegram.ext", "httpcore"]:
    l = logging.getLogger(lib)
    l.setLevel(logging.WARNING)
    l.addFilter(PollingFilter())

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, "Credentials", ".env"), override=True)
except ImportError:
    pass

# ── Import Modular System Packages ───────────────────────────────────────────
from Import_Modules.phase1_imports import run_phase1_ingestion
from Import_Modules.phase2_imports import run_phase2_orchestration
from Downloader_Modules.scheduled_scraper_manager import run_scheduled_scraper_batch, get_rotated_max_two_accounts
from Main_Modules.master_ai_editor import MasterAIEditor
from Publishing_Modules.queue_publisher import PublishQueue
from Telegram_Storage_Modules.telegram_session_manager import session_manager
from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
from Telegram_Storage_Modules import download_telegram_document
from Core_Modules import MAX_RETRIES

vault_indexer = TelegramVaultIndexer()


# ── Platform Selection Keyboard & State ───────────────────────────────────────
user_selected_platform = {}  # {chat_id: "instagram" | "youtube" | "tiktok" | "direct"}
user_pending_text = {}       # {chat_id: "target_id_string"}
user_pending_reedit_session = {}  # {chat_id: "session_id"} for custom text feedback
_wizard_sessions = {}        # {chat_id: {"wizard": str, "step": int, "data": dict}}

def build_reedit_options_keyboard(session_id: str):
    """
    Builds 5-button re-edit feedback menu for aggressive Gemini prompt polishing:
      1. Change Music / Beat Alignment
      2. Aggressive Shot Reshaping (Dopamine Cuts)
      3. Polish Climax & Ending Hook
      4. Fix Intro & First 3s Hook
      5. Custom User Feedback Directive
    """
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = [
            [InlineKeyboardButton("🎵 Change Music & Rhythm", callback_data=f"ro_music_{session_id}")],
            [InlineKeyboardButton("⚡ Aggressive Shot Reshaping (Dopamine Cuts)", callback_data=f"ro_shots_{session_id}")],
            [InlineKeyboardButton("🎬 Polish Climax & Ending", callback_data=f"ro_climax_{session_id}")],
            [InlineKeyboardButton("🎣 Fix Intro & Hook (First 3s)", callback_data=f"ro_hook_{session_id}")],
            [InlineKeyboardButton("🎙️ Add AI Voiceover Narration", callback_data=f"ro_voiceover_{session_id}")],
            [InlineKeyboardButton("✏️ Custom Suggestion...", callback_data=f"ro_custom_{session_id}")],
        ]
        return InlineKeyboardMarkup(keyboard)
    except ImportError:
        return None

def build_best_attempt_comparison_keyboard(session_id: str, total_attempts: int = 5):
    """
    Builds rating keyboard for selecting best attempt 1 to 5 after MAX_RETRIES limit.
    """
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = [
            [
                InlineKeyboardButton("Attempt 1", callback_data=f"pick_att_1_{session_id}"),
                InlineKeyboardButton("Attempt 2", callback_data=f"pick_att_2_{session_id}"),
            ],
            [
                InlineKeyboardButton("Attempt 3", callback_data=f"pick_att_3_{session_id}"),
                InlineKeyboardButton("Attempt 4", callback_data=f"pick_att_4_{session_id}"),
            ],
            [
                InlineKeyboardButton("Attempt 5", callback_data=f"pick_att_5_{session_id}"),
            ],
            [
                InlineKeyboardButton("❌ All 5 Attempts Are Bad", callback_data=f"pick_att_none_{session_id}"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)
    except ImportError:
        return None

def build_platform_selection_keyboard():
    """
    Builds the Main Menu 6-button keyboard:
      - [ ⚙️ Auto Input Setup ]        callback_data="menu_auto_setup"
      - [ 📜 Active Accounts ]          callback_data="menu_list_accounts"
      - [ 🚀 Run Scraper Batch Now ]    callback_data="menu_run_batch_now"
      - [ 📱 Add Socials ]              callback_data="menu_add_socials"
      - [ 🔑 Assign Credentials ]      callback_data="menu_credentials"
      - [ 🔗 Direct URL / Manual ]     callback_data="menu_direct_url"
    """
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = [
            [
                InlineKeyboardButton("⚙️ Auto Input Setup", callback_data="menu_auto_setup"),
                InlineKeyboardButton("📜 Active Accounts", callback_data="menu_list_accounts"),
            ],
            [
                InlineKeyboardButton("🚀 Run Scraper Batch Now", callback_data="menu_run_batch_now"),
                InlineKeyboardButton("📱 Add Socials", callback_data="menu_add_socials"),
            ],
            [
                InlineKeyboardButton("🔑 Assign Credentials", callback_data="menu_credentials"),
                InlineKeyboardButton("🔗 Direct URL / Manual", callback_data="menu_direct_url"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)
    except ImportError:
        return None


def build_active_accounts_keyboard_and_text(chat_id: Optional[Any] = None):
    """Builds text summary and inline buttons for active accounts list with 30-day expiration info, filtered per user."""
    from Downloader_Modules.scheduled_scraper_manager import get_active_accounts_metadata
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    meta = get_active_accounts_metadata(for_chat_id=chat_id)
    if not meta:
        text = (
            "📋 **Active Target Accounts (0):**\n\n"
            "No active target accounts configured.\n\n"
            "Use ⚙️ Auto Input Setup or `/addaccount @handle` to add creator handles to scrape."
        )
        kbd = InlineKeyboardMarkup([
            [InlineKeyboardButton("⚙️ Auto Input Setup", callback_data="menu_auto_setup")],
            [InlineKeyboardButton("↩️ Main Menu", callback_data="menu_main")]
        ])
        return text, kbd

    lines = []
    keyboard_rows = []
    for a in meta:
        h = a["handle"]
        d_left = a["days_left"]
        lines.append(f"• `@{h}` — ⏳ **{d_left} days remaining** (out of 30)")
        keyboard_rows.append([InlineKeyboardButton(f"🗑️ Remove @{h}", callback_data=f"remove_acc_{h}")])

    keyboard_rows.append([InlineKeyboardButton("🚀 Run Scraper Batch Now", callback_data="menu_run_batch_now")])
    keyboard_rows.append([
        InlineKeyboardButton("⚙️ Auto Input Setup", callback_data="menu_auto_setup"),
        InlineKeyboardButton("↩️ Main Menu", callback_data="menu_main")
    ])

    text = f"📋 **Active Target Accounts ({len(meta)}):**\n\n" + "\n".join(lines) + "\n\n*Accounts automatically expire and are purged after 30 days.*"
    return text, InlineKeyboardMarkup(keyboard_rows)


_global_bot_instance = None


def send_telegram_text_sync(
    chat_id: Any,
    text: str,
    parse_mode: Optional[str] = "Markdown",
    reply_markup: Optional[Any] = None,
) -> bool:
    """
    Synchronously dispatches a text message to a Telegram chat using HTTP POST via urllib.
    Safe to call from background worker threads without asyncio event loop conflicts.
    Falls back to plain text if Markdown parsing fails.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token or not chat_id:
        return False

    import urllib.request
    import urllib.error
    import json

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": str(chat_id), "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_markup is not None:
        if hasattr(reply_markup, "to_dict"):
            payload["reply_markup"] = reply_markup.to_dict()
        elif isinstance(reply_markup, (dict, list)):
            payload["reply_markup"] = reply_markup

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as he:
        # Fallback to plain text if Markdown format had parsing errors (HTTP 400 Bad Request)
        if parse_mode and he.code == 400:
            try:
                payload.pop("parse_mode", None)
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return resp.status == 200
            except Exception as e_fb:
                logger.warning(f"⚠️ Telegram plain-text fallback sendMessage error for {chat_id}: {e_fb}")
        else:
            logger.warning(f"⚠️ Telegram sendMessage HTTPError {he.code} for {chat_id}: {he}")
        return False
    except Exception as e:
        logger.warning(f"⚠️ Failed to send Telegram notification to {chat_id}: {e}")
        return False


async def _send_video_safe_main(
    bot_obj,
    chat_id,
    video_path_or_file,
    caption="",
    reply_markup=None,
    supports_streaming=True,
    max_retries=3,
):
    """
    Robust video uploader wrapper with exponential backoff retries and high write/read timeouts (600s).
    Prevents httpx.WriteTimeout / telegram.error.TimedOut crashes on large media or slow networks.
    """
    from telegram.error import TimedOut, NetworkError
    import asyncio

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            if isinstance(video_path_or_file, str):
                if os.path.isfile(video_path_or_file):
                    with open(video_path_or_file, "rb") as vf:
                        return await bot_obj.send_video(
                            chat_id=int(chat_id),
                            video=vf,
                            caption=caption,
                            reply_markup=reply_markup,
                            supports_streaming=supports_streaming,
                            read_timeout=600,
                            write_timeout=600,
                            connect_timeout=120
                        )
                elif video_path_or_file.startswith(("http://", "https://")):
                    return await bot_obj.send_video(
                        chat_id=int(chat_id),
                        video=video_path_or_file,
                        caption=caption,
                        reply_markup=reply_markup,
                        supports_streaming=supports_streaming,
                        read_timeout=600,
                        write_timeout=600,
                        connect_timeout=120
                    )
                elif "/" not in video_path_or_file and "\\" not in video_path_or_file and len(video_path_or_file) > 20:
                    # Valid Telegram file_id string
                    return await bot_obj.send_video(
                        chat_id=int(chat_id),
                        video=video_path_or_file,
                        caption=caption,
                        reply_markup=reply_markup,
                        supports_streaming=supports_streaming,
                        read_timeout=600,
                        write_timeout=600,
                        connect_timeout=120
                    )
                else:
                    raise FileNotFoundError(f"Video file missing on disk: {video_path_or_file}")
            elif hasattr(video_path_or_file, "read"):
                video_path_or_file.seek(0)
                return await bot_obj.send_video(
                    chat_id=int(chat_id),
                    video=video_path_or_file,
                    caption=caption,
                    reply_markup=reply_markup,
                    supports_streaming=supports_streaming,
                    read_timeout=600,
                    write_timeout=600,
                    connect_timeout=120
                )
            else:
                return await bot_obj.send_video(
                    chat_id=int(chat_id),
                    video=video_path_or_file,
                    caption=caption,
                    reply_markup=reply_markup,
                    supports_streaming=supports_streaming,
                    read_timeout=600,
                    write_timeout=600,
                    connect_timeout=120
                )
        except (TimedOut, NetworkError, Exception) as err:
            last_exc = err
            if isinstance(err, FileNotFoundError):
                logger.error(f"❌ [TELEGRAM UPLOAD FAILED] {err}")
                raise err
            if attempt < max_retries:
                logger.warning(
                    f"⚠️ [TELEGRAM UPLOAD TIMEOUT] Attempt {attempt}/{max_retries} failed for chat {chat_id}: {err}. "
                    f"Retrying in {attempt * 3.0:.1f}s with 600s timeout..."
                )
                await asyncio.sleep(attempt * 3.0)
            else:
                logger.error(
                    f"❌ [TELEGRAM UPLOAD FAILED] All {max_retries} attempts failed for chat {chat_id}: {err}"
                )
                raise err


async def _async_trigger_immediate_batch(bot, chat_id: int):
    """Triggers an immediate background scraper batch for the requesting user and notifies Telegram chat."""
    import asyncio
    from Downloader_Modules.scheduled_scraper_manager import run_scheduled_scraper_batch, get_rotated_max_two_accounts
    try:
        global _global_bot_instance
        effective_bot = bot if (bot and hasattr(bot, "send_message")) else _global_bot_instance
        if not effective_bot:
            logger.error("❌ No active Telegram Bot instance available for immediate batch.")
            return

        targets = get_rotated_max_two_accounts(max_accounts=2, for_chat_id=chat_id)
        if not targets:
            await effective_bot.send_message(
                chat_id=chat_id,
                text="⚠️ No active accounts configured for scraping. Please use ⚙️ **Auto Input Setup** or `/addaccount @handle` to add accounts first."
            )
            return

        await effective_bot.send_message(
            chat_id=chat_id,
            text=f"🚀 **Initial Scrape & Edit Batch Started!**\n\nScraping top reels for: `{', '.join(['@' + t for t in targets])}`...\n*(You will receive master reel review cards here shortly)*"
        )
        
        loop = asyncio.get_running_loop()
        rendered = await loop.run_in_executor(None, lambda: run_scheduled_scraper_batch(max_accounts=2, for_chat_id=chat_id))
        
        if rendered:
            for r_file in rendered:
                try:
                    sess_id = session_manager.create_session(video_path=r_file, requestor_chat_id=chat_id)
                    keyboard = build_telegram_session_keyboard(session_id=sess_id)
                    sent_msg = await _send_video_safe_main(
                        effective_bot,
                        chat_id,
                        r_file,
                        caption=f"🎬 **Master Edit Complete!**\n📁 `{os.path.basename(r_file)}`\n🆔 `Session: {sess_id}`",
                        reply_markup=keyboard
                    )
                    if sent_msg:
                        session_manager.update_message_id(sess_id, sent_msg.message_id)
                except Exception as _e:
                    logger.warning(f"Error sending batch video: {_e}")
        else:
            await effective_bot.send_message(chat_id=chat_id, text="ℹ️ Batch cycle complete: 0 new clips downloaded.")
    except Exception as e:
        logger.error(f"Error executing immediate batch: {e}")


def build_back_button_keyboard():
    """Builds a single Back/Cancel button keyboard."""
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        return InlineKeyboardMarkup([[InlineKeyboardButton("↩️ Back to Main Menu / Cancel", callback_data="platform_cancel")]])
    except ImportError:
        return None


# ── Telegram Inline Keyboard Builder ─────────────────────────────────────────

def build_telegram_session_keyboard(session_id: str, shortcode: Optional[str] = None):
    """
    Builds 5-Button Inline Keyboard for Telegram Master Reel Review:
      Row 1: [✅ Approve & Set Title]  [🚀 Post Immediately]
      Row 2: [❌ Bad / Re-Edit]        [🗑️ Reject & Discard]
      Row 3: [📥 Raw Downloaded Video]
    """
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        trigger_id = shortcode
        if not trigger_id:
            try:
                from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
                s_data = TelegramSessionManager().get_session(session_id)
                if s_data and s_data.get("clip_id"):
                    trigger_id = s_data["clip_id"]
            except Exception:
                pass
        if not trigger_id:
            trigger_id = session_id

        keyboard = [
            [
                InlineKeyboardButton("✅ Approve & Set Title", callback_data=f"approve_title_{session_id}"),
                InlineKeyboardButton("🚀 Post Immediately", callback_data=f"approve_post_{session_id}"),
            ],
            [
                InlineKeyboardButton("❌ Bad / Re-Edit", callback_data=f"reedit_{session_id}"),
                InlineKeyboardButton("🗑️ Reject & Discard", callback_data=f"reject_{session_id}"),
            ],
            [
                InlineKeyboardButton("📥 Raw Downloaded Video", callback_data=f"raw_video_{trigger_id}"),
            ],
            [
                InlineKeyboardButton("🧼 Raw Watermark Cleaned", callback_data=f"clean_video_{trigger_id}"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)
    except ImportError:
        logger.warning("python-telegram-bot package not installed; inline buttons unavailable.")
        return None


def build_awaiting_title_keyboard(session_id: str):
    """
    Builds 2-Button Inline Keyboard for Approved Reel Awaiting Custom Title:
      Row 1: [↩️ Back]  [🗑️ Reject & Discard]
    """
    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = [
            [
                InlineKeyboardButton("↩️ Back", callback_data=f"back_review_{session_id}"),
                InlineKeyboardButton("🗑️ Reject & Discard", callback_data=f"reject_{session_id}"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)
    except Exception:
        return None


# ── Telegram Callback Handlers ───────────────────────────────────────────────

async def handle_telegram_callback(update, context):
    """
    Routes Telegram Inline Button Clicks:
      - Platform Selection: platform_instagram, platform_youtube, platform_tiktok, platform_direct, platform_cancel
      - Reel Review: approve_title_<id>, approve_post_<id>, reedit_<id>, reject_<id>
    """
    query = update.callback_query
    if not query:
        return

    try:
        await query.answer()
    except Exception:
        pass

    data = query.data
    user_id = query.from_user.id
    chat_id = query.message.chat_id if query.message else user_id
    logger.info(f"📲 [TELEGRAM BOT] Received Callback Click: '{data}' from User ID: {user_id}")

    # ── Main Menu Navigation ──────────────────────────────────────────────────
    if data == "platform_cancel" or data == "menu_main":
        _wizard_sessions.pop(chat_id, None)
        user_selected_platform.pop(chat_id, None)
        user_pending_text.pop(chat_id, None)
        keyboard = build_platform_selection_keyboard()
        await query.edit_message_text(
            text="👋 **Back to Main Menu!**\n\n👇 **Choose what you want to do:**",
            reply_markup=keyboard
        )
        return

    if data == "menu_list_accounts":
        text, kbd = build_active_accounts_keyboard_and_text(chat_id=chat_id)
        await query.edit_message_text(text=text, reply_markup=kbd)
        return

    if data.startswith("remove_acc_"):
        handle = data.replace("remove_acc_", "").strip()
        from Downloader_Modules.scheduled_scraper_manager import remove_source_account
        removed = remove_source_account(handle, requestor_chat_id=chat_id)
        text, kbd = build_active_accounts_keyboard_and_text(chat_id=chat_id)
        notice = f"🗑️ Removed `@{handle}`!\n\n" if removed else f"⚠️ Could not remove `@{handle}` (permission denied or account not found).\n\n"
        await query.edit_message_text(text=notice + text, reply_markup=kbd)
        return

    if data == "menu_run_batch_now":
        from Downloader_Modules.scheduled_scraper_manager import get_rotated_max_two_accounts
        targets = get_rotated_max_two_accounts(max_accounts=2, for_chat_id=chat_id)
        if targets:
            goal = f"Ingest reels for creator accounts: {', '.join(['@' + t for t in targets])}, edit each with beat-synced rhythm and BGM, and audit quality."
        else:
            goal = "Run scheduled content scraper batch for active creator accounts, render reels, and audit."
        logger.info(f"🤖 [UNIVERSAL AGENT] Routing batch scrape trigger to Gemini Director: '{goal}'")
        await _dispatch_agent_goal(update, goal, context=context)
        return

    # ── ⚙️ Auto Input Setup Wizard ────────────────────────────────────────────
    if data == "menu_auto_setup":
        _wizard_sessions[chat_id] = {"wizard": "auto_setup", "step": 1, "data": {}}
        back_kbd = build_back_button_keyboard()
        await query.edit_message_text(
            text=(
                "⚙️ **Auto Input Setup — Step 1/6: Source Account IDs**\n\n"
                "Send the platform handles you want to scrape.\n\n"
                "📌 **Format** (one per line or all together):\n"
                "  `instagram @creator_handle`\n"
                "  `youtube @ChannelName`\n"
                "  `tiktok @tiktokuser`\n\n"
                "You can add up to **2 accounts total** across any mix of platforms.\n"
                "Send them now 👇"
            ),
            reply_markup=back_kbd
        )
        return

    # ── 📱 Add Multiple Socials Wizard ────────────────────────────────────────
    if data == "menu_add_socials":
        _wizard_sessions[chat_id] = {"wizard": "add_socials", "step": 1, "data": {}}
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        kbd = InlineKeyboardMarkup([
            [InlineKeyboardButton("📺 YouTube", callback_data="socials_youtube"),
             InlineKeyboardButton("📸 Instagram / Meta", callback_data="socials_instagram")],
            [InlineKeyboardButton("↩️ Back to Main Menu / Cancel", callback_data="menu_main")]
        ])
        await query.edit_message_text(
            text=(
                "📱 **Add Multiple Socials — Step 1: Choose Platform**\n\n"
                "Which social account do you want to connect?\n"
                "*(You can repeat this for each platform)*"
            ),
            reply_markup=kbd
        )
        return

    if data == "socials_youtube":
        sess = _wizard_sessions.setdefault(chat_id, {"wizard": "add_socials", "step": 2, "data": {}})
        sess["step"] = 2
        sess["platform"] = "youtube"
        back_kbd = build_back_button_keyboard()
        await query.edit_message_text(
            text=(
                "📺 **YouTube Setup — Step 2: Upload Client Secret**\n\n"
                "1️⃣ Go to Google Cloud Console → APIs & Services → Credentials\n"
                "2️⃣ Download your **OAuth 2.0 client secret JSON** file\n"
                "3️⃣ **Upload the `client_secret.json` file here** (as a file attachment)\n\n"
                "The bot will save it and start the YouTube OAuth sign-in flow automatically!"
            ),
            reply_markup=back_kbd
        )
        return

    if data == "socials_instagram":
        sess = _wizard_sessions.setdefault(chat_id, {"wizard": "add_socials", "step": 2, "data": {}})
        sess["step"] = 2
        sess["platform"] = "instagram"
        back_kbd = build_back_button_keyboard()
        await query.edit_message_text(
            text=(
                "📸 **Instagram / Meta Setup — Step 2: Send Credentials**\n\n"
                "Send all 4 Meta credentials in **one message** using this exact format:\n\n"
                "`/metagraphapi PAGE_ID=your_page_id PAGE_TOKEN=EAAMxxx IG_ID=your_ig_business_id IG_TOKEN=EAAMxxx`\n\n"
                "📌 **Where to find these:**\n"
                "• Meta Business Suite → Settings → Page Access Tokens\n"
                "• Instagram Graph API → Business Account ID\n\n"
                "⏱️ This session will accept your credentials for **5 minutes**."
            ),
            reply_markup=back_kbd
        )
        return

    if data == "socials_overflow_yes":
        sess = _wizard_sessions.get(chat_id, {})
        sess.setdefault("data", {})["overflow_publish"] = True
        back_kbd = build_back_button_keyboard()
        await query.edit_message_text(
            text=(
                "✅ **Overflow Publishing ON** — if daily limit is hit on one account, bot will post to the next connected social.\n\n"
                "📌 **Step 4: Branding / Watermark Name**\n\n"
                "What branding name should appear on your watermark?\n"
                "*(Example: `MyBrand TV`, `CelebSpot`, `Viral Shorts`)*\n\n"
                "Send your brand name now 👇"
            ),
            reply_markup=back_kbd
        )
        if chat_id in _wizard_sessions:
            _wizard_sessions[chat_id]["step"] = 4
        return

    if data == "socials_overflow_no":
        sess = _wizard_sessions.get(chat_id, {})
        sess.setdefault("data", {})["overflow_publish"] = False
        back_kbd = build_back_button_keyboard()
        await query.edit_message_text(
            text=(
                "❌ **Overflow Publishing OFF** — bot will only post to the primary account and stop at the daily limit.\n\n"
                "📌 **Step 4: Branding / Watermark Name**\n\n"
                "What branding name should appear on your watermark?\n"
                "*(Example: `MyBrand TV`, `CelebSpot`, `Viral Shorts`)*\n\n"
                "Send your brand name now 👇"
            ),
            reply_markup=back_kbd
        )
        if chat_id in _wizard_sessions:
            _wizard_sessions[chat_id]["step"] = 4
        return

    # ── 🔑 Assign Credentials Menu ──────────────────────────────────────────
    if data == "menu_credentials":
        try:
            from Telegram_Storage_Modules.telegram_user_manager import format_user_credentials_summary
            summary_text = format_user_credentials_summary(str(chat_id))
            back_kbd = build_back_button_keyboard()
            await query.edit_message_text(
                text=summary_text,
                reply_markup=back_kbd
            )
        except Exception as _e:
            logger.warning(f"Error in menu_credentials callback: {_e}")
        return

    # ── 🔗 Direct URL / Manual ─────────────────────────────────────────────────
    if data == "menu_direct_url":
        user_selected_platform[chat_id] = "direct"
        back_kbd = build_back_button_keyboard()
        await query.edit_message_text(
            text=(
                "🔗 **Direct URL / Manual Mode**\n\n"
                "📎 **Paste any video URL** (Instagram Reel, YouTube Short, TikTok, etc.)\n"
                "📁 Or **upload a video file** (`.mp4`) directly here.\n\n"
                "The bot will download it, run the full AI Master Edit pipeline, and send the rendered reel back to you!"
            ),
            reply_markup=back_kbd
        )
        return

    # ── Legacy platform_ prefix (kept for backwards compat) ──────────────────
    if data.startswith("platform_"):
        chosen_p = data.replace("platform_", "").strip()
        user_selected_platform[chat_id] = chosen_p
        pending_handle = user_pending_text.pop(chat_id, None)
        if pending_handle and chosen_p != "direct":
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"🎯 **[STEP 1/3] Target Handle Received!**\n\n"
                    f"👤 **Creator**: `@{pending_handle}`\n"
                    f"🌐 **Platform**: `{chosen_p.title()}`\n"
                    f"⚙️ **Status**: Scraping top reels & starting AI editing pipeline..."
                )
            )
            _dispatch_pipeline_in_background(
                mode="auto",
                target_accounts=[pending_handle],
                platform=chosen_p,
                requestor_chat_id=chat_id
            )
            return
        back_kbd = build_back_button_keyboard()
        prompts = {
            "instagram": "📸 **Instagram Mode**\n\nSend the creator handle (e.g. `@creator_handle`):",
            "youtube": "🔴 **YouTube Mode**\n\nSend the channel handle (e.g. `@ChannelName`):",
            "tiktok": "🎵 **TikTok Mode**\n\nSend the creator handle (e.g. `@tiktokuser`):",
            "direct": "🌐 **Direct URL Mode**\n\nPaste any video URL or upload a video file:"
        }
        await query.edit_message_text(text=prompts.get(chosen_p, "Send target handle or URL below:"), reply_markup=back_kbd)
        return

    if data.startswith("raw_video_"):
        trigger_id = data.replace("raw_video_", "").strip()
        logger.info(f"📥 [CALLBACK] User requested raw downloaded video for: '{trigger_id}' from Chat ID: {chat_id}")
        try:
            await query.answer("📥 Retrieving raw video from vault...", show_alert=False)
        except Exception:
            pass

        sent = await vault_indexer.send_raw_video_to_user_chat(
            bot=context.bot,
            chat_id=chat_id,
            identifier=trigger_id
        )
        if not sent:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ **Raw Video Unavailable**: Could not locate raw video for `{trigger_id}` on disk or in Telegram Storage Vault."
            )
        return

    if data.startswith("clean_video_"):
        trigger_id = data.replace("clean_video_", "").strip()
        logger.info(f"🧼 [CALLBACK] User requested raw watermark cleaned video for: '{trigger_id}' from Chat ID: {chat_id}")
        try:
            await query.answer("🧼 Retrieving watermark-cleaned video...", show_alert=False)
        except Exception:
            pass

        sent = await vault_indexer.send_clean_video_to_user_chat(
            bot=context.bot,
            chat_id=chat_id,
            identifier=trigger_id
        )
        if not sent:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ **Watermark Cleaned Video Unavailable**: Could not locate clean video for `{trigger_id}` on disk or in Telegram Storage Vault."
            )
        return

    if data.startswith("approve_title_"):
        session_id = data.replace("approve_title_", "").strip()
        session_manager.set_awaiting_title(session_id)
        sess = session_manager.get_session(session_id)
        clip_name = os.path.basename(sess["video_path"]) if sess else "reel"
        curr_text = query.message.caption or query.message.text or f"📁 `{clip_name}`"

        new_text = (
            f"{curr_text}\n\n"
            f"✅ **Approved!**\n"
            f"✏️ **Send your custom title below:**\n"
            f"📌 **Compulsory Format**: `<User Title> <Affiliate URL>` (or just `<User Title>`)\n"
            f"*(Example: `My Product Review https://amzn.to/example`)*\n\n"
            f"⚠️ *Note: Direct links alone are strictly rejected! A user title is compulsory.*"
        )
        awaiting_kbd = build_awaiting_title_keyboard(session_id)

        try:
            if query.message.video or query.message.document or query.message.photo:
                await query.edit_message_caption(caption=new_text, reply_markup=awaiting_kbd)
            else:
                await query.edit_message_text(text=new_text, reply_markup=awaiting_kbd)
        except Exception as _ce:
            logger.warning(f"⚠️ Callback caption edit warning: {_ce}")
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=new_text,
                reply_markup=awaiting_kbd
            )
        return

    if data.startswith("back_review_"):
        session_id = data.replace("back_review_", "").strip()
        sess = session_manager.revert_awaiting_title(session_id)
        if not sess:
            sess = session_manager.get_session(session_id)
            if sess:
                sess["status"] = "AWAITING_REVIEW"
                session_manager._save_sessions()

        clip_id = sess.get("clip_id") if sess else None
        video_path = sess.get("video_path") if sess else ""
        bname = os.path.basename(video_path) if video_path else "reel"

        orig_text = query.message.caption or query.message.text or f"📁 `{bname}`"
        if "✅ **Approved!**" in orig_text:
            orig_text = orig_text.split("✅ **Approved!**")[0].rstrip()
        if not orig_text.strip():
            orig_text = f"🎉 **AI Master Edit Complete!**\n📁 `{bname}`\n🆔 `Session: {session_id}`"

        review_keyboard = build_telegram_session_keyboard(session_id=session_id, shortcode=clip_id)

        try:
            if query.message.video or query.message.document or query.message.photo:
                await query.edit_message_caption(caption=orig_text, reply_markup=review_keyboard)
            else:
                await query.edit_message_text(text=orig_text, reply_markup=review_keyboard)
            await query.answer("↩️ Returned to review options.", show_alert=False)
        except Exception as _b_err:
            logger.warning(f"⚠️ Callback back_review edit warning: {_b_err}")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"↩️ **Returned to Review Options for `{bname}`:**",
                reply_markup=review_keyboard
            )
        return

    elif data.startswith("approve_post_"):
        session_id = data.replace("approve_post_", "").strip()
        sess = session_manager.get_session(session_id)
        
        video_path = sess.get("video_path") if sess else None
        clip_id = sess.get("clip_id") if sess else (os.path.basename(os.path.dirname(video_path)) if video_path else session_id)
        creator_niche = sess.get("creator", "General") if sess else "General"

        # If local video_path is missing, retrieve from Telegram Storage Group Vault
        if not video_path or not os.path.exists(video_path):
            logger.info(f"📡 [VAULT RETRIEVAL] Local processed video missing for '{session_id}' / '{clip_id}' — hydrating from Storage Group...")
            recovered_reel = vault_indexer.hydrate_processed_reel_from_vault(shortcode_or_session_id=clip_id or session_id)
            if recovered_reel and os.path.exists(recovered_reel):
                video_path = recovered_reel
                if sess:
                    sess["video_path"] = recovered_reel

        # Load clip intelligence from store or folder
        intel_data = {}
        if video_path and os.path.exists(video_path):
            try:
                from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore
                store = ClipIntelligenceStore()
                clip_folder = os.path.dirname(video_path)
                loaded_intel = store.load(clip_id, clip_folder=clip_folder)
                if loaded_intel:
                    intel_data = loaded_intel
            except Exception as _st_err:
                logger.warning(f"⚠️ Could not load clip intelligence for {clip_id}: {_st_err}")
        # Extract context payloads — support both ClipIntelligenceStore schema and pool v3 social entry schema
        v_context = (intel_data.get("visual_context")
                     or intel_data.get("visual_data", {}).get("gemini_visual_output")
                     or intel_data.get("visual_data", {}).get("clip_intelligence") or {})
        if isinstance(v_context, dict):
            video_context_str = (v_context.get("description") or v_context.get("visual_description")
                                 or v_context.get("scene_context", {}).get("description", "")
                                 or f"Reel for {creator_niche}")
        else:
            video_context_str = str(v_context) or f"Reel for {creator_niche}"

        # Build raw_meta — start from phase1 then enrich with pool v3 social fields (additive, no overrides)
        raw_meta = intel_data.get("phase1", {}) or {}
        if intel_data:
            raw_meta.setdefault("creator_handle", creator_niche)
            raw_meta.setdefault("raw_caption", intel_data.get("caption") or (sess.get("title", "") if sess else ""))
            raw_meta["caption"]        = intel_data.get("caption", "")
            raw_meta["hashtags"]       = intel_data.get("hashtags", [])
            raw_meta["taggedUsers"]    = intel_data.get("taggedUsers", [])
            raw_meta["likesCount"]     = intel_data.get("likesCount")
            raw_meta["videoViewCount"] = intel_data.get("videoViewCount")
            raw_meta["shortcode"]      = intel_data.get("shortcode", "")
        if not raw_meta and sess:
            raw_meta = {"creator_handle": creator_niche, "raw_caption": sess.get("title", "")}

        # Generate SEO content via platform_seo_generator
        seo_res = {}
        try:
            from Gemini_Modules.platform_seo_generator import generate_platform_seo
            seo_res = generate_platform_seo(
                video_context=video_context_str,
                user_title=sess.get("title", "") if sess else "",
                brand_info=creator_niche,
                cache=intel_data,
                metadata=raw_meta
            )
        except Exception as _seo_err:
            logger.warning(f"⚠️ [PlatformSEO] Failed to generate SEO for session {session_id}: {_seo_err}")

        # Extract generated platform fields
        main_subject = seo_res.get("main_subject") or "Viral Reel"
        yt_info = seo_res.get("platforms", {}).get("youtube", {})
        approved_title = yt_info.get("title") or f"{main_subject} 🌟"
        
        tags_list = yt_info.get("hashtags", [])
        if not tags_list:
            ig_tags = seo_res.get("platforms", {}).get("instagram", {}).get("hashtags", [])
            tags_list = ig_tags or ["#viral", "#shorts", "#trending"]
        tags_str = " ".join(tags_list)

        ig_info = seo_res.get("platforms", {}).get("instagram", {})
        ig_desc = ig_info.get("description") or ig_info.get("title") or approved_title

        main_subject = seo_res.get("main_subject") or "Featured Reel"
        try:
            from Gemini_Modules.platform_seo_generator import strip_system_and_tracking_tokens, clean_entity_name
            main_subject = strip_system_and_tracking_tokens(clean_entity_name(main_subject))
            approved_title = strip_system_and_tracking_tokens(approved_title)
            tags_str = strip_system_and_tracking_tokens(tags_str)
        except Exception:
            pass

        # Persist approved title and SEO results
        sess = session_manager.set_approved_title(session_id, custom_title=approved_title)
        if sess:
            sess["seo_result"] = seo_res

        curr_text = query.message.caption or query.message.text or "Master Reel"
        new_text = (
            f"{curr_text}\n\n"
            f"🚀 **APPROVED & DISPATCHED WITH AI SEO METADATA!**\n"
            f"⭐ **Main Subject**: `{main_subject}`\n"
            f"📝 **YouTube Title**: `{approved_title}`\n"
            f"🏷️ **Hashtags**: `{tags_str[:80]}`"
        )

        try:
            if query.message.video or query.message.document or query.message.photo:
                await query.edit_message_caption(caption=new_text)
            else:
                await query.edit_message_text(text=new_text)
        except Exception:
            await context.bot.send_message(chat_id=query.message.chat_id, text=new_text)

        if video_path and os.path.exists(video_path):
            PublishQueue.add(video_path, channel_title=approved_title, channel_folder=creator_niche)
            import asyncio
            loop = asyncio.get_running_loop()
            cb_user_id = str(query.from_user.id) if query.from_user else str(query.message.chat_id)
            _dispatch_phase4_publishing_in_background(
                bot_obj=context.bot,
                chat_id=query.message.chat_id,
                loop=loop,
                video_path=video_path,
                title=approved_title,
                niche=creator_niche,
                tags=tags_str,
                description=ig_desc,
                user_id=cb_user_id,
                seo_data=seo_res
            )

    elif data.startswith("reedit_"):
        session_id = data.replace("reedit_", "").strip()
        should_retry, count = session_manager.register_retry(user_id)

        if not should_retry:
            keyboard = build_best_attempt_comparison_keyboard(session_id, total_attempts=MAX_RETRIES)
            new_text = (
                f"🛑 **MAXIMUM RETRY LIMIT REACHED ({MAX_RETRIES}/{MAX_RETRIES} Attempts)**\n\n"
                f"Which attempt was the best (1 to {MAX_RETRIES})?\n"
                f"Select your star rating below to save the winning master edit and commit learning weights to RAG memory:"
            )
            try:
                if query.message.video or query.message.document or query.message.photo:
                    await query.edit_message_caption(caption=new_text, reply_markup=keyboard)
                else:
                    await query.edit_message_text(text=new_text, reply_markup=keyboard)
            except Exception as _m_err:
                logger.warning(f"Max retries menu warning: {_m_err}")
            return

        keyboard = build_reedit_options_keyboard(session_id)
        new_text = (
            f"🛠️ **RE-EDIT FEEDBACK DIRECTIVE (Attempt {count}/{MAX_RETRIES})**\n\n"
            f"What specific improvement should Gemini make for retry #{count}?\n"
            f"Select an option below to inject an aggressive correction prompt:"
        )
        try:
            if query.message.video or query.message.document or query.message.photo:
                await query.edit_message_caption(caption=new_text, reply_markup=keyboard)
            else:
                await query.edit_message_text(text=new_text, reply_markup=keyboard)
        except Exception as _re_e:
            logger.warning(f"Re-edit menu warning: {_re_e}")

    elif data.startswith("pick_att_"):
        # Format: pick_att_1_<session_id> or pick_att_none_<session_id>
        parts = data.split("_", 3)
        choice_str = parts[2]
        session_id = parts[3] if len(parts) > 3 else ""

        sess = session_manager.get_session(session_id)
        clip_id = sess.get("clip_id", "clip") if sess else "clip"
        history = sess.get("attempt_history", []) if sess else []

        if choice_str == "none":
            from Core_Modules import record_multi_attempt_feedback, purge_full_clip_and_assets
            record_multi_attempt_feedback(clip_id, winning_attempt_idx=None, total_attempts=MAX_RETRIES, attempt_paths=history)
            purge_res = purge_full_clip_and_assets(
                clip_id=clip_id,
                video_path=sess.get("video_path") if sess else None,
                attempt_history=history,
                selected_audio=sess.get("selected_audio") if sess else None
            )
            new_text = (
                f"🗑️ **ALL {MAX_RETRIES} ATTEMPTS MARKED BAD & PURGED.**\n\n"
                f"Deleted video files, audio WAVs, source folder ({purge_res.get('purged_count')} assets), and purged index from pool_metadata.json."
            )
        else:
            try:
                winning_idx = int(choice_str) - 1
            except ValueError:
                winning_idx = 0

            from Core_Modules import record_multi_attempt_feedback
            record_multi_attempt_feedback(clip_id, winning_attempt_idx=winning_idx, total_attempts=MAX_RETRIES, attempt_paths=history)
            winning_path = history[winning_idx] if (winning_idx is not None and winning_idx < len(history)) else (sess.get("video_path") if sess else None)

            if sess and winning_path:
                PublishQueue.add(winning_path, channel_title=sess.get("title", "Viral Reel"), channel_folder=sess.get("creator", "General"))

            new_text = (
                f"🏆 **ATTEMPT {winning_idx + 1} SAVED AS BEST MASTER EDIT!**\n\n"
                f"⭐ Gold pattern weight (1.5x) committed to RAG pool_metadata.json for Attempt {winning_idx + 1}.\n"
                f"Dispatched winning reel to Phase 4 publishing queue!"
            )

        try:
            if query.message.video or query.message.document or query.message.photo:
                await query.edit_message_caption(caption=new_text)
            else:
                await query.edit_message_text(text=new_text)
        except Exception as _p_err:
            logger.warning(f"Attempt pick caption warning: {_p_err}")

    elif data.startswith("ro_"):
        # Format: ro_<type>_<session_id>
        parts = data.split("_", 2)
        opt_type = parts[1]
        session_id = parts[2] if len(parts) > 2 else ""

        preset_directives = {
            "music": "Change Music & Beat Alignment — switch to a faster BGM track and align cuts precisely to audio drops.",
            "shots": "Aggressive Shot Reshaping — force maximum cut frequency, dopamine-spike shot pacing, and rapid speed ramps for intense visual energy.",
            "climax": "Polish Climax & Ending Hook — sharpen the final 5-10 seconds of the video, ensure a high-satisfaction closing transition.",
            "hook": "Fix Intro & Hook (First 3s) — re-sample and emphasize the initial 3 seconds for maximum immediate visual grab.",
            "voiceover": "Add AI Voiceover Narration — synthesize and overlay punchy storytelling voiceover narration on top of the reel with ducked BGM."
        }

        if opt_type in preset_directives:
            directive = preset_directives[opt_type]
            goal = f"Re-edit session '{session_id}' with directive: '{directive}'"
            logger.info(f"🤖 [UNIVERSAL AGENT] Routing retry button '{opt_type}' directly to Gemini Director: '{goal}'")
            await _dispatch_agent_goal(update, goal, context=context)
        elif opt_type == "custom":
            user_pending_reedit_session[chat_id] = session_id
            await context.bot.send_message(
                chat_id=chat_id,
                text="✏️ **Custom Re-edit Directive Received!**\n\nPlease reply with your specific feedback instructions for Gemini (e.g. *'Make cuts faster during code section and use a cinematic zoom on the face'*):"
            )

    elif data.startswith("reject_"):
        session_id = data.replace("reject_", "").strip()
        sess = session_manager.set_rejected(session_id)
        curr_text = query.message.caption or query.message.text or "Master Reel"
        new_text = f"{curr_text}\n\n🗑️ **REJECTED & DISCARDED BY ADMIN.**\nAll audio, video assets, and metadata pool index purged!"

        try:
            if query.message.video or query.message.document or query.message.photo:
                await query.edit_message_caption(caption=new_text)
            else:
                await query.edit_message_text(text=new_text)
        except Exception:
            pass

        if sess:
            from Core_Modules import purge_full_clip_and_assets
            purge_res = purge_full_clip_and_assets(
                clip_id=sess.get("clip_id"),
                video_path=sess.get("video_path"),
                attempt_history=sess.get("attempt_history", []),
                selected_audio=sess.get("selected_audio")
            )
            logger.info(f"🗑️ Purged all assets for rejected session {session_id}: {purge_res.get('purged_count')} items removed")
async def execute_reedit_with_directive(query, context, session_id: str, directive: str, chat_id: int):
    """Executes re-edit by dispatching directly through the Autonomous Gemini Director."""
    logger.info(f"🎬 [EXECUTE RE-EDIT] Delegating re-edit for session '{session_id}' to Gemini Director | Directive: '{directive}'")
    goal = f"Re-edit session '{session_id}' with directive: '{directive}'"
    fake_update = type("FakeUpdate", (), {
        "message": (query.message if (query and hasattr(query, "message")) else None),
        "callback_query": query,
        "effective_message": (query.message if (query and hasattr(query, "message")) else None)
    })()
    await _dispatch_agent_goal(fake_update, goal, context=context)


# ── Telegram Command Handlers ─────────────────────────────────────────────────

async def handle_telegram_start(update, context):
    """
    Handles /start command: Welcomes user & shows Platform Selection Keyboard.
    """
    msg = update.message
    if not msg:
        return

    chat_id = msg.chat.id
    user_selected_platform.pop(chat_id, None)
    user_pending_text.pop(chat_id, None)

    keyboard = build_platform_selection_keyboard()
    await msg.reply_text(
        "🎬 **Master AI Reel Editor & Publisher**\n\n"
        "📌 **1. Manage Target Creator Accounts:**\n"
        "• ➕ **Add Account**: `/addaccount @creator_handle`  \n"
        "  *(Example: `/addaccount @creator_handle`)*\n"
        "• 📜 **List Active Accounts**: `/listaccounts`\n"
        "• 🗑️ **Remove Account**: `/removeaccount @creator_handle`\n\n"
        "⚡ **2. Instant AI Reel Generation:**\n"
        "• Paste any Instagram Reel, YouTube Short, or TikTok link to edit & render immediately.\n\n"
        "⏰ **3. Automated Scheduling & Publishing:**\n"
        "• Automated content scraping runs daily at `07:00` & `19:00`.\n"
        "• Rendered reels are broadcast at `07:30` & `19:30` across YouTube, Instagram, & Facebook.\n\n"
        "🔑 **4. Personal API Keys (Optional):**\n"
        "• `/setapify <token>` | `/setgemini <key>`\n\n"
        "🤖 **5. Autonomous Gemini Director:**\n"
        "• `/agent <your goal>` — Run autonomous multi-step scraping, rendering, and publishing with Gemini 2.5 Flash.\n"
        "  *(Example: `/agent Scrape @nike 1 reel, edit with cinematic beats and publish`)*\n\n"
        "👇 **Select your target platform below to begin:**",
        reply_markup=keyboard
    )


async def cmd_ytcode(update, context):
    """
    /ytcode          -> Triggers YouTube auth refresh (sends Google sign-in link to Telegram)
    /ytcode <code>   -> Submits auth code/URL/JSON back to complete YouTube OAuth flow
    """
    msg = update.effective_message
    user_id = update.effective_user.id

    if not context.args:
        # Trigger background auth script
        await msg.reply_text(
            "🔄 **Triggering YouTube OAuth Authentication...**\n\n"
            "The Google sign-in link will appear here in a moment.\n"
            "1️⃣ Tap the Google Sign-in link\n"
            "2️⃣ Sign in and copy the entire `http://localhost/?code=...` URL\n"
            "3️⃣ Paste that URL directly into this chat (or `/ytcode YOUR_URL`) to authorize!",
            parse_mode="Markdown"
        )
        import threading
        import subprocess
        def _run_auth():
            try:
                auth_script = os.path.join(_REPO_ROOT, "scripts", "auth_youtube.py")
                subprocess.run(
                    [sys.executable, auth_script, "--admin-id", str(user_id)],
                    cwd=_REPO_ROOT,
                    timeout=300
                )
            except Exception as _ae:
                logger.error(f"auth_youtube background execution error: {_ae}")
        threading.Thread(target=_run_auth, daemon=True).start()
        return

    # User provided code/URL/JSON via args
    raw = " ".join(context.args).strip()
    if raw.startswith("{") and raw.endswith("}"):
        try:
            from Telegram_Storage_Modules.telegram_user_manager import set_user_youtube_token
            set_user_youtube_token(str(user_id), raw)
            await msg.reply_text("✅ YouTube OAuth Token Synced as your secret")
            return
        except Exception as _e:
            await msg.reply_text(f"⚠️ Error saving YouTube token JSON: {_e}")
            return

    await process_yt_auth_code_input(msg, raw, user_id=str(user_id))


async def process_yt_auth_code_input(msg, raw_text: str, user_id: Optional[str] = None):
    """Processes pasted localhost OAuth redirect URL or raw code."""
    code = raw_text.strip()
    if code.startswith("http"):
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(code)
        qs = parse_qs(parsed.query)
        extracted = qs.get("code", [None])[0]
        if extracted:
            code = extracted
            await msg.reply_text("✅ **Extracted OAuth code from URL!** Exchanging for YouTube token...")
        else:
            await msg.reply_text("❌ No 'code=' parameter found in that URL. Please paste the entire `http://localhost/?code=...` URL.")
            return
    else:
        await msg.reply_text("✅ **OAuth code received!** Exchanging for YouTube token...")

    try:
        cred_dir = os.path.join(_REPO_ROOT, "Credentials")
        os.makedirs(cred_dir, exist_ok=True)
        # Always write standard code file for backward compatibility
        code_file = os.path.join(cred_dir, "yt_auth_code.txt")
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)

        # Also write isolated per-user auth code file if user_id is provided
        if user_id:
            user_code_file = os.path.join(cred_dir, f"yt_auth_code_{user_id}.txt")
            with open(user_code_file, "w", encoding="utf-8") as f:
                f.write(code)
            logger.info(f"🔑 Saved YouTube auth code to {user_code_file} and {code_file}")
        else:
            logger.info(f"🔑 Saved YouTube auth code to {code_file}")
    except Exception as _e:
        await msg.reply_text(f"❌ Failed to save code: {_e}")


# ── Telegram Message & Media Handler ─────────────────────────────────────────

ACTIVE_PIPELINE_JOBS = set()
_MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_PIPELINE_JOBS", "2"))
_PIPELINE_SEMAPHORE = threading.BoundedSemaphore(_MAX_CONCURRENT_JOBS)

async def _check_and_notify_busy_state(msg, user_id_str: str):
    """If another job is actively rendering and user has no personal API key, prompt them to add key for high speed."""
    if len(ACTIVE_PIPELINE_JOBS) > 0:
        try:
            from Telegram_Storage_Modules.telegram_user_manager import get_user_apify_token, get_user_gemini_key
            has_apify = bool(get_user_apify_token(user_id_str))
            has_gemini = bool(get_user_gemini_key(user_id_str))
            if not has_apify and not has_gemini:
                await msg.reply_text(
                    "⏳ **[QUEUE NOTICE] Shared Engine Busy**\n\n"
                    "Shared keys are currently rendering another job. Your request is queued!\n\n"
                    "⚡ **Want to skip the queue & process INSTANTLY at max speed?**\n"
                    "Add your personal API key:\n"
                    "• 🔑 `/setapify <your_apify_token>`\n"
                    "• 🤖 `/setgemini <your_gemini_key>`\n\n"
                    "*(Users with personal keys bypass all shared queues!)*"
                )
        except Exception as _e:
            logger.debug("Notice on busy state check: %s", _e)

def _dispatch_pipeline_in_background(**kwargs):
    """Spawns non-blocking daemon thread with bounded concurrency so Telegram bot loop never freezes."""
    job_id = f"{kwargs.get('requestor_chat_id')}_{time.time()}"
    ACTIVE_PIPELINE_JOBS.add(job_id)
    logger.info(f"🚀 [PIPELINE DISPATCH] Spawning background worker thread for requestor {kwargs.get('requestor_chat_id')} (mode={kwargs.get('mode')}, url={kwargs.get('url')})")
    def _worker():
        acquired = False
        try:
            _PIPELINE_SEMAPHORE.acquire()
            acquired = True
            logger.info(f"▶️ [PIPELINE WORKER START] Worker thread executing job '{job_id}' (active concurrency slot acquired)")
            run_master_pipeline(**kwargs)
        except Exception as _pe:
            logger.error(f"❌ Background pipeline error: {_pe}", exc_info=True)
            chat_id = kwargs.get("requestor_chat_id")
            if chat_id:
                try:
                    send_telegram_text_sync(
                        chat_id,
                        f"❌ **Pipeline Processing Error**\n\n"
                        f"An unexpected error occurred while processing your request:\n`{_pe}`\n\n"
                        f"💡 Please try submitting again or upload the video file directly."
                    )
                except Exception as _ne:
                    logger.debug(f"Notice sending background error notification: {_ne}")
        finally:
            if acquired:
                _PIPELINE_SEMAPHORE.release()
            ACTIVE_PIPELINE_JOBS.discard(job_id)
            logger.info(f"🏁 [PIPELINE WORKER FINISHED] Worker thread finished job '{job_id}' (active remaining: {len(ACTIVE_PIPELINE_JOBS)})")
            sys.stdout.flush()

    import threading
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def _dispatch_phase4_publishing_in_background(
    bot_obj,
    chat_id: int,
    loop,
    video_path: str,
    title: str,
    niche: str = "General",
    tags: str = "#viral #shorts #trending",
    description: str = "",
    user_id: Optional[str] = None,
    seo_data: Optional[Dict[str, Any]] = None
):
    """Spawns non-blocking daemon thread for Phase 4 publishing so the Telegram bot asyncio event loop never freezes."""
    def _worker():
        try:
            from Publishing_Modules.media_publisher_main import run_phase4_publishing
            pub_user_id = str(user_id) if user_id else (str(chat_id) if chat_id else None)
            pub_res = run_phase4_publishing(
                video_path=video_path,
                title=title,
                tags=tags,
                niche=niche,
                description=description,
                user_id=pub_user_id,
                seo_data=seo_data
            )
            status_lines = []
            for p_name, p_info in pub_res.get("platforms", {}).items():
                st = p_info.get("status")
                icon = "✅" if st == "success" else ("⏸️" if st == "skipped" else "❌")
                detail = p_info.get("url") or p_info.get("link") or p_info.get("message") or st
                if detail and (str(detail).startswith("http://") or str(detail).startswith("https://")):
                    status_lines.append(f"• {icon} **{p_name.upper()}**:\n{detail}")
                else:
                    status_lines.append(f"• {icon} **{p_name.upper()}**: `{detail}`")

            if bot_obj and chat_id and loop:
                import asyncio
                msg_text = (
                    f"🎉 **Phase 4 Multi-Platform Broadcasting Complete!**\n\n"
                    f"📌 **Title**: `{title}`\n\n"
                    + "\n".join(status_lines)
                )
                asyncio.run_coroutine_threadsafe(
                    bot_obj.send_message(chat_id=chat_id, text=msg_text, parse_mode="Markdown"),
                    loop
                )

            # Clean up local master reel now that Phase 4 broadcast has completed
            try:
                if video_path and os.path.isfile(video_path):
                    from Core_Modules.disk_cleaner import clean_clip_artifacts
                    c_name = os.path.basename(video_path).replace("_master.mp4", "")
                    clean_clip_artifacts(clip_id=c_name, master_video_path=video_path)
            except Exception:
                pass
        except Exception as pub_err:
            logger.error(f"❌ Phase 4 background publishing error: {pub_err}")
            if bot_obj and chat_id and loop:
                try:
                    import asyncio
                    asyncio.run_coroutine_threadsafe(
                        bot_obj.send_message(chat_id=chat_id, text=f"⚠️ Multi-platform broadcast warning: `{pub_err}`"),
                        loop
                    )
                except Exception:
                    pass

    import threading
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def parse_target_accounts_input(text: str) -> List[Dict[str, str]]:
    """
    Flexibly extracts up to 2 target accounts from user text.
    Handles:
    - Direct handles: '@nike', 'nike', 'nike, adidas'
    - Platform prefixes: 'instagram @nike', 'youtube @channel', 'tiktok @user'
    - Full profile URLs: 'https://www.instagram.com/nike/', 'https://youtube.com/@channel', 'tiktok.com/@user'
    - Multi-line, comma-separated, space-separated.
    """
    import re
    accounts = []
    seen = set()

    lines = text.strip().splitlines()
    tokens = []
    for l in lines:
        l = l.strip()
        if not l:
            continue
        if "," in l:
            tokens.extend([part.strip() for part in l.split(",") if part.strip()])
        else:
            tokens.append(l)

    for item in tokens:
        item = item.strip().lstrip("/")
        if not item:
            continue

        platform = "instagram"
        handle = ""

        # 1. Check for URL patterns
        m_ig_url = re.search(r"instagram\.com/(?:[a-zA-Z0-9_\.]+/)?([a-zA-Z0-9_\.]+)", item, re.IGNORECASE)
        if m_ig_url:
            candidate = m_ig_url.group(1).rstrip("/").split("?")[0].strip()
            if candidate and candidate not in ("p", "reel", "reels", "stories", "explore"):
                platform = "instagram"
                handle = candidate
        
        if not handle:
            m_yt_url = re.search(r"youtube\.com/(?:@|c/|channel/)?([a-zA-Z0-9_\-\.]+)", item, re.IGNORECASE)
            if m_yt_url:
                platform = "youtube"
                handle = m_yt_url.group(1).rstrip("/").split("?")[0].strip()

        if not handle:
            m_tt_url = re.search(r"tiktok\.com/@?([a-zA-Z0-9_\.]+)", item, re.IGNORECASE)
            if m_tt_url:
                platform = "tiktok"
                handle = m_tt_url.group(1).rstrip("/").split("?")[0].strip()

        # 2. Check for explicit platform prefix (e.g., 'instagram @nike' or 'youtube @channel')
        if not handle:
            for p in ["instagram", "youtube", "tiktok", "ig", "yt"]:
                if item.lower().startswith(p):
                    remainder = item[len(p):].strip().lstrip(":").strip().lstrip("@").strip()
                    if remainder:
                        platform = "instagram" if p in ("instagram", "ig") else ("youtube" if p in ("youtube", "yt") else "tiktok")
                        handle = remainder.split()[0].rstrip("/").split("?")[0]
                        break

        # 3. Handle raw handle / space separated tokens (e.g. '@nike @adidas')
        if not handle:
            parts = item.split()
            for part in parts:
                clean_p = part.strip().lstrip("@").rstrip("/").split("?")[0]
                if clean_p and clean_p.lower() not in seen and len(clean_p) >= 2:
                    seen.add(clean_p.lower())
                    accounts.append({"platform": "instagram", "handle": clean_p})
                    if len(accounts) >= 2:
                        return accounts
            continue

        clean_h = handle.strip().lstrip("@").rstrip("/").split("?")[0]
        if clean_h and clean_h.lower() not in seen and len(clean_h) >= 2:
            seen.add(clean_h.lower())
            accounts.append({"platform": platform, "handle": clean_h})
            if len(accounts) >= 2:
                break

    return accounts[:2]


async def _wizard_auto_setup_step(msg, chat_id: int, text: str, bot=None):
    """Advances the Auto Input Setup wizard through steps 1-6."""
    sess = _wizard_sessions.get(chat_id)
    if not sess or sess.get("wizard") != "auto_setup":
        return False
    step = sess.get("step", 1)
    data = sess.setdefault("data", {})
    back_kbd = build_back_button_keyboard()

    if step == 1:
        added = parse_target_accounts_input(text)
        if not added:
            await msg.reply_text(
                "⚠️ Could not parse any account IDs.\n\n"
                "You can send:\n"
                "• Account handles: `@creator_handle` (or `@nike, @adidas`)\n"
                "• Profile links: `https://www.instagram.com/creator_handle/`\n"
                "• With platform: `instagram @handle`, `youtube @channel`, `tiktok @user`\n\n"
                "Please send your creator handle(s) again 👇"
            )
            return True
        data["accounts"] = added[:2]  # max 2

        # Immediately write source accounts to source_accounts.json & sync to Telegram Storage Group Vault!
        try:
            from Downloader_Modules.scheduled_scraper_manager import add_source_account
            for acc in data["accounts"]:
                add_source_account(acc["handle"], acc["platform"], owner_chat_id=chat_id)
            logger.info("⚙️ [AUTO SETUP] Chat %s Step 1: Immediately saved %d account(s) to source_accounts.json & synced to Telegram Vault: %s", chat_id, len(data["accounts"]), [a["handle"] for a in data["accounts"]])
        except Exception as _sa_err:
            logger.warning("⚠️ add_source_account Step 1 immediate save notice: %s", _sa_err)

        acc_summary = "\n".join([f"• `{a['platform'].title()}`: `@{a['handle']}`" for a in data["accounts"]])
        sess["step"] = 2
        await msg.reply_text(
            f"✅ **Accounts saved** ({len(data['accounts'])}/2 max):\n{acc_summary}\n\n"
            f"⏰ **Step 2/6: Set Scraping & Processing Time**\n\n"
            f"When should the bot automatically scrape & render reels?\n"
            f"Send time(s) in **24h HH:MM** format:\n"
            f"*(Example: `07:00,19:00` → runs at 7 AM and 7 PM daily)*",
            reply_markup=back_kbd
        )
        return True

    if step == 2:
        raw_parts = text.strip().split(",")
        norm_parts = [normalize_time_slot(p) or p.strip() for p in raw_parts if p.strip()]
        data["scrape_times"] = ",".join(norm_parts) or text.strip()
        logger.info("⚙️ [AUTO SETUP] Chat %s Step 2: Scraping times set to: %s", chat_id, data["scrape_times"])
        sess["step"] = 3
        await msg.reply_text(
            f"✅ **Scraping times set**: `{data['scrape_times']}`\n\n"
            f"📤 **Step 3/6: Set Publishing Time**\n\n"
            f"When should the bot upload the finished reels to your social accounts?\n"
            f"Send time(s) in **24h HH:MM** format:\n"
            f"*(Example: `07:30,19:30` → posts at 7:30 AM and 7:30 PM daily)*",
            reply_markup=back_kbd
        )
        return True

    if step == 3:
        raw_parts = text.strip().split(",")
        norm_parts = [normalize_time_slot(p) or p.strip() for p in raw_parts if p.strip()]
        data["publish_times"] = ",".join(norm_parts) or text.strip()
        logger.info("⚙️ [AUTO SETUP] Chat %s Step 3: Publishing times set to: %s", chat_id, data["publish_times"])
        sess["step"] = 4
        await msg.reply_text(
            f"✅ **Publishing times set**: `{data['publish_times']}`\n\n"
            f"🎬 **Step 4/6: Clips Per Account Per Run**\n\n"
            f"How many clips should the bot scrape from each account per scheduled run?\n"
            f"*(Recommended: `5–11`, max: `11`)*\n\nSend a number 👇",
            reply_markup=back_kbd
        )
        return True

    if step == 4:
        try:
            clips = min(int(text.strip()), 11)
        except ValueError:
            clips = 5
        data["clips_per_account"] = clips
        logger.info("⚙️ [AUTO SETUP] Chat %s Step 4: Clips per account set to: %d", chat_id, clips)
        sess["step"] = 5
        await msg.reply_text(
            f"✅ **Clips per account**: `{clips}` clips per run\n\n"
            f"📅 **Step 5/6: Daily Publish Limit**\n\n"
            f"How many reels should the bot publish **per day** per social account?\n"
            f"*(Recommended: `2`, to keep accounts safe)*\n\nSend a number 👇",
            reply_markup=back_kbd
        )
        return True

    if step == 5:
        try:
            pub_limit = max(1, int(text.strip()))
        except ValueError:
            pub_limit = 2
        data["max_publish_per_day"] = pub_limit
        logger.info("⚙️ [AUTO SETUP] Chat %s Step 5: Daily publish limit set to: %d", chat_id, pub_limit)
        sess["step"] = 6
        await msg.reply_text(
            f"✅ **Daily publish limit**: `{pub_limit}` reels/day\n\n"
            f"🗓️ **Step 6/6: Days Per Week**\n\n"
            f"How many days per week should the bot run the scraping schedule?\n"
            f"*(Example: `5` = Monday–Friday, `7` = every day)*\n\nSend a number (1–7) 👇",
            reply_markup=back_kbd
        )
        return True

    if step == 6:
        try:
            days = max(1, min(7, int(text.strip())))
        except ValueError:
            days = 7
        data["days_per_week"] = days
        logger.info("⚙️ [AUTO SETUP] Chat %s Step 6: Active days per week set to: %d. Committing full configuration...", chat_id, days)

        # 1. Save to Telegram User Manager (Per-User Isolation)
        try:
            from Telegram_Storage_Modules.telegram_user_manager import set_user_schedule_times, load_all_users, save_all_users
            set_user_schedule_times(str(chat_id), data["scrape_times"])
            users = load_all_users()
            u_rec = users.setdefault(str(chat_id), {})
            u_rec["auto_input_schedule_times"] = data["scrape_times"]
            u_rec["publish_times"] = data["publish_times"]
            u_rec["clips_per_account"] = data["clips_per_account"]
            u_rec["max_publish_per_day"] = data["max_publish_per_day"]
            u_rec["days_per_week"] = data["days_per_week"]
            save_all_users(users)
        except Exception as _um_err:
            logger.warning(f"Notice saving user schedule times: {_um_err}")

        # 2. Update .env and in-memory environment safely
        env_path = os.path.join(_REPO_ROOT, "Credentials", ".env")
        try:
            os.makedirs(os.path.dirname(env_path), exist_ok=True)
            lines_env = open(env_path).readlines() if os.path.exists(env_path) else []
            existing = {l.split("=")[0]: l for l in lines_env if "=" in l}
            existing["SCRAPING_AUTO_INPUT_TIMES"] = f'SCRAPING_AUTO_INPUT_TIMES="{data["scrape_times"]}"\n'
            existing["PUBLISH_STATIC_TIMES"] = f'PUBLISH_STATIC_TIMES="{data["publish_times"]}"\n'
            existing["CLIPS_PER_ACCOUNT_PER_RUN"] = f'CLIPS_PER_ACCOUNT_PER_RUN="{data["clips_per_account"]}"\n'
            existing["MAX_PUBLISH_PER_DAY"] = f'MAX_PUBLISH_PER_DAY="{data["max_publish_per_day"]}"\n'
            existing["SCRAPE_DAYS_PER_WEEK"] = f'SCRAPE_DAYS_PER_WEEK="{data["days_per_week"]}"\n'
            with open(env_path, "w") as f:
                f.writelines(existing.values())
        except Exception as _env_err:
            logger.error(f"Failed to write auto-setup to .env: {_env_err}")

        os.environ["AUTO_INPUT_SCHEDULE_TIMES"] = data["scrape_times"]
        os.environ["SCRAPING_AUTO_INPUT_TIMES"] = data["scrape_times"]

        # 3. Save source accounts with owner_chat_id
        try:
            from Downloader_Modules.scheduled_scraper_manager import add_source_account
            for acc in data.get("accounts", []):
                add_source_account(acc["handle"], acc["platform"], owner_chat_id=chat_id)
        except Exception as _sa_err:
            logger.warning(f"⚠️ add_source_account notice: {_sa_err}")

        _wizard_sessions.pop(chat_id, None)
        accs = "\n".join([f"  • `{a['platform'].title()}`: `@{a['handle']}`" for a in data.get("accounts", [])])
        
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        completion_kbd = InlineKeyboardMarkup([
            [InlineKeyboardButton("🚀 Run Scraper Batch Now", callback_data="menu_run_batch_now")],
            [InlineKeyboardButton("📜 View Active Accounts", callback_data="menu_list_accounts")],
            [InlineKeyboardButton("↩️ Main Menu", callback_data="menu_main")]
        ])

        await msg.reply_text(
            f"🎉 **Auto Input Setup Complete!**\n\n"
            f"📋 **Summary:**\n"
            f"👤 **Accounts:**\n{accs}\n"
            f"⏰ **Scrape & Process**: `{data['scrape_times']}` daily\n"
            f"📤 **Publish to Socials**: `{data['publish_times']}` daily\n"
            f"🎬 **Clips per account per run**: `{data['clips_per_account']}`\n"
            f"📅 **Daily publish limit**: `{data['max_publish_per_day']}` reels/day\n"
            f"🗓️ **Active days per week**: `{data['days_per_week']}` days\n\n"
            f"⏳ **30-Day Limit**: Added account(s) will run for 30 days before auto-expiring.\n"
            f"🚀 **Starting initial scrape batch immediately...**",
            reply_markup=completion_kbd
        )

        try:
            import asyncio
            effective_b = bot or _global_bot_instance
            if effective_b:
                asyncio.create_task(_async_trigger_immediate_batch(effective_b, chat_id))
        except Exception as _trig_err:
            logger.warning(f"Notice triggering immediate batch: {_trig_err}")

        return True

    return False


async def _wizard_socials_step(msg, chat_id: int, text: str, is_document: bool = False):
    """Advances the Add Socials wizard through its steps."""
    sess = _wizard_sessions.get(chat_id)
    if not sess or sess.get("wizard") != "add_socials":
        return False
    step = sess.get("step", 1)
    data = sess.setdefault("data", {})
    back_kbd = build_back_button_keyboard()

    if step == 2 and sess.get("platform") == "youtube" and is_document:
        # Client secret file uploaded — save and trigger ytcode
        if msg.document:
            try:
                doc_file = await msg.document.get_file()
                content_bytes = await doc_file.download_as_bytearray()
                content_str = content_bytes.decode("utf-8")
                from Telegram_Storage_Modules.telegram_user_manager import set_user_youtube_client_secret
                set_user_youtube_client_secret(str(chat_id), content_str)
                logger.info(f"🔐 Client secret document saved for user {chat_id}")
            except Exception as _de:
                logger.warning(f"Failed to read uploaded client_secret document: {_de}")

        await msg.reply_text(
            "✅ **client_secret.json received!**\n\n"
            "🔄 Now use `/ytcode` to start the Google sign-in flow:\n"
            "1️⃣ The bot will send you a Google OAuth link\n"
            "2️⃣ Sign in, copy the full `http://localhost/?code=...` URL\n"
            "3️⃣ Paste it back here or use `/ytcode <url>`\n\n"
            "After completing OAuth, your YouTube token will be saved automatically!"
        )
        sess["step"] = 3
        return True

    if step == 3:
        # Ask overflow + branding
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        kbd = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Yes — post to next social if limit hit", callback_data="socials_overflow_yes"),
             InlineKeyboardButton("❌ No — stop at daily limit", callback_data="socials_overflow_no")],
            [InlineKeyboardButton("↩️ Back to Main Menu / Cancel", callback_data="menu_main")]
        ])
        await msg.reply_text(
            "📌 **Step 3: Overflow Publishing**\n\n"
            f"You've set a daily publish limit per account.\n"
            f"If the limit is reached, should the bot post the remaining reels to your **other connected social accounts**?",
            reply_markup=kbd
        )
        return True

    if step == 4:
        # Branding name
        brand = text.strip()
        data["watermark_brand"] = brand
        env_path = os.path.join(_REPO_ROOT, "Credentials", ".env")
        try:
            lines_env = open(env_path).readlines() if os.path.exists(env_path) else []
            existing = {l.split("=")[0]: l for l in lines_env if "=" in l}
            existing["WATERMARK_TEXT"] = f'WATERMARK_TEXT="{brand}"\n'
            existing["BRAND_WATERMARK_TEXT"] = f'BRAND_WATERMARK_TEXT="{brand}"\n'
            existing["OVERFLOW_PUBLISH"] = f'OVERFLOW_PUBLISH="{str(data.get("overflow_publish", False)).lower()}"\n'
            with open(env_path, "w") as f:
                f.writelines(existing.values())
        except Exception as _e:
            logger.error(f"Failed to write socials settings to .env: {_e}")
        _wizard_sessions.pop(chat_id, None)
        await msg.reply_text(
            f"🎉 **Socials Setup Complete!**\n\n"
            f"🏷️ **Watermark Brand Name**: `{brand}`\n"
            f"📤 **Overflow Publishing**: `{'ON ✅' if data.get('overflow_publish') else 'OFF ❌'}`\n\n"
            f"All settings saved! Your branded reels will now be posted automatically.",
            reply_markup=build_platform_selection_keyboard()
        )
        return True

    return False


async def _wizard_credentials_step(msg, chat_id: int, text: str):
    """Advances the Assign Credentials wizard through its steps."""
    sess = _wizard_sessions.get(chat_id)
    if not sess or sess.get("wizard") != "credentials":
        return False
    step = sess.get("step", 1)
    back_kbd = build_back_button_keyboard()

    if step == 1 and text.startswith("/setapify "):
        token = text.replace("/setapify ", "").strip()
        try:
            from Telegram_Storage_Modules.telegram_user_manager import set_user_apify_token
            set_user_apify_token(str(chat_id), token)
        except Exception as _e:
            logger.debug(f"setapify wizard: {_e}")
        sess["step"] = 2
        await msg.reply_text(
            f"✅ **Apify token saved!**\n\n"
            f"🤖 **Step 2/3: Gemini API Key**\n\n"
            f"Send your Google Gemini API key:\n"
            f"`/setgemini YOUR_GEMINI_KEY`\n\n"
            f"*(Get it from: https://aistudio.google.com/apikey)*",
            reply_markup=back_kbd
        )
        return True

    if step == 2 and text.startswith("/setgemini "):
        key = text.replace("/setgemini ", "").strip()
        try:
            from Telegram_Storage_Modules.telegram_user_manager import set_user_gemini_key
            set_user_gemini_key(str(chat_id), key)
        except Exception as _e:
            logger.debug(f"setgemini wizard: {_e}")
        sess["step"] = 3
        await msg.reply_text(
            f"✅ **Gemini key saved!**\n\n"
            f"📦 **Step 3/3: Telegram Storage Group Chat ID**\n\n"
            f"Send your Telegram Storage Group Chat ID:\n"
            f"`/setstoragechat -100xxxxxxxxxx`\n\n"
            f"*(Forward any message from your storage group to @userinfobot to get the Chat ID)*",
            reply_markup=back_kbd
        )
        return True

    if step == 3 and text.startswith("/setstoragechat "):
        chat_id_val = text.replace("/setstoragechat ", "").strip()
        env_path = os.path.join(_REPO_ROOT, "Credentials", ".env")
        try:
            lines_env = open(env_path).readlines() if os.path.exists(env_path) else []
            existing = {l.split("=")[0]: l for l in lines_env if "=" in l}
            existing["TELEGRAM_STORAGE_CHAT_ID"] = f'TELEGRAM_STORAGE_CHAT_ID="{chat_id_val}"\n'
            with open(env_path, "w") as f:
                f.writelines(existing.values())
        except Exception as _e:
            logger.error(f"Failed to write TELEGRAM_STORAGE_CHAT_ID: {_e}")
        _wizard_sessions.pop(chat_id, None)
        await msg.reply_text(
            f"🎉 **All Credentials Assigned!**\n\n"
            f"✅ **Apify Token**: saved\n"
            f"✅ **Gemini API Key**: saved\n"
            f"📦 **Telegram Storage Group**: `{chat_id_val}`\n\n"
            f"All keys saved securely! Your bot is ready to run.",
            reply_markup=build_platform_selection_keyboard()
        )
        return True

    return False


_BG_AGENT_TASKS: set = set()   # strong refs: asyncio only keeps weak refs, un-referenced tasks can be GC'd mid-run


def _run_reedit_direct(session_id: str, directive: str) -> Dict[str, Any]:
    """Deterministic re-edit (preset buttons / custom directive): NO LLM round trips."""
    from Agentic_Core.tool_adapters import tool_reedit_session
    t0 = time.time()
    with _PIPELINE_SEMAPHORE:
        r = tool_reedit_session(session_id=session_id, edit_directive=directive)
    ok = r.get("status") == "success"
    return {
        "status": "success" if ok else "error",
        "final_summary": r.get("message") or "Re-edit finished.",
        "message": r.get("message"),
        "turns_executed": 0,
        "execution_time_sec": round(time.time() - t0, 2),
        "rendered_video_path": r.get("rendered_video_path") or r.get("new_master_video"),
        "session_id": r.get("session_id") or session_id,
        "retry_count": r.get("retry_count"),
        "audit_passed": None,
    }


def _run_director_in_slot(goal: str, progress_callback=None) -> Dict[str, Any]:
    """Free-form goals only: LLM Director, bounded by the same concurrency semaphore as the pipeline."""
    from Agentic_Core.run_agent import run_agentic_goal
    with _PIPELINE_SEMAPHORE:
        return run_agentic_goal(goal, max_turns=10, progress_callback=progress_callback)


def _dispatch_direct_route(route, chat_id: int) -> str:
    """
    Fast path for URL / @handle / batch / upload goals: run the proven legacy pipeline in a
    semaphore-bounded worker thread (it delivers the reel + review keyboard to Telegram itself).
    Returns a short human-readable description of what was started.
    """
    kind = route.kind
    if kind == "url":
        kwargs = dict(mode="manual", url=route.url, platform=route.platform, requestor_chat_id=chat_id)
        if route.directive:
            kwargs["user_edit_directive"] = route.directive
        _dispatch_pipeline_in_background(**kwargs)
        return f"URL pipeline started ({route.platform})" + (f" with directive: {route.directive[:120]}" if route.directive else "")
    if kind in ("handle", "batch") and route.handles:
        _dispatch_pipeline_in_background(mode="auto", target_accounts=route.handles, platform=route.platform, requestor_chat_id=chat_id)
        return "Creator pipeline started for " + ", ".join("@" + h for h in route.handles)
    if kind == "batch":
        _dispatch_pipeline_in_background(mode="auto", platform=route.platform, requestor_chat_id=chat_id)
        return "Scheduled-pool batch pipeline started"
    if kind == "upload":
        if not (route.path and os.path.isfile(route.path)):
            raise FileNotFoundError(f"Uploaded video not found on disk: {route.path}")
        _dispatch_pipeline_in_background(mode="manual", input_path=route.path, platform=route.platform, requestor_chat_id=chat_id)
        return "Uploaded-video pipeline started"
    raise ValueError(f"Not a direct route: {kind}")



async def _dispatch_agent_goal(update, user_goal: str, existing_status_msg=None, context=None):
    """
    Universal dispatcher that hands off any user goal, UI button click, or custom prompt to the Gemini Autonomous Director
    in a non-blocking background task with real-time Telegram progress updates and interactive reel delivery.
    """
    query = getattr(update, "callback_query", None)
    msg = getattr(update, "message", None) or (query.message if query else None)
    chat_id = (msg.chat_id if msg else None) or (query.from_user.id if (query and query.from_user) else None)

    if not chat_id:
        logger.error("❌ [_dispatch_agent_goal] Unable to resolve chat_id from update.")
        return

    global _global_bot_instance
    effective_bot = (context.bot if (context and hasattr(context, "bot")) else None) or _global_bot_instance

    # Deterministic routing: the LLM Director is ONLY used for genuinely open-ended free text.
    from Agentic_Core.goal_router import classify_goal
    route = classify_goal(user_goal)
    _plan_line = (
        "⚡ *Direct pipeline (no AI planner) starting...*" if route.is_direct
        else "⏳ *Planning and coordinating tools with Gemini...*"
    )

    if query:
        try:
            await query.answer("🤖 Agent activated...", show_alert=False)
        except Exception:
            pass

    if not existing_status_msg:
        try:
            if query and query.message:
                status_msg = await query.message.reply_text(
                    f"🤖 **[GEMINI AGENT ACTIVATED]**\n\n"
                    f"🎯 *Goal:* `{user_goal[:300]}`\n"
                    f"{_plan_line}"
                )
            elif msg:
                status_msg = await msg.reply_text(
                    f"🤖 **[GEMINI AGENT ACTIVATED]**\n\n"
                    f"🎯 *Goal:* `{user_goal[:300]}`\n"
                    f"{_plan_line}"
                )
            else:
                status_msg = await effective_bot.send_message(
                    chat_id=chat_id,
                    text=f"🤖 **[GEMINI AGENT ACTIVATED]**\n\n🎯 *Goal:* `{user_goal[:300]}`\n{_plan_line}"
                )
        except Exception as _sm_err:
            logger.warning(f"Failed to post initial agent status: {_sm_err}")
            status_msg = None
    else:
        status_msg = existing_status_msg

    async def _run_agent_task():
        try:
            from Agentic_Core.run_agent import run_agentic_goal_async

            last_edit_time = 0

            async def _async_notify(event, data):
                nonlocal last_edit_time
                now = time.time()
                # Throttle updates to avoid Telegram flood limits (once every 3 seconds)
                if now - last_edit_time > 3.0 and status_msg:
                    last_edit_time = now
                    evt_text = f"🔄 *Status:* `{event}`"
                    if "tool" in data:
                        evt_text += f"\n⚙️ *Running Tool:* `{data['tool']}`"
                    elif "turn" in data:
                        evt_text += f"\n🧠 *Turn:* {data['turn']}/{data.get('max_turns', 10)}"
                    elif "wait" in data:
                        evt_text += f"\n⏳ *API retry {data.get('attempt')}/{data.get('max')} in {data['wait']:.0f}s ({data.get('reason')})*"
                    try:
                        await status_msg.edit_text(
                            f"🤖 **[AGENT IN PROGRESS]**\n\n"
                            f"🎯 *Goal:* `{user_goal[:300]}`\n"
                            f"{evt_text}"
                        )
                    except Exception:
                        pass

            loop = asyncio.get_running_loop()

            def _progress_cb(event, data):
                asyncio.run_coroutine_threadsafe(_async_notify(event, data), loop)

            if route.is_direct and route.kind != "reedit":
                # URL / @handle / batch / upload -> proven pipeline, zero LLM round trips
                started = await asyncio.to_thread(_dispatch_direct_route, route, chat_id)
                logger.info(f"⚡ [DIRECT ROUTE] {route.kind}: {started}")
                if status_msg:
                    try:
                        await status_msg.edit_text(
                            f"⚡ **[DIRECT PIPELINE]**\n\n🎯 *Goal:* `{user_goal[:300]}`\n✅ {started}\n"
                            f"📬 *The finished reel will be delivered here automatically.*"
                        )
                    except Exception:
                        pass
                return
            elif route.kind == "reedit":
                res = await asyncio.to_thread(_run_reedit_direct, route.session_id, route.directive)
            else:
                res = await asyncio.to_thread(_run_director_in_slot, user_goal, _progress_cb)

            status = res.get("status", "completed")
            summary = res.get("final_summary") or res.get("message") or "Goal execution completed."
            turns = res.get("turns_executed", 0)
            elapsed = res.get("execution_time_sec", 0.0)
            rendered_vid = res.get("rendered_video_path")
            sess_id = res.get("session_id")

            final_report = (
                f"{'✅ **[PRODUCTION COMPLETE]**' if status == 'success' else '⚠️ **[PRODUCTION ENDED: ' + str(status).upper() + ']**'}\n\n"
                f"🎯 *Goal:* `{user_goal[:300]}`\n"
                f"⚡ *Status:* `{status}`\n"
                f"⏱️ *Duration:* `{elapsed}s` ({turns} turns)\n\n"
                f"📋 *Summary:*\n{summary[:3000]}"
            )
            if status_msg:
                try:
                    await status_msg.edit_text(final_report)
                except Exception:
                    pass

            # ── If the Agent rendered or re-edited a video, deliver interactive review card ──
            if rendered_vid and os.path.exists(rendered_vid) and effective_bot:
                try:
                    sess = session_manager.get_session(sess_id) if sess_id else None
                    if not sess:
                        sess_id = session_manager.create_session(
                            video_path=rendered_vid,
                            requestor_chat_id=chat_id
                        )
                        sess = session_manager.get_session(sess_id)

                    clip_id = (sess.get("clip_id") if sess else None) or os.path.splitext(os.path.basename(rendered_vid))[0].replace("_master", "")
                    count = (sess.get("retry_count") if sess else 1) or 1
                    keyboard = build_telegram_session_keyboard(session_id=sess_id, shortcode=clip_id)

                    sent_video_msg = await _send_video_safe_main(
                        effective_bot,
                        chat_id,
                        rendered_vid,
                        caption=f"🎉 **AI Master Edit Delivered by Agent!** (Attempt #{count})\n📁 `{os.path.basename(rendered_vid)}`\n🆔 `Session: {sess_id}`",
                        reply_markup=keyboard,
                        supports_streaming=True
                    )
                    if sent_video_msg and sess_id:
                        session_manager.update_message_id(sess_id, sent_video_msg.message_id)
                except Exception as _v_err:
                    logger.error(f"❌ Failed to deliver agent-rendered video to chat {chat_id}: {_v_err}")

        except Exception as e:
            logger.error(f"❌ [AGENT ERROR] Failed: {e}", exc_info=True)
            if status_msg:
                try:
                    await status_msg.edit_text(f"❌ **Agent execution failed:**\n`{str(e)}`")
                except Exception:
                    pass

    _task = asyncio.create_task(_run_agent_task())
    _BG_AGENT_TASKS.add(_task)
    _task.add_done_callback(_BG_AGENT_TASKS.discard)


async def handle_telegram_incoming_msg(update, context):
    """
    Handles user messages sent to Telegram bot:
      1. Active Session Title Capture: If user is replying to an approved reel, captures title & dispatches to PublishQueue.
      2. Universal Agent Route: Routes all URLs, creator handles, and goals directly to Gemini Director.
    """
    msg = update.message
    if not msg:
        return

    chat_id = msg.chat.id
    from_user = msg.from_user.to_dict() if msg.from_user else {}
    user_id = str(msg.from_user.id) if msg.from_user else str(chat_id)
    user_handle = msg.from_user.username if (msg.from_user and msg.from_user.username) else str(user_id)
    text_preview = (msg.text[:80] + "...") if (msg.text and len(msg.text) > 80) else (msg.text or ("[Video/File]" if (msg.video or msg.document) else "[Media]"))
    logger.info(f"📩 [TELEGRAM INCOMING] From @{user_handle} (chat_id={chat_id}): '{text_preview}'")

    # ── Active Wizard Step Handler ────────────────────────────────────────────
    active_wizard = _wizard_sessions.get(chat_id, {}).get("wizard")
    if active_wizard and msg.text and not msg.text.startswith("/start") and not msg.text.startswith("/help"):
        is_doc = bool(msg.document or msg.video)
        text_in = msg.text.strip() if msg.text else ""
        if active_wizard == "auto_setup":
            if await _wizard_auto_setup_step(msg, chat_id, text_in, bot=context.bot):
                return
        elif active_wizard == "add_socials":
            sess = _wizard_sessions.get(chat_id, {})
            if sess.get("step") == 2 and sess.get("platform") == "instagram" and text_in.startswith("/metagraphapi "):
                raw = text_in.replace("/metagraphapi ", "").strip()
                env_vars = {}
                for part in raw.split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        env_vars[k.strip()] = v.strip()
                env_path = os.path.join(_REPO_ROOT, "Credentials", ".env")
                try:
                    lines_env = open(env_path).readlines() if os.path.exists(env_path) else []
                    existing = {l.split("=")[0]: l for l in lines_env if "=" in l}
                    for k, v in env_vars.items():
                        existing[k] = f'{k}="{v}"\n'
                    with open(env_path, "w") as f:
                        f.writelines(existing.values())
                except Exception as _e:
                    logger.error(f"metagraphapi wizard: {_e}")
                saved = ", ".join([f"`{k}`" for k in env_vars])
                await msg.reply_text(f"✅ **Meta credentials saved**: {saved}")
                _wizard_sessions[chat_id]["step"] = 3
                if await _wizard_socials_step(msg, chat_id, ""):
                    return
                return
            if await _wizard_socials_step(msg, chat_id, text_in, is_doc):
                return
        elif active_wizard == "credentials":
            if await _wizard_credentials_step(msg, chat_id, text_in):
                return
    if active_wizard and (msg.document or msg.video):
        if active_wizard == "add_socials":
            if await _wizard_socials_step(msg, chat_id, "", is_document=True):
                return

    # Check if user pasted a localhost OAuth redirect URL or auth code
    if msg.text and ("localhost" in msg.text or "code=" in msg.text) and ("http://" in msg.text or "https://" in msg.text):
        effective_uid = str(update.effective_user.id if update.effective_user else chat_id)
        await process_yt_auth_code_input(msg, msg.text.strip(), user_id=effective_uid)
        return

    # Check if we are waiting for custom re-edit feedback from user
    if chat_id in user_pending_reedit_session and msg.text and not msg.text.startswith("/"):
        session_id = user_pending_reedit_session.pop(chat_id)
        custom_directive = msg.text.strip()
        goal = f"Re-edit session '{session_id}' with custom user directive: '{custom_directive}'"
        logger.info(f"🤖 [UNIVERSAL AGENT] Routing custom prompt directly to Gemini Director: '{goal}'")
        await _dispatch_agent_goal(update, goal, context=context)
        return

    # Check if we are waiting for a custom title / hint / affiliate URL from user
    pending_sess = session_manager.get_pending_title_session(chat_id)
    if pending_sess and msg.text:
        user_input_raw = msg.text.strip()
        cmd_check = user_input_raw.lower()

        # Handle explicit rejection or cancellation in text
        if cmd_check in ("/reject", "reject", "/rejet", "rejet"):
            sess_id = pending_sess["session_id"]
            sess = session_manager.set_rejected(sess_id)
            if sess:
                from Core_Modules import purge_full_clip_and_assets
                purge_res = purge_full_clip_and_assets(
                    clip_id=sess.get("clip_id"),
                    video_path=sess.get("video_path"),
                    attempt_history=sess.get("attempt_history", []),
                    selected_audio=sess.get("selected_audio")
                )
                logger.info(f"🗑️ Purged assets for session {sess_id} via text reject: {purge_res.get('purged_count')} items removed")
            await msg.reply_text("🗑️ **REJECTED & DISCARDED BY ADMIN.**\nAll audio, video assets, and metadata pool index purged!")
            return

        if cmd_check in ("/cancel", "cancel", "/back", "back"):
            sess_id = pending_sess["session_id"]
            session_manager.revert_awaiting_title(sess_id)
            sess = session_manager.get_session(sess_id)
            clip_id = sess.get("clip_id") if sess else None
            review_kbd = build_telegram_session_keyboard(session_id=sess_id, shortcode=clip_id)
            await msg.reply_text("↩️ **Cancelled title input.** Returned reel to review options.", reply_markup=review_kbd)
            return

        if not user_input_raw.startswith("/"):
            sess_id = pending_sess["session_id"]
            sess = session_manager.get_session(sess_id)

            # 1. Detect and extract all optional affiliate URLs from user input format <User Title> <url1> <url2>...
            extracted_urls = re.findall(r'https?://[^\s<>"]+', user_input_raw)
            user_hint_text = user_input_raw
            for u in extracted_urls:
                user_hint_text = user_hint_text.replace(u, "")
            user_hint_text = re.sub(r'\s+', ' ', user_hint_text).strip()

            affiliate_url = "\n".join(extracted_urls) if extracted_urls else None

            # 🛑 COMPULSORY RULE: Direct links alone are strictly forbidden! User title is compulsory.
            if not user_hint_text or len(user_hint_text) < 2:
                cancel_kbd = build_awaiting_title_keyboard(sess_id)
                await msg.reply_text(
                    "❌ **Bare link(s) alone are NOT allowed!**\n\n"
                    "You MUST provide a **User Title** before the link(s).\n\n"
                    "📌 **Compulsory Format**: `<User Title> <Link 1> <Link 2> ...` (or just `<User Title>`)\n"
                    "💡 *Example*: `Amazon vs Meesho Review https://amzn.to/item1 https://meesho.com/item2`\n\n"
                    "Please send your title and link(s) again in the correct format 👇",
                    reply_markup=cancel_kbd
                )
                return

        video_path = sess.get("video_path") if sess else None
        clip_id = sess.get("clip_id") if sess else (os.path.basename(os.path.dirname(video_path)) if video_path else sess_id)
        creator_niche = sess.get("creator", "General") if sess else "General"

        # Hydrate processed reel from Telegram Storage Group Vault if local file is missing
        if not video_path or not os.path.exists(video_path):
            logger.info(f"📡 [VAULT RETRIEVAL] Local processed video missing for title publish ('{sess_id}' / '{clip_id}') — hydrating from Storage Group...")
            recovered_reel = vault_indexer.hydrate_processed_reel_from_vault(shortcode_or_session_id=clip_id or sess_id)
            if recovered_reel and os.path.exists(recovered_reel):
                video_path = recovered_reel
                if sess:
                    sess["video_path"] = recovered_reel

        # 2. Load clip intelligence
        intel_data = {}
        if video_path and os.path.exists(video_path):
            try:
                from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore
                store = ClipIntelligenceStore()
                loaded_intel = store.load(clip_id, clip_folder=os.path.dirname(video_path))
                if loaded_intel:
                    intel_data = loaded_intel
            except Exception as _st_err:
                logger.warning(f"⚠️ Could not load clip intelligence for {clip_id}: {_st_err}")

        # Extract context payloads — support both ClipIntelligenceStore schema and pool v3 social entry schema
        v_context = (intel_data.get("visual_context")
                     or intel_data.get("visual_data", {}).get("gemini_visual_output")
                     or intel_data.get("visual_data", {}).get("clip_intelligence") or {})
        if isinstance(v_context, dict):
            video_context_str = (v_context.get("description") or v_context.get("visual_description")
                                 or v_context.get("scene_context", {}).get("description", "")
                                 or f"Reel for {creator_niche}")
        else:
            video_context_str = str(v_context) or f"Reel for {creator_niche}"

        # Build raw_meta — start from phase1 then enrich with pool v3 social fields (additive, no overrides)
        raw_meta = intel_data.get("phase1", {}) or {}
        if intel_data:
            raw_meta.setdefault("creator_handle", creator_niche)
            raw_meta.setdefault("raw_caption", intel_data.get("caption") or user_hint_text)
            raw_meta["caption"]        = intel_data.get("caption", "")
            raw_meta["hashtags"]       = intel_data.get("hashtags", [])
            raw_meta["taggedUsers"]    = intel_data.get("taggedUsers", [])
            raw_meta["likesCount"]     = intel_data.get("likesCount")
            raw_meta["videoViewCount"] = intel_data.get("videoViewCount")
            raw_meta["shortcode"]      = intel_data.get("shortcode", "")
        if not raw_meta:
            raw_meta = {"creator_handle": creator_niche, "raw_caption": user_hint_text}

        # 3. Generate SEO content with User Hint & Affiliate Link Integration
        seo_res = {}
        try:
            from Gemini_Modules.platform_seo_generator import generate_platform_seo
            seo_res = generate_platform_seo(
                video_context=video_context_str,
                user_title=user_hint_text,
                brand_info=creator_niche,
                cache=intel_data,
                metadata=raw_meta,
                affiliate_link=affiliate_url,
                user_hint=user_hint_text
            )
        except Exception as _seo_err:
            logger.warning(f"⚠️ [PlatformSEO] Failed to generate commercial SEO for session {sess_id}: {_seo_err}")

        try:
            from Gemini_Modules.platform_seo_generator import strip_system_and_tracking_tokens, clean_entity_name
            main_subject = strip_system_and_tracking_tokens(clean_entity_name(seo_res.get("main_subject") or user_hint_text or "Viral Reel"))
        except Exception:
            main_subject = seo_res.get("main_subject") or user_hint_text or "Viral Reel"

        yt_info = seo_res.get("platforms", {}).get("youtube", {})
        raw_title = yt_info.get("title") or f"{main_subject} 🌟"
        try:
            from Gemini_Modules.platform_seo_generator import strip_system_and_tracking_tokens
            approved_title = strip_system_and_tracking_tokens(raw_title)
        except Exception:
            approved_title = raw_title
        
        tags_list = yt_info.get("hashtags", []) or seo_res.get("platforms", {}).get("instagram", {}).get("hashtags", []) or ["#viral", "#shorts"]
        tags_str = " ".join(tags_list)
        try:
            from Gemini_Modules.platform_seo_generator import strip_system_and_tracking_tokens
            tags_str = strip_system_and_tracking_tokens(tags_str)
        except Exception:
            pass

        ig_info = seo_res.get("platforms", {}).get("instagram", {})
        ig_desc = ig_info.get("description") or ig_info.get("title") or approved_title

        # Persist approved title and SEO results
        updated_sess = session_manager.set_approved_title(sess_id, approved_title)
        if updated_sess:
            updated_sess["seo_result"] = seo_res

        if updated_sess:
            v_path = updated_sess.get("video_path") or video_path
            if not v_path or not os.path.exists(v_path):
                v_path = vault_indexer.hydrate_processed_reel_from_vault(shortcode_or_session_id=clip_id or sess_id)

            if not v_path or not os.path.exists(v_path):
                await msg.reply_text("❌ **Publish Failed**: Processed short video file could not be found locally or in Telegram Storage Vault.")
                return

            PublishQueue.add(v_path, channel_title=approved_title, channel_folder=creator_niche)
            logger.info(f"🚀 Custom Title & Affiliate SEO saved for session {sess_id}: '{approved_title}'. Triggering Phase 4...")

            if extracted_urls:
                if len(extracted_urls) == 1:
                    aff_status = f"\n🛒 **Affiliate Promotion Link**: `{extracted_urls[0]}`\n⚖️ **Policy Disclosure**: Included (#ad #affiliate)"
                else:
                    links_fmt = "\n".join([f"  • `{u}`" for u in extracted_urls])
                    aff_status = f"\n🛒 **Affiliate Promotion Links ({len(extracted_urls)})**:\n{links_fmt}\n⚖️ **Policy Disclosure**: Included (#ad #affiliate)"
            else:
                aff_status = ""

            await msg.reply_text(
                f"🚀 **Commercial AI Title Saved!** Starting 4-Platform Broadcasting...\n\n"
                f"⭐ **Main Subject**: `{main_subject}`\n"
                f"📌 **YouTube Title**: `{approved_title}`\n"
                f"🏷️ **Hashtags**: `{tags_str[:80]}`"
                f"{aff_status}\n"
                f"📁 **Reel**: `{os.path.basename(v_path)}`"
            )

            import asyncio
            loop = asyncio.get_running_loop()
            _dispatch_phase4_publishing_in_background(
                bot_obj=context.bot,
                chat_id=chat_id,
                loop=loop,
                video_path=v_path,
                title=approved_title,
                niche=creator_niche,
                tags=tags_str,
                description=ig_desc,
                user_id=user_id,
                seo_data=seo_res
            )
            return

    target_url = None
    local_video_path = None
    target_accs = None

    # Determine active platform choice (check if user explicitly clicked a platform button)
    has_platform_choice = chat_id in user_selected_platform
    chosen_platform = user_selected_platform.pop(chat_id, "instagram")

    GREETINGS = {"hello", "hi", "hey", "test", "help", "bot", "start", "menu", "status", "yo", "hola"}

    if msg.text:
        text = msg.text.strip()

        if text.startswith("/"):
            if text.startswith("/start") or text.startswith("/help") or text.startswith("/mode"):
                await handle_telegram_start(update, context)
            return

        # Handle simple greetings by showing the help menu
        if text.lower() in GREETINGS:
            await handle_telegram_start(update, context)
            return

        # ── UNIVERSAL AGENTIC ROUTING ──────────────────────────────────────────
        # Every user text input (URL, account handle, natural language prompt)
        # routes directly through the Gemini Autonomous Director without needing /agent.
        logger.info(f"🤖 [UNIVERSAL AGENT] Routing user text directly to Gemini Director: '{text[:80]}'")
        await _dispatch_agent_goal(update, text)
        return

    elif msg.video or msg.document:
        doc = msg.document
        orig_filename = (doc.file_name if doc else "") or ""
        ext = os.path.splitext(orig_filename)[1].lower()
        mime = (doc.mime_type if doc else "").lower()

        # Check if non-video document was received
        if doc and not (mime.startswith("video/") or ext in (".mp4", ".mov", ".avi", ".mkv", ".webm")):
            await msg.reply_text("⚠️ **Unsupported Document Format**\nPlease upload a valid video file (.mp4, .mov, .mkv, .webm).")
            return

        # Normal Video Upload
        await _check_and_notify_busy_state(msg, str(from_user.get("id") or chat_id))
        status_msg = await msg.reply_text(
            "📥 **[VIDEO UPLOAD RECEIVED]**\n\n"
            "📁 **Status**: Downloading uploaded video file to local staging..."
        )
        media_obj = msg.video or msg.document
        temp_dir = os.path.join(_REPO_ROOT, "downloads", f"telegram_{chat_id}_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
        local_video_path = os.path.join(temp_dir, "video.mp4")
        try:
            await download_telegram_document(media_obj, msg, local_video_path, status_msg=status_msg)
        except Exception as dl_err:
            logger.error(f"❌ Failed to download user video: {dl_err}", exc_info=True)
            try:
                await status_msg.edit_text(f"❌ **Download Failed**\n\nCould not download video: {dl_err}")
            except Exception:
                pass
            return

        upload_goal = f"Process uploaded video at '{local_video_path}', edit with beat-synced rhythm and BGM, audit quality, and publish to social platforms."
        logger.info(f"🤖 [UNIVERSAL AGENT] Routing uploaded video to Gemini Director: '{upload_goal}'")
        await _dispatch_agent_goal(update, upload_goal, existing_status_msg=status_msg)
        return



# ── Full End-to-End Execution Pipeline ───────────────────────────────────────

def run_master_pipeline(
    mode: str = "auto",
    url: Optional[str] = None,
    input_path: Optional[str] = None,
    target_accounts: Optional[List[str]] = None,
    platform: str = "instagram",
    requestor_chat_id: Optional[int] = None,
    user_edit_directive: Optional[str] = None,
    excluded_audio: Optional[Union[str, List[str], Set[str]]] = None
) -> Dict[str, Any]:
    """
    Executes End-to-End Master Pipeline across all 4 Phases.
    Supports URL inputs, local file paths, or target account list.
    """
    logger.info("==================================================================")
    logger.info("🚀 [MASTER PIPELINE] Starting End-to-End Master Execution Cycle")
    logger.info("==================================================================")

    rendered_master_reels = []

    target_clip_dirs = None

    # ── Telegram Vault First-Look Check (Bypass Phase 1 Downloader) ───────────
    if url:
        try:
            v_entry = vault_indexer.find_entry_by_shortcode(url) or vault_indexer.lookup_downloaded_source(url)
            if v_entry:
                clean_fid = (
                    v_entry.get("media_file_ids", {}).get("wm_clean_file_id") or
                    v_entry.get("wm_clean_file_id")
                )
                raw_fid = (
                    v_entry.get("media_file_ids", {}).get("raw_video_file_id") or
                    v_entry.get("raw_video_file_id") or
                    v_entry.get("raw_file_id")
                )
                if clean_fid or raw_fid:
                    sc = v_entry.get("shortcode") or "vault_clip"
                    clip_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads", f"manual_{sc}")
                    os.makedirs(clip_dir, exist_ok=True)
                    hydrated_path = vault_indexer.hydrate_raw_video_from_vault(url, dest_dir=clip_dir, check_clean_first=True)
                    if hydrated_path and os.path.exists(hydrated_path) and os.path.getsize(hydrated_path) > 1024:
                        _kind = "watermark-cleaned" if "clean" in os.path.basename(hydrated_path).lower() else "raw source"
                        logger.info(f"⚡ [VAULT FIRST-LOOK HIT] Retrieved {_kind} from Telegram Vault -> {hydrated_path}. Skipping Phase 1 downloader!")
                        std_vid = os.path.join(clip_dir, "video.mp4")
                        if not os.path.exists(std_vid):
                            try:
                                import shutil
                                shutil.copy2(hydrated_path, std_vid)
                            except Exception:
                                pass
                        target_clip_dirs = [clip_dir]
                        # Also retrieve the already-extracted audio from Telegram Vault
                        try:
                            vault_indexer.hydrate_extracted_audio_from_vault(url, dest_dir=clip_dir)
                        except Exception as _ea_err:
                            logger.debug(f"[MASTER PIPELINE] Extracted audio vault retrieval notice: {_ea_err}")
        except Exception as _vh_err:
            logger.debug(f"[MASTER PIPELINE] Vault check error: {_vh_err}")

    # Phase 1: Ingestion (when mode == 'auto', or when URL/target_accounts provided and not already in vault)
    if not target_clip_dirs and (mode == "auto" or url or target_accounts):
        logger.info(f"📥 [PHASE 1] Running Content Ingestion for target: {target_accounts or url or 'auto-pool'} (platform='{platform}')...")
        ingest_res = run_phase1_ingestion(mode=mode, url=url, limit_per_account=3, target_accounts=target_accounts, platform=platform)
        if not ingest_res.get("success") and not input_path:
            logger.warning("⚠️ [MASTER PIPELINE] Ingestion completed with no new clips to process.")
            if requestor_chat_id:
                err_detail = ingest_res.get("error") or "All download strategies failed"
                if url:
                    fail_msg = (
                        "❌ **Download Failed**\n\n"
                        f"🔗 **URL**: {url}\n"
                        f"🌐 **Platform**: {platform.title()}\n\n"
                        f"⚠️ **Reason**: {err_detail}\n\n"
                        "💡 **Possible Causes & Solutions**:\n"
                        "• Post is private, restricted, login-gated, or age-gated\n"
                        "• Post has expired or was removed by the creator\n"
                        "• **Direct Video Upload**: You can download/save the video on your device and upload the video file directly into this chat!"
                    )
                elif target_accounts:
                    fail_msg = (
                        "⚠️ **Ingestion Failed**\n\n"
                        f"👤 **Target Account(s)**: `{', '.join(target_accounts)}`\n"
                        f"🌐 **Platform**: {platform.title()}\n\n"
                        f"⚠️ **Reason**: {err_detail}\n\n"
                        "💡 No active or downloadable reels found for the specified account(s)."
                    )
                else:
                    fail_msg = (
                        "⚠️ **Ingestion Notice**\n\n"
                        "No new clips found to process in the auto pool."
                    )
                send_telegram_text_sync(requestor_chat_id, fail_msg)
            return {"success": False, "rendered_files": [], "error": ingest_res.get("error")}
        
        dl_files = ingest_res.get("downloaded_files", [])
        if dl_files:
            target_clip_dirs = list(set(os.path.dirname(f) for f in dl_files if os.path.exists(f)))
            logger.info(f"   🎯 Targeted ingestion isolated {len(target_clip_dirs)} folder(s): {[os.path.basename(d) for d in target_clip_dirs]}")

    if not target_clip_dirs and not input_path:
        logger.warning("⚠️ [MASTER PIPELINE] No valid clip directories or input file found to edit.")
        if requestor_chat_id:
            fail_msg = (
                "❌ **Processing Failed**\n\n"
                f"🔗 **Target**: {url or (target_accounts and ', '.join(target_accounts)) or 'Unknown'}\n\n"
                "⚠️ **Reason**: No downloaded clips available for editing.\n\n"
                "💡 **Tip**: If downloading from a link fails, you can upload the video file directly into this chat!"
            )
            send_telegram_text_sync(requestor_chat_id, fail_msg)
        return {"success": False, "rendered_files": [], "error": "No valid clip directories or input file found"}

    # Phase 2 & 3: Master AI Perception & Render Orchestrator
    active_clips_registered = []
    try:
        from Main_Modules.phase2_main import run_phase2_orchestration
        from Core_Modules.disk_cleaner import mark_clip_active, mark_clip_inactive

        # Register active clips so background cleaners never purge their working directories
        if target_clip_dirs:
            for d in target_clip_dirs:
                cname = os.path.basename(d)
                mark_clip_active(cname)
                active_clips_registered.append(cname)
        elif input_path:
            cname = os.path.splitext(os.path.basename(input_path))[0]
            mark_clip_active(cname)
            active_clips_registered.append(cname)

        # ── Scalable Dynamic Multi-User Isolation ─────────────────────────────
        # Every Telegram user (User A, User B, User N) gets their rendered reel
        # delivered dynamically to their own chat window.
        # CLI/scheduled runs without a requestor_chat_id remain local (no spam).
        target_chat = requestor_chat_id
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

        def _on_clip_rendered(reel_path: str):
            """Immediately delivers each rendered clip to the requesting Telegram user."""
            PublishQueue.add(reel_path, channel_title="General", channel_folder="General")
            logger.info(f"🚀 [PHASE 4] Dispatched reel '{os.path.basename(reel_path)}' to publish queue.")
            reel_cid = os.path.basename(reel_path).replace("_master.mp4", "")
            if target_chat:
                session_manager.record_rendered_attempt(target_chat, reel_path, clip_id=reel_cid)

            if bot_token and target_chat:
                try:
                    import asyncio
                    from telegram import Bot
                    from telegram.request import HTTPXRequest

                    async def _send_single():
                        storage_group = os.getenv("TELEGRAM_STORAGE_GROUP_ID")
                        req = HTTPXRequest(
                            connection_pool_size=16,
                            read_timeout=600.0,
                            write_timeout=600.0,
                            connect_timeout=120.0,
                            pool_timeout=120.0
                        )
                        bot = Bot(token=bot_token, request=req)
                        active_reel_path = reel_path
                        real_cid = os.path.basename(active_reel_path).replace("_master.mp4", "")
                        clean_cid = real_cid.replace("manual_", "").replace("auto_", "").strip()
                        possible_raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads", real_cid, "video.mp4")

                        # ── PHASE 2.1 WATERMARK GATE (bypass if inpainting occurred upfront) ──
                        watermark_output = os.path.splitext(active_reel_path)[0] + "_wm_cleaned.mp4"
                        _inpainted_upfront = False

                        try:
                            from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore
                            _store = ClipIntelligenceStore()
                            _intel_d = _store.load(real_cid)
                            if _intel_d:
                                _inpainted_upfront = bool(
                                    _intel_d.get("forensic", {}).get("inpainted_upfront") or
                                    _intel_d.get("output", {}).get("mode") == "SINGLE_PASS" or
                                    _intel_d.get("output", {}).get("status") == "SUCCESS" or
                                    _intel_d.get("editing_plan")
                                )
                        except Exception as _ce:
                            logger.debug(f"[PHASE 2.1] Store lookup note: {_ce}")

                        if not _inpainted_upfront:
                            # Fallback to direct path inspection
                            _possible_intel_paths = [
                                os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads", real_cid, ".clip_intelligence.json"),
                                os.path.join(os.path.dirname(active_reel_path), ".clip_intelligence.json"),
                                os.path.join(os.path.dirname(active_reel_path).replace("Processed Shorts", "downloads"), real_cid, ".clip_intelligence.json")
                            ]
                            for intel_file in _possible_intel_paths:
                                if os.path.exists(intel_file):
                                    try:
                                        with open(intel_file, "r", encoding="utf-8") as _f:
                                            _id_data = json.load(_f)
                                        _inpainted_upfront = bool(
                                            _id_data.get("forensic", {}).get("inpainted_upfront") or
                                            _id_data.get("output", {}).get("mode") == "SINGLE_PASS" or
                                            _id_data.get("editing_plan")
                                        )
                                        break
                                    except Exception:
                                        continue

                        # Master reels (_master.mp4) rendered via Phase 2 single pass are already clean
                        if "_master.mp4" in os.path.basename(active_reel_path):
                            _inpainted_upfront = True

                        if _inpainted_upfront:
                            logger.info(f"⏭️ [PHASE 2.1] Skipping post-render watermark inpainting for {os.path.basename(active_reel_path)} (0 Gemini calls).")
                        else:
                            try:
                                from Watermark_and_Inpainting.watermark_main import run_watermark_removal
                                logger.info(f"🧼 [PHASE 2.1] Running watermark cleanup on rendered master before Telegram delivery: {os.path.basename(active_reel_path)}")
                                cleaned_video_path, watermark_log = run_watermark_removal(
                                    input_path=active_reel_path,
                                    output_path=watermark_output,
                                    keywords="",
                                    retry_level=0,
                                    video_path=active_reel_path
                                )
                                if cleaned_video_path and os.path.exists(cleaned_video_path):
                                    active_reel_path = cleaned_video_path
                                    logger.info(f"✅ [PHASE 2.1] Watermark gate completed. Telegram payload now uses: {os.path.basename(active_reel_path)}")
                            except Exception as _wm_err:
                                logger.warning(f"⚠️ [PHASE 2.1] Watermark gate failed, continuing with original render: {_wm_err}")


                        _clean_cid = real_cid.replace("manual_", "").strip()
                        _social_link = url or (f"https://instagram.com/reel/{_clean_cid}" if _clean_cid else None)
                        _sel_audio_track = _intel_d.get("audio_data", {}).get("selected_bgm_track") if _intel_d else None
                        sess_id = session_manager.create_session(
                            video_path=active_reel_path,
                            clip_id=real_cid,
                            raw_video_path=possible_raw if os.path.exists(possible_raw) else None,
                            requestor_chat_id=requestor_chat_id,
                            selected_audio=_sel_audio_track,
                            social_url=_social_link
                        )
                        keyboard = build_telegram_session_keyboard(session_id=sess_id, shortcode=real_cid)
                        sent_msg = await _send_video_safe_main(
                            bot,
                            target_chat,
                            active_reel_path,
                            caption=f"🎉 **AI Master Edit Complete!**\n📁 `{os.path.basename(active_reel_path)}`\n🆔 `Session: {sess_id}`",
                            reply_markup=keyboard,
                            supports_streaming=True
                        )
                        if sent_msg:
                            session_manager.update_message_id(sess_id, sent_msg.message_id)

                        # Storage Group Backup & Master Vault Index Sync
                        if storage_group:
                            try:
                                master_file_id = sent_msg.video.file_id if sent_msg and sent_msg.video else None
                                if str(target_chat) != str(storage_group):
                                    sg_msg = await _send_video_safe_main(
                                        bot,
                                        storage_group,
                                        active_reel_path,
                                        caption=f"📦 **[VAULT BACKUP]** Master Reel Backup\n📁 `{os.path.basename(active_reel_path)}`\n🆔 `Session: {sess_id}`",
                                        supports_streaming=True
                                    )
                                    if sg_msg and sg_msg.video:
                                        master_file_id = sg_msg.video.file_id

                                # Load clip intelligence and lyric intelligence from disk
                                clip_intel = {}
                                lyric_intel = {}
                                beat_math = {}
                                base_dir = os.path.dirname(os.path.abspath(__file__))
                                
                                intel_file = os.path.join(base_dir, "downloads", real_cid, ".clip_intelligence.json")
                                if os.path.exists(intel_file):
                                    try:
                                        with open(intel_file, "r", encoding="utf-8") as f:
                                            clip_intel = json.load(f)
                                    except Exception: pass

                                beats_file = os.path.join(base_dir, "Original_audio", "beats", f"{real_cid}_lyric.json")
                                if os.path.exists(beats_file):
                                    try:
                                        with open(beats_file, "r", encoding="utf-8") as f:
                                            lyric_intel = json.load(f)
                                    except Exception: pass

                                audio_analysis_file = os.path.join(base_dir, "downloads", real_cid, "audio_analysis.json")
                                if os.path.exists(audio_analysis_file):
                                    try:
                                        with open(audio_analysis_file, "r", encoding="utf-8") as f:
                                            beat_math = json.load(f)
                                    except Exception: pass

                                possible_audio = os.path.join(base_dir, "downloads", real_cid, "video_extracted.wav")
                                if not os.path.exists(possible_audio):
                                    clip_dir_path = os.path.join(base_dir, "downloads", real_cid)
                                    if os.path.exists(clip_dir_path):
                                        for f_item in os.listdir(clip_dir_path):
                                            if f_item.endswith(".wav") or f_item.endswith(".mp3") or f_item.endswith(".m4a"):
                                                possible_audio = os.path.join(clip_dir_path, f_item)
                                                break

                                clean_cid = real_cid.replace("manual_", "").strip()
                                social_link = url or f"https://instagram.com/reel/{clean_cid}"

                                # 1. Record Column 2 Source Download entry & Upload Raw Source + Audio to Vault
                                rec = await vault_indexer.record_downloaded_source(
                                    bot=bot,
                                    social_url=social_link,
                                    session_id=sess_id,
                                    raw_video_path=possible_raw if os.path.exists(possible_raw) else None,
                                    audio_path=possible_audio if os.path.exists(possible_audio) else None,
                                    beat_math=beat_math,
                                    pin_now=False
                                )
                                if rec and rec.get("raw_video_file_id"):
                                    session_manager.update_raw_video_file_id(sess_id, rec["raw_video_file_id"])
                                    session_manager.update_raw_video_file_id(real_cid, rec["raw_video_file_id"])
                                    session_manager.update_raw_video_file_id(clean_cid, rec["raw_video_file_id"])

                                if master_file_id:
                                    session_manager.update_processed_file_id(sess_id, master_file_id)
                                    session_manager.update_processed_file_id(real_cid, master_file_id)
                                    session_manager.update_processed_file_id(clean_cid, master_file_id)

                                # 2. Record Column 1 Master Reel entry with full intel and pin index
                                await vault_indexer.record_processed_reel(
                                    bot=bot,
                                    session_id=sess_id,
                                    social_url=social_link,
                                    custom_title=None,
                                    master_video_path=active_reel_path,
                                    clip_intel=clip_intel,
                                    lyric_intel=lyric_intel,
                                    master_file_id=master_file_id,
                                )

                            except Exception as _sg_e:
                                if "Chat not found" in str(_sg_e) or "chat not found" in str(_sg_e).lower():
                                    logger.warning(f"⚠️ Vault storage group backup skipped: Storage group chat not found. Check TELEGRAM_STORAGE_GROUP_ID in .env")
                                else:
                                    logger.warning(f"⚠️ Vault storage group backup warning: {_sg_e}")
                            finally:
                                # 3. Ephemeral Scratchpad Eraser: Wipes temporary downloads & beat cache.
                                # Preserves local master reel on disk until user review & publishing completes.
                                try:
                                    from Core_Modules.disk_cleaner import clean_clip_artifacts
                                    clean_clip_artifacts(real_cid)
                                    clean_clip_artifacts(clean_cid)
                                    logger.info(f"🧹 [ZERO-LEAK DISK CLEANER] Wiped ephemeral scratchpad artifacts for '{real_cid}'")
                                except Exception as _cln_e:
                                    logger.warning(f"⚠️ [ZERO-LEAK DISK CLEANER] Ephemeral wipe notice: {_cln_e}")

                    import threading

                    def _run_in_thread():
                        asyncio.run(_send_single())

                    t = threading.Thread(target=_run_in_thread, daemon=True)
                    t.start()
                    logger.info(f"📲 [REALTIME TELEGRAM DELIVERY] Dispatched delivery task for '{os.path.basename(reel_path)}' to Telegram!")
                except Exception as _deliv_err:
                    logger.error(f"❌ Realtime Telegram delivery failed for {os.path.basename(reel_path)}: {_deliv_err}")

        phase2_res = run_phase2_orchestration(
            input_path=input_path,
            target_dirs=target_clip_dirs,
            on_rendered_callback=_on_clip_rendered,
            user_edit_directive=user_edit_directive,
            excluded_audio=excluded_audio
        )
        rendered_master_reels = phase2_res.get("rendered_files", [])
        logger.info(f"🎬 [PHASE 2 & 3 COMPLETE] {len(rendered_master_reels)} master reel(s) ready.")

    except Exception as p2_err:
        logger.error(f"❌ [MASTER PIPELINE FAILED] Phase 2 error: {p2_err}")
        if requestor_chat_id:
            try:
                send_telegram_text_sync(
                    requestor_chat_id,
                    f"❌ **AI Editing Failed**\n\n"
                    f"⚠️ An error occurred during video editing:\n`{p2_err}`\n\n"
                    f"💡 Please try again or submit a different video file directly."
                )
            except Exception as _e_msg:
                logger.debug(f"Notice sending phase 2 failure message: {_e_msg}")
    finally:
        # Release active clip protection now that Phase 2 rendering has concluded
        try:
            from Core_Modules.disk_cleaner import mark_clip_inactive
            for cname in active_clips_registered:
                mark_clip_inactive(cname)
        except Exception:
            pass

    if not rendered_master_reels and requestor_chat_id and (url or input_path):
        try:
            send_telegram_text_sync(
                requestor_chat_id,
                "⚠️ **AI Editing Completed with No Output**\n\n"
                "The AI editor could not generate a rendered reel from the provided input.\n\n"
                "💡 Please verify the video and try uploading the video file directly into this chat."
            )
        except Exception as _e_msg:
            logger.debug(f"Notice sending zero-render failure message: {_e_msg}")

    return {"success": len(rendered_master_reels) > 0, "rendered_files": rendered_master_reels}


# ── Dual Pipeline Schedulers (Scraping vs Publishing) ────────────────────────

def normalize_time_slot(slot_str: str) -> Optional[str]:
    """Normalizes time string into standard HH:MM 24-hour format."""
    try:
        parts = slot_str.strip().split(":")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
    except Exception:
        pass
    return None


def parse_scraping_auto_input_times() -> List[str]:
    """
    Parses AUTO_INPUT_SCHEDULE_TIMES from .env and aggregates per-user schedule times
    from telegram_users.json so any registered user's configured schedule slot fires.
    Defaults to ['06:00', '19:00'].
    """
    raw = os.getenv("AUTO_INPUT_SCHEDULE_TIMES") or os.getenv("SCRAPING_AUTO_INPUT_TIMES") or os.getenv("APIFY_SCRAPE_SCHEDULE_TIMES") or "06:00,19:00"
    slots = set()
    for piece in raw.split(","):
        norm = normalize_time_slot(piece)
        if norm:
            slots.add(norm)

    # Hydrate per-user schedule times from telegram_users.json
    try:
        from Telegram_Storage_Modules.telegram_user_manager import load_all_users
        users = load_all_users()
        for u in users.values():
            u_sched = u.get("auto_input_schedule_times") or ""
            if u_sched:
                for piece in str(u_sched).split(","):
                    norm = normalize_time_slot(piece)
                    if norm:
                        slots.add(norm)
    except Exception as _ue:
        logger.debug("Notice parsing user schedule times: %s", _ue)

    return sorted(list(slots)) if slots else ["06:00", "19:00"]


def parse_static_publish_times() -> List[str]:
    """
    Legacy compatibility helper.
    Publishing is dynamic and triggers automatically inside the 06:00 & 19:00 execution cycle.
    """
    return parse_scraping_auto_input_times()



def resolve_app_timezone():
    """
    Returns a timezone object and readable label.
    Supports IANA names via zoneinfo or fallback offsets (defaults to Asia/Kolkata / IST).
    """
    import datetime
    tz_env = (os.getenv("APP_TIMEZONE") or os.getenv("TZ") or "Asia/Kolkata").strip()
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz_env), tz_env
    except Exception:
        pass

    known_offsets = {
        "asia/kolkata": (5, 30, "IST"),
        "kolkata": (5, 30, "IST"),
        "ist": (5, 30, "IST"),
        "utc": (0, 0, "UTC"),
        "gmt": (0, 0, "GMT"),
    }
    key = tz_env.lower()
    if key in known_offsets:
        h, m, name = known_offsets[key]
        return datetime.timezone(datetime.timedelta(hours=h, minutes=m), name=name), name

    return datetime.timezone(datetime.timedelta(hours=5, minutes=30), name="IST"), "IST"


def get_seconds_until_next_slot(slots: List[str]) -> tuple[str, float, str]:
    """
    Calculates the next target time slot and seconds remaining until it fires.
    Timezone-aware: respects APP_TIMEZONE / TZ (defaults to Asia/Kolkata / IST).
    """
    import datetime
    tz, tz_label = resolve_app_timezone()
    now = datetime.datetime.now(tz)
    today_date = now.date()

    candidates = []
    for slot_str in slots:
        try:
            h, m = map(int, slot_str.split(":"))
            slot_dt = datetime.datetime.combine(today_date, datetime.time(hour=h, minute=m), tzinfo=tz)
            if slot_dt <= now:
                slot_dt += datetime.timedelta(days=1)
            candidates.append((slot_str, slot_dt))
        except Exception:
            pass

    if not candidates:
        return ("12:00", 3600.0, tz_label)

    candidates.sort(key=lambda x: x[1])
    next_slot, next_dt = candidates[0]
    delay_s = max(1.0, (next_dt - now).total_seconds())
    return (next_slot, delay_s, tz_label)


def run_scheduled_pipeline_loop():
    """
    Blocking CLI Loop for Static Time Scheduler.
    Rotates max 2 source accounts and runs master pipeline automatically at each configured slot.
    """
    logger.info("==================================================================")
    logger.info("⏰ [SCRAPING SCHEDULER] Active auto-input slots loop starting...")
    logger.info("==================================================================")

    while True:
        # Dynamically reload slots from .env on every check
        slots = parse_scraping_auto_input_times()
        next_slot, delay_s, tz_label = get_seconds_until_next_slot(slots)
        hours_left = delay_s / 3600.0
        logger.info(f"⏳ [SCHEDULER] Next scheduled run at {next_slot} {tz_label} (in {hours_left:.2f} hours / {delay_s:.0f}s)...")
        # Sleep in chunks to emit heartbeat every 15 minutes
        remaining_s = delay_s
        while remaining_s > 0:
            to_sleep = min(remaining_s, 900.0)
            time.sleep(to_sleep)
            remaining_s -= to_sleep
            if remaining_s > 0:
                h_left = remaining_s / 3600.0
                logger.info(f"💓 [DAEMON HEARTBEAT] Scheduler active | Next run at {next_slot} {tz_label} (in {h_left:.2f}h / {remaining_s:.0f}s)...")
                sys.stdout.flush()

        logger.info(f"⏰ [SCHEDULER] Trigger time reached ({next_slot})! Executing max 2-account scraper batch...")
        try:
            rendered = run_scheduled_scraper_batch(max_accounts=2)
            logger.info(f"✅ [SCHEDULER] Completed batch run: {len(rendered)} reel(s) rendered.")
        except Exception as e:
            logger.error(f"❌ [SCHEDULER] Pipeline execution error: {e}")


async def _async_static_scheduler_task(bot_app=None):
    """
    Async background task running alongside Telegram Bot polling.
    Rotates max 2 accounts per scheduled slot, renders reels, creates sessions, and delivers to the account owner and admin.
    """
    import asyncio
    logger.info("⏰ [ASYNC SCRAPER SCHEDULER] Async scraper scheduler task active...")

    while True:
        # Dynamically reload slots from .env on every check
        slots = parse_scraping_auto_input_times()
        next_slot, delay_s, tz_label = get_seconds_until_next_slot(slots)
        hours_left = delay_s / 3600.0
        logger.info(f"⏳ [ASYNC SCHEDULER] Next run scheduled at {next_slot} {tz_label} (in {hours_left:.2f} hours)...")
        # Sleep in chunks to emit heartbeat every 15 minutes so console logs remain alive
        remaining_s = delay_s
        while remaining_s > 0:
            to_sleep = min(remaining_s, 900.0)
            await asyncio.sleep(to_sleep)
            remaining_s -= to_sleep
            if remaining_s > 0:
                h_left = remaining_s / 3600.0
                logger.info(f"💓 [ASYNC SCHEDULER HEARTBEAT] Telegram polling active | Next run at {next_slot} {tz_label} (in {h_left:.2f}h / {remaining_s:.0f}s) | Active background jobs: {len(ACTIVE_PIPELINE_JOBS)}")
                sys.stdout.flush()

        logger.info(f"⏰ [ASYNC SCHEDULER] Trigger time reached ({next_slot})! Starting 2-account ingestion & AI edit cycle...")
        try:
            rendered = run_scheduled_scraper_batch(max_accounts=2)
            if bot_app and rendered:
                from Downloader_Modules.scheduled_scraper_manager import get_account_owner_for_file
                admin_raw = os.getenv("TELEGRAM_ADMIN_ID")
                admin_ids = [x.strip() for x in str(admin_raw).split(",") if x.strip()] if admin_raw else []

                for r_file in rendered:
                    try:
                        owner_id = get_account_owner_for_file(r_file)
                        req_chat = int(owner_id) if (owner_id and str(owner_id).isdigit()) else None
                        sess_id = session_manager.create_session(video_path=r_file, requestor_chat_id=req_chat)
                        keyboard = build_telegram_session_keyboard(session_id=sess_id)

                        # Determine list of target chats to notify:
                        # 1. Normal user who configured the account receives the review card directly!
                        target_recipients = set()
                        if req_chat:
                            target_recipients.add(str(req_chat))
                        # 2. Admins also receive copy
                        for aid in admin_ids:
                            target_recipients.add(str(aid))

                        for target_chat in target_recipients:
                            try:
                                sent_msg = await _send_video_safe_main(
                                    bot_app.bot,
                                    target_chat,
                                    r_file,
                                    caption=(
                                        f"⏰ **[SCHEDULED SLOT {next_slot}]** Master Edit Complete!\n"
                                        f"📁 `{os.path.basename(r_file)}`\n"
                                        f"🆔 `Session: {sess_id}`"
                                    ),
                                    reply_markup=keyboard
                                )
                                if sent_msg and (str(target_chat) == str(req_chat) or not req_chat):
                                    session_manager.update_message_id(sess_id, sent_msg.message_id)
                            except Exception as _push_e:
                                logger.warning(f"Failed to send scheduled video to recipient {target_chat}: {_push_e}")
                    except Exception as _e:
                        logger.warning(f"Failed to process scheduled video {r_file}: {_e}")
        except Exception as e:
            logger.error(f"❌ [ASYNC SCHEDULER] Error during scheduled run: {e}")


def start_telegram_bot_service():
    """
    Launches continuous Telegram Bot Polling Listener.
    Waits for user input on Telegram (URL, video upload, or target account ID).
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token.startswith("your_"):
        logger.error("❌ TELEGRAM_BOT_TOKEN environment variable is missing or unconfigured in .env! (Set TELEGRAM_BOT_TOKEN in GitHub Secrets or .env)")
        sys.exit(1)

    try:
        from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters
        from telegram.request import HTTPXRequest

        logger.info("🤖 [TELEGRAM BOT] Starting Silent Long-Polling Telegram Listener...")

        request = HTTPXRequest(
            connection_pool_size=8,
            read_timeout=300.0,
            write_timeout=300.0,
            connect_timeout=30.0
        )

        async def _on_startup(application):
            try:
                global _global_bot_instance
                _global_bot_instance = application.bot
                # Hydrate all vault JSON databases (master index, users, audio pool) from Telegram Storage Group
                logger.info("📦 [VAULT CLOUD HYDRATION] Hydrating vault index, user sessions, and pool metadata from Telegram...")
                vault_indexer.hydrate_all_vault_jsons_on_startup()
                await vault_indexer.sync_vault_index_from_telegram(application.bot)

                bot_user = await application.bot.get_me()
                bot_id = str(bot_user.id)

                # Register native Telegram Bot Command Menu for auto-complete in chat
                try:
                    from telegram import BotCommand
                    commands = [
                        BotCommand("start", "Show welcome & platform selection menu"),
                        BotCommand("myconfig", "View your SaaS credential checklist & setup commands"),
                        BotCommand("autosetup", "Run interactive 6-step setup wizard"),
                        BotCommand("setgemini", "Set your Google Gemini API Key"),
                        BotCommand("setapify", "Set your Apify Scraper Token"),
                        BotCommand("setbranding", "Set your custom brand watermark text"),
                        BotCommand("setgroup", "Set your Telegram Public Group ID"),
                        BotCommand("setschedule", "Set daily scraping schedule times (HH:MM)"),
                        BotCommand("instagramtoken", "Set Instagram Access Token"),
                        BotCommand("instagramid", "Set Instagram Business Account ID"),
                        BotCommand("facebookid", "Set Facebook Page ID"),
                        BotCommand("setytclient", "Set YouTube client_secret.json"),
                        BotCommand("ytcode", "Set YouTube OAuth code/token"),
                        BotCommand("tiktoktoken", "Set TikTok Access Token"),
                        BotCommand("addaccount", "Add Instagram/YouTube handle to scrape"),
                        BotCommand("removeaccount", "Remove a handle from scraping list"),
                        BotCommand("listaccounts", "List active source handles"),
                        BotCommand("reject", "Reject & discard current reel"),
                        BotCommand("cancel", "Cancel title input or active wizard"),
                    ]
                    await application.bot.set_my_commands(commands)
                    logger.info("📋 Registered Telegram Bot Command Menu with Telegram API")
                except Exception as _cmd_err:
                    logger.warning(f"⚠️ set_my_commands notice: {_cmd_err}")

                admin_raw = os.getenv("TELEGRAM_ADMIN_ID")
                if admin_raw:
                    admin_ids = [x.strip() for x in str(admin_raw).split(",") if x.strip()]
                    keyboard = build_platform_selection_keyboard()
                    for aid in admin_ids:
                        if aid == bot_id:
                            continue
                        try:
                            await application.bot.send_message(
                                chat_id=int(aid),
                                text=(
                                    "🚀 **Master AI Video Factory Bot Active!**\n\n"
                                    "🎯 **Select a platform below to begin bulk scraping & editing**, or send a creator handle / video link:\n"
                                    "• 📸 **Instagram**: `@username` or `/addaccount @handle`\n"
                                    "• 🔴 **YouTube**: `@ChannelName`\n"
                                    "• 🎵 **TikTok**: `@tiktokuser`\n"
                                    "• 🌐 **Direct URL / File**: Paste link or upload video\n\n"
                                    "👇 **Select your target platform below:**"
                                ),
                                reply_markup=keyboard
                            )
                            logger.info(f"📲 Automatically pushed Platform Selection Menu to Telegram Admin ID: {aid}")
                        except Exception as _st_err:
                            error_str = str(_st_err)
                            if "can't send messages to the bot" in error_str.lower() or "bot can't initiate" in error_str.lower():
                                logger.info(f"ℹ️ Startup push skipped for {aid}: TELEGRAM_ADMIN_ID is set to bot ID or user has not sent /start yet.")
                            elif "Chat not found" in error_str or "chat not found" in error_str.lower():
                                logger.warning(f"⚠️ Startup welcome push warning: Admin chat ID {aid} not found. Bot may not have access to this chat, or the ID is incorrect.")
                            else:
                                logger.warning(f"⚠️ Startup welcome push warning for {aid}: {_st_err}")

                # Spawn background auto-input schedule task alongside Telegram bot polling
                asyncio.create_task(_async_static_scheduler_task(application))
            except Exception as _init_err:
                logger.warning(f"⚠️ Startup check exception: {_init_err}")

        app = ApplicationBuilder().token(token).request(request).post_init(_on_startup).build()
        app.add_handler(CommandHandler("start", handle_telegram_start))
        app.add_handler(CommandHandler("help", handle_telegram_start))
        app.add_handler(CallbackQueryHandler(handle_telegram_callback))

        # ── Wizard shortcut commands ──────────────────────────────────────────
        async def _cmd_auto_setup(update, context):
            sess = _wizard_sessions.setdefault(update.effective_chat.id, {"wizard": "auto_setup", "step": 1, "data": {}})
            sess.update({"wizard": "auto_setup", "step": 1, "data": {}})
            await update.message.reply_text(
                "⚙️ **Auto Input Setup — Step 1/6: Source Account IDs**\n\n"
                "Send the creator handles or profile URLs you want to scrape.\n\n"
                "📌 **Accepted Formats**:\n"
                "  • `@creator_handle` (or `@nike, @adidas`)\n"
                "  • `https://www.instagram.com/creator_handle/`\n"
                "  • `instagram @handle`, `youtube @channel`, `tiktok @user`\n\n"
                "You can add up to **2 accounts total** across any mix of platforms.\n"
                "Send them now 👇",
                reply_markup=build_back_button_keyboard()
            )
        async def _cmd_addaccount(update, context):
            args_text = " ".join(context.args).strip() if context.args else ""
            if not args_text:
                await update.message.reply_text("Usage: `/addaccount @handle` (or `/addaccount https://instagram.com/handle`)")
                return
            parsed = parse_target_accounts_input(args_text)
            handle = parsed[0]["handle"] if parsed else args_text.lstrip("@").strip()
            plat = parsed[0]["platform"] if parsed else "instagram"
            try:
                from Downloader_Modules.scheduled_scraper_manager import add_source_account
                add_source_account(handle, plat, owner_chat_id=update.effective_chat.id)
            except Exception as _e:
                logger.debug(f"addaccount: {_e}")
            await update.message.reply_text(f"✅ Account `@{handle}` ({plat.title()}) added! Use ⚙️ Auto Input Setup to configure full schedule.")
        async def _cmd_removeaccount(update, context):
            args_text = " ".join(context.args).strip() if context.args else ""
            handle = args_text.lstrip("@").strip()
            try:
                from Downloader_Modules.scheduled_scraper_manager import remove_source_account
                removed = remove_source_account(handle, requestor_chat_id=update.effective_chat.id)
                if removed:
                    await update.message.reply_text(f"🗑️ Account `@{handle}` removed!")
                else:
                    await update.message.reply_text(f"⚠️ Could not remove `@{handle}` (permission denied or account not found).")
            except Exception as _e:
                await update.message.reply_text(f"⚠️ Could not remove `@{handle}`: {_e}")
        async def _cmd_listaccounts(update, context):
            try:
                text, kbd = build_active_accounts_keyboard_and_text(chat_id=update.effective_chat.id)
                await update.message.reply_text(text=text, reply_markup=kbd)
            except Exception as _e:
                await update.message.reply_text(f"⚠️ Could not list accounts: {_e}")
        async def _cmd_setapify(update, context):
            args_text = " ".join(context.args).strip() if context.args else ""
            if not args_text:
                await update.message.reply_text("Usage: `/setapify YOUR_APIFY_TOKEN`")
                return
            try:
                from Telegram_Storage_Modules.telegram_user_manager import set_user_apify_token
                set_user_apify_token(str(update.effective_chat.id), args_text)
            except Exception as _e:
                logger.debug(f"setapify: {_e}")
            await update.message.reply_text("✅ Apify API Token Synced as your secret")
        async def _cmd_setgemini(update, context):
            args_text = " ".join(context.args).strip() if context.args else ""
            if not args_text:
                await update.message.reply_text("Usage: `/setgemini YOUR_GEMINI_KEY`")
                return
            try:
                from Telegram_Storage_Modules.telegram_user_manager import set_user_gemini_key
                set_user_gemini_key(str(update.effective_chat.id), args_text)
            except Exception as _e:
                logger.debug(f"setgemini: {_e}")
            await update.message.reply_text("✅ Gemini API Key Synced as your secret")
        async def _cmd_ytcode(update, context):
            await cmd_ytcode(update, context)

        async def _cmd_setbranding(update, context):
            args_text = " ".join(context.args).strip() if context.args else ""
            if not args_text:
                await update.message.reply_text("Usage: `/setbranding your_brand_watermark`")
                return
            try:
                from Telegram_Storage_Modules.telegram_user_manager import set_user_branding
                set_user_branding(str(update.effective_chat.id), args_text)
                await update.message.reply_text(f"🏷️ **Personal branding watermark updated to:** `{args_text}`\nPersonal Branding Watermark Synced as your secret")
            except Exception as _e:
                await update.message.reply_text(f"⚠️ Error saving branding watermark: {_e}")

        async def _cmd_setgroup(update, context):
            args_text = " ".join(context.args).strip() if context.args else ""
            if not args_text:
                await update.message.reply_text("Usage: `/setgroup YOUR_PUBLIC_GROUP_ID` (e.g., -1001234567890)")
                return
            try:
                from Telegram_Storage_Modules.telegram_user_manager import set_user_public_group_id
                ok = set_user_public_group_id(str(update.effective_chat.id), args_text)
                if ok:
                    msg = f"📢 **Public Group ID updated to:** `{args_text}`\nPublic Group ID Synced as your secret"
                else:
                    msg = f"📢 **Public Group ID updated to:** `{args_text}`\nPublic Group ID Synced as your secret"
                await update.message.reply_text(msg)
            except Exception as _e:
                await update.message.reply_text(f"⚠️ Error saving group ID: {_e}")

        async def _cmd_setschedule(update, context):
            args_text = " ".join(context.args).strip() if context.args else ""
            if not args_text:
                await update.message.reply_text("Usage: `/setschedule HH:MM,HH:MM` (e.g., `/setschedule 06:00,19:00`)")
                return
            try:
                from Telegram_Storage_Modules.telegram_user_manager import set_user_schedule_times
                set_user_schedule_times(str(update.effective_chat.id), args_text)
                await update.message.reply_text(f"⏰ **Schedule times updated to:** `{args_text}`\nSchedule Times Synced as your secret")
            except Exception as _e:
                await update.message.reply_text(f"⚠️ Error saving schedule times: {_e}")

        async def _cmd_setytclient(update, context):
            args_text = " ".join(context.args).strip() if context.args else ""
            if not args_text:
                await update.message.reply_text("Usage: `/setytclient YOUR_CLIENT_SECRET_JSON`")
                return
            try:
                from Telegram_Storage_Modules.telegram_user_manager import set_user_youtube_client_secret
                uid = str(update.effective_user.id if update.effective_user else update.effective_chat.id)
                set_user_youtube_client_secret(uid, args_text)
                await update.message.reply_text("🔐 **YouTube Client Secret Synced as your secret**")
            except Exception as _e:
                await update.message.reply_text(f"⚠️ Error saving YouTube client secret: {_e}")

        async def _cmd_instagramtoken(update, context):
            args_text = " ".join(context.args).strip() if context.args else ""
            if not args_text:
                await update.message.reply_text("Usage: `/instagramtoken YOUR_INSTAGRAM_ACCESS_TOKEN`")
                return
            try:
                from Telegram_Storage_Modules.telegram_user_manager import set_user_instagram_token
                set_user_instagram_token(str(update.effective_chat.id), args_text)
                await update.message.reply_text("📸 **Instagram Access Token Synced as your secret**")
            except Exception as _e:
                await update.message.reply_text(f"⚠️ Error saving Instagram token: {_e}")

        async def _cmd_instagramid(update, context):
            args_text = " ".join(context.args).strip() if context.args else ""
            if not args_text:
                await update.message.reply_text("Usage: `/instagramid YOUR_INSTAGRAM_BUSINESS_ACCOUNT_ID`")
                return
            try:
                from Telegram_Storage_Modules.telegram_user_manager import set_user_instagram_id
                set_user_instagram_id(str(update.effective_chat.id), args_text)
                await update.message.reply_text(f"🆔 **Instagram Business Account ID updated to:** `{args_text}`\nInstagram Business Account ID Synced as your secret")
            except Exception as _e:
                await update.message.reply_text(f"⚠️ Error saving Instagram Business ID: {_e}")

        async def _cmd_facebookid(update, context):
            args_text = " ".join(context.args).strip() if context.args else ""
            if not args_text:
                await update.message.reply_text("Usage: `/facebookid YOUR_FACEBOOK_PAGE_ID`")
                return
            try:
                from Telegram_Storage_Modules.telegram_user_manager import set_user_facebook_id
                set_user_facebook_id(str(update.effective_chat.id), args_text)
                await update.message.reply_text(f"📘 **Facebook Page ID updated to:** `{args_text}`\nFacebook Page ID Synced as your secret")
            except Exception as _e:
                await update.message.reply_text(f"⚠️ Error saving Facebook Page ID: {_e}")

        async def _cmd_facebooktoken(update, context):
            args_text = " ".join(context.args).strip() if context.args else ""
            if not args_text:
                await update.message.reply_text("Usage: `/facebooktoken YOUR_FACEBOOK_PAGE_ACCESS_TOKEN`")
                return
            try:
                from Telegram_Storage_Modules.telegram_user_manager import set_user_facebook_token
                set_user_facebook_token(str(update.effective_chat.id), args_text)
                await update.message.reply_text("🔑 **Facebook Page Access Token Synced as your secret**")
            except Exception as _e:
                await update.message.reply_text(f"⚠️ Error saving Facebook token: {_e}")

        async def _cmd_tiktoktoken(update, context):
            args_text = " ".join(context.args).strip() if context.args else ""
            if not args_text:
                await update.message.reply_text("Usage: `/tiktoktoken YOUR_TIKTOK_ACCESS_TOKEN`")
                return
            try:
                from Telegram_Storage_Modules.telegram_user_manager import set_user_tiktok_token
                set_user_tiktok_token(str(update.effective_chat.id), args_text)
                await update.message.reply_text("🎵 **TikTok Access Token Synced as your secret**")
            except Exception as _e:
                await update.message.reply_text(f"⚠️ Error saving TikTok token: {_e}")

        async def _cmd_myconfig(update, context):
            try:
                from Telegram_Storage_Modules.telegram_user_manager import format_user_credentials_summary
                summary_text = format_user_credentials_summary(str(update.effective_chat.id))
                await update.message.reply_text(summary_text)
            except Exception as _e:
                await update.message.reply_text(f"⚠️ Error fetching config: {_e}")

        async def _cmd_reject(update, context):
            chat_id = update.effective_chat.id
            pending_sess = session_manager.get_pending_title_session(chat_id)
            if not pending_sess:
                for s in reversed(list(session_manager.sessions.values())):
                    if s.get("status") in ("AWAITING_REVIEW", "AWAITING_TITLE"):
                        if not s.get("requestor_chat_id") or str(s.get("requestor_chat_id")) == str(chat_id):
                            pending_sess = s
                            break
            if pending_sess:
                sess_id = pending_sess["session_id"]
                sess = session_manager.set_rejected(sess_id)
                if sess:
                    from Core_Modules import purge_full_clip_and_assets
                    purge_res = purge_full_clip_and_assets(
                        clip_id=sess.get("clip_id"),
                        video_path=sess.get("video_path"),
                        attempt_history=sess.get("attempt_history", []),
                        selected_audio=sess.get("selected_audio")
                    )
                    logger.info(f"🗑️ Purged assets for session {sess_id} via /reject: {purge_res.get('purged_count')} items removed")
                await update.message.reply_text("🗑️ **REJECTED & DISCARDED BY ADMIN.**\nAll audio, video assets, and metadata pool index purged!")
            else:
                await update.message.reply_text("ℹ️ No active review or pending title session found to reject.")

        async def _cmd_cancel(update, context):
            chat_id = update.effective_chat.id
            pending_sess = session_manager.get_pending_title_session(chat_id)
            if pending_sess and pending_sess.get("status") == "AWAITING_TITLE":
                sess_id = pending_sess["session_id"]
                session_manager.revert_awaiting_title(sess_id)
                sess = session_manager.get_session(sess_id)
                clip_id = sess.get("clip_id") if sess else None
                review_kbd = build_telegram_session_keyboard(session_id=sess_id, shortcode=clip_id)
                await update.message.reply_text("↩️ **Cancelled title input.** Returned reel to review options.", reply_markup=review_kbd)
            else:
                _wizard_sessions.pop(chat_id, None)
                user_pending_reedit_session.pop(chat_id, None)
                await update.message.reply_text("↩️ Cancelled active operation.")

        async def _cmd_agent(update, context):
            """
            /agent <goal>
            Autonomous ReAct agent powered by Gemini 2.5 Flash.
            Dispatches non-blocking async pipeline task with live progress reporting.
            """
            user_goal = " ".join(context.args).strip() if context.args else ""
            if not user_goal:
                await update.message.reply_text(
                    "🤖 **Autonomous Creative Director (Gemini 2.5 Flash)**\n\n"
                    "Instruct the agent with any video production goal:\n\n"
                    "📌 **Usage:**\n"
                    "`/agent <your goal description>`\n\n"
                    "💡 **Examples:**\n"
                    "• `/agent Scrape @nike top 1 reel, edit with cinematic beats and publish`\n"
                    "• `/agent Ingest https://instagram.com/reel/Cxxxxxx/ and render with high energy hook`\n"
                    "• `/agent Scrape @cristiano 1 reel and audit dopamine retention`"
                )
                return

            await _dispatch_agent_goal(update, user_goal)

        app.add_handler(CommandHandler("agent", _cmd_agent))
        app.add_handler(CommandHandler("director", _cmd_agent))
        app.add_handler(CommandHandler("autosetup", _cmd_auto_setup))
        app.add_handler(CommandHandler("addaccount", _cmd_addaccount))
        app.add_handler(CommandHandler("removeaccount", _cmd_removeaccount))
        app.add_handler(CommandHandler("listaccounts", _cmd_listaccounts))
        app.add_handler(CommandHandler("setapify", _cmd_setapify))
        app.add_handler(CommandHandler("setgemini", _cmd_setgemini))
        app.add_handler(CommandHandler("ytcode", cmd_ytcode))
        app.add_handler(CommandHandler("setbranding", _cmd_setbranding))
        app.add_handler(CommandHandler("setgroup", _cmd_setgroup))
        app.add_handler(CommandHandler("setschedule", _cmd_setschedule))
        app.add_handler(CommandHandler("setytclient", _cmd_setytclient))
        app.add_handler(CommandHandler("instagramtoken", _cmd_instagramtoken))
        app.add_handler(CommandHandler("instagramid", _cmd_instagramid))
        app.add_handler(CommandHandler("facebookid", _cmd_facebookid))
        app.add_handler(CommandHandler("facebooktoken", _cmd_facebooktoken))
        app.add_handler(CommandHandler("tiktoktoken", _cmd_tiktoktoken))
        app.add_handler(CommandHandler("myconfig", _cmd_myconfig))
        app.add_handler(CommandHandler("reject", _cmd_reject))
        app.add_handler(CommandHandler("rejet", _cmd_reject))
        app.add_handler(CommandHandler("cancel", _cmd_cancel))
        app.add_handler(CommandHandler("back", _cmd_cancel))
        app.add_handler(CommandHandler("setstoragechat", handle_telegram_incoming_msg))
        app.add_handler(CommandHandler("metagraphapi", handle_telegram_incoming_msg))

        async def _cmd_unknown(update, context):
            if update.message:
                cmd = update.message.text.split()[0] if update.message.text else "command"
                uid = update.effective_user.id if update.effective_user else "unknown"
                logger.warning(f"🛡️ [SECURITY GUARD] Unknown command '{cmd}' blocked from User {uid}")
                await update.message.reply_text(
                    f"⚠️ **Unknown command:** `{cmd}`\n\n"
                    f"Send `/start` to view the platform menu, or `/help` for assistance."
                )

        app.add_handler(MessageHandler(filters.COMMAND, _cmd_unknown))
        app.add_handler(MessageHandler((filters.TEXT & ~filters.COMMAND) | filters.VIDEO | filters.Document.ALL, handle_telegram_incoming_msg))

        try:
            from Core_Modules.disk_cleaner import sweep_all_stale_storage, check_storage_circuit_breaker
            check_storage_circuit_breaker(min_free_gb=3.0)
            sweep_all_stale_storage(max_age_hours=2)
        except Exception as _sw_err:
            logger.debug(f"Startup disk sweeper notice: {_sw_err}")

        # Start periodic background storage guardian thread (sweeps every 30 mins)
        def _storage_maintenance_worker():
            import time
            while True:
                time.sleep(1800)  # 30 minutes
                try:
                    from Core_Modules.disk_cleaner import sweep_all_stale_storage, check_storage_circuit_breaker
                    check_storage_circuit_breaker(min_free_gb=3.0)
                    sweep_all_stale_storage(max_age_hours=2)
                except Exception as _m_err:
                    logger.debug(f"Periodic storage sweeper notice: {_m_err}")

        import threading
        _m_thread = threading.Thread(target=_storage_maintenance_worker, daemon=True, name="StorageGuardianThread")
        _m_thread.start()
        logger.info("🛡️ [STORAGE GUARDIAN] Periodic 30-minute storage sweeper thread active.")

        # CI Session Timer: ONLY active inside GitHub Actions!
        # On Oracle Cloud (or local server), GITHUB_ACTIONS is not set, so this timer NEVER runs
        # and the bot runs 24/7/365 continuously without interruption.
        if os.getenv("GITHUB_ACTIONS") == "true":
            max_run_h = os.getenv("MAX_RUN_HOURS", "3.95")
            try:
                run_seconds = float(max_run_h) * 3600.0
                def _ci_auto_stop_worker():
                    time.sleep(run_seconds)
                    logger.info(f"🛑 [CI AUTO-STOP] Configured max runtime of {max_run_h}h reached ({run_seconds/60:.0f}m). Gracefully exiting...")
                    os._exit(0)

                _ci_thread = threading.Thread(target=_ci_auto_stop_worker, daemon=True, name="CIAutoStopThread")
                _ci_thread.start()
                logger.info(f"⏱️ [CI SESSION TIMER] Auto-stop timer active: Runner will shut down in {max_run_h}h ({run_seconds/60:.0f} mins).")
            except Exception as _te:
                logger.warning(f"Failed to start CI auto-stop timer: {_te}")

        logger.info("✅ Telegram Bot Active & Listening! Platform Selection Menu dispatched to admin chat.")
        app.run_polling(poll_interval=2.0, drop_pending_updates=False)
    except KeyboardInterrupt:
        logger.info("\n🛑 [SHUTDOWN] Ctrl+C / KeyboardInterrupt received. Telegram Bot stopped gracefully.")
        sys.exit(0)
    except Exception as bot_err:
        logger.error(f"❌ Failed to start Telegram bot polling: {bot_err}")
        sys.exit(1)


# ── CLI Entrypoint ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Master AI Video Factory Orchestrator & Telegram Bot")
        parser.add_argument("input", type=str, nargs="?", default=None, help="Target video URL, local .mp4 file, or clip folder")
        parser.add_argument("--mode", type=str, choices=["auto", "manual", "batch"], default=None, help="Ingestion mode ('auto', 'manual', or 'batch')")
        parser.add_argument("--url", "-u", type=str, default=None, help="Target video URL for manual ingestion")
        parser.add_argument("--target-accounts", "-t", type=str, default=None, help="Target Instagram handle(s) to scrape (comma-separated, max 2)")
        parser.add_argument("--bot", action="store_true", help="Launch Telegram Bot polling listener mode + timed auto-input scheduler")
        parser.add_argument("--legacy", action="store_true", help="Force legacy procedural pipeline instead of Universal Agent")
        parser.add_argument("--batch", action="store_true", help="Run a single scheduled batch scrape & edit cycle and exit")

        args = parser.parse_args()

        # Resolve mode from argument or environment variable
        env_mode = os.getenv("DISPATCH_MODE")
        is_batch_run = args.batch or (args.mode == "batch") or (env_mode == "batch") or (os.getenv("CI_BATCH_MODE") == "1")

        if is_batch_run:
            from Downloader_Modules.scheduled_scraper_manager import run_scheduled_scraper_batch
            logger.info("🚀 [BATCH RUN] Triggering single scheduled batch scrape & master edit cycle...")
            rendered = run_scheduled_scraper_batch(max_accounts=2)
            logger.info(f"✅ [BATCH RUN] Batch completed successfully: {len(rendered)} reel(s) rendered.")
            sys.exit(0)

        mode_val = args.mode or (env_mode if env_mode in ["auto", "manual"] else None)

        env_target_accs = os.getenv("DISPATCH_TARGET_ACCOUNTS")
        env_target_url = os.getenv("DISPATCH_TARGET_URL")

        # AUTO mode = continuous 24/7 daemon (Telegram bot polling + async static scheduler task)
        # MANUAL mode = targeted single run for specific URL or accounts
        is_manual_targeted = (mode_val == "manual") or bool(args.input or args.url or args.target_accounts or env_target_url or env_target_accs)

        if not is_manual_targeted:
            # Launch 24/7 continuous daemon (Telegram Bot + background scraper scheduler)
            start_telegram_bot_service()
        else:
            target_input = args.input or args.url or env_target_url
            target_url = None
            target_file = None
            target_accs = None

            raw_target_accs = args.target_accounts or env_target_accs
            if raw_target_accs:
                target_accs = [a.strip().lstrip("@") for a in raw_target_accs.split(",") if a.strip()][:2]
            elif target_input:
                clean_in = target_input.strip().strip("'").strip('"')
                if clean_in.startswith("http://") or clean_in.startswith("https://") or "instagram.com" in clean_in:
                    target_url = clean_in
                elif os.path.exists(clean_in):
                    target_file = clean_in
                else:
                    # Validate if positional input argument is a valid Instagram handle (letters, numbers, dots, underscores)
                    import re
                    handle_candidate = clean_in.lstrip("@")
                    if re.match(r"^[A-Za-z0-9._]{1,30}$", handle_candidate):
                        target_accs = [handle_candidate]
                    else:
                        logger.warning(f"⚠️ Invalid target input or handle format provided: '{clean_in}'. Defaulting to target account pool.")

            mode_to_use = mode_val or ("manual" if (target_url or target_file) else "auto")
            try:
                from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
                logger.info("📡 [STARTUP] Syncing master vault index & pool_metadata.json from Telegram Storage Group...")
                TelegramVaultIndexer().hydrate_all_vault_jsons_on_startup()
            except Exception as _h_err:
                logger.warning(f"⚠️ Startup vault hydration notice: {_h_err}")

            try:
                from Core_Modules.disk_cleaner import sweep_all_stale_storage, check_storage_circuit_breaker
                check_storage_circuit_breaker(min_free_gb=3.0)
                sweep_all_stale_storage(max_age_hours=2)
            except Exception as _sw_err:
                logger.debug(f"Startup disk sweeper notice: {_sw_err}")

            _cli_use_agent = False
            if not args.legacy and target_input and not (target_url or target_file or target_accs):
                from Agentic_Core.goal_router import classify_goal
                _cli_use_agent = classify_goal(target_input).kind == "freeform"
            if _cli_use_agent:
                from Agentic_Core.run_agent import run_agentic_goal
                logger.info(f"🤖 [CLI AGENT] Free-form goal -> LLM Director: '{target_input}'")
                run_agentic_goal(target_input)
            else:
                run_master_pipeline(mode=mode_to_use, url=target_url, input_path=target_file, target_accounts=target_accs)

    except KeyboardInterrupt:
        logger.info("\n🛑 [SHUTDOWN] Ctrl+C / KeyboardInterrupt received. System stopped gracefully.")
        sys.exit(0)
