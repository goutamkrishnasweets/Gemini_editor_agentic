"""
Telegram_Storage_Modules/telegram_http.py
===========================================
Minimal raw HTTP wrapper for the Telegram Bot API with MTProto Large File Streaming.

This is the primary module in the vault stack that talks to Telegram directly.
Uses stdlib urllib where possible so it works without any extra packages.
Falls back to `requests` for streaming downloads if available.
Supports Pyrogram MTProto fallback for files > 20MB (up to 2GB).
"""

import os
import json
import uuid
import logging
import asyncio
import concurrent.futures
import urllib.request
import urllib.error
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("vault.telegram_http")

# Safely pre-import pyrogram if present to avoid thread-local import race/missing loop issues
try:
    from pyrogram import Client as _PyroClient
except Exception:
    _PyroClient = None


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    """Ensures the current thread has an active event loop set, creating one if needed."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def _run_coro_safely(coro_or_factory):
    """
    Safely executes an async coroutine or coroutine factory synchronously from any context:
      - Active running event loop (e.g. main asyncio loop) -> isolated thread worker with a new event loop
      - Worker thread without event loop -> creates and binds thread event loop
      - Worker thread with inactive event loop -> runs loop.run_until_complete
    Prevents 'cannot reuse already awaited coroutine' errors by creating/running fresh coroutines.
    """
    is_callable = callable(coro_or_factory) and not asyncio.iscoroutine(coro_or_factory)
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop and running_loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            def _thread_worker():
                thr_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(thr_loop)
                try:
                    c = coro_or_factory() if is_callable else coro_or_factory
                    return thr_loop.run_until_complete(c)
                finally:
                    thr_loop.close()
            return executor.submit(_thread_worker).result()
    else:
        loop = _ensure_event_loop()
        c = coro_or_factory() if is_callable else coro_or_factory
        return loop.run_until_complete(c)


# Automatically load environment variables from Credentials/.env, .env, or telegram_config.json
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_telegram_credentials() -> None:
    """Loads Telegram credentials from .env, Credentials/.env, telegram_config.json, or TELEGRAM_CONFIG_JSON."""
    try:
        from dotenv import load_dotenv
        for env_path in [
            os.path.join(_REPO_ROOT, "Credentials", ".env"),
            os.path.join(_REPO_ROOT, ".env"),
        ]:
            if os.path.exists(env_path):
                load_dotenv(env_path, override=False)
    except ImportError:
        pass

    def _apply_config_dict(cfg: Dict[str, Any]) -> None:
        key_mappings = {
            "TELEGRAM_API_ID": ["TELEGRAM_API_ID", "api_id", "telegram_api_id", "app_id"],
            "TELEGRAM_API_HASH": ["TELEGRAM_API_HASH", "api_hash", "telegram_api_hash", "app_hash"],
            "TELEGRAM_BOT_TOKEN": ["TELEGRAM_BOT_TOKEN", "bot_token", "token"],
            "TELEGRAM_STORAGE_GROUP_ID": ["TELEGRAM_STORAGE_GROUP_ID", "storage_group_id", "group_id", "chat_id"],
            "TELEGRAM_ADMIN_ID": ["TELEGRAM_ADMIN_ID", "admin_id"],
            "TELEGRAM_PUBLIC_GROUP_ID": ["TELEGRAM_PUBLIC_GROUP_ID", "public_group_id"],
        }
        for target_env, candidates in key_mappings.items():
            if not os.getenv(target_env):
                for c in candidates:
                    val = cfg.get(c)
                    if val is not None and str(val).strip():
                        os.environ[target_env] = str(val).strip()
                        break

    for cfg_path in [
        os.path.join(_REPO_ROOT, "Credentials", "telegram_config.json"),
        os.path.join(_REPO_ROOT, "telegram_config.json"),
    ]:
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if isinstance(cfg, dict):
                        _apply_config_dict(cfg)
            except Exception as e:
                logger.debug("Notice loading %s: %s", cfg_path, e)

    raw_json = os.getenv("TELEGRAM_CONFIG_JSON", "").strip()
    if raw_json and raw_json.startswith("{"):
        try:
            cfg = json.loads(raw_json)
            if isinstance(cfg, dict):
                _apply_config_dict(cfg)
        except Exception as e:
            logger.debug("Notice parsing TELEGRAM_CONFIG_JSON: %s", e)


_load_telegram_credentials()


def _token() -> str:
    t = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not t:
        raise RuntimeError("TELEGRAM_BOT_TOKEN env var is not set")
    return t


def _group_id() -> str:
    g = os.getenv("TELEGRAM_STORAGE_GROUP_ID", "").strip()
    if not g:
        raise RuntimeError("TELEGRAM_STORAGE_GROUP_ID env var is not set")
    return g


def _api_url(method: str) -> str:
    return f"https://api.telegram.org/bot{_token()}/{method}"


DEFAULT_TELEGRAM_API_ID = 2040
DEFAULT_TELEGRAM_API_HASH = "b1844dd134348e3da5c715c5acc94cf8"
ANDROID_TELEGRAM_API_ID = 6
ANDROID_TELEGRAM_API_HASH = "eb06d4abfb49dc3eeb1aeb98ae0f581e"


def get_mtproto_credentials() -> Tuple[int, str]:
    """
    Resolves MTProto api_id and api_hash.
    Priority:
      1. Environment variables (TELEGRAM_API_ID, TELEGRAM_API_HASH)
      2. Credentials/telegram_config.json or TELEGRAM_CONFIG_JSON
      3. Built-in official Telegram Desktop / Android client credentials fallback
    """
    _load_telegram_credentials()
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")

    clean_id = str(api_id).strip() if api_id is not None else ""
    clean_hash = str(api_hash).strip() if api_hash is not None else ""

    if clean_id == "6":
        hash_val = clean_hash if (clean_hash and clean_hash not in ("dummy", "None", "dummy_hash")) else ANDROID_TELEGRAM_API_HASH
        return (6, hash_val)

    if clean_id and clean_id not in ("", "dummy", "None", "0"):
        try:
            int_id = int(clean_id)
            if clean_hash and clean_hash not in ("", "dummy", "None", "dummy_hash"):
                return (int_id, clean_hash)
        except ValueError:
            pass

    return (DEFAULT_TELEGRAM_API_ID, DEFAULT_TELEGRAM_API_HASH)


def is_mtproto_configured() -> bool:
    """Returns True if Pyrogram is available and MTProto can be used."""
    try:
        import pyrogram  # noqa: F401
    except ImportError:
        return False
    try:
        t = _token()
        if not t:
            return False
    except Exception:
        return False
    return True


# ── UPLOAD ────────────────────────────────────────────────────────────────────

def upload_file_with_pyrogram(
    local_path: str,
    chat_id: Optional[str] = None,
    caption: str = "",
    file_name: Optional[str] = None,
    as_video: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    MTProto upload fallback using Pyrogram for files > 20MB / 50MB (up to 2GB).
    Bypasses Telegram Bot API HTTP 413 (50MB) upload limits with TgCrypto acceleration.
    Returns a dict with {"ok": True, "result": {"message_id": ..., "video": {...}, "document": {...}}}
    matching standard Telegram Bot API response structure.
    """
    chat_id = chat_id or _group_id()
    if not os.path.exists(local_path):
        logger.error("[telegram_http] upload_file_with_pyrogram: file not found: %s", local_path)
        return None

    filename = file_name or os.path.basename(local_path)
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    logger.info("[telegram_http] Initiating Pyrogram MTProto upload for %s (size: %.1f MB) -> chat %s...",
                filename, file_size_mb, chat_id)
    try:
        _ensure_event_loop()
        from pyrogram import Client

        token = _token()
        if not is_mtproto_configured():
            logger.warning("[telegram_http] Pyrogram MTProto is not available. Skipping Pyrogram MTProto upload.")
            return None
        api_id, api_hash = get_mtproto_credentials()

        async def _async_upload():
            session_name = f"vault_pyro_up_{uuid.uuid4().hex[:8]}"
            async with Client(
                session_name,
                api_id=int(api_id),
                api_hash=api_hash,
                bot_token=token,
                in_memory=True
            ) as app:
                target_chat = int(chat_id) if (isinstance(chat_id, int) or (isinstance(chat_id, str) and chat_id.lstrip("-").isdigit())) else chat_id
                
                ext = os.path.splitext(filename)[1].lower()
                is_video = as_video and ext in [".mp4", ".mkv", ".mov", ".webm", ".avi"]

                if is_video:
                    msg = await app.send_video(
                        chat_id=target_chat,
                        video=local_path,
                        caption=caption[:1024] if caption else None,
                        file_name=filename,
                        supports_streaming=True
                    )
                else:
                    msg = await app.send_document(
                        chat_id=target_chat,
                        document=local_path,
                        caption=caption[:1024] if caption else None,
                        file_name=filename
                    )

                if msg:
                    video_dict = {}
                    doc_dict = {}
                    if msg.video:
                        video_dict = {
                            "file_id": msg.video.file_id,
                            "file_unique_id": msg.video.file_unique_id,
                            "file_name": getattr(msg.video, "file_name", filename),
                            "file_size": msg.video.file_size,
                        }
                    if msg.document:
                        doc_dict = {
                            "file_id": msg.document.file_id,
                            "file_unique_id": msg.document.file_unique_id,
                            "file_name": getattr(msg.document, "file_name", filename),
                            "file_size": msg.document.file_size,
                        }

                    res_payload = {
                        "message_id": msg.id,
                        "video": video_dict,
                        "document": doc_dict
                    }
                    logger.info("[telegram_http] Pyrogram MTProto upload successful for %s -> msg_id: %s", filename, msg.id)
                    return {"ok": True, "result": res_payload}
                return None

        return _run_coro_safely(_async_upload)

    except Exception as e:
        logger.warning("[telegram_http] Pyrogram MTProto upload failed for %s: %s", filename, e)
        return None


def send_document(
    local_path: str,
    chat_id: Optional[str] = None,
    caption: str = "",
    max_retries: int = 3,
    custom_filename: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Uploads local_path as a document to the Telegram storage group.
    Returns the raw Telegram message dict on success, or None on failure.
    Automatically routes files >= 45MB to Pyrogram MTProto (up to 2GB).
    Retries up to max_retries on transient socket/connection errors.
    """
    import time
    chat_id = chat_id or _group_id()
    if not os.path.exists(local_path):
        logger.error("[telegram_http] send_document: file not found: %s", local_path)
        return None

    filename = custom_filename or os.path.basename(local_path)
    file_size = os.path.getsize(local_path)

    # 50MB HTTP limit proactive check: route >= 45MB to Pyrogram MTProto
    if file_size >= 45 * 1024 * 1024:
        logger.info("[telegram_http] File size %.1fMB >= 45MB. Delegating to Pyrogram MTProto upload directly...", file_size / (1024 * 1024))
        pyro_res = upload_file_with_pyrogram(local_path, chat_id=chat_id, caption=caption, file_name=filename, as_video=False)
        if pyro_res and pyro_res.get("ok"):
            return pyro_res.get("result")

    for attempt in range(1, max_retries + 1):
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body = bytearray()

        def _add_field(name: str, value: str) -> None:
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.extend(f"{value}\r\n".encode())

        _add_field("chat_id", chat_id)
        if caption:
            _add_field("caption", caption[:1024])

        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="document"; filename="{filename}"\r\n'.encode())
        body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
        with open(local_path, "rb") as f:
            body.extend(f.read())
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())

        req = urllib.request.Request(
            _api_url("sendDocument"),
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                     "User-Agent": "AMTCE-Vault/2.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode())
                if result.get("ok"):
                    return result.get("result")
                logger.warning("[telegram_http] sendDocument not OK: %s", result)
                return None
        except urllib.error.HTTPError as he:
            # Check for HTTP 429 Too Many Requests -> sleep retry_after
            if he.code == 429:
                retry_after = 5
                try:
                    err_data = json.loads(he.read().decode())
                    retry_after = err_data.get("parameters", {}).get("retry_after", 5)
                except Exception:
                    pass
                if attempt < max_retries:
                    logger.warning("[telegram_http] HTTP 429 Rate Limit for %s. Sleeping %.1fs before retry...", filename, float(retry_after) + 0.5)
                    time.sleep(float(retry_after) + 0.5)
                    continue

            # Check for HTTP 413 Request Entity Too Large -> trigger Pyrogram immediately
            if he.code == 413:
                logger.warning("[telegram_http] HTTP 413 encountered for %s. Falling back to Pyrogram MTProto upload...", filename)
                pyro_res = upload_file_with_pyrogram(local_path, chat_id=chat_id, caption=caption, file_name=filename, as_video=False)
                if pyro_res and pyro_res.get("ok"):
                    return pyro_res.get("result")

            if attempt < max_retries:
                logger.warning("[telegram_http] send_document attempt %d failed for %s (%s) — retrying...", attempt, filename, he)
                time.sleep(2.0 * attempt)
            else:
                logger.warning("[telegram_http] send_document failed after %d attempts for %s: %s. Trying Pyrogram MTProto fallback...", max_retries, filename, he)
                pyro_res = upload_file_with_pyrogram(local_path, chat_id=chat_id, caption=caption, file_name=filename, as_video=False)
                if pyro_res and pyro_res.get("ok"):
                    return pyro_res.get("result")
                return None
        except Exception as e:
            # Check for HTTP 413 Request Entity Too Large -> trigger Pyrogram immediately
            if "413" in str(e) or "Too Large" in str(e):
                logger.warning("[telegram_http] HTTP 413 encountered for %s. Falling back to Pyrogram MTProto upload...", filename)
                pyro_res = upload_file_with_pyrogram(local_path, chat_id=chat_id, caption=caption, file_name=filename, as_video=False)
                if pyro_res and pyro_res.get("ok"):
                    return pyro_res.get("result")

            if attempt < max_retries:
                logger.warning("[telegram_http] send_document attempt %d failed for %s (%s) — retrying...", attempt, filename, e)
                time.sleep(2.0 * attempt)
            else:
                logger.warning("[telegram_http] send_document failed after %d attempts for %s: %s. Trying Pyrogram MTProto fallback...", max_retries, filename, e)
                pyro_res = upload_file_with_pyrogram(local_path, chat_id=chat_id, caption=caption, file_name=filename, as_video=False)
                if pyro_res and pyro_res.get("ok"):
                    return pyro_res.get("result")
                return None


def extract_file_id(message: Dict[str, Any]) -> Optional[str]:
    """Extracts the file_id from any Telegram message dict regardless of media type."""
    doc = message.get("document") or {}
    if doc.get("file_id"):
        return doc["file_id"]
    for kind in ("video", "audio"):
        v = message.get(kind) or {}
        if isinstance(v, dict) and v.get("file_id"):
            return v["file_id"]
    photo = message.get("photo")
    if isinstance(photo, list) and photo:
        return photo[-1].get("file_id")
    return None


# ── DOWNLOAD ──────────────────────────────────────────────────────────────────

def _download_with_pyrogram(file_id: str, dest_path: str) -> bool:
    """
    MTProto download fallback using Pyrogram for files > 20MB.
    Bypasses Telegram Bot API getFile size restrictions up to 2GB.
    """
    logger.info("[telegram_http] Initiating Pyrogram MTProto download for large file_id=%s...", file_id[:12])
    if not is_mtproto_configured():
        logger.warning("[telegram_http] Pyrogram MTProto is not available. Skipping Pyrogram MTProto download.")
        return False

    try:
        _ensure_event_loop()
        from pyrogram import Client

        token = _token()
        api_id, api_hash = get_mtproto_credentials()

        async def _async_download():
            session_name = f"vault_pyro_dl_{uuid.uuid4().hex[:8]}"
            async with Client(
                session_name,
                api_id=int(api_id),
                api_hash=api_hash,
                bot_token=token,
                in_memory=True
            ) as app:
                os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
                tmp_path = dest_path + ".tmp"
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                downloaded_file = await app.download_media(message=file_id, file_name=tmp_path)
                actual_path = None
                try:
                    if downloaded_file and isinstance(downloaded_file, (str, os.PathLike)) and os.path.exists(str(downloaded_file)):
                        actual_path = str(downloaded_file)
                except Exception:
                    pass
                if not actual_path and os.path.exists(tmp_path):
                    actual_path = tmp_path
                if actual_path and os.path.exists(actual_path) and os.path.getsize(actual_path) > 0:
                    if os.path.abspath(actual_path) != os.path.abspath(dest_path):
                        if os.path.exists(dest_path):
                            try:
                                os.remove(dest_path)
                            except Exception:
                                pass
                        os.replace(actual_path, dest_path)
                    logger.info("[telegram_http] Pyrogram MTProto download successful -> %s", dest_path)
                    return True
                return False

        return _run_coro_safely(_async_download)

    except Exception as e:
        logger.warning("[telegram_http] Pyrogram MTProto download failed for file_id=%s: %s", file_id[:12], e)
        return False


_OVERSIZED_FILE_IDS = set()

def is_file_oversized(file_id: Optional[str]) -> bool:
    """Returns True if the file_id was flagged as exceeding Telegram's 20MB Bot API getFile limit."""
    if not file_id:
        return False
    return file_id in _OVERSIZED_FILE_IDS


def download_file_by_id(file_id: str, dest_path: str) -> bool:
    """
    Downloads a Telegram file by file_id to dest_path.

    Automatically falls back to Pyrogram MTProto download if:
      - File is known to be oversized (>20MB)
      - HTTP 400: Telegram's 20MB Bot API getFile limit is hit.
    """
    token = _token()
    headers = {"User-Agent": "AMTCE-Vault/2.0"}

    # Proactive MTProto routing for files already confirmed to exceed 20MB Bot API limit
    if is_file_oversized(file_id) and is_mtproto_configured():
        logger.info("[telegram_http] File '%s' is known to exceed 20MB Bot API limit. Delegating directly to Pyrogram MTProto...", file_id[:12])
        return _download_with_pyrogram(file_id, dest_path)

    # --- Attempt 1: requests (better SSL on Windows) ---
    try:
        import requests as req_lib
        r = req_lib.get(
            f"https://api.telegram.org/bot{token}/getFile",
            params={"file_id": file_id},
            headers=headers,
            timeout=15,
        )
        if r.status_code == 400:
            _OVERSIZED_FILE_IDS.add(file_id)
            if not is_mtproto_configured():
                logger.warning("[telegram_http] 20MB getFile limit hit for file_id=%s and MTProto is not configured. Download impossible via Bot API.", file_id[:12])
                return False
            logger.warning("[telegram_http] 20MB getFile limit hit for file_id=%s. Switching to Pyrogram MTProto fallback...", file_id[:12])
            return _download_with_pyrogram(file_id, dest_path)
        r.raise_for_status()
        body = r.json()
        if not body.get("ok"):
            if not is_mtproto_configured():
                return False
            return _download_with_pyrogram(file_id, dest_path)
        dl_url = f"https://api.telegram.org/file/bot{token}/{body['result']['file_path']}"
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        tmp = dest_path + ".tmp"
        with req_lib.get(dl_url, headers=headers, timeout=60, stream=True) as dl:
            dl.raise_for_status()
            with open(tmp, "wb") as out:
                for chunk in dl.iter_content(8192):
                    out.write(chunk)
        os.replace(tmp, dest_path)
        return True
    except ImportError:
        pass  # fall through to urllib
    except Exception as e:
        logger.debug("[telegram_http] requests download failed, trying urllib: %s", e)

    # --- Attempt 2: urllib fallback with 3 retries ---
    import time
    for attempt in range(1, 4):
        try:
            get_url = f"https://api.telegram.org/bot{token}/getFile?file_id={file_id}"
            req = urllib.request.Request(get_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode())
            if not body.get("ok"):
                if not is_mtproto_configured():
                    return False
                return _download_with_pyrogram(file_id, dest_path)
            dl_url = f"https://api.telegram.org/file/bot{token}/{body['result']['file_path']}"
            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
            tmp = dest_path + ".tmp"
            dl_req = urllib.request.Request(dl_url, headers=headers)
            with urllib.request.urlopen(dl_req, timeout=60) as dl, open(tmp, "wb") as out:
                out.write(dl.read())
            os.replace(tmp, dest_path)
            return True
        except urllib.error.HTTPError as he:
            if he.code == 400:
                _OVERSIZED_FILE_IDS.add(file_id)
                if not is_mtproto_configured():
                    logger.warning("[telegram_http] 20MB getFile limit (urllib) hit for file_id=%s and MTProto is not configured. Download impossible via Bot API.", file_id[:12])
                    return False
                logger.warning("[telegram_http] 20MB getFile limit (urllib) hit for file_id=%s. Switching to Pyrogram MTProto fallback...", file_id[:12])
                return _download_with_pyrogram(file_id, dest_path)
            if attempt == 3:
                logger.warning("[telegram_http] download failed after 3 attempts: %s", he)
        except Exception as e:
            if attempt == 3:
                logger.warning("[telegram_http] download failed: %s", e)
            time.sleep(1.0)

    if is_mtproto_configured():
        return _download_with_pyrogram(file_id, dest_path)
    return False


# ── PIN / CHAT ─────────────────────────────────────────────────────────────────

def pin_message(message_id: int, chat_id: Optional[str] = None) -> bool:
    """Pins a message in the storage group. Returns True on success."""
    chat_id = chat_id or _group_id()
    body = json.dumps({"chat_id": chat_id, "message_id": message_id, "disable_notification": True}).encode()
    req = urllib.request.Request(
        _api_url("pinChatMessage"),
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "AMTCE-Vault/2.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode()).get("ok", False)
    except Exception as e:
        logger.warning("[telegram_http] pinChatMessage failed: %s", e)
        return False


def get_pinned_message(chat_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Returns the currently-pinned message dict for the storage group, or None."""
    chat_id = chat_id or _group_id()
    url = f"{_api_url('getChat')}?chat_id={chat_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            body = json.loads(resp.read().decode())
            if body.get("ok"):
                return body["result"].get("pinned_message")
    except Exception as e:
        logger.warning("[telegram_http] get_pinned_message failed: %s", e)
    return None
