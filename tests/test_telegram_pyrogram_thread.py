import os
import sys
import threading
from unittest.mock import patch, AsyncMock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import Telegram_Storage_Modules.telegram_http as th


def test_ensure_event_loop_in_worker_thread():
    result = {}

    def worker():
        loop = th._ensure_event_loop()
        result["loop_created"] = loop is not None
        result["is_closed"] = loop.is_closed()

    t = threading.Thread(target=worker, name="Thread-Test-Worker")
    t.start()
    t.join()

    assert result.get("loop_created") is True
    assert result.get("is_closed") is False


def test_run_coro_safely_in_worker_thread():
    result = {}

    def worker():
        async def my_coro():
            return 42

        val = th._run_coro_safely(my_coro())
        result["val"] = val

    t = threading.Thread(target=worker, name="Thread-Test-Worker")
    t.start()
    t.join()

    assert result.get("val") == 42


def test_download_with_pyrogram_in_worker_thread():
    result = {}

    def worker():
        with patch("Telegram_Storage_Modules.telegram_http._token", return_value="dummy_token"):
            mock_app = AsyncMock()
            mock_app.download_media.return_value = "dummy.mp4"
            with patch("pyrogram.Client.__aenter__", new_callable=AsyncMock, return_value=mock_app):
                with patch("pyrogram.Client.__aexit__", new_callable=AsyncMock, return_value=None):
                    with patch("os.path.exists", return_value=True):
                        with patch("os.path.getsize", return_value=1024):
                            with patch("os.replace", return_value=None):
                                res = th._download_with_pyrogram("BAACAgUAAyEGdummy", "test_dest.mp4")
                                result["success"] = res

    t = threading.Thread(target=worker, name="Thread-17 (_worker)")
    t.start()
    t.join()

    assert result.get("success") is True


def test_hydrate_clean_video_from_vault_in_worker_thread(tmp_path):
    from unittest.mock import MagicMock
    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer

    result = {}

    def worker():
        tvi = TelegramVaultIndexer()
        fake_entry = {
            "shortcode": "Dc8dzXbIIq7",
            "wm_clean_file_id": "BAACAgUAAyEG_clean",
            "media_file_ids": {"wm_clean_file_id": "BAACAgUAAyEG_clean"}
        }
        dest_dir = str(tmp_path / "downloads" / "manual_Dc8dzXbIIq7")

        with patch.object(tvi, "find_entry_by_shortcode", return_value=fake_entry):
            with patch("Telegram_Storage_Modules.telegram_http._token", return_value="dummy_token"):
                with patch("requests.get") as mock_req_get:
                    mock_req_get.return_value.status_code = 400
                    
                    mock_app = MagicMock()
                    async def fake_download(message, file_name):
                        with open(file_name, "wb") as f:
                            f.write(b"CLEAN_VIDEO_DATA" * 100)
                        return file_name
                    mock_app.download_media = AsyncMock(side_effect=fake_download)

                    with patch("pyrogram.Client.__aenter__", new_callable=AsyncMock, return_value=mock_app):
                        with patch("pyrogram.Client.__aexit__", new_callable=AsyncMock, return_value=None):
                            res = tvi.hydrate_clean_video_from_vault("Dc8dzXbIIq7", dest_dir=dest_dir)
                            result["res"] = res

    t = threading.Thread(target=worker, name="Thread-17 (_worker)")
    t.start()
    t.join()

    assert result.get("res") is not None
    assert os.path.exists(result["res"])
    assert "manual_clean_Dc8dzXbIIq7.mp4" in result["res"]


def test_hydrate_clean_video_auto_mode(tmp_path):
    from unittest.mock import MagicMock
    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer

    result = {}

    def worker():
        tvi = TelegramVaultIndexer()
        fake_entry = {
            "shortcode": "auto_test123",
            "wm_clean_file_id": "BAACAgUAAyEG_clean_auto",
            "media_file_ids": {"wm_clean_file_id": "BAACAgUAAyEG_clean_auto"}
        }
        dest_dir = str(tmp_path / "downloads" / "auto_auto_test123")

        with patch.object(tvi, "find_entry_by_shortcode", return_value=fake_entry):
            with patch("Telegram_Storage_Modules.telegram_http._token", return_value="dummy_token"):
                with patch("requests.get") as mock_req_get:
                    mock_req_get.return_value.status_code = 400
                    
                    mock_app = MagicMock()
                    async def fake_download(message, file_name):
                        with open(file_name, "wb") as f:
                            f.write(b"CLEAN_AUTO_VIDEO_DATA" * 100)
                        return file_name
                    mock_app.download_media = AsyncMock(side_effect=fake_download)

                    with patch("pyrogram.Client.__aenter__", new_callable=AsyncMock, return_value=mock_app):
                        with patch("pyrogram.Client.__aexit__", new_callable=AsyncMock, return_value=None):
                            res = tvi.hydrate_clean_video_from_vault("auto_test123", dest_dir=dest_dir)
                            result["res"] = res

    t = threading.Thread(target=worker, name="Thread-17 (_worker)")
    t.start()
    t.join()

    assert result.get("res") is not None
    assert os.path.exists(result["res"])
    assert "auto_clean_test123.mp4" in result["res"]


def test_hydrate_raw_video_retrieves_clean_first(tmp_path):
    from unittest.mock import MagicMock
    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer

    result = {}

    def worker():
        tvi = TelegramVaultIndexer()
        fake_entry = {
            "shortcode": "Dc8dzXbIIq7",
            "wm_clean_file_id": "BAACAgUAAyEG_clean",
            "raw_video_file_id": "BAACAgUAAyEG_raw",
            "media_file_ids": {
                "wm_clean_file_id": "BAACAgUAAyEG_clean",
                "raw_video_file_id": "BAACAgUAAyEG_raw"
            }
        }
        dest_dir = str(tmp_path / "downloads" / "manual_Dc8dzXbIIq7")

        with patch.object(tvi, "find_entry_by_shortcode", return_value=fake_entry):
            with patch("Telegram_Storage_Modules.telegram_http._token", return_value="dummy_token"):
                with patch("requests.get") as mock_req_get:
                    mock_req_get.return_value.status_code = 400
                    
                    mock_app = MagicMock()
                    async def fake_download(message, file_name):
                        with open(file_name, "wb") as f:
                            f.write(b"CLEAN_VIDEO_DATA" * 100)
                        return file_name
                    mock_app.download_media = AsyncMock(side_effect=fake_download)

                    with patch("pyrogram.Client.__aenter__", new_callable=AsyncMock, return_value=mock_app):
                        with patch("pyrogram.Client.__aexit__", new_callable=AsyncMock, return_value=None):
                            res = tvi.hydrate_raw_video_from_vault("Dc8dzXbIIq7", dest_dir=dest_dir, check_clean_first=True)
                            result["res"] = res

    t = threading.Thread(target=worker, name="Thread-17 (_worker)")
    t.start()
    t.join()

    assert result.get("res") is not None
    assert "manual_clean_Dc8dzXbIIq7.mp4" in result["res"]


def test_build_telegram_session_keyboard_has_clean_button():
    from main import build_telegram_session_keyboard
    kb = build_telegram_session_keyboard(session_id="sess_12345", shortcode="Dc8dzXbIIq7")
    assert kb is not None
    # Verify Row 4 button exists with callback clean_video_Dc8dzXbIIq7
    all_buttons = [btn for row in kb.inline_keyboard for btn in row]
    clean_btn = next((b for b in all_buttons if "clean_video_" in b.callback_data), None)
    assert clean_btn is not None
    assert clean_btn.callback_data == "clean_video_Dc8dzXbIIq7"
    assert "Raw Watermark Cleaned" in clean_btn.text


def test_send_clean_video_to_user_chat_via_file_id():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer

    tvi = TelegramVaultIndexer()
    fake_entry = {
        "shortcode": "Dc8dzXbIIq7",
        "wm_clean_file_id": "BAACAgUAAyEG_valid_clean_fid",
        "media_file_ids": {"wm_clean_file_id": "BAACAgUAAyEG_valid_clean_fid"}
    }

    mock_bot = MagicMock()
    mock_bot.send_video = AsyncMock(return_value=True)

    with patch.object(tvi, "find_entry_by_shortcode", return_value=fake_entry):
        sent = asyncio.run(tvi.send_clean_video_to_user_chat(
            bot=mock_bot,
            chat_id=12345678,
            identifier="manual_Dc8dzXbIIq7"
        ))
        assert sent is True
        mock_bot.send_video.assert_called_once()
        call_kwargs = mock_bot.send_video.call_args[1]
        assert call_kwargs["video"] == "BAACAgUAAyEG_valid_clean_fid"
        assert "Raw Watermark Cleaned Video" in call_kwargs["caption"]


def test_verify_master_render_rejects_failed_synthesis(tmp_path):
    import importlib
    step07 = importlib.import_module("Phase_2.07_master_render")
    verify_master_render = step07.verify_master_render

    fake_master = tmp_path / "manual_test_master.mp4"
    with open(fake_master, "wb") as f:
        f.write(b"EXISTING_STALE_RENDER_DATA" * 5000)

    synthesis_failed = {"status": "FAILED", "error": "FFmpeg exit code -22: Output same as Input"}
    res = verify_master_render(str(fake_master), synthesis_result=synthesis_failed)
    assert res["success"] is False
    assert "Synthesis failed" in res["error"]


def test_hydrate_clean_video_rejects_master_reel_in_session(tmp_path):
    from unittest.mock import MagicMock
    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer

    tvi = TelegramVaultIndexer()
    stale_master = tmp_path / "downloads" / "Processed Shorts" / "manual_Dc8dzXbIIq7_master.mp4"
    stale_master.parent.mkdir(parents=True, exist_ok=True)
    with open(stale_master, "wb") as f:
        f.write(b"MASTER_REEL_DATA" * 100)

    fake_session = {
        "video_path": str(stale_master),
        "clean_video_path": str(stale_master),
    }

    mock_sm = MagicMock()
    mock_sm.get_session.return_value = fake_session

    with patch("Telegram_Storage_Modules.telegram_session_manager.TelegramSessionManager", return_value=mock_sm):
        with patch.object(tvi, "find_entry_by_shortcode", return_value=None):
            with patch.object(tvi, "lookup_downloaded_source", return_value=None):
                clean_path = tvi.hydrate_clean_video_from_vault("manual_Dc8dzXbIIq7")
                # Must NOT return the master reel!
                assert clean_path is None or "_master.mp4" not in clean_path


def test_hydrate_clean_video_never_returns_raw_file(tmp_path):
    from unittest.mock import MagicMock
    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer

    tvi = TelegramVaultIndexer()
    test_dir = tmp_path / "downloads" / "manual_Dc8dzXbIIq7"
    test_dir.mkdir(parents=True, exist_ok=True)
    raw_file = test_dir / "manual_Dc8dzXbIIq7.mp4"
    raw_video = test_dir / "video.mp4"
    with open(raw_file, "wb") as f:
        f.write(b"RAW_DATA" * 100)
    with open(raw_video, "wb") as f:
        f.write(b"RAW_VIDEO_DATA" * 100)

    # When no clean video exists and vault returns no file_id, it must NOT return the raw files!
    with patch.object(tvi, "find_entry_by_shortcode", return_value=None):
        with patch.object(tvi, "lookup_downloaded_source", return_value=None):
            clean_res = tvi.hydrate_clean_video_from_vault("manual_Dc8dzXbIIq7", dest_dir=str(test_dir))
            assert clean_res is None or "_clean" in clean_res or "inpainted" in clean_res
            if clean_res:
                assert "manual_Dc8dzXbIIq7.mp4" != os.path.basename(clean_res)
                assert "video.mp4" != os.path.basename(clean_res)


def test_send_clean_video_never_sends_raw_file(tmp_path):
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer

    tvi = TelegramVaultIndexer()
    test_dir = tmp_path / "downloads" / "manual_Dc8dzXbIIq7"
    test_dir.mkdir(parents=True, exist_ok=True)
    raw_file = test_dir / "manual_Dc8dzXbIIq7.mp4"
    with open(raw_file, "wb") as f:
        f.write(b"RAW_VIDEO_ONLY" * 100)

    mock_bot = MagicMock()
    mock_bot.send_video = AsyncMock(return_value=True)

    with patch.object(tvi, "find_entry_by_shortcode", return_value=None):
        with patch.object(tvi, "lookup_downloaded_source", return_value=None):
            # Point _REPO_ROOT to tmp_path
            with patch("Telegram_Storage_Modules.telegram_vault_indexer._REPO_ROOT", str(tmp_path)):
                with patch("Watermark_and_Inpainting.watermark_main.run_watermark_removal") as mock_wm:
                    clean_target = str(test_dir / "manual_clean_Dc8dzXbIIq7.mp4")
                    with open(clean_target, "wb") as cf:
                        cf.write(b"CLEAN_RESULT_DATA" * 100)
                    mock_wm.return_value = (clean_target, "cleaned")

                    sent = asyncio.run(tvi.send_clean_video_to_user_chat(
                        bot=mock_bot,
                        chat_id=12345678,
                        identifier="manual_Dc8dzXbIIq7"
                    ))
                    assert sent is True
                    mock_bot.send_video.assert_called_once()
                    call_kwargs = mock_bot.send_video.call_args[1]
                    assert "manual_clean_Dc8dzXbIIq7.mp4" in call_kwargs["caption"]
                    assert "manual_Dc8dzXbIIq7.mp4`" not in call_kwargs["caption"]

