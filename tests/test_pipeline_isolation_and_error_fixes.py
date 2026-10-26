import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Modules.disk_cleaner import (
    mark_clip_active,
    mark_clip_inactive,
    is_clip_active,
    clean_clip_artifacts,
)
from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager


def test_active_clip_protection():
    cid = "test_clip_12345"
    assert not is_clip_active(cid)
    
    mark_clip_active(cid)
    assert is_clip_active(cid)
    assert is_clip_active(f"manual_{cid}")
    
    # Verify clean_clip_artifacts skips active clips
    res = clean_clip_artifacts(cid)
    assert res["status"] == "skipped"
    assert res["reason"] == "active_processing"
    
    mark_clip_inactive(cid)
    assert not is_clip_active(cid)


def test_session_attempt_history_isolation():
    manager = TelegramSessionManager()
    manager.sessions = {}
    
    # Create session for clip A
    sess_a = manager.create_session(
        video_path="Processed Shorts/clip_A_master.mp4",
        clip_id="manual_clip_A",
        requestor_chat_id=12345,
    )
    
    # Record attempt for clip B belonging to the same user chat
    manager.record_rendered_attempt(
        chat_id_or_sess_id=12345,
        video_path="Processed Shorts/clip_B_master.mp4",
        clip_id="manual_clip_B"
    )
    
    # Verify clip A's session was NOT polluted with clip B's master reel
    sess_a_data = manager.get_session(sess_a)
    history = sess_a_data.get("attempt_history", [])
    assert "Processed Shorts/clip_B_master.mp4" not in history


def test_send_video_safe_missing_file_raises_filenotfound():
    import asyncio
    from main import _send_video_safe_main
    
    class DummyBot:
        async def send_video(self, *args, **kwargs):
            return None

    # Should immediately raise FileNotFoundError instead of trying HTTP URL parsing
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            _send_video_safe_main(
                DummyBot(),
                chat_id=123,
                video_path_or_file="/nonexistent/path/manual_De4MlrTooCR_master.mp4",
                max_retries=1
            )
        )


def test_pin_message_handles_429_retry_backoff():
    import json
    import io
    import urllib.error
    from unittest.mock import patch, MagicMock
    from Telegram_Storage_Modules.telegram_http import pin_message

    err_body = json.dumps({"ok": False, "error_code": 429, "parameters": {"retry_after": 0.05}}).encode("utf-8")
    
    mock_429 = urllib.error.HTTPError(
        url="https://api.telegram.org/botTOKEN/pinChatMessage",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=io.BytesIO(err_body)
    )

    mock_ok_resp = MagicMock()
    mock_ok_resp.read.return_value = json.dumps({"ok": True, "result": True}).encode("utf-8")
    mock_ok_resp.__enter__.return_value = mock_ok_resp

    with patch("urllib.request.urlopen", side_effect=[mock_429, mock_ok_resp]) as mock_urlopen, \
         patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "123456:FAKE_TOKEN"}):
        res = pin_message(message_id=99999, chat_id="-100123456789", max_retries=3)
        assert res is True
        assert mock_urlopen.call_count == 2


def test_vault_indexer_pin_debounce():
    import time
    from unittest.mock import patch, MagicMock
    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
    import Telegram_Storage_Modules.telegram_vault_indexer as tvi

    tvi._LAST_VAULT_PIN_TIMESTAMP = time.time()
    indexer = TelegramVaultIndexer()
    
    with patch("Telegram_Storage_Modules.telegram_http.send_document") as mock_send, \
         patch("Telegram_Storage_Modules.telegram_http.pin_message") as mock_pin, \
         patch.dict("os.environ", {"TELEGRAM_STORAGE_GROUP_ID": "-100123"}):
        # Within cooldown window, should skip without calling send_document or pin_message
        indexer.upload_and_pin_vault_index_sync(force=False)
        assert mock_send.call_count == 0
        assert mock_pin.call_count == 0


def test_telegram_session_manager_shared_debounce():
    import time
    from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager

    now = time.time()
    TelegramSessionManager._last_vault_sync_time = now

    sm1 = TelegramSessionManager()
    sm2 = TelegramSessionManager()

    # Both instances must share the same class-level debounce timestamp
    assert sm1._last_vault_sync_time == now
    assert sm2._last_vault_sync_time == now
    assert (time.time() - TelegramSessionManager._last_vault_sync_time) < sm1._VAULT_SYNC_DEBOUNCE_SEC

