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

