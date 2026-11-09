import os
import time
import unittest
import tempfile
import json
from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
from main import _is_source_video_url


class TestSessionManagerAndUrlRouting(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sessions_file = os.path.join(self.temp_dir, "test_sessions.json")

    def tearDown(self):
        try:
            if os.path.exists(self.sessions_file):
                os.remove(self.sessions_file)
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_01_strict_chat_id_isolation(self):
        """User A waiting for title should NEVER be returned to User B."""
        mgr = TelegramSessionManager(sessions_file=self.sessions_file)
        # Create session for User A (chat_id: 11111)
        sess_id = mgr.create_session(
            video_path="/dummy/path/master.mp4",
            clip_id="clip_A",
            requestor_chat_id=11111
        )
        mgr.set_awaiting_title(sess_id)

        # 1. User B (chat_id: 22222) queries -> MUST return None!
        pending_b = mgr.get_pending_title_session(chat_id=22222)
        self.assertIsNone(pending_b, "User B was incorrectly assigned User A's pending title session!")

        # 2. User A (chat_id: 11111) queries -> MUST return Session A
        pending_a = mgr.get_pending_title_session(chat_id=11111)
        self.assertIsNotNone(pending_a)
        self.assertEqual(pending_a["session_id"], sess_id)

    def test_02_orphaned_session_without_chat_id_not_hijacked(self):
        """A session with no requestor_chat_id must NEVER hijack real Telegram users."""
        mgr = TelegramSessionManager(sessions_file=self.sessions_file)
        sess_id = mgr.create_session(
            video_path="/dummy/path/master.mp4",
            clip_id="clip_orphan",
            requestor_chat_id=None
        )
        # Force status to AWAITING_TITLE
        mgr.sessions[sess_id]["status"] = "AWAITING_TITLE"
        mgr._save_sessions()

        # Query from any user chat_id -> Must return None!
        pending = mgr.get_pending_title_session(chat_id=987654)
        self.assertIsNone(pending)

    def test_03_ttl_timeout_auto_revert_to_review(self):
        """Stale AWAITING_TITLE session older than 30 mins must auto-revert to AWAITING_REVIEW."""
        mgr = TelegramSessionManager(sessions_file=self.sessions_file)
        sess_id = mgr.create_session(
            video_path="/dummy/path/master.mp4",
            clip_id="clip_old",
            requestor_chat_id=55555
        )
        mgr.set_awaiting_title(sess_id)
        # Manually backdate updated_at by 40 minutes (2400s)
        mgr.sessions[sess_id]["updated_at"] = time.time() - 2400
        mgr._save_sessions()

        # Query with 1800s TTL -> Must auto-revert and return None
        pending = mgr.get_pending_title_session(chat_id=55555, max_age_seconds=1800.0)
        self.assertIsNone(pending, "Expired session was not timed out!")

        # Verify status in store reverted to AWAITING_REVIEW
        sess = mgr.get_session(sess_id)
        self.assertEqual(sess["status"], "AWAITING_REVIEW")

    def test_04_source_video_url_classifier(self):
        """Verify _is_source_video_url distinguishes video URLs from affiliate product links."""
        # Source video URLs:
        self.assertTrue(_is_source_video_url("https://www.instagram.com/reel/Dfg9Y1GDQiM/"))
        self.assertTrue(_is_source_video_url("https://instagram.com/reel/DfULIxos0kb/?cplk=Z245MWFrMzllOXFo"))
        self.assertTrue(_is_source_video_url("https://www.instagram.com/p/C-xyz123/"))
        self.assertTrue(_is_source_video_url("https://youtube.com/shorts/3fG8jK1mNop"))
        self.assertTrue(_is_source_video_url("https://youtu.be/3fG8jK1mNop"))
        self.assertTrue(_is_source_video_url("https://www.youtube.com/watch?v=3fG8jK1mNop"))
        self.assertTrue(_is_source_video_url("https://www.tiktok.com/@creator/video/7123456789012345678"))
        self.assertTrue(_is_source_video_url("https://vt.tiktok.com/ZSxxxx/"))
        self.assertTrue(_is_source_video_url("https://facebook.com/reel/123456789"))
        self.assertTrue(_is_source_video_url("https://fb.watch/12345/"))

        # E-commerce affiliate product links (MUST BE FALSE):
        self.assertFalse(_is_source_video_url("https://amzn.to/3xyz"))
        self.assertFalse(_is_source_video_url("https://www.amazon.in/dp/B08N5WRWNW"))
        self.assertFalse(_is_source_video_url("https://www.meesho.com/s/p/123xyz"))
        self.assertFalse(_is_source_video_url("https://fkrt.it/enkr123"))
        self.assertFalse(_is_source_video_url("https://wishlink.com/item/123"))
        self.assertFalse(_is_source_video_url("https://ekaro.in/enkr456"))
        self.assertFalse(_is_source_video_url("My Viral Video Title"))


if __name__ == "__main__":
    unittest.main()

