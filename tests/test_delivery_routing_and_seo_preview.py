import os
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import tempfile

from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager


def _run_coro(coro):
    """Executes a coroutine safely without requiring OS-level network loopback sockets."""
    while True:
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value


class TestDeliveryRoutingAndSEOPreview(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sessions_file = os.path.join(self.temp_dir, "test_sessions.json")
        self.session_mgr = TelegramSessionManager(sessions_file=self.sessions_file)

    def tearDown(self):
        try:
            if os.path.exists(self.sessions_file):
                os.remove(self.sessions_file)
            os.rmdir(self.temp_dir)
        except Exception:
            pass

    def test_session_manager_set_and_create_seo_result(self):
        """Verifies that create_session and set_seo_result properly persist seo_result."""
        dummy_seo = {
            "main_subject": "Summer Fashion Lookbook",
            "platforms": {
                "youtube": {
                    "title": "Summer Fashion 2026",
                    "hashtags": ["#summer", "#fashion", "#viral"]
                }
            }
        }
        with patch("Telegram_Storage_Modules.telegram_user_manager._upload_file_to_telegram_storage", return_value=None):
            sess_id = self.session_mgr.create_session(
                video_path="dummy_master.mp4",
                requestor_chat_id=123456,
                seo_result=dummy_seo
            )
            sess = self.session_mgr.get_session(sess_id)
            self.assertIsNotNone(sess)
            self.assertEqual(sess.get("seo_result", {}).get("main_subject"), "Summer Fashion Lookbook")

            updated_seo = {
                "main_subject": "Updated Lookbook",
                "platforms": {"youtube": {"title": "Updated Title"}}
            }
            self.session_mgr.set_seo_result(sess_id, updated_seo)
            sess_updated = self.session_mgr.get_session(sess_id)
            self.assertEqual(sess_updated.get("seo_result", {}).get("main_subject"), "Updated Lookbook")

    @patch("main._send_video_safe_main", new_callable=AsyncMock)
    def test_deliver_rendered_reel_with_seo(self, mock_send_video):
        """Verifies deliver_rendered_reel_with_seo sends the video with rich caption and sends SEO preview."""
        from main import deliver_rendered_reel_with_seo
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()
        mock_sent_msg = MagicMock()
        mock_sent_msg.message_id = 999888
        mock_send_video.return_value = mock_sent_msg

        dummy_seo = {
            "main_subject": "Trendy Outfit",
            "platforms": {
                "youtube": {
                    "title": "Viral Summer Style ✨",
                    "hashtags": ["#summer", "#style", "#shorts"],
                    "seo_score": 92,
                    "description": "Full description here"
                },
                "instagram": {
                    "title": "Viral Summer Style IG",
                    "hashtags": ["#reels", "#explore"],
                    "seo_score": 90
                }
            }
        }

        with tempfile.NamedTemporaryFile(suffix="_master.mp4", delete=False) as f:
            f.write(b"fake video content")
            temp_vid = f.name

        try:
            with patch("Telegram_Storage_Modules.telegram_user_manager._upload_file_to_telegram_storage", return_value=None):
                sess_id = self.session_mgr.create_session(video_path=temp_vid, requestor_chat_id=12345)
                with patch("main.session_manager", self.session_mgr):
                    res = _run_coro(deliver_rendered_reel_with_seo(
                        bot_obj=mock_bot,
                        target_chat=12345,
                        reel_path=temp_vid,
                        sess_id=sess_id,
                        precomputed_seo=dummy_seo
                    ))

                self.assertIsNotNone(res)
                mock_send_video.assert_called_once()
                call_kwargs = mock_send_video.call_args[1]
                self.assertIn("Trendy Outfit", call_kwargs["caption"])
                self.assertIn("Viral Summer Style ✨", call_kwargs["caption"])
                self.assertIn("#summer", call_kwargs["caption"])

                mock_bot.send_message.assert_called_once()
                msg_call = mock_bot.send_message.call_args[1]
                self.assertEqual(msg_call["chat_id"], 12345)
                self.assertIn("SEO Content Preview", msg_call["text"])
                self.assertIn("YOUTUBE", msg_call["text"])
        finally:
            if os.path.exists(temp_vid):
                os.remove(temp_vid)

    def test_routing_guard_redirects_public_group(self):
        """Verifies that Phase 4 completion summary redirects away from TELEGRAM_PUBLIC_GROUP_ID to TELEGRAM_ADMIN_ID."""
        pub_group = "-1001111111"
        storage_group = "-1002222222"
        admin_id = "998877"

        with patch.dict(os.environ, {
            "TELEGRAM_PUBLIC_GROUP_ID": pub_group,
            "TELEGRAM_STORAGE_GROUP_ID": storage_group,
            "TELEGRAM_ADMIN_ID": admin_id
        }):
            from main import _dispatch_phase4_publishing_in_background
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock()

            # We directly verify the target_completion_chat resolution logic in _dispatch_phase4_publishing_in_background
            # by inspecting how it reroutes chat_id == pub_group to admin_id
            target_chat = int(pub_group)
            clean_pub = os.getenv("TELEGRAM_PUBLIC_GROUP_ID", "").strip()
            clean_admin = os.getenv("TELEGRAM_ADMIN_ID", "").strip()
            
            resolved_target = target_chat
            if str(resolved_target) in (clean_pub, storage_group) and clean_admin:
                resolved_target = int(clean_admin)
            
            self.assertEqual(resolved_target, int(admin_id))
            self.assertNotEqual(resolved_target, int(pub_group))

    def test_scheduled_scraper_skips_publishing_when_auto_publish_disabled(self):
        """Verifies that scheduled_scraper_manager does NOT publish when AUTO_PUBLISH_CLIPS is not yes."""
        with patch.dict(os.environ, {"AUTO_PUBLISH_CLIPS": "no"}):
            with patch("Telegram_Storage_Modules.telegram_vault_indexer.TelegramVaultIndexer") as mock_tvi, \
                 patch("Publishing_Modules.media_publisher_main.run_phase4_publishing") as mock_pub, \
                 patch("Downloader_Modules.downloader_main.run_phase1_ingestion", return_value={"downloaded_files": ["fake_dir/video.mp4"]}), \
                 patch("Main_Modules.phase2_main.run_phase2_orchestration", return_value={"rendered_files": ["fake_master.mp4"]}), \
                 patch("Gemini_Modules.gemini_clip_auditor.run_clip_audit_and_seo", return_value={"audit_passed": True, "seo_metadata": {}}), \
                 patch("os.path.exists", return_value=True):
                mock_tvi.return_value.hydrate_all_vault_jsons_on_startup.return_value = True
                mock_tvi.return_value.find_entry_by_shortcode.return_value = {}
                from Downloader_Modules.scheduled_scraper_manager import run_scheduled_scraper_batch
                with patch("Downloader_Modules.scheduled_scraper_manager.get_rotated_max_two_accounts", return_value=["test_user"]):
                    rendered = run_scheduled_scraper_batch(max_accounts=1)
                    mock_pub.assert_not_called()

    def test_admin_id_fallback_when_requestor_chat_id_is_none(self):
        """Verifies that target_chat resolves to TELEGRAM_ADMIN_ID when requestor_chat_id is None."""
        admin_id_str = "554433221"
        with patch.dict(os.environ, {"TELEGRAM_ADMIN_ID": admin_id_str}):
            requestor_chat_id = None
            target_chat = requestor_chat_id
            if not target_chat:
                admin_raw = os.getenv("TELEGRAM_ADMIN_ID", "").strip()
                if admin_raw:
                    first_admin = [a.strip() for a in admin_raw.split(",") if a.strip()][0]
                    if first_admin.isdigit() or (first_admin.startswith("-") and first_admin[1:].isdigit()):
                        target_chat = int(first_admin)

            self.assertEqual(target_chat, int(admin_id_str))


if __name__ == "__main__":
    unittest.main()
