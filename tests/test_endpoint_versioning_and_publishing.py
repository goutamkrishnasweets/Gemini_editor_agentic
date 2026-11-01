"""
tests/test_endpoint_versioning_and_publishing.py
================================================
Unit tests verifying:
  1. Dynamic Meta Graph API endpoint versioning & URL construction.
  2. Dynamic YouTube Data API versioning.
  3. Telegram User Manager version update & persistence logic.
  4. Admin-only visibility of endpoint controls in /myconfig summary.
  5. Multi-platform media publishing module integration.
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

# Ensure repo root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _run_coro(coro):
    """Executes a coroutine safely without requiring OS-level network loopback sockets."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    raise RuntimeError("Coroutine did not complete synchronously")


class TestEndpointVersioning(unittest.TestCase):

    def setUp(self):
        # Save env state
        self._orig_meta_ver = os.environ.get("META_GRAPH_VERSION")
        self._orig_yt_ver = os.environ.get("YOUTUBE_API_VERSION")

    def tearDown(self):
        # Restore env state
        if self._orig_meta_ver is not None:
            os.environ["META_GRAPH_VERSION"] = self._orig_meta_ver
        else:
            os.environ.pop("META_GRAPH_VERSION", None)

        if self._orig_yt_ver is not None:
            os.environ["YOUTUBE_API_VERSION"] = self._orig_yt_ver
        else:
            os.environ.pop("YOUTUBE_API_VERSION", None)

    def test_meta_graph_versioning_dynamic(self):
        from Publishing_Modules.meta_uploader import (
            get_meta_graph_version,
            set_meta_graph_version,
            get_graph_api_url,
            get_rupload_video_url,
            GRAPH_API_URL
        )

        # Set to v21.0
        set_meta_graph_version("v21.0")
        self.assertEqual(get_meta_graph_version(), "v21.0")
        self.assertEqual(get_graph_api_url(), "https://graph.facebook.com/v21.0")
        self.assertEqual(get_rupload_video_url("999"), "https://rupload.facebook.com/video-upload/v21.0/999")
        self.assertIn("v21.0", f"{GRAPH_API_URL}/me")

        # Update to v22.0 dynamically
        set_meta_graph_version("22.0")  # Test auto-prefixing 'v'
        self.assertEqual(get_meta_graph_version(), "v22.0")
        self.assertEqual(get_graph_api_url(), "https://graph.facebook.com/v22.0")
        self.assertEqual(get_rupload_video_url("999"), "https://rupload.facebook.com/video-upload/v22.0/999")
        self.assertIn("v22.0", f"{GRAPH_API_URL}/me")

    def test_youtube_api_versioning_dynamic(self):
        from Publishing_Modules.uploader import (
            get_youtube_api_version,
            set_youtube_api_version
        )

        set_youtube_api_version("v3")
        self.assertEqual(get_youtube_api_version(), "v3")

        # Change to v4
        set_youtube_api_version("4")
        self.assertEqual(get_youtube_api_version(), "v4")

    def test_telegram_user_manager_endpoint_updates(self):
        from Telegram_Storage_Modules.telegram_user_manager import (
            get_meta_graph_api_version,
            update_meta_graph_api_version,
            get_youtube_api_version,
            update_youtube_api_version,
            query_gemini_recommended_endpoint_version,
            format_user_credentials_summary
        )

        # Meta update
        ok, msg = update_meta_graph_api_version("v21.0")
        self.assertTrue(ok)
        self.assertEqual(get_meta_graph_api_version(), "v21.0")

        # YouTube update
        ok, msg = update_youtube_api_version("v3")
        self.assertTrue(ok)
        self.assertEqual(get_youtube_api_version(), "v3")

        # Gemini recommendation query returns valid schema even on fallback
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}):
            meta_rec = query_gemini_recommended_endpoint_version("meta")
            self.assertIn("version", meta_rec)
            self.assertTrue(meta_rec["version"].startswith("v"))

            yt_rec = query_gemini_recommended_endpoint_version("youtube")
            self.assertIn("version", yt_rec)
            self.assertTrue(yt_rec["version"].startswith("v"))

        # Admin summary contains endpoint section
        with patch("Telegram_Storage_Modules.telegram_user_manager._is_admin_user", return_value=True):
            summary = format_user_credentials_summary("admin_test_123")
            self.assertIn("/updatemetaendpoints", summary)
            self.assertIn("/updateyoutubeendpoints", summary)
            self.assertIn("Meta Graph API Version", summary)
            self.assertIn("/setmetaappid", summary)
            self.assertIn("/setmetaappsecret", summary)

        # Meta app setter functions
        from Telegram_Storage_Modules.telegram_user_manager import set_user_meta_app, set_user_meta_app_id, set_user_meta_app_secret
        with patch("Telegram_Storage_Modules.telegram_user_manager.sync_user_secret_to_github"):
            self.assertTrue(set_user_meta_app("test_user_789", "app_123", "secret_456"))
            self.assertEqual(os.environ.get("META_APP_ID"), "app_123")
            self.assertEqual(os.environ.get("META_APP_SECRET"), "secret_456")

            self.assertTrue(set_user_meta_app_id("test_user_789", "app_999"))
            self.assertEqual(os.environ.get("META_APP_ID"), "app_999")

            self.assertTrue(set_user_meta_app_secret("test_user_789", "secret_888"))
            self.assertEqual(os.environ.get("META_APP_SECRET"), "secret_888")

        # Non-admin summary does NOT contain admin endpoint commands
        with patch("Telegram_Storage_Modules.telegram_user_manager._is_admin_user", return_value=False):
            summary_non_admin = format_user_credentials_summary("normal_user_456")
            self.assertNotIn("/updatemetaendpoints", summary_non_admin)
            self.assertNotIn("/updateyoutubeendpoints", summary_non_admin)

    def test_media_publisher_main_exports_and_structure(self):
        import Publishing_Modules
        self.assertTrue(hasattr(Publishing_Modules, "run_phase4_publishing"))
        self.assertTrue(hasattr(Publishing_Modules, "run_phase4_publishing_async"))
        self.assertTrue(hasattr(Publishing_Modules, "publish_to_youtube"))
        self.assertTrue(hasattr(Publishing_Modules, "publish_to_meta"))
        self.assertTrue(hasattr(Publishing_Modules, "get_meta_graph_version"))
        self.assertTrue(hasattr(Publishing_Modules, "get_youtube_api_version"))

    def test_meta_oauth_url_builder(self):
        from Publishing_Modules.meta_uploader import build_meta_oauth_url
        url = build_meta_oauth_url(app_id="123456789", redirect_uri="http://localhost:8080/callback")
        self.assertIn("client_id=123456789", url)
        self.assertIn("redirect_uri=http%3A%2F%2Flocalhost%3A8080%2Fcallback", url)
        self.assertIn("scope=instagram_basic", url)
        self.assertIn("instagram_content_publish", url)
        self.assertIn("pages_manage_posts", url)

    def test_waterfall_primary_success(self):
        import Publishing_Modules.media_publisher_main as mpm
        with patch.object(mpm, "is_make_configured", return_value=True), \
             patch.object(mpm, "publish_to_make", return_value={"status": "success", "media_url": "https://example.com/v.mp4"}), \
             patch.object(mpm, "publish_to_youtube", new_callable=AsyncMock) as mock_yt, \
             patch.object(mpm, "publish_to_meta", new_callable=AsyncMock) as mock_meta, \
             patch.object(mpm, "publish_to_telegram", new_callable=AsyncMock) as mock_tg:
            
            mock_tg.return_value = {"status": "success"}
            res = _run_coro(mpm.run_phase4_publishing_async(video_path=__file__, title="Test Title"))
            self.assertTrue(res["success"])
            self.assertEqual(res["platforms"]["make"]["status"], "success")
            mock_yt.assert_not_called()
            mock_meta.assert_not_called()

    def test_waterfall_failover_to_local(self):
        import Publishing_Modules.media_publisher_main as mpm
        with patch.object(mpm, "is_make_configured", return_value=True), \
             patch.object(mpm, "publish_to_make", return_value={"status": "failed", "error": "HTTP 410 Gone"}), \
             patch.object(mpm, "publish_to_youtube", new_callable=AsyncMock) as mock_yt, \
             patch.object(mpm, "publish_to_meta", new_callable=AsyncMock) as mock_meta, \
             patch.object(mpm, "publish_to_telegram", new_callable=AsyncMock) as mock_tg:
            
            mock_yt.return_value = {"status": "success", "video_id": "yt_123"}
            mock_meta.return_value = {"instagram": {"status": "success", "id": "ig_456"}}
            mock_tg.return_value = {"status": "success"}

            res = _run_coro(mpm.run_phase4_publishing_async(video_path=__file__, title="Test Title"))
            self.assertTrue(res["success"])
            self.assertEqual(res["platforms"]["make"]["status"], "fallback_triggered")
            mock_yt.assert_called_once()
            mock_meta.assert_called_once()


if __name__ == "__main__":
    unittest.main()

