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
import unittest
from unittest.mock import patch, MagicMock

# Ensure repo root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


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


if __name__ == "__main__":
    unittest.main()
