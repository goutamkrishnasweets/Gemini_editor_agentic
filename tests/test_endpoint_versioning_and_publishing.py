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
        # Save complete env state
        self._orig_env = dict(os.environ)
        # Patch telegram vault sync so unit tests never hit network
        self._vault_patcher = patch("Telegram_Storage_Modules.telegram_user_manager.sync_users_json_to_telegram_vault")
        self._vault_patcher.start()

    def tearDown(self):
        # Stop vault patcher
        self._vault_patcher.stop()
        # Restore complete env state
        os.environ.clear()
        os.environ.update(self._orig_env)

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
        from Telegram_Storage_Modules.telegram_user_manager import (
            set_user_meta_app,
            set_user_meta_app_id,
            set_user_meta_app_secret,
            get_user_meta_app_id,
            get_user_meta_app_secret,
        )
        with patch("Telegram_Storage_Modules.telegram_user_manager.sync_user_secret_to_github"):
            # When admin sets it, root os.environ is updated
            with patch("Telegram_Storage_Modules.telegram_user_manager._is_admin_user", return_value=True):
                self.assertTrue(set_user_meta_app("admin_test_123", "app_123", "secret_456"))
                self.assertEqual(os.environ.get("META_APP_ID"), "app_123")
                self.assertEqual(os.environ.get("META_APP_SECRET"), "secret_456")

                self.assertTrue(set_user_meta_app_id("admin_test_123", "app_999"))
                self.assertEqual(os.environ.get("META_APP_ID"), "app_999")

                self.assertTrue(set_user_meta_app_secret("admin_test_123", "secret_888"))
                self.assertEqual(os.environ.get("META_APP_SECRET"), "secret_888")

            # When non-admin sets it, root os.environ is NOT polluted, but user-specific getter returns it
            with patch("Telegram_Storage_Modules.telegram_user_manager._is_admin_user", return_value=False):
                os.environ.pop("META_APP_ID", None)
                os.environ.pop("META_APP_SECRET", None)
                self.assertTrue(set_user_meta_app("normal_user_456", "user_app_777", "user_secret_777"))
                self.assertIsNone(os.environ.get("META_APP_ID"))
                self.assertEqual(get_user_meta_app_id("normal_user_456"), "user_app_777")
                self.assertEqual(get_user_meta_app_secret("normal_user_456"), "user_secret_777")
                # Another non-admin user gets None (strict zero fallback to admin credentials)
                self.assertIsNone(get_user_meta_app_id("unknown_user_999"))

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

    def test_save_and_sync_meta_credentials_strict_isolation(self):
        """Verifies that save_and_sync_meta_credentials isolates normal user credentials and never overwrites admin credentials."""
        from Publishing_Modules.meta_uploader import save_and_sync_meta_credentials

        fake_auth = {
            "success": True,
            "meta_page_id": "normal_page_111",
            "meta_page_token": "normal_tok_222",
            "ig_business_id": "normal_ig_333",
            "ig_business_token": "normal_igtok_444",
            "page_name": "Normal Page",
            "ig_username": "normal_ig",
        }

        # Clear root environment variables
        os.environ["META_PAGE_ID"] = "admin_page_999"
        os.environ["META_PAGE_TOKEN"] = "admin_tok_999"

        with patch("Telegram_Storage_Modules.telegram_user_manager._is_admin_user", return_value=False), \
             patch("Telegram_Storage_Modules.telegram_user_manager._persist_env_var_to_disk") as mock_persist, \
             patch("Telegram_Storage_Modules.telegram_user_manager.sync_user_secret_to_github") as mock_user_sync, \
             patch("Telegram_Storage_Modules.telegram_user_manager.save_all_users") as mock_save_users:

            ok, msg = save_and_sync_meta_credentials(fake_auth, user_id_str="normal_user_888")
            self.assertTrue(ok)

            # CRITICAL ASSERTION: Root admin environment was NOT overwritten
            self.assertEqual(os.environ.get("META_PAGE_ID"), "admin_page_999")
            self.assertEqual(os.environ.get("META_PAGE_TOKEN"), "admin_tok_999")

            # CRITICAL ASSERTION: _persist_env_var_to_disk was NEVER called for non-admin
            mock_persist.assert_not_called()

            # CRITICAL ASSERTION: User-isolated runtime env WAS set
            self.assertEqual(os.environ.get("USER_normal_user_888_META_PAGE_ID"), "normal_page_111")
            self.assertEqual(os.environ.get("USER_normal_user_888_META_PAGE_TOKEN"), "normal_tok_222")

            # CRITICAL ASSERTION: sync_user_secret_to_github was called with user ID
            mock_user_sync.assert_any_call("normal_user_888", "META_PAGE_ID", "normal_page_111")
            mock_user_sync.assert_any_call("normal_user_888", "META_PAGE_TOKEN", "normal_tok_222")
            mock_user_sync.assert_any_call("normal_user_888", "IG_BUSINESS_ID", "normal_ig_333")

    def test_user_setters_and_getters_strict_isolation(self):
        """Verifies that all user credential getters & setters strictly isolate non-admin vs admin."""
        from Telegram_Storage_Modules import telegram_user_manager as tum

        # Setup root env vars (admin credentials)
        os.environ["IG_BUSINESS_TOKEN"] = "root_ig_tok"
        os.environ["IG_BUSINESS_ID"] = "root_ig_id"
        os.environ["META_PAGE_ID"] = "root_fb_id"
        os.environ["META_PAGE_TOKEN"] = "root_fb_tok"
        os.environ["TIKTOK_ACCESS_TOKEN"] = "root_tt_tok"
        os.environ["TOKEN_JSON"] = '{"root": "yt"}'
        os.environ["CLIENT_SECRET_JSON"] = '{"root": "secret"}'
        os.environ["TELEGRAM_PUBLIC_GROUP_ID"] = "-1009999999"
        os.environ["AUTO_INPUT_SCHEDULE_TIMES"] = "09:00,18:00"
        os.environ["APIFY_API_TOKEN"] = "root_apify_tok"
        os.environ["GEMINI_API_KEY"] = "root_gemini_key"
        os.environ["BRAND_WATERMARK_TEXT"] = "@RootAdmin"

        non_admin_uid = "5551234"
        admin_uid = "9998887"

        # Patch _is_admin_user so non_admin_uid is False, admin_uid is True
        def mock_is_admin(uid):
            return str(uid) == admin_uid

        with patch.object(tum, "_is_admin_user", side_effect=mock_is_admin), \
             patch.object(tum, "_persist_env_var_to_disk") as mock_persist, \
             patch.object(tum, "save_all_users"), \
             patch.object(tum, "sync_user_secret_to_github") as mock_sync:

            # 1. Non-admin with NO credentials must get None or empty string, NEVER root admin values
            self.assertIsNone(tum.get_user_instagram_token(non_admin_uid))
            self.assertIsNone(tum.get_user_instagram_id(non_admin_uid))
            self.assertIsNone(tum.get_user_facebook_id(non_admin_uid))
            self.assertIsNone(tum.get_user_facebook_token(non_admin_uid))
            self.assertIsNone(tum.get_user_tiktok_token(non_admin_uid))
            self.assertIsNone(tum.get_user_youtube_token_json(non_admin_uid))
            self.assertIsNone(tum.get_user_youtube_client_secret(non_admin_uid))
            self.assertIsNone(tum.get_user_public_group(non_admin_uid))
            self.assertIsNone(tum.get_user_schedule_times(non_admin_uid))
            self.assertIsNone(tum.get_user_apify_token(non_admin_uid))
            self.assertIsNone(tum.get_user_gemini_key(non_admin_uid))
            self.assertEqual(tum.get_user_branding(non_admin_uid), "")

            # 2. Admin with no personal override gets root credentials
            self.assertEqual(tum.get_user_instagram_token(admin_uid), "root_ig_tok")
            self.assertEqual(tum.get_user_instagram_id(admin_uid), "root_ig_id")
            self.assertEqual(tum.get_user_facebook_id(admin_uid), "root_fb_id")
            self.assertEqual(tum.get_user_facebook_token(admin_uid), "root_fb_tok")
            self.assertEqual(tum.get_user_tiktok_token(admin_uid), "root_tt_tok")
            self.assertEqual(tum.get_user_youtube_token_json(admin_uid), '{"root": "yt"}')
            self.assertEqual(tum.get_user_public_group(admin_uid), "-1009999999")
            self.assertEqual(tum.get_user_branding(admin_uid), "@RootAdmin")

            # 3. Non-admin calls setters -> sets USER_<uid>_<VAR>, does NOT overwrite root, does NOT persist to disk
            tum.set_user_instagram_token(non_admin_uid, "user_ig_tok")
            tum.set_user_instagram_id(non_admin_uid, "user_ig_id")
            tum.set_user_facebook_id(non_admin_uid, "user_fb_id")
            tum.set_user_facebook_token(non_admin_uid, "user_fb_tok")
            tum.set_user_tiktok_token(non_admin_uid, "user_tt_tok")
            tum.set_user_branding(non_admin_uid, "@UserBrand")
            tum.set_user_public_group_id(non_admin_uid, "-1005555555")

            mock_persist.assert_not_called()
            self.assertEqual(os.environ.get(f"USER_{non_admin_uid}_IG_BUSINESS_TOKEN"), "user_ig_tok")
            self.assertEqual(os.environ.get(f"USER_{non_admin_uid}_IG_BUSINESS_ID"), "user_ig_id")
            self.assertEqual(os.environ.get(f"USER_{non_admin_uid}_META_PAGE_ID"), "user_fb_id")
            self.assertEqual(os.environ.get(f"USER_{non_admin_uid}_META_PAGE_TOKEN"), "user_fb_tok")
            self.assertEqual(os.environ.get(f"USER_{non_admin_uid}_TIKTOK_ACCESS_TOKEN"), "user_tt_tok")
            self.assertEqual(os.environ.get(f"USER_{non_admin_uid}_BRAND_WATERMARK_TEXT"), "@UserBrand")
            self.assertEqual(os.environ.get(f"USER_{non_admin_uid}_TELEGRAM_PUBLIC_GROUP_ID"), "-1005555555")

            # Root credentials intact
            self.assertEqual(os.environ.get("IG_BUSINESS_TOKEN"), "root_ig_tok")
            self.assertEqual(os.environ.get("META_PAGE_ID"), "root_fb_id")
            self.assertEqual(os.environ.get("BRAND_WATERMARK_TEXT"), "@RootAdmin")

            # Now getters return personal values for non-admin
            self.assertEqual(tum.get_user_instagram_token(non_admin_uid), "user_ig_tok")
            self.assertEqual(tum.get_user_instagram_id(non_admin_uid), "user_ig_id")
            self.assertEqual(tum.get_user_facebook_id(non_admin_uid), "user_fb_id")
            self.assertEqual(tum.get_user_facebook_token(non_admin_uid), "user_fb_tok")
            self.assertEqual(tum.get_user_tiktok_token(non_admin_uid), "user_tt_tok")
            self.assertEqual(tum.get_user_branding(non_admin_uid), "@UserBrand")
            self.assertEqual(tum.get_user_public_group(non_admin_uid), "-1005555555")

    def test_apify_client_user_isolation(self):
        """Verifies that Apify client strictly requires user's personal token and never shares tokens."""
        from Downloader_Modules import apify_downloader as ad

        non_admin_uid = "777888"
        with patch("Telegram_Storage_Modules.telegram_user_manager._is_admin_user", return_value=False):
            # Non-admin without personal token raises RuntimeError
            with patch("Telegram_Storage_Modules.telegram_user_manager.get_user_apify_token", return_value=None):
                with self.assertRaises(RuntimeError) as ctx:
                    ad._get_client(requesting_user_id=non_admin_uid)
                self.assertIn("APIFY_API_TOKEN is not configured for this user", str(ctx.exception))

            # Non-admin with personal token creates client with their token
            with patch("Telegram_Storage_Modules.telegram_user_manager.get_user_apify_token", return_value="personal_apify_123"), \
                 patch("apify_client.ApifyClient") as mock_client:
                ad._get_client(requesting_user_id=non_admin_uid)
                mock_client.assert_called_once_with("personal_apify_123")

    def test_publish_to_telegram_never_falls_back_to_storage_vault(self):
        """Verifies that publish_to_telegram never leaks into TELEGRAM_STORAGE_GROUP_ID."""
        from Publishing_Modules import media_publisher_main as mpm

        env_patch = {
            "TELEGRAM_BOT_TOKEN": "fake_bot_token",
            "TELEGRAM_STORAGE_GROUP_ID": "-100999888777"
        }
        non_admin_uid = "333444"
        with patch.dict(os.environ, env_patch), \
             patch("Telegram_Storage_Modules.telegram_user_manager._is_admin_user", return_value=False), \
             patch("Telegram_Storage_Modules.telegram_user_manager.get_user_public_group", return_value=None):
            os.environ.pop("TELEGRAM_PUBLIC_GROUP_ID", None)

            res = _run_coro(mpm.publish_to_telegram(
                video_path=__file__,
                title="Test Video",
                user_id=non_admin_uid
            ))
            # Must skip cleanly instead of broadcasting to private vault
            self.assertEqual(res.get("status"), "skipped")
            self.assertIn("not configured", res.get("reason", ""))


if __name__ == "__main__":
    unittest.main()


# ══════════════════════════════════════════════════════════════════════════════
# Trail Reels Feature Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTrailReelsManager(unittest.TestCase):
    """Unit tests for Publishing_Modules.instagram_features.trail_reels.TrailReelsManager"""

    def setUp(self):
        self._orig_env = dict(os.environ)
        self._vault_patcher = patch(
            "Telegram_Storage_Modules.telegram_user_manager.sync_users_json_to_telegram_vault"
        )
        self._vault_patcher.start()

    def tearDown(self):
        self._vault_patcher.stop()
        os.environ.clear()
        os.environ.update(self._orig_env)

    def _mgr(self):
        from Publishing_Modules.instagram_features.trail_reels import TrailReelsManager
        return TrailReelsManager()

    # ── Command Parsing ──────────────────────────────────────────────────────

    def test_trail_command_basic(self):
        """Clean title with /trail_reels → is_trial=True, token stripped."""
        clean, aff, is_trial = self._mgr().extract_trail_command("My Viral Video /trail_reels")
        self.assertTrue(is_trial)
        self.assertEqual(clean, "My Viral Video")
        self.assertIsNone(aff)
        self.assertNotIn("/trail_reels", clean)

    def test_trail_command_with_affiliate_url(self):
        """Title + URL + /trail_reels → all three parsed correctly."""
        clean, aff, is_trial = self._mgr().extract_trail_command(
            "My Product Review https://amzn.to/example /trail_reels"
        )
        self.assertTrue(is_trial)
        self.assertEqual(clean, "My Product Review")
        self.assertEqual(aff, "https://amzn.to/example")
        self.assertNotIn("/trail_reels", clean)
        self.assertNotIn("/trail_reels", aff or "")

    def test_trail_command_typo_alias(self):
        """/trial_reels (typo) is accepted as an alias for /trail_reels."""
        clean, aff, is_trial = self._mgr().extract_trail_command("Cool Title /trial_reels")
        self.assertTrue(is_trial, "Typo /trial_reels must be treated as /trail_reels")
        self.assertEqual(clean, "Cool Title")

    def test_trail_command_no_command(self):
        """Plain title without /trail_reels → is_trial=False."""
        clean, aff, is_trial = self._mgr().extract_trail_command("Just A Normal Title")
        self.assertFalse(is_trial)
        self.assertEqual(clean, "Just A Normal Title")

    def test_trail_command_case_insensitive(self):
        """/TRAIL_REELS in uppercase is also accepted."""
        clean, aff, is_trial = self._mgr().extract_trail_command("Title /TRAIL_REELS")
        self.assertTrue(is_trial)

    def test_trail_command_token_never_in_seo_output(self):
        """/trail_reels must be completely stripped — never in clean_title."""
        for variant in ["/trail_reels", "/trial_reels", "/TRAIL_REELS"]:
            clean, _, _ = self._mgr().extract_trail_command(f"Test Title {variant}")
            self.assertNotIn("trail_reels", clean.lower(),
                             f"Command token leaked into clean_title for variant: {variant}")
            self.assertNotIn("trial_reels", clean.lower(),
                             f"Command token leaked into clean_title for variant: {variant}")

    # ── Meta Container Injection ─────────────────────────────────────────────

    def test_apply_trial_params_when_trial(self):
        """apply_trial_params injects trial_params with MANUAL strategy when is_trial=True."""
        import json
        init_data = {"media_type": "REELS", "caption": "test", "access_token": "tok"}
        result = self._mgr().apply_trial_params(init_data, is_trial=True)
        self.assertIn("trial_params", result)
        parsed = json.loads(result["trial_params"])
        self.assertEqual(parsed.get("graduation_strategy"), "MANUAL")

    def test_apply_trial_params_skipped_when_not_trial(self):
        """apply_trial_params does NOT modify init_data when is_trial=False."""
        init_data = {"media_type": "REELS", "caption": "test", "access_token": "tok"}
        result = self._mgr().apply_trial_params(init_data, is_trial=False)
        self.assertNotIn("trial_params", result)

    # ── Ineligible Account Fallback ──────────────────────────────────────────

    def test_trial_ineligible_fallback_removes_trial_params(self):
        """handle_trial_ineligible_fallback removes trial_params from init_data."""
        import json
        init_data = {
            "media_type": "REELS",
            "trial_params": json.dumps({"graduation_strategy": "MANUAL"}),
            "access_token": "tok",
        }
        result = self._mgr().handle_trial_ineligible_fallback(init_data)
        self.assertNotIn("trial_params", result,
                         "trial_params must be removed on ineligible fallback")
        # Remaining fields must still be present (safe for re-submit)
        self.assertIn("media_type", result)
        self.assertIn("access_token", result)

    def test_trial_ineligible_fallback_no_error_on_clean_dict(self):
        """handle_trial_ineligible_fallback is safe even if trial_params was never present."""
        init_data = {"media_type": "REELS", "access_token": "tok"}
        result = self._mgr().handle_trial_ineligible_fallback(init_data)
        self.assertEqual(result.get("media_type"), "REELS")

    def test_is_trial_rejection_error_detects_trial_in_message(self):
        """_is_trial_rejection_error returns True for typical Meta trial rejection shape."""
        from Publishing_Modules.instagram_features.trail_reels import TrailReelsManager
        err_response = {
            "error": {
                "message": "trial_params not supported for this account type",
                "code": 100
            }
        }
        self.assertTrue(TrailReelsManager._is_trial_rejection_error(err_response))

    def test_is_trial_rejection_error_false_for_unrelated_errors(self):
        """_is_trial_rejection_error returns False for unrelated Meta errors."""
        from Publishing_Modules.instagram_features.trail_reels import TrailReelsManager
        err_response = {
            "error": {
                "message": "Invalid OAuth access token.",
                "code": 190
            }
        }
        self.assertFalse(TrailReelsManager._is_trial_rejection_error(err_response))

    # ── Daily Velocity Tracking ──────────────────────────────────────────────

    def test_velocity_check_returns_no_warning_below_threshold(self):
        """check_daily_velocity returns is_warning=False when count < 10."""
        with patch(
            "Publishing_Modules.instagram_features.trail_reels.TrailReelsManager._load_daily_count",
            return_value=5
        ):
            count, warned = self._mgr().check_daily_velocity("99999")
        self.assertEqual(count, 5)
        self.assertFalse(warned)

    def test_velocity_check_returns_warning_at_threshold(self):
        """check_daily_velocity returns is_warning=True when count >= 10."""
        with patch(
            "Publishing_Modules.instagram_features.trail_reels.TrailReelsManager._load_daily_count",
            return_value=10
        ):
            count, warned = self._mgr().check_daily_velocity("99999")
        self.assertEqual(count, 10)
        self.assertTrue(warned)

    def test_velocity_check_bypass_is_non_blocking(self):
        """Velocity check never raises or hard-blocks — only warns."""
        with patch(
            "Publishing_Modules.instagram_features.trail_reels.TrailReelsManager._load_daily_count",
            side_effect=Exception("simulated DB failure")
        ):
            # Must not raise — returns (0, False) on any error
            try:
                count, warned = self._mgr().check_daily_velocity("99999")
                self.assertFalse(warned, "Must not warn on error — safe default")
            except Exception as e:
                self.fail(f"check_daily_velocity raised unexpectedly: {e}")


class TestPlatformSEOTrailReelsIntegration(unittest.TestCase):
    """Integration tests: /trail_reels token is stripped before Gemini SEO call."""

    def test_trail_token_stripped_from_seo_inputs(self):
        """generate_platform_seo must strip /trail_reels before calling Gemini."""
        # We only test that the SEO function does NOT crash and the token is gone
        # from the returned result. We mock the LLM call to avoid network.
        with patch(
            "Gemini_Modules.platform_seo_generator._llm_json",
            return_value=None
        ):
            from Gemini_Modules.platform_seo_generator import generate_platform_seo
            result = generate_platform_seo(
                video_context="A product review reel.",
                user_title="My Product Review /trail_reels",
                user_hint="My Product Review /trail_reels",
                platforms=["instagram"],
            )
        # The /trail_reels slash-command token must NOT appear anywhere in the result
        result_str = str(result)
        self.assertNotIn("/trail_reels", result_str.lower(),
                         "/trail_reels slash command token must never appear in SEO output")
        self.assertNotIn("/trial_reels", result_str.lower(),
                         "/trial_reels alias token must never appear in SEO output")

        # Specific user-facing fields must not contain the command
        ig_plat = result.get("platforms", {}).get("instagram", {})
        self.assertNotIn("trail_reels", (ig_plat.get("title") or "").lower())
        self.assertNotIn("trail_reels", (ig_plat.get("description") or "").lower())
        self.assertNotIn("trail_reels", (result.get("main_subject") or "").lower())

        # meta_features.trail_reels must be True
        self.assertTrue(
            result.get("meta_features", {}).get("trail_reels"),
            "meta_features.trail_reels must be True when /trail_reels was in input"
        )

    def test_no_trail_command_no_trial_meta_feature(self):
        """When /trail_reels is absent, meta_features.trail_reels must be False."""
        with patch(
            "Gemini_Modules.platform_seo_generator._llm_json",
            return_value=None
        ):
            from Gemini_Modules.platform_seo_generator import generate_platform_seo
            result = generate_platform_seo(
                video_context="A lifestyle reel.",
                user_title="Just A Normal Title",
                platforms=["instagram"],
            )
        self.assertFalse(
            result.get("meta_features", {}).get("trail_reels"),
            "meta_features.trail_reels must be False for standard (non-trial) reel"
        )


