"""
tests / test_multiuser_api_quota_and_concurrency.py
===================================================
Tests for:
1. Immediate parallel execution of multiple users in _dispatch_pipeline_in_background (Zero waiting list).
2. Per-user Gemini API Key and Apify Token resolution and propagation.
3. Vanguard Governor api_key routing ensuring personal quotas are preserved.
4. Container command construction ensuring user-specific API keys are forwarded.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import unittest
from unittest.mock import patch, MagicMock
import threading
import time

from main import _dispatch_pipeline_in_background, ACTIVE_PIPELINE_TARGETS, ACTIVE_PIPELINE_JOBS
from Core_Modules.container_job_manager import ContainerJobManager


class TestMultiUserQuotaAndConcurrency(unittest.TestCase):

    def setUp(self):
        ACTIVE_PIPELINE_TARGETS.clear()
        ACTIVE_PIPELINE_JOBS.clear()

    def tearDown(self):
        ACTIVE_PIPELINE_TARGETS.clear()
        ACTIVE_PIPELINE_JOBS.clear()

    def test_zero_waiting_list_parallel_execution(self):
        """Verify that 3 distinct users submitting simultaneously all start immediately in parallel without waiting."""
        started_events = {
            "user_1": threading.Event(),
            "user_2": threading.Event(),
            "user_3": threading.Event(),
        }
        release_event = threading.Event()
        running_users = []
        lock = threading.Lock()

        def _mock_pipeline(**kwargs):
            uid = kwargs.get("requestor_chat_id")
            with lock:
                running_users.append(uid)
            if uid in started_events:
                started_events[uid].set()
            release_event.wait(timeout=2.0)

        with patch("main.run_master_pipeline", side_effect=_mock_pipeline):
            # Dispatch 3 different users simultaneously
            _dispatch_pipeline_in_background(mode="manual", url="https://instagram.com/reel/clip1/", requestor_chat_id="user_1")
            _dispatch_pipeline_in_background(mode="manual", url="https://instagram.com/reel/clip2/", requestor_chat_id="user_2")
            _dispatch_pipeline_in_background(mode="manual", url="https://instagram.com/reel/clip3/", requestor_chat_id="user_3")

            # All 3 must start in parallel immediately
            self.assertTrue(started_events["user_1"].wait(timeout=1.5), "user_1 did not start immediately!")
            self.assertTrue(started_events["user_2"].wait(timeout=1.5), "user_2 did not start immediately!")
            self.assertTrue(started_events["user_3"].wait(timeout=1.5), "user_3 did not start immediately!")

            with lock:
                self.assertEqual(len(running_users), 3, "All 3 users should be executing concurrently!")

            # Release workers
            release_event.set()

    def test_per_user_key_resolution_in_dispatch(self):
        """Verify that user's personal gemini_api_key and apify_api_token are resolved from telegram_users.json."""
        captured_kwargs = {}
        done_event = threading.Event()

        def _capture_pipeline(**kwargs):
            captured_kwargs.update(kwargs)
            done_event.set()

        mock_users = {
            "999888": {
                "gemini_api_key": "AIzaSyCustomUserGeminiKey_999",
                "apify_api_token": "apify_api_custom_user_token_999"
            }
        }

        with patch("Telegram_Storage_Modules.telegram_user_manager.load_all_users", return_value=mock_users), \
             patch("main.run_master_pipeline", side_effect=_capture_pipeline):

            _dispatch_pipeline_in_background(mode="manual", url="https://instagram.com/reel/test_clip/", requestor_chat_id="999888")

            self.assertTrue(done_event.wait(timeout=1.5))
            self.assertEqual(captured_kwargs.get("user_gemini_key"), "AIzaSyCustomUserGeminiKey_999")
            self.assertEqual(captured_kwargs.get("user_apify_token"), "apify_api_custom_user_token_999")

    def test_gemini_governor_uses_custom_user_api_key(self):
        """Verify that Vanguard Governor generate() passes the user's specific key to genai client."""
        from Gemini_Modules.gemini_router_module.gemini_governor import gemini_router

        custom_key = "AIzaSySpecificUserPersonalKey123"

        with patch("google.genai.Client") as mock_client_cls:
            mock_client_instance = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = '{"status": "ok"}'
            mock_client_instance.models.generate_content.return_value = mock_resp
            mock_client_cls.return_value = mock_client_instance

            # Call generate with custom api_key
            res = gemini_router.generate(
                task_type="vision",
                prompt=["Analyze video"],
                module_name="test_user_quota",
                api_key=custom_key
            )

            # Check that Client was instantiated with the user's personal key
            mock_client_cls.assert_called()
            client_call_kwargs = mock_client_cls.call_args[1]
            self.assertEqual(client_call_kwargs.get("api_key"), custom_key)

    def test_container_command_forwards_user_personal_keys(self):
        """Verify that ContainerJobManager forwards user's personal gemini_api_key and apify_token from telegram_users.json."""
        cjm = ContainerJobManager.instance()
        mock_users = {
            "user_777": {
                "gemini_api_key": "AIzaSyUser777DedicatedKey",
                "apify_api_token": "apify_token_user_777"
            }
        }

        with patch("Telegram_Storage_Modules.telegram_user_manager.load_all_users", return_value=mock_users):
            cmd = cjm._build_docker_cmd("gemini_test_container", "user_777", "user_777", max_accounts=1)

            # Verify that personal keys are injected into docker run -e flags
            self.assertIn("-e", cmd)
            self.assertIn("GEMINI_API_KEY=AIzaSyUser777DedicatedKey", cmd)
            self.assertIn("APIFY_API_TOKEN=apify_token_user_777", cmd)

    def test_governor_does_not_ban_models_on_file_access_403(self):
        """Verify that a 403 on a File resource does NOT permanently ban healthy Gemini models."""
        from Gemini_Modules.gemini_router_module.gemini_governor import gemini_router

        model = "gemini-3.5-flash-lite"
        with gemini_router.state_lock:
            if model in gemini_router.model_states:
                gemini_router.model_states[model]["status"] = "ACTIVE"
                gemini_router.model_states[model]["ban_remaining_seconds"] = 0

        # Simulate Google Files 403 error message
        file_403_msg = "403 PERMISSION_DENIED. {'error': {'code': 403, 'message': 'You do not have permission to access the File 96lo0v0gsnwx or it may not exist.', 'status': 'PERMISSION_DENIED'}}"
        
        # Call mark_model_banned with the file error
        gemini_router.mark_model_banned(model, error_type="403", error_message=file_403_msg)

        # Model should still be ACTIVE, NOT BANNED!
        with gemini_router.state_lock:
            state = gemini_router.model_states.get(model, {})
            self.assertEqual(state.get("status"), "ACTIVE", "Model should NOT be banned on File resource errors!")

    def test_get_user_gemini_key_env_var_and_admin_fallback(self):
        """Verify that get_user_gemini_key checks USER_<id>_GEMINI_API_KEY env var and admin fallback."""
        from Telegram_Storage_Modules.telegram_user_manager import get_user_gemini_key, get_user_apify_token

        with patch("Telegram_Storage_Modules.telegram_user_manager.load_all_users", return_value={}), \
             patch.dict(os.environ, {
                 "USER_12345_GEMINI_API_KEY": "AIzaSyUser12345SecretKey",
                 "USER_12345_APIFY_API_TOKEN": "apify_token_user_12345",
                 "GEMINI_API_KEY": "AIzaSyAdminRootKey",
                 "TELEGRAM_ADMIN_ID": "admin_999"
             }):

            # Non-admin user with USER_<id>_ env var
            key = get_user_gemini_key("12345")
            tok = get_user_apify_token("12345")
            self.assertEqual(key, "AIzaSyUser12345SecretKey")
            self.assertEqual(tok, "apify_token_user_12345")

            # Admin user fallback to root key
            admin_key = get_user_gemini_key("admin_999")
            self.assertEqual(admin_key, "AIzaSyAdminRootKey")


if __name__ == "__main__":
    unittest.main()

