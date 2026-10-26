import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# Add repo root to path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Telegram_Storage_Modules.telegram_user_manager import (
    is_admin_user,
    set_user_youtube_token,
    set_user_youtube_client_secret,
    sync_user_secret_to_github,
)


def test_is_admin_user_detection(monkeypatch):
    """Admin detection strictly recognizes admin IDs and rejects non-admin IDs."""
    monkeypatch.setenv("TELEGRAM_ADMIN_ID", "111111,222222")

    assert is_admin_user("111111") is True
    assert is_admin_user("222222") is True
    assert is_admin_user("999999") is False
    assert is_admin_user("admin") is False


def test_set_user_youtube_token_admin_vs_non_admin(tmp_path, monkeypatch):
    """Verifies admin writes to root disk files while non-admin never touches root disk files."""
    monkeypatch.setenv("TELEGRAM_ADMIN_ID", "100")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("Telegram_Storage_Modules.telegram_user_manager.DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("Telegram_Storage_Modules.telegram_user_manager.USERS_JSON_PATH", str(tmp_path / "data" / "telegram_users.json"))

    fake_token = json.dumps({"token": "fake_access", "refresh_token": "fake_refresh"})

    with patch("Telegram_Storage_Modules.telegram_user_manager.sync_user_secret_to_github") as mock_sync, \
         patch("Telegram_Storage_Modules.telegram_user_manager.sync_users_json_to_telegram_vault"):
        
        # 1. Non-admin user sets token
        res_user = set_user_youtube_token("200", fake_token)
        assert res_user is True

        # Verify root disk files were NOT created
        assert not os.path.exists("Credentials/token.json")
        assert not os.path.exists("Credentials/youtube/token.json")

        # Verify telegram_users.json has user's token
        with open(tmp_path / "data" / "telegram_users.json", "r", encoding="utf-8") as f:
            users_data = json.load(f)
        assert users_data["200"]["youtube_token_json"] == fake_token

        # 2. Admin sets token
        res_admin = set_user_youtube_token("100", fake_token)
        assert res_admin is True

        # Verify root disk files WERE created for admin
        assert os.path.exists("Credentials/token.json")
        assert os.path.exists("Credentials/youtube/token.json")


def test_sync_user_secret_to_github_isolation(monkeypatch):
    """Verifies admin updates canonical secret while non-admin gets USER_<id>_ prefix."""
    monkeypatch.setenv("TELEGRAM_ADMIN_ID", "100")

    with patch("Utilities.github_secret_updater.sync_custom_secret_to_github", return_value=True) as mock_sync:
        # Non-admin
        sync_user_secret_to_github("200", "TOKEN_JSON", "secret_val")
        mock_sync.assert_called_with("USER_200_TOKEN_JSON", "secret_val")

        mock_sync.reset_mock()

        # Admin
        sync_user_secret_to_github("100", "TOKEN_JSON", "secret_val")
        mock_sync.assert_called_with("TOKEN_JSON", "secret_val")


def test_auth_youtube_isolation_non_admin(tmp_path, monkeypatch):
    """Verifies scripts.auth_youtube never touches root token.json or root secrets when called for non-admin."""
    monkeypatch.setenv("TELEGRAM_ADMIN_ID", "100")
    monkeypatch.chdir(tmp_path)

    import scripts.auth_youtube as auth_yt

    fake_token = json.dumps({"token": "user_token_abc", "refresh_token": "user_refresh_123"})
    user_token_path = "Credentials/user_999_token.json"

    with patch.object(auth_yt, "sync_token_to_github_secret") as mock_gh_sync, \
         patch("Telegram_Storage_Modules.telegram_user_manager.set_user_youtube_token") as mock_set_token, \
         patch.object(auth_yt, "_send_telegram"):

        # Simulate _try_device_flow success for non-admin user 999
        with patch("urllib.request.urlopen") as mock_url:
            # Device code request mock
            mock_resp1 = MagicMock()
            mock_resp1.read.return_value = json.dumps({
                "device_code": "dev123",
                "user_code": "usr123",
                "verification_url": "https://google.com/device",
                "expires_in": 10,
                "interval": 1
            }).encode("utf-8")

            # Poll response mock
            mock_resp2 = MagicMock()
            mock_resp2.read.return_value = json.dumps({
                "access_token": "user_access_token",
                "refresh_token": "user_refresh_token"
            }).encode("utf-8")

            mock_url.side_effect = [mock_resp1, mock_resp2]

            handled = auth_yt._try_device_flow(
                client_id="cid123",
                client_secret="sec123",
                tg_token="fake_bot_token",
                tg_admin="999",  # non-admin
                token_path=user_token_path
            )

            assert handled is True
            # Root secret sync MUST NOT be called for non-admin
            mock_gh_sync.assert_not_called()

            # Root files MUST NOT exist
            assert not os.path.exists("Credentials/token.json")
            assert not os.path.exists("Credentials/youtube/token.json")

            # User isolated token file MUST exist
            assert os.path.exists(user_token_path)

            # set_user_youtube_token MUST be called for user 999
            mock_set_token.assert_called_once()
            called_uid, _ = mock_set_token.call_args[0]
            assert called_uid == "999"


def test_auth_youtube_isolation_admin(tmp_path, monkeypatch):
    """Verifies scripts.auth_youtube properly writes root token and syncs root secret for admin."""
    monkeypatch.setenv("TELEGRAM_ADMIN_ID", "100")
    monkeypatch.chdir(tmp_path)

    import scripts.auth_youtube as auth_yt

    admin_token_path = "Credentials/token.json"

    with patch.object(auth_yt, "sync_token_to_github_secret") as mock_gh_sync, \
         patch("Telegram_Storage_Modules.telegram_user_manager.set_user_youtube_token") as mock_set_token, \
         patch.object(auth_yt, "_send_telegram"):

        with patch("urllib.request.urlopen") as mock_url:
            mock_resp1 = MagicMock()
            mock_resp1.read.return_value = json.dumps({
                "device_code": "dev123",
                "user_code": "usr123",
                "verification_url": "https://google.com/device",
                "expires_in": 10,
                "interval": 1
            }).encode("utf-8")

            mock_resp2 = MagicMock()
            mock_resp2.read.return_value = json.dumps({
                "access_token": "admin_access_token",
                "refresh_token": "admin_refresh_token"
            }).encode("utf-8")

            mock_url.side_effect = [mock_resp1, mock_resp2]

            handled = auth_yt._try_device_flow(
                client_id="cid123",
                client_secret="sec123",
                tg_token="fake_bot_token",
                tg_admin="100",  # admin
                token_path=admin_token_path
            )

            assert handled is True
            # Root secret sync MUST be called for admin
            mock_gh_sync.assert_called_once()

            # Root file MUST exist
            assert os.path.exists(admin_token_path)

            # set_user_youtube_token MUST be called for admin 100
            mock_set_token.assert_called_once()
            called_uid, _ = mock_set_token.call_args[0]
            assert called_uid == "100"


def test_process_yt_auth_code_input_creates_isolated_file(tmp_path, monkeypatch):
    """Verifies process_yt_auth_code_input creates user-specific code file."""
    import asyncio
    from unittest.mock import AsyncMock
    monkeypatch.setattr("main._REPO_ROOT", str(tmp_path))

    from main import process_yt_auth_code_input

    mock_msg = MagicMock()
    mock_msg.reply_text = AsyncMock()

    # Paste code with user_id="888"
    asyncio.run(process_yt_auth_code_input(mock_msg, "4/0Aean...", user_id="888"))

    code_file = tmp_path / "Credentials" / "yt_auth_code.txt"
    user_file = tmp_path / "Credentials" / "yt_auth_code_888.txt"

    assert code_file.exists()
    assert user_file.exists()
    assert user_file.read_text(encoding="utf-8") == "4/0Aean..."
