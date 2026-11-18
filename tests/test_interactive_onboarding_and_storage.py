"""
tests/test_interactive_onboarding_and_storage.py
=================================================
Validates zero-mandatory .env configuration, interactive onboarding resolver,
storage vault group ID normalization and persistence, and MTProto credential saving.
"""

import os
import json
import pytest
from unittest.mock import patch

from Telegram_Storage_Modules.telegram_user_manager import (
    normalize_telegram_chat_id,
    set_user_storage_group_id,
    get_user_storage_group_id,
    set_user_mtproto,
    format_user_credentials_summary,
)
from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
from main import (
    normalize_telegram_group_id,
    load_bot_config_into_env,
    _save_bot_config_and_env,
    resolve_bot_token,
    BOT_CONFIG_PATH,
)


def test_normalize_telegram_group_ids():
    """Verifies that positive, negative, and supergroup IDs normalize properly."""
    assert normalize_telegram_chat_id("5393269483") == "-5393269483"
    assert normalize_telegram_chat_id("-5393269483") == "-5393269483"
    assert normalize_telegram_chat_id("-1001234567890") == "-1001234567890"
    assert normalize_telegram_chat_id("1001234567890") == "-1001234567890"
    assert normalize_telegram_chat_id("") == ""
    assert normalize_telegram_chat_id("   ") == ""

    # Test main.py's normalize_telegram_group_id helper
    assert normalize_telegram_group_id("987654321") == "-987654321"
    assert normalize_telegram_group_id("-987654321") == "-987654321"
    assert normalize_telegram_group_id("") == ""


def test_set_and_get_user_storage_group_id(tmp_path, monkeypatch):
    """Verifies user storage group ID persistence and retrieval hierarchy."""
    fake_users_path = str(tmp_path / "telegram_users.json")
    monkeypatch.setattr("Telegram_Storage_Modules.telegram_user_manager.USERS_JSON_PATH", fake_users_path)
    monkeypatch.delenv("TELEGRAM_STORAGE_GROUP_ID", raising=False)
    monkeypatch.delenv("USER_12345_TELEGRAM_STORAGE_GROUP_ID", raising=False)

    user_id = "12345"
    storage_id = "5393269483"

    # Save
    saved = set_user_storage_group_id(user_id, storage_id)
    assert saved is True
    assert os.getenv("TELEGRAM_STORAGE_GROUP_ID") == "-5393269483"

    # Retrieve by user_id
    resolved = get_user_storage_group_id(user_id)
    assert resolved == "-5393269483"

    # Retrieve global fallback
    resolved_global = get_user_storage_group_id(None)
    assert resolved_global == "-5393269483"


def test_set_user_mtproto_credentials(tmp_path, monkeypatch):
    """Verifies MTProto API_ID and API_HASH saving."""
    fake_users_path = str(tmp_path / "telegram_users.json")
    monkeypatch.setattr("Telegram_Storage_Modules.telegram_user_manager.USERS_JSON_PATH", fake_users_path)

    saved = set_user_mtproto("admin", "30635574", "9a2ef355bf2d729020aab7749739aaf7")
    assert saved is True
    assert os.getenv("TELEGRAM_API_ID") == "30635574"
    assert os.getenv("TELEGRAM_API_HASH") == "9a2ef355bf2d729020aab7749739aaf7"


def test_bot_config_persistence_and_loading(tmp_path, monkeypatch):
    """Verifies saving to and loading from bot_config.json."""
    fake_config_path = str(tmp_path / "bot_config.json")
    monkeypatch.setattr("main.BOT_CONFIG_PATH", fake_config_path)

    # Save credentials
    _save_bot_config_and_env(
        token="123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ",
        storage_gid="-5393269483",
        api_id="11223344",
        api_hash="aabbccddeeff00112233",
    )

    assert os.path.exists(fake_config_path)
    with open(fake_config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["bot_token"] == "123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ"
    assert data["storage_group_id"] == "-5393269483"
    assert data["telegram_api_id"] == "11223344"
    assert data["telegram_api_hash"] == "aabbccddeeff00112233"

    # Clear env vars and test load_bot_config_into_env
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_STORAGE_GROUP_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_API_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_API_HASH", raising=False)

    load_bot_config_into_env()

    assert os.getenv("TELEGRAM_BOT_TOKEN") == "123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ"
    assert os.getenv("TELEGRAM_STORAGE_GROUP_ID") == "-5393269483"
    assert os.getenv("TELEGRAM_API_ID") == "11223344"
    assert os.getenv("TELEGRAM_API_HASH") == "aabbccddeeff00112233"


def test_resolve_bot_token_immediate_when_already_set(monkeypatch):
    """Verifies resolve_bot_token returns immediately if already configured."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ")
    monkeypatch.setenv("TELEGRAM_STORAGE_GROUP_ID", "-1001234567890")

    token = resolve_bot_token()
    assert token == "123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ"


def test_resolve_bot_token_non_interactive_ci_safe(tmp_path, monkeypatch):
    """Verifies resolve_bot_token never hangs in non-interactive CI environments."""
    fake_config_path = str(tmp_path / "bot_config.json")
    monkeypatch.setattr("main.BOT_CONFIG_PATH", fake_config_path)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_STORAGE_GROUP_ID", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    # Should exit or return without freezing
    token = resolve_bot_token()
    assert token == ""


def test_vault_indexer_get_storage_group_id(monkeypatch):
    """Verifies TelegramVaultIndexer resolves storage group ID correctly."""
    monkeypatch.setenv("TELEGRAM_STORAGE_GROUP_ID", "-5393269483")
    indexer = TelegramVaultIndexer()
    assert indexer.get_storage_group_id() == "-5393269483"


def test_format_user_credentials_summary_includes_vault_and_mtproto():
    """Verifies that /myconfig checklist includes the new storage vault and MTProto rows."""
    summary = format_user_credentials_summary("admin")
    assert "Storage Vault Group ID:" in summary
    assert "/setstorage" in summary
    assert "MTProto 2GB Bypass Engine:" in summary
    assert "/setmtproto" in summary
