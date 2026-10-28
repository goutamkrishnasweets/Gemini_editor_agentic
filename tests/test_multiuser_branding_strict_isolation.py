import os
import sys
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Telegram_Storage_Modules.telegram_user_manager import (
    set_user_branding,
    get_user_branding,
)
from Gemini_Modules.gemini_ffmpeg_synthesis import FFmpegCommandGenerator, GeminiFFmpegEngine


def test_user_branding_applied_strictly_to_that_user_only(tmp_path, monkeypatch):
    """
    Verifies:
    1. If User A runs /setbranding 'BrandA', BrandA is applied ONLY to User A's inputs.
    2. User B (no branding set) gets 100% clean video with ZERO branding.
    3. User A's brand NEVER leaks to User B or anonymous inputs.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("Telegram_Storage_Modules.telegram_user_manager.DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("Telegram_Storage_Modules.telegram_user_manager.USERS_JSON_PATH", str(tmp_path / "data" / "telegram_users.json"))
    monkeypatch.setenv("TELEGRAM_ADMIN_ID", "999")
    monkeypatch.setenv("BRAND_WATERMARK_TEXT", "AdminSecretBrand")

    fake_video = str(tmp_path / "input.mp4")
    with open(fake_video, "w") as f:
        f.write("dummy video")
    fake_out = str(tmp_path / "output.mp4")

    with patch("Telegram_Storage_Modules.telegram_user_manager.sync_user_secret_to_github"), \
         patch("Telegram_Storage_Modules.telegram_user_manager.sync_users_json_to_telegram_vault"):

        # User A sets custom brand
        set_user_branding("User_A", "BrandAlpha")

        # Verify User A gets BrandAlpha
        assert get_user_branding("User_A") == "BrandAlpha"

        # Verify User B gets empty string (NO branding set)
        assert get_user_branding("User_B") == ""

        # Verify anonymous gets empty string
        assert get_user_branding(None) == ""
        assert get_user_branding("") == ""

        gen = FFmpegCommandGenerator()

        # Synthesis for User A -> MUST contain BrandAlpha, MUST NOT contain AdminSecretBrand
        plan_user_a = gen.build_single_pass_filtergraph(
            input_path=fake_video,
            output_path=fake_out,
            extra_inputs={"requestor_chat_id": "User_A", "brand_text": get_user_branding("User_A")},
            gemini_operations=[]
        )
        cmd_a = " ".join(plan_user_a["cmd_list"])
        assert "BrandAlpha" in cmd_a
        assert "AdminSecretBrand" not in cmd_a

        # Synthesis for User B -> MUST NOT contain BrandAlpha, AdminSecretBrand, AMTCE, or drawtext!
        plan_user_b = gen.build_single_pass_filtergraph(
            input_path=fake_video,
            output_path=fake_out,
            extra_inputs={"requestor_chat_id": "User_B", "brand_text": get_user_branding("User_B")},
            gemini_operations=[]
        )
        cmd_b = " ".join(plan_user_b["cmd_list"])
        assert "BrandAlpha" not in cmd_b
        assert "AdminSecretBrand" not in cmd_b
        assert "AMTCE" not in cmd_b
        assert "drawtext" not in cmd_b

        # Synthesis for Anonymous / Unassigned input -> MUST NOT contain any branding!
        plan_anon = gen.build_single_pass_filtergraph(
            input_path=fake_video,
            output_path=fake_out,
            extra_inputs=None,
            gemini_operations=[]
        )
        cmd_anon = " ".join(plan_anon["cmd_list"])
        assert "BrandAlpha" not in cmd_anon
        assert "AdminSecretBrand" not in cmd_anon
        assert "AMTCE" not in cmd_anon
        assert "drawtext" not in cmd_anon
