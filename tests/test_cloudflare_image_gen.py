"""
tests/test_cloudflare_image_gen.py
==================================
Unit tests for Cloudflare Workers AI Free Image Generation (FLUX Schnell / SDXL),
Hero visual prompt parsing in Forensic Analyzer, precision frame extraction, and credential management.
"""

import os
import json
import base64
import pytest
from unittest.mock import patch, MagicMock

from Image_Modules.cloudflare_image_gen import (
    CloudflareImageGenerator,
    extract_single_hero_frame,
    generate_hero_cover_for_clip,
    DEFAULT_CLOUDFLARE_MODEL,
)
from Telegram_Storage_Modules.telegram_user_manager import (
    set_user_cloudflare,
    get_user_cloudflare,
    format_user_credentials_summary,
)
from Gemini_Modules.forensic_analyzer import ForensicVideoAnalyzer, _default_result


def test_cloudflare_generator_credential_hydration(monkeypatch):
    """Verifies that credentials resolve from environment variables."""
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "cf_acc_1234567890")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf_tok_abcdefghijklmn")

    client = CloudflareImageGenerator()
    assert client.is_configured() is True
    assert client.account_id == "cf_acc_1234567890"
    assert client.api_token == "cf_tok_abcdefghijklmn"


def test_cloudflare_generator_unconfigured(monkeypatch):
    """Verifies is_configured returns False when credentials are missing."""
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)

    client = CloudflareImageGenerator(account_id="", api_token="")
    assert client.is_configured() is False


def test_cloudflare_generate_image_binary_stream(tmp_path, monkeypatch):
    """Verifies generation when Cloudflare returns direct binary image bytes (FLUX Schnell)."""
    fake_png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRfakeimagecontent"
    out_file = str(tmp_path / "hero_cover.png")

    mock_resp = MagicMock()
    mock_resp.headers.get.return_value = "image/png"
    mock_resp.read.return_value = fake_png_bytes
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        client = CloudflareImageGenerator(account_id="acc123456789", api_token="tok123456789")
        res_path = client.generate_image(
            prompt="Hyper-realistic 8k medium close-up, 85mm lens, cinematic rim light",
            output_path=out_file,
            steps=4,
        )

    assert res_path == out_file
    assert os.path.isfile(out_file)
    with open(out_file, "rb") as f:
        assert f.read() == fake_png_bytes


def test_cloudflare_generate_image_base64_json(tmp_path, monkeypatch):
    """Verifies generation when Cloudflare returns JSON with base64 encoded image."""
    fake_img = b"fake_base64_jpeg_content"
    b64_str = base64.b64encode(fake_img).decode("utf-8")
    json_resp = json.dumps({"result": {"image": b64_str}, "success": True}).encode("utf-8")
    out_file = str(tmp_path / "hero_cover_b64.png")

    mock_resp = MagicMock()
    mock_resp.headers.get.return_value = "application/json"
    mock_resp.read.return_value = json_resp
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        client = CloudflareImageGenerator(account_id="acc123456789", api_token="tok123456789")
        res_path = client.generate_image(
            prompt="Ultra-detailed cinematic portrait with volumetric Tyndall rays",
            output_path=out_file,
            model=DEFAULT_CLOUDFLARE_MODEL,
        )

    assert res_path == out_file
    assert os.path.isfile(out_file)
    with open(out_file, "rb") as f:
        assert f.read() == fake_img


def test_cloudflare_generate_hero_cover_pipeline(tmp_path, monkeypatch):
    """Verifies end-to-end generate_hero_cover_for_clip metadata and fallbacks."""
    fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRdummycover"
    mock_resp = MagicMock()
    mock_resp.headers.get.return_value = "image/png"
    mock_resp.read.return_value = fake_png
    mock_resp.__enter__.return_value = mock_resp

    clip_dir = str(tmp_path / "clip_01")
    os.makedirs(clip_dir, exist_ok=True)

    hero_visual = {
        "timestamp_sec": 3.45,
        "selection_reason": "Peak emotional reaction and dynamic gaze",
        "micro_detail_image_prompt": "Hyper-realistic 8k medium shot, 85mm f/1.4 lens, natural skin pores, dramatic rim light",
        "negative_prompt": "blurry, smooth plastic skin, cartoon",
    }

    with patch("Image_Modules.cloudflare_image_gen.extract_single_hero_frame", return_value=str(tmp_path / "raw.jpg")), \
         patch("urllib.request.urlopen", return_value=mock_resp):
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc123456789")
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok123456789")

        meta = generate_hero_cover_for_clip(
            video_path="dummy.mp4",
            clip_folder=clip_dir,
            hero_visual=hero_visual,
        )

    assert meta["timestamp_sec"] == 3.45
    assert meta["ai_generated"] is True
    assert meta["status"] == "ai_image_regenerated"
    assert os.path.isfile(meta["hero_cover_generated"])


def test_forensic_analyzer_hero_visual_schema():
    """Verifies that ForensicVideoAnalyzer default result and parser preserve hero_visual structure."""
    default = _default_result()
    assert "hero_visual" in default
    assert "timestamp_sec" in default["hero_visual"]
    assert "micro_detail_image_prompt" in default["hero_visual"]


def test_set_and_get_user_cloudflare(tmp_path, monkeypatch):
    """Verifies set_user_cloudflare saves credentials and get_user_cloudflare retrieves them."""
    fake_users = str(tmp_path / "telegram_users.json")
    monkeypatch.setattr("Telegram_Storage_Modules.telegram_user_manager.USERS_JSON_PATH", fake_users)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)

    ok = set_user_cloudflare("123456", "cf_acc_99887766", "cf_tok_aabbccddeeff")
    assert ok is True
    assert os.getenv("CLOUDFLARE_ACCOUNT_ID") == "cf_acc_99887766"
    assert os.getenv("CLOUDFLARE_API_TOKEN") == "cf_tok_aabbccddeeff"

    acc, tok = get_user_cloudflare("123456")
    assert acc == "cf_acc_99887766"
    assert tok == "cf_tok_aabbccddeeff"

    summary = format_user_credentials_summary("123456")
    assert "Cloudflare Workers AI" in summary
    assert "/setcloudflare" in summary
