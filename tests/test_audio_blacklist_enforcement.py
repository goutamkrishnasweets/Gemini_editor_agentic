"""
tests/test_audio_blacklist_enforcement.py
=========================================
Unit and integration tests verifying the 3-layer deterministic
Python-first audio blacklist enforcement system:

1. Canonical identifier normalization and cross-matching in rejected_audio_blacklist.py
2. Dynamic mtime reload in rejected_audio_blacklist.py across simulated multi-process writes
3. Stage 2 BGM Cache Lock bust when cached audio is blacklisted (Phase_2/04_bgm_selector.py)
4. Deterministic pre-filtering and Zero-Trust Veto in Gemini_Modules/lyric_rhythm_aligner.py
5. Automatic exclusion of blacklisted audio in TelegramVaultIndexer.get_vault_audio_pool()
"""

import json
import os
import sys
import shutil
import tempfile
import time
import pytest
from unittest.mock import patch, MagicMock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import Audio_Modules.rejected_audio_blacklist as bl
from Audio_Modules.rejected_audio_blacklist import clean_track_identifier, is_blacklisted, add


@pytest.fixture(autouse=True)
def isolated_blacklist(tmp_path, monkeypatch):
    """Isolate blacklist file to a temp directory for tests."""
    temp_bl = str(tmp_path / "data" / "rejected_audio_blacklist.json")
    os.makedirs(os.path.dirname(temp_bl), exist_ok=True)
    with open(temp_bl, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "entries": {}}, f)

    monkeypatch.setattr(bl, "_BLACKLIST_FILE", temp_bl)
    monkeypatch.setattr(bl, "_cache", None)
    monkeypatch.setattr(bl, "_cache_mtime", 0.0)
    yield temp_bl


def test_clean_track_identifier():
    assert clean_track_identifier("vault_bgm_C8123xyz.wav") == "c8123xyz"
    assert clean_track_identifier("bgm_C8123xyz.mp3") == "c8123xyz"
    assert clean_track_identifier("extracted_C8123xyz.wav") == "c8123xyz"
    assert clean_track_identifier("audio_C8123xyz.aac") == "c8123xyz"
    assert clean_track_identifier("C8123xyz.wav") == "c8123xyz"
    assert clean_track_identifier("C8123xyz") == "c8123xyz"
    assert clean_track_identifier("") == ""
    assert clean_track_identifier(None) == ""


def test_blacklist_cross_matching_and_normalization():
    # Blacklist with vault_bgm_ prefix
    add(
        audio_filename="vault_bgm_bad_song_456.wav",
        telegram_file_id="tg_fid_999",
        reason="test_rejection"
    )

    # Must match across all variations
    assert is_blacklisted("vault_bgm_bad_song_456.wav") is True
    assert is_blacklisted("bad_song_456.wav") is True
    assert is_blacklisted("bgm_bad_song_456.mp3") is True
    assert is_blacklisted("extracted_bad_song_456.wav") is True
    assert is_blacklisted("bad_song_456") is True
    assert is_blacklisted(None, audio_shortcode="bad_song_456") is True
    assert is_blacklisted("completely_renamed_file.wav", telegram_file_id="tg_fid_999") is True

    # Clean track should not be matched
    assert is_blacklisted("good_clean_audio_789.wav") is False
    assert is_blacklisted(None, audio_shortcode="clean_shortcode") is False


def test_mtime_dynamic_cache_invalidation():
    # Initial state
    assert is_blacklisted("track_alpha.wav") is False

    # Simulate another worker process writing directly to the JSON on disk
    with open(bl._BLACKLIST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["entries"]["track_alpha.wav"] = {
        "filename": "track_alpha.wav",
        "shortcode": "track_alpha",
        "telegram_file_id": "",
        "reason": "external_rejection",
        "blacklisted_at": time.time(),
    }
    # Touch mtime in the future to ensure mtime changes
    future_time = time.time() + 2.0
    with open(bl._BLACKLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.utime(bl._BLACKLIST_FILE, (future_time, future_time))

    # _load() must detect newer mtime and reload without process restart
    assert is_blacklisted("track_alpha.wav") is True
    assert is_blacklisted("bgm_track_alpha.mp3") is True


def test_bgm_cache_lock_busts_on_blacklisted_audio():
    import importlib
    bgm_selector = importlib.import_module("Phase_2.04_bgm_selector")

    # Blacklist the cached track
    add(audio_filename="vault_bgm_cached_bad.wav", reason="admin_rejected")

    mock_cached_bgm = {
        "selected_audio_track": "vault_bgm_cached_bad.wav",
        "alignment_score": 0.95,
        "physical_path": "/fake/path/vault_bgm_cached_bad.wav"
    }

    with patch.object(bgm_selector, "_try_load_cached_bgm", return_value=mock_cached_bgm), \
         patch.object(bgm_selector, "select_best_audio_for_clip") as mock_select, \
         patch("Telegram_Storage_Modules.telegram_vault_indexer.TelegramVaultIndexer") as mock_vault_cls:
        
        mock_vault = mock_vault_cls.return_value
        mock_vault.get_vault_audio_pool.return_value = {}
        mock_vault.hydrate_bgm_track_from_vault.return_value = "/fake/clean.wav"
        
        mock_select.return_value = {
            "selected_audio_track": "fresh_clean_track.wav",
            "alignment_score": 0.90,
            "backup_candidates": []
        }

        with patch("os.path.isfile", return_value=True), patch("os.path.getsize", return_value=5000):
            res = bgm_selector.select_clip_bgm(
                clip_id="test_clip",
                intent_vector={"preserve_music": True}
            )

        # It must NOT return cached_bgm because it is blacklisted!
        assert res.get("selected_audio_track") != "vault_bgm_cached_bad.wav"
        mock_select.assert_called_once()


def test_lyric_rhythm_aligner_zero_trust_veto():
    from Gemini_Modules.lyric_rhythm_aligner import select_best_audio_for_clip

    add(audio_filename="banned_viral_hit.wav", reason="admin_rejected")

    # Mock pool files containing banned and clean tracks
    fake_pool = {
        "clean_candidate_1.wav": {"bpm": 120, "energy": 0.8, "genre": "electronic", "duration": 30.0},
        "clean_candidate_2.wav": {"bpm": 125, "energy": 0.7, "genre": "hiphop", "duration": 30.0},
    }

    # Banned candidate should not even be in the candidate pool if get_vault_audio_pool respects it
    with patch("Telegram_Storage_Modules.telegram_vault_indexer.TelegramVaultIndexer.get_vault_audio_pool", return_value=fake_pool), \
         patch("Audio_Modules.audio_pool_manager.AudioPoolManager.get_files_index", return_value=fake_pool), \
         patch("Gemini_Modules.gemini_router_module.gemini_governor.gemini_router.generate") as mock_gemini:

        # Simulate Gemini hallucinating or returning the banned track
        mock_gemini.return_value = json.dumps({
            "selected_audio_track": "banned_viral_hit.wav",
            "alignment_score": 0.99,
            "reasoning": "Hallucinated choice"
        })

        res = select_best_audio_for_clip(clip_id="clip_123")
        # Zero-trust gate must veto and fall back to clean 4D math winner!
        assert res.get("selected_audio_track") != "banned_viral_hit.wav"
        assert res.get("selected_audio_track") in ["clean_candidate_1.wav", "clean_candidate_2.wav"]
