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


def test_clean_track_identifier_urls_and_edge_cases():
    # URL trailing slashes and query params
    assert clean_track_identifier("https://www.instagram.com/reel/Deb2IaFE67-/") == "deb2iafe67-"
    assert clean_track_identifier("https://www.instagram.com/reel/Deb2IaFE67-/?utm_source=ig_web_copy_link") == "deb2iafe67-"
    assert clean_track_identifier("https://www.instagram.com/p/Deb2IaFE67-") == "deb2iafe67-"
    
    # manual_ prefix
    assert clean_track_identifier("manual_Deb2IaFE67-") == "deb2iafe67-"
    assert clean_track_identifier("manual_DXKHhNfhxwX") == "dxkhhnfhxwx"
    
    # File patterns
    assert clean_track_identifier("Deb2IaFE67-.wav") == "deb2iafe67-"
    assert clean_track_identifier("extracted_Deb2IaFE67-.wav") == "deb2iafe67-"
    assert clean_track_identifier("vault_bgm_Deb2IaFE67-.mp3") == "deb2iafe67-"


def test_dual_input_purge_blacklists_both_reel_and_bgm(tmp_path, monkeypatch):
    from Audio_Modules.audio_pool_manager import AudioPoolManager

    temp_pool_dir = tmp_path / "pool_data"
    temp_pool_dir.mkdir(parents=True, exist_ok=True)
    temp_meta = str(temp_pool_dir / "pool_metadata.json")

    reel_sc = "Deb2IaFE67-"
    reel_url = f"https://www.instagram.com/reel/{reel_sc}/"
    bgm_sc = "SomeOtherBGM"
    bgm_fname = f"vault_bgm_{bgm_sc}.wav"

    mock_meta_data = {
        "version": 3,
        "files": {
            "social_media_id": {
                reel_url: {
                    "social_media_id": reel_url,
                    "shortcode": reel_sc,
                    "selected_audio": bgm_fname,
                    "media_file_ids": {"extracted_audio_file_id": "reel_audio_fid_111"}
                }
            },
            bgm_fname: {
                "file_id": "bgm_fid_222",
                "shortcode": bgm_sc
            }
        }
    }
    with open(temp_meta, "w", encoding="utf-8") as f:
        json.dump(mock_meta_data, f)

    pm = AudioPoolManager(base_dir=str(temp_pool_dir))
    pm.metadata = mock_meta_data
    pm._save_metadata(sync_to_vault=False)

    # Pre-condition: neither is blacklisted
    assert is_blacklisted(reel_sc) is False
    assert is_blacklisted(bgm_fname) is False

    # Execute dual-input purge
    result = pm.purge_media_and_audio_entries(
        reel_identifier=f"manual_{reel_sc}",
        selected_audio_name=bgm_fname
    )

    assert result["status"] == "success"

    # POST-CONDITION 1: The rejected reel's own extracted audio is blacklisted!
    assert is_blacklisted(reel_sc) is True
    assert is_blacklisted(f"extracted_{reel_sc}.wav") is True
    assert is_blacklisted(f"{reel_sc}.wav") is True

    # POST-CONDITION 2: The selected BGM track is blacklisted!
    assert is_blacklisted(bgm_fname) is True
    assert is_blacklisted(bgm_sc) is True
    assert is_blacklisted("random_name.wav", telegram_file_id="bgm_fid_222") is True


def test_vault_hydration_skips_and_sanitizes_blacklisted_tracks(tmp_path, monkeypatch):
    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
    from Audio_Modules.audio_pool_manager import AudioPoolManager

    temp_pool_dir = tmp_path / "pool_data_hydrate"
    temp_pool_dir.mkdir(parents=True, exist_ok=True)
    temp_meta = str(temp_pool_dir / "pool_metadata.json")

    with open(temp_meta, "w", encoding="utf-8") as f:
        json.dump({"version": 3, "files": {}}, f)

    # Blacklist Deb2IaFE67-
    add(audio_filename="Deb2IaFE67-.wav", reason="admin_rejected")

    indexer = TelegramVaultIndexer()
    indexer.vault_index = {
        "column_1_processed_reels": {
            "by_session_id": {
                "sess_old_1": {
                    "audio_data": {
                        "selected_audio_track": "Deb2IaFE67-.wav",
                        "pool_metadata": {
                            "selected_audio_track": "Deb2IaFE67-.wav",
                            "shortcode": "Deb2IaFE67-",
                            "file_id": "fid_blacklisted"
                        }
                    }
                },
                "sess_old_2": {
                    "audio_data": {
                        "selected_audio_track": "clean_track_999.wav",
                        "pool_metadata": {
                            "selected_audio_track": "clean_track_999.wav",
                            "shortcode": "clean_track_999",
                            "file_id": "fid_clean_999"
                        }
                    }
                }
            }
        }
    }

    # Patch AudioPoolManager to use our temp metadata directory
    orig_apm_init = AudioPoolManager.__init__
    def mock_apm_init(self, base_dir=None):
        orig_apm_init(self, base_dir=str(temp_pool_dir))

    monkeypatch.setattr(AudioPoolManager, "__init__", mock_apm_init)

    indexer._hydrate_local_caches()

    # Verify that clean_track_999 was hydrated, but Deb2IaFE67-.wav was BLOCKED
    with open(temp_meta, "r", encoding="utf-8") as f:
        hydrated_meta = json.load(f)

    files = hydrated_meta.get("files", {})
    assert "Deb2IaFE67-.wav" not in files
    assert "clean_track_999.wav" in files

