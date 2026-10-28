"""
tests/test_sibling_audio_isolation_and_vault_guard.py
====================================================
Verifies:
1. Sibling clips downloaded concurrently in downloads/ NEVER swap audios.
2. If no curated BGM is in the pool, each clip safely preserves its own original audio.
3. AudioPoolManager circuit breaker: an empty local pool NEVER overwrites an existing pool_metadata_file_id in master_vault_index.json.
4. TelegramVaultIndexer downloads pool_metadata.json directly to disk before initializing AudioPoolManager.
"""

import os
import sys
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def test_sibling_clips_never_swap_audio():
    """Verify that when 2 sibling clips exist in downloads/, neither accepts the other's audio as BGM."""
    with tempfile.TemporaryDirectory() as td:
        dl_dir = os.path.join(td, "downloads")
        clip1_dir = os.path.join(dl_dir, "manual_DfAVqq2xX5Q")
        clip2_dir = os.path.join(dl_dir, "manual_DfAWcxWSqsy")
        os.makedirs(clip1_dir, exist_ok=True)
        os.makedirs(clip2_dir, exist_ok=True)

        # Create extracted audio for both
        clip1_audio = os.path.join(clip1_dir, "video_extracted.wav")
        clip2_audio = os.path.join(clip2_dir, "video_extracted.wav")
        with open(clip1_audio, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 2000)
        with open(clip2_audio, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 2000)

        # Mock pool_files containing both clips as raw entries
        mock_pool = {
            "DfAVqq2xX5Q.wav": {
                "shortcode": "DfAVqq2xX5Q",
                "file_id": "FID_1",
                "is_source_extract": True,
                "is_unusable_bgm": True,
                "is_curated_bgm": False,
                "duration": 15.0,
            },
            "DfAWcxWSqsy.wav": {
                "shortcode": "DfAWcxWSqsy",
                "file_id": "FID_2",
                "is_source_extract": True,
                "is_unusable_bgm": True,
                "is_curated_bgm": False,
                "duration": 15.0,
            },
        }

        with patch("Gemini_Modules.lyric_rhythm_aligner._REPO_ROOT", td), \
             patch("Gemini_Modules.lyric_rhythm_aligner.classify_video_archetype", return_value={"name": "ENERGETIC"}), \
             patch("Telegram_Storage_Modules.telegram_vault_indexer.TelegramVaultIndexer.get_vault_audio_pool", return_value=mock_pool), \
             patch("Audio_Modules.audio_pool_manager.AudioPoolManager.get_files_index", return_value={}):

            from Gemini_Modules.lyric_rhythm_aligner import select_best_audio_for_clip

            # Run selector for clip 1
            res1 = select_best_audio_for_clip("manual_DfAVqq2xX5Q", clip_folder=clip1_dir)
            # Clip 1 must NOT select DfAWcxWSqsy.wav! It must preserve its own audio
            assert res1.get("selected_audio_track") != "DfAWcxWSqsy.wav", "Clip 1 stole Clip 2's audio!"
            assert res1.get("physical_path") == clip1_audio, "Clip 1 failed to preserve its own audio!"

            # Run selector for clip 2
            res2 = select_best_audio_for_clip("manual_DfAWcxWSqsy", clip_folder=clip2_dir)
            # Clip 2 must NOT select DfAVqq2xX5Q.wav! It must preserve its own audio
            assert res2.get("selected_audio_track") != "DfAVqq2xX5Q.wav", "Clip 2 stole Clip 1's audio!"
            assert res2.get("physical_path") == clip2_audio, "Clip 2 failed to preserve its own audio!"


def test_vault_circuit_breaker_prevents_empty_clobber():
    """Verify that an empty/wiped local pool_metadata.json never overwrites an existing valid cloud pointer."""
    with tempfile.TemporaryDirectory() as td:
        oa_dir = os.path.join(td, "Original_audio")
        data_dir = os.path.join(td, "data")
        os.makedirs(oa_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Mock master_vault_index with an existing valid pool_metadata_file_id
        index_file = os.path.join(data_dir, "master_vault_index.json")
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump({
                "pool_metadata_file_id": "VALID_EXISTING_CLOUD_FID_12345",
                "metadata_pool_file_id": "VALID_EXISTING_CLOUD_FID_12345",
            }, f)

        # Mock empty local pool_metadata.json
        meta_file = os.path.join(oa_dir, "pool_metadata.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"version": 3, "files": {}}, f)

        with patch("Telegram_Storage_Modules.telegram_vault_indexer._REPO_ROOT", td), \
             patch("Telegram_Storage_Modules.telegram_vault_indexer.DATA_DIR", data_dir), \
             patch("Audio_Modules.audio_pool_manager._REPO_ROOT", td), \
             patch("Telegram_Storage_Modules.telegram_http.send_document", return_value={"ok": True, "result": {"document": {"file_id": "NEW_EMPTY_FID"}}}):

            from Audio_Modules.audio_pool_manager import AudioPoolManager
            from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer

            pm = AudioPoolManager(base_dir=oa_dir)
            pm.metadata = {"version": 3, "files": {}}
            pm._sync_to_telegram_vault(force=True)

            # Reload indexer and verify that the valid existing pointer was PRESERVED
            indexer = TelegramVaultIndexer()
            assert indexer.vault_index.get("pool_metadata_file_id") == "VALID_EXISTING_CLOUD_FID_12345", \
                "Circuit breaker failed! Empty local metadata clobbered valid cloud file_id!"


def test_startup_hydration_downloads_before_audio_pool_manager_init():
    """Verify that hydrate_all_vault_jsons_on_startup downloads directly to disk without clobbering."""
    with tempfile.TemporaryDirectory() as td:
        oa_dir = os.path.join(td, "Original_audio")
        data_dir = os.path.join(td, "data")
        os.makedirs(oa_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        meta_target = os.path.join(oa_dir, "pool_metadata.json")
        assert not os.path.exists(meta_target), "File should not exist yet"

        # Mock cloud download writing 16 mock tracks
        def mock_download(fid, dest):
            with open(dest, "w", encoding="utf-8") as f:
                json.dump({"version": 3, "files": {f"track_{i}.mp3": {"file_id": f"fid_{i}"} for i in range(16)}}, f)
            return True

        with patch("Telegram_Storage_Modules.telegram_vault_indexer._REPO_ROOT", td), \
             patch("Telegram_Storage_Modules.telegram_vault_indexer.DATA_DIR", data_dir):

            from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer

            indexer = TelegramVaultIndexer()
            indexer.vault_index = {"pool_metadata_file_id": "CLOUD_POOL_FID"}
            indexer.sync_pinned_index_from_telegram_sync = MagicMock(return_value=True)
            indexer.download_vault_file_by_id = MagicMock(side_effect=mock_download)

            res = indexer.hydrate_all_vault_jsons_on_startup(force=True)
            assert res.get("pool_metadata") is True, "pool_metadata download failed"
            assert os.path.exists(meta_target), "pool_metadata.json was not created on disk"

            with open(meta_target, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert len(loaded.get("files", {})) == 16, "All 16 tracks must be present on disk"
