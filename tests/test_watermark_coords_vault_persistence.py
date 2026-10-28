import os
import sys
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Audio_Modules.audio_pool_manager import AudioPoolManager
from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer


@pytest.fixture
def mock_pool_env(tmp_path):
    td = str(tmp_path)
    orig_dir = os.path.join(td, "Original_audio")
    os.makedirs(orig_dir, exist_ok=True)
    meta_path = os.path.join(orig_dir, "pool_metadata.json")

    initial_meta = {
        "version": 3.0,
        "files": {
            "social_media_id": {
                "https://www.instagram.com/reel/TEST_SC_123/": {
                    "shortcode": "TEST_SC_123",
                    "social_media_id": "https://www.instagram.com/reel/TEST_SC_123/",
                    "media_file_ids": {
                        "raw_video_file_id": "raw_123",
                        "wm_clean_file_id": None
                    },
                    "visual_data": {}
                }
            }
        },
        "clips": {}
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(initial_meta, f, indent=2)

    with patch("Audio_Modules.audio_pool_manager._REPO_ROOT", td), \
         patch("Telegram_Storage_Modules.telegram_vault_indexer._REPO_ROOT", td), \
         patch("Audio_Modules.audio_pool_manager.AudioPoolManager._sync_to_telegram_vault", return_value=True):
        yield {
            "meta_path": meta_path,
            "orig_dir": orig_dir,
            "td": td
        }


def test_update_inpainted_clean_source_persists_coords_and_vectors(mock_pool_env):
    """Verify that update_inpainted_clean_source_in_vault persists watermark_boxes and opencv_vectors to pool_metadata.json."""
    indexer = TelegramVaultIndexer()

    clean_video = os.path.join(mock_pool_env["td"], "video_inpainted_clean.mp4")
    with open(clean_video, "wb") as f:
        f.write(b"dummy clean mp4 video content" * 100)

    sample_boxes = [
        {
            "x": 640,
            "y": 120,
            "w": 320,
            "h": 90,
            "video_width": 1080,
            "video_height": 1920,
            "opencv_vectors": {
                "inpaint_radius": 13,
                "mask_padding_x": 12,
                "mask_padding_y": 8,
                "effective_x": 628,
                "effective_y": 112,
                "effective_w": 344,
                "effective_h": 106,
                "inpaint_method": "Navier_Stokes_EdgeIntegrator",
                "background_texture": "complex"
            }
        }
    ]

    indexer.update_inpainted_clean_source_in_vault(
        clean_video_path=clean_video,
        clip_folder_name="TEST_SC_123",
        watermark_boxes=sample_boxes
    )

    with open(mock_pool_env["meta_path"], "r", encoding="utf-8") as f:
        saved_meta = json.load(f)

    entry = saved_meta["files"]["social_media_id"]["https://www.instagram.com/reel/TEST_SC_123/"]
    assert "visual_data" in entry
    assert entry["visual_data"]["watermark_boxes"] == sample_boxes
    assert entry["visual_data"]["opencv_vectors"]["effective_x"] == 628
    assert entry["visual_data"]["opencv_vectors"]["effective_w"] == 344
    assert entry["watermark_boxes"] == sample_boxes
    assert entry["opencv_vectors"]["effective_h"] == 106


def test_hydrate_clean_video_reconstructs_coords_sidecar(mock_pool_env):
    """Verify that hydrate_clean_video_from_vault reconstructs .coords.json when downloading from vault."""
    with open(mock_pool_env["meta_path"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    entry = meta["files"]["social_media_id"]["https://www.instagram.com/reel/TEST_SC_123/"]
    entry["media_file_ids"]["wm_clean_file_id"] = "clean_fid_999"
    entry["wm_clean_file_id"] = "clean_fid_999"
    entry["visual_data"]["watermark_boxes"] = [
        {
            "x": 500,
            "y": 100,
            "w": 200,
            "h": 50,
            "video_width": 1080,
            "video_height": 1920,
            "opencv_vectors": {
                "effective_x": 488,
                "effective_y": 92,
                "effective_w": 224,
                "effective_h": 66
            }
        }
    ]

    with open(mock_pool_env["meta_path"], "w", encoding="utf-8") as f:
        json.dump(meta, f)

    indexer = TelegramVaultIndexer()

    dest_dir = os.path.join(mock_pool_env["td"], "downloads", "manual_TEST_SC_123")
    os.makedirs(dest_dir, exist_ok=True)

    def fake_download(fid, target_path):
        with open(target_path, "wb") as f:
            f.write(b"vault clean mp4 content" * 100)
        return True

    with patch.object(indexer, "download_vault_file_by_id", side_effect=fake_download):
        hydrated_path = indexer.hydrate_clean_video_from_vault("TEST_SC_123", dest_dir=dest_dir)

    assert hydrated_path is not None
    assert os.path.exists(hydrated_path)

    sidecar_path = hydrated_path + ".coords.json"
    assert os.path.exists(sidecar_path), "Coords sidecar must be reconstructed on hydration!"

    with open(sidecar_path, "r", encoding="utf-8") as sc_f:
        sc_data = json.load(sc_f)

    assert "watermark_boxes" in sc_data
    assert len(sc_data["watermark_boxes"]) == 1
    assert sc_data["watermark_boxes"][0]["x"] == 500
    assert sc_data["watermark_boxes"][0]["opencv_vectors"]["effective_x"] == 488


def test_hydrate_harvested_clip_metadata_reads_coords_file(mock_pool_env):
    """Verify that hydrate_harvested_clip_metadata detects .coords.json in download folders and syncs to pool_metadata."""
    downloads_dir = os.path.join(mock_pool_env["td"], "downloads")
    clip_dir = os.path.join(downloads_dir, "manual_TEST_SC_123")
    os.makedirs(clip_dir, exist_ok=True)

    video_f = os.path.join(clip_dir, "video.mp4")
    with open(video_f, "wb") as f:
        f.write(b"dummy video data" * 50)

    coords_f = os.path.join(clip_dir, "video_inpainted_clean.mp4.coords.json")
    coords_payload = {
        "watermark_boxes": [
            {
                "x": 700,
                "y": 150,
                "w": 180,
                "h": 40,
                "opencv_vectors": {
                    "effective_x": 688,
                    "effective_y": 142,
                    "effective_w": 204,
                    "effective_h": 56
                }
            }
        ]
    }
    with open(coords_f, "w", encoding="utf-8") as f:
        json.dump(coords_payload, f)

    meta_f = os.path.join(clip_dir, "metadata.json")
    with open(meta_f, "w", encoding="utf-8") as f:
        json.dump({"shortcode": "TEST_SC_123", "url": "https://www.instagram.com/reel/TEST_SC_123/"}, f)

    pm = AudioPoolManager()
    pm.hydrate_harvested_clip_metadata()

    with open(mock_pool_env["meta_path"], "r", encoding="utf-8") as f:
        updated_meta = json.load(f)

    entry = updated_meta["files"]["social_media_id"]["https://www.instagram.com/reel/TEST_SC_123/"]
    assert "visual_data" in entry
    assert "watermark_boxes" in entry["visual_data"]
    assert entry["visual_data"]["watermark_boxes"][0]["x"] == 700
    assert entry["visual_data"]["opencv_vectors"]["effective_x"] == 688


def test_branding_overlay_recovers_coords_from_pool_metadata(mock_pool_env):
    """Verify that branding overlay in gemini_ffmpeg_synthesis recovers watermark boxes from pool_metadata when local coords file is absent."""
    # Seed metadata with watermark coordinates
    with open(mock_pool_env["meta_path"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    entry = meta["files"]["social_media_id"]["https://www.instagram.com/reel/TEST_SC_123/"]
    entry["visual_data"]["watermark_boxes"] = [
        {
            "x": 800,
            "y": 200,
            "w": 150,
            "h": 60,
            "video_width": 1080,
            "video_height": 1920,
            "opencv_vectors": {
                "effective_x": 788,
                "effective_y": 192,
                "effective_w": 174,
                "effective_h": 76
            }
        }
    ]
    with open(mock_pool_env["meta_path"], "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Input video path with NO .coords.json sidecar
    fake_video = os.path.join(mock_pool_env["td"], "downloads", "manual_TEST_SC_123", "video_inpainted_clean.mp4")
    os.makedirs(os.path.dirname(fake_video), exist_ok=True)
    with open(fake_video, "wb") as f:
        f.write(b"video" * 100)

    # Simulate Step E branding coordinates recovery
    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer, extract_clean_shortcode
    tvi = TelegramVaultIndexer()
    p_dir = os.path.basename(os.path.dirname(fake_video))
    sc_guess = extract_clean_shortcode(p_dir) or p_dir.replace("manual_", "").replace("auto_", "").strip()
    recovered_entry = tvi.find_entry_by_shortcode(sc_guess)

    assert recovered_entry is not None
    recovered_wm = (
        recovered_entry.get("visual_data", {}).get("watermark_boxes")
        or recovered_entry.get("watermark_boxes")
    )
    assert recovered_wm is not None
    assert len(recovered_wm) == 1
    assert recovered_wm[0]["x"] == 800
    assert recovered_wm[0]["opencv_vectors"]["effective_x"] == 788

