"""
tests/test_inpaint_directive_and_cache_invalidation.py
======================================================
Verifies:
1. Resilient typo-tolerant inpaint directive parsing ('Inapint flimygyan', 'delogo', etc.).
2. Accurate extraction and canonicalization of target watermark keywords.
3. Vault hydration behavior with check_clean_first=False vs True.
4. Clean cache invalidation on inpaint request so stale clean videos are never reused.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
from unittest.mock import MagicMock, patch

from Watermark_and_Inpainting.watermark_directive_helper import (
    is_inpaint_directive,
    extract_watermark_keywords,
)


def test_is_inpaint_directive_typo_resilience():
    positive_cases = [
        "Inapint flimygyan",
        "inpaint filmygyan",
        "in-paint logo",
        "inapaint the watermark",
        "inpint this",
        "inpainting please",
        "inapinting logo",
        "delogo video",
        "de-logo please",
        "remove watermark",
        "clean the watermrk",
        "erase logo",
        "logo still visible in bottom right",
        "inpaint not in correct position",
        "missed the corner logo",
        "viral bhayani logo",
        "voompla watermark",
    ]
    for case in positive_cases:
        assert is_inpaint_directive(case) is True, f"Failed on positive case: {case}"

    negative_cases = [
        "change music to energetic rock",
        "make cut faster",
        "increase video volume",
        "replace audio track",
        "cinematic zoom effect",
        "",
        None,
    ]
    for case in negative_cases:
        assert is_inpaint_directive(case) is False, f"Failed on negative case: {case}"


def test_extract_watermark_keywords():
    assert extract_watermark_keywords("Inapint custombrand") == "custombrand"
    assert extract_watermark_keywords("inpaint filmygyan") == "filmygyan"
    assert extract_watermark_keywords("remove viralbhayani logo") == "viralbhayani"
    assert extract_watermark_keywords("custom_channel_name") == "custom_channel_name"
    
    # Generic directives return fallback
    generic = extract_watermark_keywords("remove watermark")
    assert "watermark" in generic or "logo" in generic


def test_vault_hydrate_respects_check_clean_first(tmp_path):
    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer

    indexer = TelegramVaultIndexer()
    test_dir = str(tmp_path / "downloads" / "manual_test123")
    os.makedirs(test_dir, exist_ok=True)

    # Place a clean video in test_dir
    clean_path = os.path.join(test_dir, "manual_clean_test123.mp4")
    with open(clean_path, "wb") as f:
        f.write(b"FAKE_CLEAN_VIDEO" * 100)

    # When check_clean_first=True, it should discover clean video
    with patch.object(indexer, "hydrate_clean_video_from_vault", return_value=clean_path):
        res_clean = indexer.hydrate_raw_video_from_vault("test123", dest_dir=test_dir, check_clean_first=True)
        assert res_clean == clean_path

    # Place a raw video in test_dir
    raw_path = os.path.join(test_dir, "manual_raw_test123.mp4")
    with open(raw_path, "wb") as f:
        f.write(b"FAKE_RAW_VIDEO" * 100)

    # When check_clean_first=False, it MUST NOT return clean video, it must return raw video!
    res_raw = indexer.hydrate_raw_video_from_vault("test123", dest_dir=test_dir, check_clean_first=False)
    assert res_raw == raw_path
    assert "_clean" not in res_raw


def test_tool_reedit_session_invalidates_clean_cache_on_inpaint(tmp_path):
    from Agentic_Core.tool_adapters import tool_reedit_session

    clip_id = "manual_DeWVSzUy1f7"
    recovery_dir = str(tmp_path / "downloads" / clip_id)
    os.makedirs(recovery_dir, exist_ok=True)

    # Put a stale clean file and coords file
    stale_clean = os.path.join(recovery_dir, "manual_clean_DeWVSzUy1f7.mp4")
    with open(stale_clean, "wb") as f:
        f.write(b"OLD_CLEAN_DATA" * 50)
    stale_coords = stale_clean + ".coords.json"
    with open(stale_coords, "w") as f:
        f.write('{"watermark_boxes": []}')

    raw_file = os.path.join(recovery_dir, "video.mp4")
    with open(raw_file, "wb") as f:
        f.write(b"TRUE_RAW_DATA" * 100)

    fake_session = {
        "session_id": "sess_test",
        "clip_id": clip_id,
        "raw_video_path": raw_file,
        "video_path": stale_clean,
    }

    mock_sm = MagicMock()
    mock_sm.get_session.return_value = fake_session
    mock_sm.register_retry.return_value = (True, 2)

    with patch("Telegram_Storage_Modules.telegram_session_manager.TelegramSessionManager", return_value=mock_sm):
        with patch("Telegram_Storage_Modules.telegram_vault_indexer.TelegramVaultIndexer") as mock_vi_cls:
            mock_vi = mock_vi_cls.return_value
            mock_vi.find_entry_by_shortcode.return_value = {"shortcode": "DeWVSzUy1f7"}
            mock_vi.hydrate_raw_video_from_vault.return_value = raw_file

            with patch("Phase_2.phase2_orchestrator.run_phase2_pipeline") as mock_p2:
                mock_p2.return_value = {"rendered_files": [os.path.join(recovery_dir, "master.mp4")]}

                res = tool_reedit_session(
                    session_id="sess_test",
                    edit_directive="Inapint flimygyan"
                )

                assert res["status"] == "success"
                # Verify check_clean_first was set to False because directive requested inpainting
                mock_vi.hydrate_raw_video_from_vault.assert_called_once()
                call_kwargs = mock_vi.hydrate_raw_video_from_vault.call_args[1]
                assert call_kwargs.get("check_clean_first") is False
                
                # Verify stale clean was deleted
                assert not os.path.exists(stale_clean)
                assert not os.path.exists(stale_coords)
