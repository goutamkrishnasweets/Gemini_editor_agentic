import os
import json
import time
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from Audio_Modules.rejected_audio_blacklist import (
    add as bl_add,
    is_blacklisted,
    clean_track_identifier,
    get_blacklisted_shortcodes,
)
from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
from Core_Modules.purger import purge_full_clip_and_assets
from Audio_Modules.audio_pool_manager import AudioPoolManager
from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer


class TestAudioRejectionAndFatigue(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)

        # Mock blacklist file location to temporary directory
        self.bl_file = os.path.join(self.test_dir.name, "rejected_audio_blacklist.json")
        self.sessions_file = os.path.join(self.test_dir.name, "telegram_sessions.json")
        self.metadata_file = os.path.join(self.test_dir.name, "pool_metadata.json")

    def test_clean_track_identifier_and_prefix_stripping(self):
        """Verify clean_track_identifier handles dictionaries, telegram_ prefixes, and shortcodes."""
        self.assertEqual(clean_track_identifier("telegram_***_1793909978.wav"), "***_1793909978")
        self.assertEqual(clean_track_identifier("vault_bgm_DfX19qZx1gn.wav"), "dfx19qzx1gn")
        self.assertEqual(clean_track_identifier("extracted_manual_DfULIxos0kb.wav"), "dfulixos0kb")
        self.assertEqual(clean_track_identifier({"track_name": "telegram_DfX19qZx1gn.wav"}), "dfx19qzx1gn")
        self.assertEqual(
            clean_track_identifier("https://www.instagram.com/reel/DfULIxos0kb/?cplk=Z245MWFrMzllOXFo"),
            "dfulixos0kb"
        )

    @patch("Audio_Modules.rejected_audio_blacklist._BLACKLIST_FILE")
    @patch("Audio_Modules.rejected_audio_blacklist._cache", None)
    @patch("Audio_Modules.rejected_audio_blacklist._cache_mtime", 0.0)
    @patch("Audio_Modules.rejected_audio_blacklist.sync_blacklist_to_vault")
    def test_rejection_and_discard_permanently_blacklists_audio(self, mock_sync, mock_bl_path):
        mock_bl_path.__str__ = lambda x: self.bl_file
        with patch("Audio_Modules.rejected_audio_blacklist._BLACKLIST_FILE", self.bl_file):
            track_name = "telegram_DfX19qZx1gn_1793909978.wav"
            file_id = "BQACAgUAAyEGAAT"
            shortcode = "DfX19qZx1gn"

            # 1. Add rejected audio directly or via dict
            bl_add(
                audio_filename=track_name,
                audio_shortcode=shortcode,
                telegram_file_id=file_id,
                reason="admin_rejected",
                sync_to_vault=False
            )

            # 2. Verify is_blacklisted detects all forms
            self.assertTrue(is_blacklisted(track_name))
            self.assertTrue(is_blacklisted(f"vault_bgm_{shortcode}.wav"))
            self.assertTrue(is_blacklisted(f"extracted_{shortcode}.wav"))
            self.assertTrue(is_blacklisted(f"{shortcode}.wav"))
            self.assertTrue(is_blacklisted({"track_name": track_name}))
            self.assertTrue(is_blacklisted(None, audio_shortcode=shortcode))
            self.assertTrue(is_blacklisted(None, telegram_file_id=file_id))

            # Non-blacklisted track should pass
            self.assertFalse(is_blacklisted("completely_different_track.mp3"))

    @patch("Telegram_Storage_Modules.telegram_session_manager.SESSIONS_FILE")
    def test_session_manager_set_rejected_and_purger_integration(self, mock_sess_file):
        with patch("Telegram_Storage_Modules.telegram_session_manager.SESSIONS_FILE", self.sessions_file):
            sm = TelegramSessionManager()
            sm.sessions_file = self.sessions_file
            sess_id = "sess_1794068728_6695"
            clip_id = "manual_DfULIxos0kb"
            chosen_audio = "telegram_DfX19qZx1gn_1793909978.wav"

            # Create session
            sess_id = sm.create_session(
                video_path="dummy_path.mp4",
                clip_id=clip_id,
                selected_audio=chosen_audio
            )

            # Mark rejected
            rejected_sess = sm.set_rejected(sess_id)
            self.assertIsNotNone(rejected_sess)
            self.assertEqual(rejected_sess.get("status"), "REJECTED")
            self.assertEqual(rejected_sess.get("selected_audio"), chosen_audio)

            # Test purger recovers selected_audio and purges
            with patch("Audio_Modules.audio_pool_manager.get_pool_manager") as mock_gpm, \
                 patch("Audio_Modules.rejected_audio_blacklist.add") as mock_bl_add:
                mock_apm = MagicMock()
                mock_gpm.return_value = mock_apm

                purge_res = purge_full_clip_and_assets(
                    clip_id=clip_id,
                    selected_audio={"track_name": chosen_audio}
                )

                self.assertEqual(purge_res.get("status"), "success")
                # Verify pool manager purge was called with the recovered audio name
                mock_apm.purge_media_and_audio_entries.assert_called_once()
                call_args = mock_apm.purge_media_and_audio_entries.call_args
                self.assertEqual(call_args[1].get("selected_audio_name"), chosen_audio)

    def test_audio_usage_and_recency_tracking(self):
        """Verify harvested audio accumulates usage_count and last_used."""
        metadata = {
            "version": 1,
            "files": {
                "social_media_id": {
                    "https://www.instagram.com/reel/DfX19qZx1gn/": {
                        "shortcode": "dfx19qzx1gn",
                        "usage_count": 0,
                        "last_used": 0,
                        "audio_data": {
                            "usage_count": 0,
                            "last_used": 0
                        }
                    }
                }
            }
        }
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        apm = AudioPoolManager(base_dir=self.test_dir.name)
        apm.metadata_file = self.metadata_file
        apm.metadata = metadata

        # Use the audio
        apm.use_audio("telegram_dfx19qzx1gn.wav")

        entry = apm.metadata["files"]["social_media_id"]["https://www.instagram.com/reel/DfX19qZx1gn/"]
        self.assertGreater(entry["usage_count"], 0)
        self.assertGreater(entry["last_used"], 0)
        self.assertGreater(entry["audio_data"]["usage_count"], 0)
        self.assertGreater(entry["audio_data"]["last_used"], 0)


if __name__ == "__main__":
    unittest.main()
