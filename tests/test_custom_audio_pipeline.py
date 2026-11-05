"""
tests/test_custom_audio_pipeline.py
===================================
Unit tests for User-Provided Custom Audio Pipeline:
  1. Agentic_Core/goal_router.py: Dual-URL & /audio parsing.
  2. Audio_Modules/custom_audio_extractor.py: Multi-tier fail-safe extraction & fallback.
  3. Phase_2/04_bgm_selector.py: Tier 0 bypass when custom audio is present (zero token waste).
  4. Phase_2/05_rhythm_timeline.py:
       - Scenario A (Audio < Video): Micro-reel dynamic target duration condensation.
       - Scenario B (Audio >= Video): Golden sliding window calculation.
       - Short custom audio bypass (<10s allowed).
  5. Phase_2/06_ffmpeg_synthesis.py: Propagates target_duration and audio_start_offset into filtergraph.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Agentic_Core.goal_router import classify_goal, Route
from Audio_Modules.custom_audio_extractor import CustomAudioExtractor
import importlib
step04 = importlib.import_module("Phase_2.04_bgm_selector")
step05 = importlib.import_module("Phase_2.05_rhythm_timeline")
step06 = importlib.import_module("Phase_2.06_ffmpeg_synthesis")


class TestGoalRouterCustomAudio(unittest.TestCase):
    """Verifies that goal_router accurately classifies video URLs with custom audio tracks."""

    def test_single_url_no_audio(self):
        goal = "https://www.instagram.com/reel/C123456789/"
        route = classify_goal(goal)
        self.assertEqual(route.kind, "url")
        self.assertEqual(route.url, "https://www.instagram.com/reel/C123456789/")
        self.assertIsNone(route.custom_audio_url)
        self.assertEqual(route.directive, "")

    def test_dual_url_implicit(self):
        goal = "https://www.instagram.com/reel/VideoClip123/ https://www.instagram.com/reel/AudioTrack456/"
        route = classify_goal(goal)
        self.assertEqual(route.kind, "url")
        self.assertEqual(route.url, "https://www.instagram.com/reel/VideoClip123/")
        self.assertEqual(route.custom_audio_url, "https://www.instagram.com/reel/AudioTrack456/")
        self.assertEqual(route.directive, "")

    def test_slash_audio_tag(self):
        goal = "https://www.instagram.com/reel/VideoClip123/ /audio https://www.instagram.com/reel/AudioTrack456/ make it snappy"
        route = classify_goal(goal)
        self.assertEqual(route.kind, "url")
        self.assertEqual(route.url, "https://www.instagram.com/reel/VideoClip123/")
        self.assertEqual(route.custom_audio_url, "https://www.instagram.com/reel/AudioTrack456/")
        self.assertEqual(route.directive, "make it snappy")

    def test_upload_with_slash_audio(self):
        goal = "process uploaded video at 'D:/videos/test.mp4' /audio https://www.instagram.com/reel/AudioTrack456/"
        route = classify_goal(goal)
        self.assertEqual(route.kind, "upload")
        self.assertEqual(route.path, "D:/videos/test.mp4")
        self.assertEqual(route.custom_audio_url, "https://www.instagram.com/reel/AudioTrack456/")


class TestCustomAudioExtractor(unittest.TestCase):
    """Verifies multi-tier extraction behavior and non-blocking fallback to None."""

    def setUp(self):
        self.extractor = CustomAudioExtractor()

    def test_empty_or_invalid_url(self):
        self.assertIsNone(self.extractor.extract_custom_audio(""))
        self.assertIsNone(self.extractor.extract_custom_audio(None))

    @patch("Telegram_Storage_Modules.telegram_vault_indexer.TelegramVaultIndexer")
    def test_tier1_vault_first_look_success(self, mock_vault_cls):
        mock_vault = MagicMock()
        mock_vault_cls.return_value = mock_vault
        mock_vault.hydrate_extracted_audio_from_vault.return_value = "D:/dummy/vault_audio.wav"

        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=50_000), \
             patch.object(self.extractor, "_sanitize_to_standard_wav", return_value="D:/dummy/user_custom_audio.wav"):
            res = self.extractor.extract_custom_audio("https://www.instagram.com/reel/Existing123/")
            self.assertEqual(res, "D:/dummy/user_custom_audio.wav")

    def test_all_tiers_fail_returns_none(self):
        with patch.object(self.extractor, "_extract_via_ytdlp", return_value=None), \
             patch.object(self.extractor, "_extract_via_apify", return_value=None):
            res = self.extractor.extract_custom_audio("https://www.instagram.com/reel/NonExistent999/")
            self.assertIsNone(res, "When all extraction tiers fail, must return None (fail-safe to Gemini pool)")


class TestBgmSelectorBypass(unittest.TestCase):
    """Verifies that Step 04 bypasses Gemini Call 2 BGM Selector when custom audio is present."""

    def test_custom_audio_path_bypass(self):
        dummy_audio = "D:/dummy/custom_audio.wav"
        with patch("os.path.isfile", return_value=True), \
             patch("os.path.getsize", return_value=60_000):
            res = step04.select_clip_bgm(
                clip_id="test_clip",
                custom_audio_path=dummy_audio
            )
            self.assertEqual(res["status"], "user_provided")
            self.assertTrue(res["is_user_custom_audio"])
            self.assertEqual(res["selected_audio_track"], "custom_audio.wav")
            self.assertEqual(res["physical_path"], os.path.abspath(dummy_audio))

    def test_auto_detect_user_custom_audio_in_clip_folder(self):
        with patch("os.path.isfile", side_effect=lambda p: "user_custom_audio.wav" in p), \
             patch("os.path.getsize", return_value=45_000):
            res = step04.select_clip_bgm(
                clip_id="test_clip",
                clip_folder="D:/dummy/clip_folder"
            )
            self.assertEqual(res["status"], "user_provided")
            self.assertTrue(res["is_user_custom_audio"])
            self.assertEqual(res["selected_audio_track"], "user_custom_audio.wav")


class TestRhythmTimelineScenarios(unittest.TestCase):
    """Verifies Scenario A (Audio < Video) and Scenario B (Audio >= Video) dynamic timeline direction."""

    @patch("Phase_2.05_rhythm_timeline.analyze_music")
    @patch("Phase_2.05_rhythm_timeline.compute_routing_parameters")
    def test_scenario_a_audio_shorter_than_video(self, mock_routing, mock_analyze):
        mock_analyze.return_value = {"tempo_bpm": 128.0, "is_unusable": False, "emotional_peak_moments": [2.0, 5.0]}
        mock_routing.return_value = {"speed_factor": 1.25, "recommended_editing_mode": "hype"}

        # Video = 24.0s, Audio = 8.5s
        def mock_dur(path):
            if "custom" in path:
                return 8.5
            return 24.0

        with patch("os.path.isfile", return_value=True), \
             patch("Audio_Modules.rhythm_timeline_builder.RhythmTimelineBuilder._get_duration", side_effect=mock_dur), \
             patch("Audio_Modules.rhythm_timeline_builder.RhythmTimelineBuilder.build_timeline", return_value=[{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.5}]):

            res = step05.build_rhythm_timeline(
                video_path="D:/dummy/video.mp4",
                selected_bgm_path="D:/dummy/user_custom_audio.wav",
                forensic_context={"is_user_custom_audio": True}
            )

            # Scenario A: Target duration must match short audio (8.5s), offset must be 0.0s
            self.assertEqual(res["target_duration"], 8.5)
            self.assertEqual(res["audio_start_offset"], 0.0)
            self.assertTrue(res["is_user_custom_audio"])

    @patch("Phase_2.05_rhythm_timeline.analyze_music")
    @patch("Phase_2.05_rhythm_timeline.compute_routing_parameters")
    def test_scenario_b_audio_longer_than_video(self, mock_routing, mock_analyze):
        # Audio = 45.0s, Video = 15.0s, Chorus starts at 14.0s
        mock_analyze.return_value = {
            "tempo_bpm": 120.0,
            "is_unusable": False,
            "sections": [
                {"type": "verse", "start": 0.0, "end": 14.0, "energy": 0.4},
                {"type": "chorus", "start": 14.0, "end": 29.0, "energy": 0.95}
            ]
        }
        mock_routing.return_value = {"speed_factor": 1.0, "recommended_editing_mode": "hype"}

        def mock_dur(path):
            if "custom" in path or "bgm" in path:
                return 45.0
            return 15.0

        with patch("os.path.isfile", return_value=True), \
             patch("Audio_Modules.rhythm_timeline_builder.RhythmTimelineBuilder._get_duration", side_effect=mock_dur), \
             patch("Audio_Modules.rhythm_timeline_builder.RhythmTimelineBuilder.build_timeline", return_value=[{"start": 0.0, "end": 15.0}]):

            res = step05.build_rhythm_timeline(
                video_path="D:/dummy/video.mp4",
                selected_bgm_path="D:/dummy/user_custom_audio.wav",
                forensic_context={"is_user_custom_audio": True}
            )

            # Scenario B: Target duration is video length (15.0s), Golden sliding window snaps to chorus (14.0s)
            self.assertEqual(res["target_duration"], 15.0)
            self.assertEqual(res["audio_start_offset"], 14.0)

    @patch("Phase_2.05_rhythm_timeline.analyze_music")
    @patch("Phase_2.05_rhythm_timeline.compute_routing_parameters")
    def test_short_custom_audio_not_rejected(self, mock_routing, mock_analyze):
        # Custom audio is only 7.0s (less than the standard 10.0s reject threshold)
        mock_analyze.return_value = {"tempo_bpm": 120.0, "is_unusable": False}
        mock_routing.return_value = {"speed_factor": 1.0}

        def mock_dur(path):
            if "custom" in path:
                return 7.0
            return 20.0

        with patch("os.path.isfile", return_value=True), \
             patch("Audio_Modules.rhythm_timeline_builder.RhythmTimelineBuilder._get_duration", side_effect=mock_dur), \
             patch("Audio_Modules.rhythm_timeline_builder.RhythmTimelineBuilder.build_timeline", return_value=[{"start": 0.0, "end": 7.0}]):

            res = step05.build_rhythm_timeline(
                video_path="D:/dummy/video.mp4",
                selected_bgm_path="D:/dummy/user_custom_audio.wav",
                forensic_context={"is_user_custom_audio": True}
            )

            # Must NOT be auto-rejected or swapped!
            self.assertEqual(res["selected_bgm_path"], "D:/dummy/user_custom_audio.wav")
            self.assertEqual(res["target_duration"], 7.0)


class TestSynthesisOffsetPropagation(unittest.TestCase):
    """Verifies that audio_start_offset propagates through step06 into extra_inputs."""

    @patch("Gemini_Modules.gemini_ffmpeg_synthesis.GeminiFFmpegEngine.run_full_pipeline")
    def test_audio_offset_passed_to_extra_inputs(self, mock_run_full):
        mock_run_full.return_value = {"status": "SUCCESS"}

        with patch("os.path.exists", return_value=True):
            step06.synthesize_editing_plan(
                video_path="D:/dummy/video.mp4",
                output_path="D:/dummy/master.mp4",
                selected_bgm_path="D:/dummy/user_custom_audio.wav",
                target_duration=9.0,
                audio_start_offset=4.5
            )

            self.assertTrue(mock_run_full.called)
            kwargs = mock_run_full.call_args[1]
            extra = kwargs.get("extra_inputs", {})
            self.assertEqual(extra.get("audio_start_offset"), 4.5)
            self.assertEqual(extra.get("audio_start_time"), 4.5)
            self.assertEqual(extra.get("target_duration"), 9.0)


if __name__ == "__main__":
    unittest.main()
