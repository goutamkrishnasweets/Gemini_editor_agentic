import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Gemini_Modules.gemini_ffmpeg_synthesis import FFmpegCommandGenerator, GeminiFFmpegEngine
from main import ACTIVE_PIPELINE_TARGETS, _dispatch_pipeline_in_background


class TestSynthesisAtomicAndAudioPropagation(unittest.TestCase):

    def test_audio_propagation_into_single_pass_filtergraph(self):
        """Test that extra_inputs['audio'] is automatically picked up by build_single_pass_filtergraph."""
        cmd_gen = FFmpegCommandGenerator()
        dummy_video = os.path.join(_REPO_ROOT, "input_test.mp4")
        dummy_audio = os.path.join(_REPO_ROOT, "bgm_test.wav")

        # Fake existence of dummy_video and dummy_audio for the test
        with patch("os.path.exists", side_effect=lambda p: True), \
             patch("os.path.isfile", side_effect=lambda p: True), \
             patch.object(cmd_gen, "_get_media_duration", return_value=15.0), \
             patch.object(cmd_gen, "_has_audio_stream", return_value=True):

            res = cmd_gen.build_single_pass_filtergraph(
                input_path=dummy_video,
                output_path="out.mp4",
                micro_shots=[{"start": 0.0, "end": 10.0}],
                bgm_path=None,  # Not passed directly!
                extra_inputs={"audio": dummy_audio}  # Passed via extra_inputs
            )

            # Check that dummy_audio was added as an input [-i, dummy_audio]
            cmd = res["cmd_list"]
            self.assertIn("-i", cmd)
            self.assertIn(dummy_audio, cmd)
            # Verify faststart flag is present in final encoder flags
            self.assertIn("-movflags", cmd)
            self.assertIn("+faststart", cmd)

    def test_duration_lock_repair_includes_faststart(self):
        """Test that _enforce_duration_lock includes -movflags +faststart when remuxing drift."""
        engine = GeminiFFmpegEngine()
        with patch("os.path.isfile", return_value=True), \
             patch("os.path.exists", return_value=True), \
             patch("subprocess.run") as mock_subproc:

            # Return video duration 10.0s, audio duration 12.0s (>50ms drift)
            mock_subproc.side_effect = [
                # probe_src
                MagicMock(stdout='{"format": {"duration": "10.0"}}'),
                # probe_v
                MagicMock(stdout='{"format": {"duration": "10.0"}}'),
                # probe_a
                MagicMock(stdout='{"streams": [{"duration": "12.0"}]}'),
                # repair_cmd
                MagicMock(returncode=0)
            ]

            engine._enforce_duration_lock("src.mp4", "out.mp4")

            # Check repair_cmd arguments
            repair_call_args = mock_subproc.call_args[0][0]
            self.assertIn("-movflags", repair_call_args)
            self.assertIn("+faststart", repair_call_args)
            self.assertIn("-shortest", repair_call_args)

    def test_pipeline_dispatch_deduplication(self):
        """Test that duplicate URLs in _dispatch_pipeline_in_background are dropped."""
        import threading
        ACTIVE_PIPELINE_TARGETS.clear()
        target_url = "https://www.instagram.com/reel/DZW2MoYo7GC/"
        gate_event = threading.Event()
        release_event = threading.Event()

        def _blocking_run(**kwargs):
            gate_event.set()
            release_event.wait(timeout=2.0)

        # Mock run_master_pipeline and send_telegram_text_sync
        with patch("main.run_master_pipeline", side_effect=_blocking_run), \
             patch("main.send_telegram_text_sync") as mock_notify:

            # First dispatch
            _dispatch_pipeline_in_background(mode="manual", url=target_url, requestor_chat_id=12345)
            self.assertTrue(gate_event.wait(timeout=1.0))
            self.assertIn("url_https://www.instagram.com/reel/dzw2moyo7gc", ACTIVE_PIPELINE_TARGETS)

            # Second duplicate dispatch while first is active
            _dispatch_pipeline_in_background(mode="manual", url=target_url, requestor_chat_id=12345)

            # Second call should have sent notice and dropped execution
            mock_notify.assert_called_once()
            self.assertIn("already currently being edited", mock_notify.call_args[0][1])

            # Release first worker
            release_event.set()

        # Cleanup
        ACTIVE_PIPELINE_TARGETS.clear()


if __name__ == "__main__":
    unittest.main()
