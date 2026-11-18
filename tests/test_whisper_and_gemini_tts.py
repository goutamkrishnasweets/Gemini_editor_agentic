import inspect
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Audio_Modules import whisper_transcriber
from Audio_Modules import speech_boundary_detector
from Audio_Modules.narrator import TTSEngine
from Gemini_Modules import lyric_rhythm_aligner
from scripts import meeting_audio_pipeline


def test_whisper_defaults_are_tiny():
    """All Whisper entrypoints must default to 'tiny' since Gemini corrects transcripts."""
    assert whisper_transcriber.WHISPER_MODEL_SIZE == "tiny"
    assert lyric_rhythm_aligner.WHISPER_MODEL_SIZE == "tiny"

    sig_get = inspect.signature(speech_boundary_detector._get_whisper_model)
    assert sig_get.parameters["model_size"].default == "tiny"

    sig_meeting = inspect.signature(meeting_audio_pipeline.run_meeting_pipeline)
    assert sig_meeting.parameters["model_size"].default == "tiny"


def test_resolve_gemini_voice_mapping():
    tts = TTSEngine()
    assert tts._resolve_gemini_voice("Kore") == "Kore"
    assert tts._resolve_gemini_voice("puck") == "Puck"
    assert tts._resolve_gemini_voice("ml-IN-MidhunNeural") == "Puck"
    assert tts._resolve_gemini_voice("en-US-AvaNeural") == "Kore"


def test_gemini_tts_is_primary_and_converts_pcm_to_mp3(tmp_path):
    """Gemini TTS runs first, converts 24kHz PCM to valid MP3, and skips edge-tts fallback."""
    tts = TTSEngine()

    # 0.5s of 24kHz 16-bit mono PCM (24000 bytes)
    pcm_bytes = b"\x00\x10" * 12000
    fake_part = SimpleNamespace(
        inline_data=SimpleNamespace(data=pcm_bytes, mime_type="audio/L16;codec=pcm;rate=24000")
    )
    fake_resp = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[fake_part]))]
    )
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_resp

    with patch("Audio_Modules.narrator.TEMP_DIR", str(tmp_path)), \
         patch("google.genai.Client", return_value=fake_client), \
         patch.object(tts, "_amake_voiceover", new_callable=AsyncMock) as mock_edge:
        out_path = tts.generate_voiceover(
            text="Check out this incredible look.",
            voice="en-US-AvaNeural",
            filename="test_gemini_primary.mp3",
            api_key="dummy-key",
        )

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 500
    # Gemini TTS was called with the TTS model and AUDIO modality
    assert fake_client.models.generate_content.called
    call_kwargs = fake_client.models.generate_content.call_args.kwargs
    assert "tts" in call_kwargs["model"].lower()
    # Edge-TTS fallback must NOT have been called when Gemini TTS succeeded
    assert not mock_edge.called


def test_gemini_tts_falls_back_to_edge_tts_on_failure(tmp_path):
    """When Gemini TTS returns False, TTSEngine falls back to edge-tts."""
    tts = TTSEngine()

    async def fake_edge(text, voice, output_path, rate="+0%", pitch="+0Hz"):
        with open(output_path, "wb") as f:
            f.write(b"ID3" + b"\x00" * 2048)

    class _SyncFakeLoop:
        def is_closed(self):
            return False

        def run_until_complete(self, coro):
            try:
                coro.send(None)
            except StopIteration as stop:
                return stop.value

    with patch("Audio_Modules.narrator.TEMP_DIR", str(tmp_path)), \
         patch("Audio_Modules.narrator.asyncio.get_event_loop", return_value=_SyncFakeLoop()), \
         patch.object(tts, "_generate_gemini_tts", return_value=False) as mock_gemini, \
         patch.object(tts, "_generate_pocket_tts", return_value=False), \
         patch.object(tts, "_amake_voiceover", side_effect=fake_edge) as mock_edge:
        out_path = tts.generate_voiceover(
            text="Fallback test line.",
            voice="en-US-AvaNeural",
            filename="test_fallback.mp3",
        )

    assert mock_gemini.called
    assert mock_edge.called
    assert os.path.exists(out_path)
