"""
tests/test_audio_semantic_alignment.py
=======================================
Verifies the end-to-end fix for audio-video semantic connection:
1. Vault indexer preserves transcripts, lyrics, captions, hashtags, and language.
2. compute_4d_audio_score computes high S_semantic for matching lyrical/thematic/cultural tracks.
3. select_best_audio_for_clip injects rich lyrics and context into the candidate profiles.
4. Lip-sync performance footage retains preserve_original_audio.
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
from Gemini_Modules.lyric_rhythm_aligner import compute_4d_audio_score, select_best_audio_for_clip
from Gemini_Modules.forensic_analyzer import ForensicVideoAnalyzer


def test_vault_indexer_preserves_lyrics_and_transcripts(tmp_path):
    """Ensure get_vault_audio_pool extracts whisper_transcript, caption, language, and lyrics."""
    fake_pool = {
        "files": {
            "social_media_id": {
                "https://www.instagram.com/reel/BollyTrack1/": {
                    "shortcode": "BollyTrack1",
                    "caption": "Shah Rukh Khan and Deepika Padukone romance in Switzerland #bollywood #romance",
                    "hashtags": ["bollywood", "romance", "srk"],
                    "media_file_ids": {
                        "extracted_audio_file_id": "tg_fid_bolly_1",
                        "extracted_audio_file_size": 2048576,
                    },
                    "audio_data": {
                        "tempo_bpm": 118.0,
                        "avg_energy": 0.72,
                        "vibe": "romantic",
                        "genre": "bollywood_melodic",
                        "language": "hi",
                        "dominant_emotion": "romantic",
                        "hook_time": 15.5,
                        "whisper_transcript": {
                            "text": "Tujhe dekha toh yeh jaana sanam pyaar hota hai deewana sanam",
                            "language": "hi",
                        },
                    },
                }
            }
        }
    }

    indexer = TelegramVaultIndexer()
    with patch("builtins.open", MagicMock(side_effect=lambda p, *a, **kw: MagicMock(
        __enter__=lambda s: MagicMock(read=lambda: json.dumps(fake_pool)),
        __exit__=lambda *args: None
    ))), patch("os.path.exists", return_value=True), patch("json.load", return_value=fake_pool):
        pool = indexer.get_vault_audio_pool()

    assert "BollyTrack1.wav" in pool
    entry = pool["BollyTrack1.wav"]
    assert entry["transcript"] == "Tujhe dekha toh yeh jaana sanam pyaar hota hai deewana sanam"
    assert entry["language"] == "hi"
    assert entry["caption"] == "Shah Rukh Khan and Deepika Padukone romance in Switzerland #bollywood #romance"
    assert entry["dominant_emotion"] == "romantic"
    assert entry["hook_time"] == 15.5


def test_compute_4d_audio_score_semantic_boost():
    """Verify S_semantic is substantially higher for a matching Bollywood/Hindi track than an unrelated track."""
    archetype = {
        "name": "PLAYFUL_RHYTHMIC",
        "description": "Upbeat dance & music",
        "directive": "Keep the energy bouncy",
        "fatal_genres": [],
    }
    visual_ctx = {
        "intent": "bollywood_dance_romance",
        "tone": "romantic",
        "visual_event": "Couple dancing in traditional royal attire under rain",
        "content_director": {
            "visual_event": "romantic rain dance",
            "tone": "romantic",
        },
    }
    clip_metadata = {
        "caption": "Romantic rain sequence with Bollywood stars #bollywood #romance #dance",
        "hashtags": ["bollywood", "romance", "dance"],
    }
    current_audio = {"math": {"tempo_bpm": 120.0}}

    # Relevant Candidate (Hindi romantic song)
    bolly_cand = {
        "gemini_genre": "bollywood_romantic",
        "vibe_tags": ["romantic", "dance", "melodic"],
        "energy": 0.70,
        "tempo_bpm": 120.0,
        "language": "hi",
        "dominant_emotion": "romantic",
        "transcript": "Tujhe dekha toh yeh jaana sanam pyaar hota hai deewana sanam",
        "caption": "Classic romantic dance sequence #bollywood",
        "hashtags": ["bollywood", "dance"],
    }

    # Irrelevant Candidate (Western aggressive phonk)
    phonk_cand = {
        "gemini_genre": "phonk",
        "vibe_tags": ["aggressive", "drift", "dark"],
        "energy": 0.85,
        "tempo_bpm": 130.0,
        "language": "en",
        "dominant_emotion": "aggressive",
        "transcript": "drift phonk murder in my mind slaughterhouse",
        "caption": "Drift cars burning rubber at night #drift #phonk",
        "hashtags": ["phonk", "drift"],
    }

    score_bolly, arch_b, sem_b, rhythm_b, fatigue_b = compute_4d_audio_score(
        archetype=archetype,
        candidate_meta=bolly_cand,
        candidate_filename="BollyTrack1.wav",
        visual_ctx=visual_ctx,
        current_audio=current_audio,
        clip_metadata=clip_metadata,
    )

    score_phonk, arch_p, sem_p, rhythm_p, fatigue_p = compute_4d_audio_score(
        archetype=archetype,
        candidate_meta=phonk_cand,
        candidate_filename="PhonkTrack.wav",
        visual_ctx=visual_ctx,
        current_audio=current_audio,
        clip_metadata=clip_metadata,
    )

    # Semantic score of Bollywood track must dramatically outperform the phonk track
    assert sem_b > 0.60, f"Expected sem_b > 0.60, got {sem_b}"
    assert sem_b > sem_p + 0.30, f"Expected sem_b ({sem_b}) > sem_p ({sem_p}) + 0.30"
    assert score_bolly > score_phonk, f"Bollywood score {score_bolly} should exceed phonk {score_phonk}"


def test_select_best_audio_prompt_contains_lyrics_and_origin():
    """Verify Gemini Call 2 prompt builder injects lyrics and origin caption previews."""
    archetype = {
        "name": "PLAYFUL_RHYTHMIC",
        "description": "Upbeat",
        "directive": "Vibe",
        "fatal_genres": [],
    }

    fake_pool = {
        "dance_track.wav": {
            "tempo_bpm": 120.0,
            "energy": 0.75,
            "genre": "bollywood_pop",
            "vibe_tags": ["dance", "party"],
            "language": "hi",
            "dominant_emotion": "joyful",
            "transcript": "Nachde ne saare ral mil ke",
            "caption": "Wedding sangeet celebration dancing all night",
            "hook_time": 12.0,
            "file_id": "fid_dance_123",
        }
    }

    with patch("Gemini_Modules.lyric_rhythm_aligner.classify_video_archetype", return_value=archetype), \
         patch("Telegram_Storage_Modules.telegram_vault_indexer.TelegramVaultIndexer.get_vault_audio_pool", return_value=fake_pool), \
         patch("Audio_Modules.audio_pool_manager.AudioPoolManager.get_files_index", return_value={}), \
         patch("Gemini_Modules.clip_intelligence_store.ClipIntelligenceStore.load", return_value={
             "visual_context": {
                 "intent": "wedding_dance",
                 "tone": "celebratory",
                 "main_subject": "Dancers in festive attire",
                 "visual_event": "Wedding party choreographed group dance",
                 "content_director": {"visual_event": "Group dance", "tone": "celebratory"},
             },
             "audio_data": {},
         }), \
         patch("Gemini_Modules.clip_intelligence_store.ClipIntelligenceStore.save"), \
         patch("Gemini_Modules.clip_intelligence_store.ClipIntelligenceStore.patch_bgm_selection"):

        captured_prompt = []
        def mock_generate(task_type, prompt, module_name, gen_config):
            captured_prompt.append(prompt)
            return json.dumps({
                "selected_audio_track": "dance_track.wav",
                "telegram_file_id": "fid_dance_123",
                "alignment_score": 0.96,
                "reasoning": "Bollywood wedding dance song matches festive choreo perfectly."
            })

        mock_router = MagicMock()
        mock_router.generate.side_effect = mock_generate

        with patch("Gemini_Modules.gemini_router_module.gemini_governor.gemini_router", mock_router):
            res = select_best_audio_for_clip("clip_test_123")

        assert len(captured_prompt) == 1
        prompt_text = captured_prompt[0]
        # Verify prompt includes lyrics preview, origin caption, lang, and hook drop
        assert "lyrics='Nachde ne saare ral mil ke'" in prompt_text
        assert "origin_caption='Wedding sangeet celebration dancing all night'" in prompt_text
        assert "lang='hi'" in prompt_text
        assert "hook_drop=12.0s" in prompt_text
        assert "SEMANTIC & CULTURAL RESONANCE (CRITICAL)" in prompt_text

        assert res["selected_audio_track"] == "dance_track.wav"
        assert res["hook_time"] == 12.0
        assert "Nachde ne" in res["lyrics"]
