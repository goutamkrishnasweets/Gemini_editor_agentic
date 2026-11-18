"""
Speech Boundary Detector (faster-whisper)

Extracts word-level timestamps and sentence boundaries from clip audio WAV files.
Identifies precise cut timestamps (word end + natural silence gaps) so Gemini and
FFmpeg synthesis can perform 100% clean, natural vocal cuts without clipping mid-word.
"""

import gc
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

# Limit OpenMP/MKL heap allocations to prevent mkl_malloc failures on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

logger = logging.getLogger("speech_boundary_detector")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_WHISPER_MODELS: Dict[str, Any] = {}
_WHISPER_LOCK = threading.Lock()


def _get_whisper_model(model_size: str = "tiny") -> Any:
    """Singleton model cache with bounded CPU threads to prevent memory leaks and MKL exhaustion."""
    with _WHISPER_LOCK:
        if model_size not in _WHISPER_MODELS:
            from faster_whisper import WhisperModel
            threads = min(4, os.cpu_count() or 4)
            logger.info(f"🎙️ [Speech Boundary] Initializing WhisperModel ({model_size}) with {threads} CPU threads...")
            _WHISPER_MODELS[model_size] = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=threads,
                num_workers=1
            )
        return _WHISPER_MODELS[model_size]


def extract_speech_boundaries(
    audio_wav_path: str,
    output_json_path: Optional[str] = None,
    model_size: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs faster-whisper on audio_wav_path to produce word-level timestamps,
    sentence structures, and safe cut boundaries.
    """
    model_size = model_size or os.getenv("WHISPER_MODEL_SIZE", "tiny")
    if not os.path.exists(audio_wav_path):
        logger.warning(f"⚠️ [Speech Boundary] WAV file not found: {audio_wav_path}")
        return {"words": [], "sentences": [], "clean_cut_timestamps": [], "has_speech": False}

    if output_json_path and os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("has_speech") is not None:
                    logger.info(f"⚡ [Speech Boundary] Loaded cached boundaries: {output_json_path}")
                    return data
        except Exception:
            pass

    # Attempt transcription with primary model, fallback to 'tiny' if memory limits hit
    models_to_try = [model_size]
    if model_size != "tiny":
        models_to_try.append("tiny")

    last_err = None
    for attempt_model in models_to_try:
        try:
            model = _get_whisper_model(attempt_model)
            logger.info(f"🎙️ [Speech Boundary] Running faster-whisper ({attempt_model}, beam=1, VAD=on) on: {os.path.basename(audio_wav_path)}")

            # Use beam_size=1 (greedy) and vad_filter=True to prevent MKL memory explosion
            segments, info = model.transcribe(
                audio_wav_path,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                beam_size=1,
                best_of=1
            )

            all_words: List[Dict[str, Any]] = []
            sentences: List[Dict[str, Any]] = []
            clean_cuts: List[Dict[str, Any]] = []

            for segment in segments:
                seg_text = segment.text.strip()
                seg_start = round(float(segment.start), 3)
                seg_end = round(float(segment.end), 3)

                current_sentence_words = []
                for w in (segment.words or []):
                    w_str = w.word.strip()
                    w_start = round(float(w.start), 3)
                    w_end = round(float(w.end), 3)
                    w_conf = round(float(w.probability), 2)

                    word_obj = {
                        "word": w_str,
                        "start": w_start,
                        "end": w_end,
                        "confidence": w_conf
                    }
                    all_words.append(word_obj)
                    current_sentence_words.append(word_obj)

                    # Check for sentence punctuation
                    is_punct = any(p in w_str for p in [".", "?", "!", ","])
                    if is_punct:
                        clean_cuts.append({
                            "timestamp_sec": w_end,
                            "word_after_which_to_cut": w_str,
                            "reason": "sentence_punctuation"
                        })

                sentences.append({
                    "text": seg_text,
                    "start": seg_start,
                    "end": seg_end,
                    "words": current_sentence_words
                })

                # Always add segment end as a valid cut point
                clean_cuts.append({
                    "timestamp_sec": seg_end,
                    "word_after_which_to_cut": current_sentence_words[-1]["word"] if current_sentence_words else "",
                    "reason": "segment_end"
                })

            # Deduplicate and sort cut points by timestamp
            seen_ts = set()
            unique_cuts = []
            for c in sorted(clean_cuts, key=lambda x: x["timestamp_sec"]):
                ts = c["timestamp_sec"]
                if ts not in seen_ts:
                    seen_ts.add(ts)
                    unique_cuts.append(c)

            result = {
                "has_speech": len(all_words) > 0,
                "language": info.language if hasattr(info, "language") else "en",
                "duration_sec": round(float(info.duration), 2) if hasattr(info, "duration") else 0.0,
                "words": all_words,
                "sentences": sentences,
                "clean_cut_timestamps": unique_cuts,
                "extracted_at": time.time(),
                "model_used": attempt_model
            }

            if output_json_path:
                os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"✅ [Speech Boundary] Saved speech boundaries ({len(all_words)} words, {len(unique_cuts)} cut points) -> {output_json_path}")

            return result

        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            gc.collect()
            if "mkl_malloc" in err_str or "memory" in err_str or isinstance(e, MemoryError):
                logger.warning(f"⚠️ [Speech Boundary] Memory pressure on model '{attempt_model}': {e}. Attempting fallback...")
                continue
            else:
                logger.error(f"❌ [Speech Boundary] Extraction error on '{attempt_model}': {e}")
                break

    logger.warning(f"⚠️ [Speech Boundary] Extraction failed on all models: {last_err}. Returning empty boundaries.")
    return {"words": [], "sentences": [], "clean_cut_timestamps": [], "has_speech": False, "error": str(last_err)}
