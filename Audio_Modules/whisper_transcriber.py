"""
Audio_Modules/whisper_transcriber.py
====================================
STAGE 1 of the transcript pipeline.

    raw audio  ->  faster-whisper  ->  draft transcript (segments + word timestamps)

This module does ONE thing: produce a draft transcript. It does not try to be
right. Stage 2 (transcript_corrector.py) listens to the same audio and fixes it.

What it adds on top of the old speech_boundary_detector.py:
  - Model is loaded ONCE per process (the old code reloaded it on every call).
  - Per-segment quality signals (avg_logprob, no_speech_prob, compression_ratio)
    and a `suspect` flag, so Stage 2 knows where Whisper is probably lying.
  - Per-word confidence is kept, so Stage 2 can mark low-confidence words.
  - condition_on_previous_text=False: stops Whisper's repetition/hallucination
    loops from snowballing across segments (very common on music).
  - Cache is keyed on audio content hash + model size, not just "file exists".

Never raises. On any failure returns an empty result with an "error" key.
"""

import hashlib
import json
import logging
import os
import threading
from typing import Any, Dict, Optional

# Limit OpenMP/MKL heap allocations to prevent mkl_malloc failures on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

logger = logging.getLogger("whisper_transcriber")

# ─── Config ───────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")   # tiny|base|small|medium|large-v3
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

_MIN_AUDIO_BYTES = 4096
_SCHEMA_VERSION = 2

# Thresholds for the `suspect` flag on a segment
_SUSPECT_NO_SPEECH_PROB = 0.6
_SUSPECT_COMPRESSION_RATIO = 2.4
_SUSPECT_AVG_LOGPROB = -1.0

# ─── Model cache (load once per process) ──────────────────────────────────────
_model_cache: Dict[tuple, Any] = {}
_model_lock = threading.Lock()


def _get_model(size: str, device: str, compute_type: str):
    key = (size, device, compute_type)
    with _model_lock:
        model = _model_cache.get(key)
        if model is None:
            from faster_whisper import WhisperModel
            threads = min(4, os.cpu_count() or 4)
            logger.info(f"🎙️ [WHISPER] Loading model '{size}' ({device}/{compute_type}, threads={threads})...")
            model = WhisperModel(size, device=device, compute_type=compute_type, cpu_threads=threads)
            _model_cache[key] = model
        return model


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _empty_result(model_size: str, error: str = "") -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "has_speech": False,
        "language": "unknown",
        "language_probability": 0.0,
        "duration_sec": 0.0,
        "model_size": model_size,
        "segments": [],
        "full_text": "",
        "_schema": _SCHEMA_VERSION,
    }
    if error:
        res["error"] = error
    return res


# ─── Main API ─────────────────────────────────────────────────────────────────

def transcribe_audio(
    audio_path: str,
    output_json_path: Optional[str] = None,
    *,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Transcribe `audio_path` with faster-whisper.

    Args:
        audio_path:       Path to a WAV/MP3/M4A file.
        output_json_path: Optional cache/output path for the result JSON.
        model_size:       Overrides WHISPER_MODEL_SIZE.
        language:         Force a language code ("hi", "en", ...). None = auto-detect.
                          Forcing it is the single best fix for Hindi/Hinglish misdetection.
        initial_prompt:   Optional vocabulary/context hint (names, song title, etc).
        force:            Ignore the cache.

    Returns:
        {
          "has_speech": bool, "language": str, "language_probability": float,
          "duration_sec": float, "model_size": str, "full_text": str,
          "segments": [
            {"id": int, "start": float, "end": float, "text": str,
             "confidence": float, "avg_logprob": float, "no_speech_prob": float,
             "compression_ratio": float, "suspect": bool,
             "words": [{"word": str, "start": float, "end": float, "confidence": float}]}
          ]
        }
    """
    size = model_size or WHISPER_MODEL_SIZE

    if not audio_path or not os.path.exists(audio_path):
        logger.warning(f"⚠️ [WHISPER] Audio not found: {audio_path}")
        return _empty_result(size, "audio_not_found")
    if os.path.getsize(audio_path) < _MIN_AUDIO_BYTES:
        logger.warning(f"⚠️ [WHISPER] Audio too small: {audio_path}")
        return _empty_result(size, "audio_too_small")

    try:
        audio_sha1 = _sha1_file(audio_path)

        # Cache hit only if audio content, model, and language request all match
        if output_json_path and not force and os.path.exists(output_json_path):
            try:
                with open(output_json_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if (
                    cached.get("_schema") == _SCHEMA_VERSION
                    and cached.get("audio_sha1") == audio_sha1
                    and cached.get("model_size") == size
                    and cached.get("requested_language") == language
                ):
                    logger.info(f"⚡ [WHISPER] Cache hit: {output_json_path}")
                    return cached
            except Exception:
                pass

        model = _get_model(size, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
        seg_iter, info = model.transcribe(
            audio_path,
            language=language,
            initial_prompt=initial_prompt,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
        )

        segments = []
        for i, seg in enumerate(seg_iter):  # generator — must be consumed here
            words = []
            for w in (seg.words or []):
                token = (w.word or "").strip()
                if not token:
                    continue
                words.append({
                    "word": token,
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    "confidence": round(float(w.probability), 3),
                })

            text = (seg.text or "").strip()
            if not text:
                continue

            if words:
                confidence = sum(w["confidence"] for w in words) / len(words)
            else:
                confidence = 2.718281828 ** float(seg.avg_logprob)  # exp(avg_logprob)

            avg_logprob = float(seg.avg_logprob)
            no_speech = float(seg.no_speech_prob)
            comp_ratio = float(seg.compression_ratio)
            suspect = (
                no_speech > _SUSPECT_NO_SPEECH_PROB
                or comp_ratio > _SUSPECT_COMPRESSION_RATIO
                or avg_logprob < _SUSPECT_AVG_LOGPROB
            )

            segments.append({
                "id": len(segments),
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": text,
                "confidence": round(min(1.0, confidence), 3),
                "avg_logprob": round(avg_logprob, 3),
                "no_speech_prob": round(no_speech, 3),
                "compression_ratio": round(comp_ratio, 3),
                "suspect": bool(suspect),
                "words": words,
            })

        result = {
            "has_speech": len(segments) > 0,
            "language": getattr(info, "language", "unknown"),
            "language_probability": round(float(getattr(info, "language_probability", 0.0)), 3),
            "duration_sec": round(float(getattr(info, "duration", 0.0)), 2),
            "model_size": size,
            "requested_language": language,
            "audio_sha1": audio_sha1,
            "full_text": " ".join(s["text"] for s in segments),
            "segments": segments,
            "_schema": _SCHEMA_VERSION,
        }

        if output_json_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
                tmp = output_json_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                os.replace(tmp, output_json_path)
            except Exception as _we:
                logger.warning(f"[WHISPER] Failed to persist transcript: {_we}")

        logger.info(
            f"✅ [WHISPER] {os.path.basename(audio_path)}: {len(segments)} segments, "
            f"lang={result['language']} ({result['language_probability']:.2f}), "
            f"suspect={sum(1 for s in segments if s['suspect'])}"
        )
        return result

    except Exception as e:
        logger.error(f"❌ [WHISPER] Transcription failed: {e}")
        return _empty_result(size, str(e))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("usage: python whisper_transcriber.py <audio_file> [language]")
        raise SystemExit(1)
    out = transcribe_audio(sys.argv[1], language=sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps(out, ensure_ascii=False, indent=2))
