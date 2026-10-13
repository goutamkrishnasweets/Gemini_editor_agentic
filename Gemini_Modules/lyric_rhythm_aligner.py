"""
Audio_Modules/lyric_rhythm_aligner.py
======================================
Musical Intelligence Report — ONE Gemini call, maximum value.

Gemini receives the raw BGM audio file PLUS two grounding-context blocks —
machine-computed beat/tempo math and a word-level ASR transcript — and returns
a full structured intelligence report that drives ALL downstream rhythm
editing decisions:

    1. Lyric timestamps + emotional weight per word/phrase
    2. Section map   (intro / verse / pre-chorus / chorus / drop / bridge / outro)
    3. Tension arc   (0-1 score per second — drives hold vs. cut decisions)
    4. Shot directives (what visual to use at which moment)
    5. Vibe tags     (feeds CreativeBrain niche alignment)
    6. Emotional peak moments (single timestamps for instant-cut triggers)

Why ONE call? Because all 6 outputs share the same audio context window —
splitting them into 6 calls would cost 6x the quota for the same source material.

Controlled by: ENABLE_LYRIC_SYNC=true (default: true)
Gracefully returns empty structure on failure / instrumental audio.

────────────────────────────────────────────────────────────────────────────
CHANGELOG (fusion-precision rewrite)
────────────────────────────────────────────────────────────────────────────
This revision fixes a gap between intent and implementation: the module was
computing beat/tempo math (STEP 1) and running faster-whisper (STEP 2), but
the math was NEVER included in the Gemini prompt — it was only spliced into
the output *after* Gemini had already responded blind. Only sentence-level
(not word-level) ASR text was passed in, capped at 25 lines. The Gemini call
also had a hard 15s wall-clock timeout that guarantees fallback on any real
audio-understanding + large-JSON-generation task, which was almost certainly
the dominant cause of low output quality, not model capability.

Fixed in this version:
  - Math (BPM, beat grid, drop timestamps) is now serialized into the prompt
    as an explicit grounding block, not merged post-hoc.
  - Word-level ASR timestamps (not sentence-capped-at-25) are passed in,
    reused from the Phase-1 speech_boundaries.json when it already exists
    instead of re-running faster-whisper a second time.
  - Gemini call timeout raised to a realistic, configurable default.
  - Fixed a NameError in the upload-cleanup `finally:` block that silently
    caused every uploaded file to leak (never actually deleted).
  - Removed the blind "quote unquoted keys" regex JSON repair, which could
    corrupt lyric text containing colons. Replaced with structural-only
    cleanup + optional `json_repair` library as a verified last resort.
  - Cached / pooled results are stamped with a prompt version so upgrading
    this module doesn't get masked by stale results forever.
────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    if os.path.exists(".env"):
        load_dotenv(".env", override=False)
    _cred_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Credentials", ".env")
    if os.path.exists(_cred_env):
        load_dotenv(_cred_env, override=False)
except Exception:
    pass

try:
    from google import genai
except Exception:
    genai = None

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("lyric_rhythm_aligner")

# ─── Config ───────────────────────────────────────────────────────────────────
ENABLE_LYRIC_SYNC = os.getenv("ENABLE_LYRIC_SYNC", "true").lower() in ("true", "1", "yes")

VALID_SECTIONS = {"intro", "verse", "pre_chorus", "chorus", "drop", "bridge", "outro"}

# Minimum audio file size — skip analysis on tiny/corrupt extracts
_MIN_AUDIO_BYTES = 32_768  # 32 KB

# Which faster-whisper model tier to use IF no Phase-1 transcript already
# exists and we have to run ASR fresh. "base" trades accuracy for speed;
# "medium" (default here) trades speed for accuracy — precision is
# the point, so this defaults to medium. Override with env if you need speed.
WHISPER_MODEL_SIZE = os.getenv("LYRIC_WHISPER_MODEL_SIZE", "medium")

# How long we actually wait for Gemini to finish the audio + math + ASR
# fusion call before giving up and falling back to BeatEngine-only output.
GEMINI_AUDIO_CALL_TIMEOUT_SEC = float(os.getenv("LYRIC_GEMINI_TIMEOUT_SEC", "150"))

# Bumped whenever the prompt/fusion logic changes meaningfully. Cached and
# pooled reports stamped with an older version are treated as stale and
# regenerated, instead of silently serving pre-fix output forever.
_PROMPT_VERSION = "2.7-shazam-fingerprint-fused"

# ─── Prompt ───────────────────────────────────────────────────────────────────

_PROMPT = """You are a world-class music supervisor and video editor with expertise in rhythm-based editing.

Listen to this audio track carefully. Extract EVERYTHING needed to edit a viral short-form video to this music.

You will also be given two machine-computed grounding-context blocks after this prompt:
  1. BEAT/TEMPO DATA — exact BPM, beat timestamps, and drop timestamps computed mathematically from
     the waveform. Treat this as ground truth for timing/rhythm; use it to sanity-check your own
     tension_arc and recommended_cut_pace values, not to override what you actually hear.
  2. ASR TRANSCRIPT REFERENCE — a word-level (or sentence-level, if word-level wasn't available)
     speech-to-text transcript with timestamps. This is machine-generated and WILL contain misheard
     words, especially for regional languages, slang, or ad-libs. Treat its TIMESTAMPS as reliable
     timing anchors for where vocals occur, but treat its WORDS only as a hint — correct the actual
     wording yourself by listening to the audio.

Return ONLY a single strict JSON object — no markdown, no explanation, no extra text.

JSON schema:
{
  "has_vocals": true | false,
  "language": "Hindi" | "English" | "Telugu" | "Tamil" | "Spanish" | "Instrumental" | ...,
  "transcript": "<string — full continuous lyrics/speech text as a single smooth string, corrected for wording/spelling. If instrumental, ''>",
  "tempo_bpm": <float — overall BPM estimate>,
  "bar_duration_sec": <float — duration of one musical bar in seconds>,
  "dominant_emotion": <string — single best emotion label: joy | love | hype | power | sadness | euphoria | nostalgia | celebration | anger | intimacy | freedom | neutral>,
  "energy_profile": "low" | "medium" | "high" | "building" | "explosive",

  "sections": [
    {
      "start": <float seconds>,
      "end": <float seconds>,
      "type": "intro" | "verse" | "pre_chorus" | "chorus" | "drop" | "bridge" | "outro" | "instrumental",
      "energy": <float 0.0-1.0>,
      "mood": <string — 1-2 word description e.g. "playful", "intense", "melancholic">,
      "recommended_cut_pace": "hold" | "slow" | "medium" | "fast" | "rapid_fire"
    }
  ],

  "tension_arc": [
    { "time": <float seconds>, "tension": <float 0.0-1.0> }
  ],

  "lyrics": [
    {
      "time": <float seconds — COPY EXACTLY from the ASR transcript timestamp. DO NOT change this value>,
      "end": <float seconds — COPY EXACTLY from the ASR transcript timestamp. DO NOT change this value>,
      "text": "<YOUR ONLY JOB HERE: correct the misheard word(s) by listening to the audio. The timestamp is already correct — only fix spelling/wording>",
      "emotion_weight": <float 0.0-1.0 — how emotionally charged this phrase is>,
      "emotion_tag": <string — joy | love | hype | power | sadness | euphoria | nostalgia | celebration | anger | intimacy | freedom | neutral>,
      "section": <string — which section this lyric falls in>,
      "asr_confidence": "high" | "medium" | "low",
      "asr_raw": "<the original whisper text verbatim, before your correction>"
    }
  ],

  "emotional_peak_moments": [<float seconds>, ...],

  "shot_directives": [
    {
      "time": <float seconds>,
      "duration": <float seconds — how long this directive applies>,
      "directive": "face_closeup" | "wide_energetic" | "fast_action" | "slow_zoom_in" | "wide_landscape" | "low_angle" | "match_cut_motion" | "hold_on_subject",
      "priority": <int 1-5 — 5 is most important>,
      "reason": "<brief reason, e.g.: 'chorus drop — maximum energy'>"
    }
  ],

  "vibe_tags": [<string>, ...],

  "instrumental_sections": [
    { "start": <float seconds>, "end": <float seconds> }
  ],

  "is_unusable": true | false,
  "unusable_reason": "<brief explanation if unusable: e.g. 'loud crowd shouting/chatter with no music', 'heavy traffic/car noise', 'trading floor shouting', 'pure mic static' or '' if usable>"
}

RULES:
- WHISPER ROLE (RELY ONLY ON TIMESTAMPS, NOT WHISPER TEXT):
  Faster-Whisper text (`asr_raw`) frequently mishears regional, Hindi, or song lyrics into wrong English words.
  DO NOT rely on Whisper's text for spelling or word accuracy. RELY ONLY ON WHISPER'S PCM TIMESTAMPS (`time` and `end`) as precision timing anchors.
  Listen directly to the uploaded raw audio track to extract the true spoken/sung words for `lyrics[].text` and `transcript`.

- transcript (EXHAUSTIVE 100% FULL-TRACK SPEECH COVERAGE):
  Provide the 100% complete, verbatim continuous speech/lyric transcript covering the ENTIRE audio track from 0.0s to the end of the audio file.
  Listen to the uploaded raw audio track across its FULL duration. Do NOT stop transcribing early even if the ASR reference block below has gaps.
  Do NOT summarize, skip, or truncate ANY spoken or sung words.
  If instrumental or no speech, set to "".

- is_unusable: set to true ONLY if the audio is unusable non-music noise (e.g. crowd chatter, shouting, car traffic noise, stock market trading hall shouting, heavy static/distortion with no usable music).

- lyrics TIMESTAMP LOCK (CRITICAL — highest priority rule):
  The `time` and `end` values in every lyrics[] entry MUST be copied verbatim from the ASR transcript
  timestamps provided below. You are FORBIDDEN from changing, rounding, shifting, or estimating these
  values. The ASR timestamps are PCM-measured — they are more accurate than anything you can infer.
  Your primary job on each lyrics entry is to determine the TRUE words in `text` by listening directly to the raw audio track at that timestamp.
  If you hear a vocal that has NO corresponding ASR timestamp, add new lyrics entries using the nearest beat timestamp from the BEAT/TEMPO DATA below.
  Do NOT truncate the lyrics array — every word spoken across the FULL audio track must appear in lyrics[] and transcript.
  Set `asr_confidence` to: "high" if whisper word sounds correct, "medium" if plausible but unsure,
  "low" if you had to significantly correct it or infer it from audio alone.

- tension_arc: provide one entry every 1 second (or every beat if BPM > 100). Tension rises into a chorus/drop, falls during verse/outro. Cross-check against the beat/drop timestamps provided below.
- sections: cover the entire track with no gaps. Every second must fall in exactly one section.
- emotional_peak_moments: timestamps where the music hits its hardest emotional/energy peak (typically the first chorus or drop). Maximum 5 entries.
- shot_directives: minimum 3, maximum 12 entries. Focus on the most critical edit decision moments.
- vibe_tags: 3-6 lowercase tags that describe the vibe (e.g. ["festive", "dance", "bollywood", "high_energy", "romantic"]).
- If the audio is very short (< 15s), still provide full structure based on what you can hear.
"""

# ─── Empty / fallback structure ───────────────────────────────────────────────

def _empty_report() -> Dict[str, Any]:
    return {
        "has_vocals": False,
        "language": "Unknown",
        "tempo_bpm": 0.0,
        "bar_duration_sec": 0.0,
        "dominant_emotion": "neutral",
        "energy_profile": "medium",
        "sections": [],
        "tension_arc": [],
        "lyrics": [],
        "emotional_peak_moments": [],
        "shot_directives": [],
        "vibe_tags": [],
        "instrumental_sections": [],
        "is_unusable": False,
        "unusable_reason": "",
        "transcript": "",
        "_source": "fallback",
    }


def _clean_json(text: str) -> str:
    """
    Strip markdown wrappers and extract a JSON object from Gemini's raw text.
    """
    if not text:
        return ""
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        else:
            text = re.sub(r"```(?:json)?", "", text).replace("```", "")
    j_start = text.find("{")
    j_end   = text.rfind("}")
    if j_start != -1 and j_end > j_start:
        text = text[j_start:j_end + 1].strip()
    else:
        text = text.strip()

    # Structural repair only: trailing commas before closing braces/brackets.
    text = re.sub(r",\s*([\]}])", r"\1", text)

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Optional last resort: pip install json_repair --break-system-packages
    try:
        from json_repair import repair_json
        repaired = repair_json(text)
        json.loads(repaired)  # verify before trusting it
        return repaired
    except Exception:
        pass

    return text


def _validate_and_enrich(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process Gemini output:
    - Clamp numeric fields to valid ranges
    - Normalize section types to known values
    - Add `shot_directive` hints derived from lyric emotion tags
    - Sort tension_arc and sections by time
    """
    # Clamp tempo
    report["tempo_bpm"] = max(0.0, float(report.get("tempo_bpm", 0.0)))
    report["bar_duration_sec"] = max(0.0, float(report.get("bar_duration_sec", 0.0)))

    # Auto-compute bar duration if Gemini skipped it
    if report["bar_duration_sec"] == 0.0 and report["tempo_bpm"] > 0:
        report["bar_duration_sec"] = round(4 * 60.0 / report["tempo_bpm"], 3)

    # Normalize sections (preserve Gemini's natural section types directly)
    for sec in report.get("sections", []):
        sec["start"] = float(sec.get("start", 0.0))
        sec["end"]   = float(sec.get("end",   0.0))
        sec["energy"] = max(0.0, min(1.0, float(sec.get("energy", 0.5))))
        sec["type"] = str(sec.get("type", "verse")).strip() or "verse"
    report["sections"] = sorted(report.get("sections", []), key=lambda x: x["start"])

    # Sort tension arc
    report["tension_arc"] = sorted(
        report.get("tension_arc", []),
        key=lambda x: float(x.get("time", 0.0))
    )

    # Clamp tension values
    for pt in report["tension_arc"]:
        pt["tension"] = max(0.0, min(1.0, float(pt.get("tension", 0.5))))

    # Sort lyrics
    report["lyrics"] = sorted(
        report.get("lyrics", []),
        key=lambda x: float(x.get("time", 0.0))
    )
    for lyric in report["lyrics"]:
        lyric["emotion_weight"] = max(0.0, min(1.0, float(lyric.get("emotion_weight", 0.5))))

    # Preserve Gemini's native shot directives directly (sorted by time)
    shot_dirs = []
    for d in report.get("shot_directives", []):
        if isinstance(d, dict):
            shot_dirs.append({
                "time": float(d.get("time", 0.0)),
                "duration": max(0.1, float(d.get("duration", 2.0))),
                "directive": str(d.get("directive", "hold_on_subject")).strip(),
                "priority": int(d.get("priority", 3)),
                "reason": str(d.get("reason", "")).strip(),
            })
    report["shot_directives"] = sorted(shot_dirs, key=lambda x: x["time"])

    # Sort emotional peaks
    report["emotional_peak_moments"] = sorted(
        [float(t) for t in report.get("emotional_peak_moments", [])]
    )

    report["is_unusable"] = bool(report.get("is_unusable", False))
    report["unusable_reason"] = str(report.get("unusable_reason", "")).strip()

    # Ensure transcript is populated as a continuous string
    transcript_val = str(report.get("transcript", "")).strip()
    if not transcript_val and report.get("lyrics"):
        transcript_val = " ".join(
            str(l.get("text", "")).strip() for l in report["lyrics"] if l.get("text")
        ).strip()
    report["transcript"] = transcript_val

    report["_source"] = "gemini"
    return report


# ─── Math + ASR grounding-context builders ─────────────────────────────────────

def _normalize_word(w: Dict) -> Optional[Dict[str, Any]]:
    """
    Defensive key-name handling for speech_boundary_detector word entries.
    """
    if not isinstance(w, dict):
        return None
    text = w.get("word") or w.get("text")
    start = w.get("start")
    if start is None:
        start = w.get("start_time")
    if start is None:
        start = w.get("time")
    end = w.get("end")
    if end is None:
        end = w.get("end_time")
    if text is None or start is None:
        return None
    try:
        start_f = float(start)
        end_f = float(end) if end is not None else start_f + 0.3
        return {"text": str(text).strip(), "start": start_f, "end": end_f}
    except (TypeError, ValueError):
        return None


def _load_or_run_whisper(audio_path: str, cache_dir: str, audio_basename: str, pool_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Reuses pre-computed pool metadata or Phase-1 word-level transcript when available.
    """
    if pool_meta and pool_meta.get("has_vocals") is False:
        logger.info("[LYRIC_ALIGNER] BGM metadata indicates instrumental track (has_vocals=False) — skipping Whisper pass.")
        return {"has_speech": False, "words": [], "sentences": []}

    if pool_meta and (pool_meta.get("transcript") or pool_meta.get("words")):
        logger.info("[LYRIC_ALIGNER] Reusing pre-computed transcript from pool_metadata.json")
        return {
            "has_speech": True,
            "words": pool_meta.get("words", []),
            "sentences": pool_meta.get("sentences", []),
            "transcript": str(pool_meta.get("transcript", ""))
        }

    clip_dir = os.path.dirname(audio_path)
    phase1_path = os.path.join(clip_dir, "speech_boundaries.json")
    if os.path.exists(phase1_path):
        try:
            with open(phase1_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("has_speech") and (data.get("words") or data.get("sentences")):
                logger.info(f"[LYRIC_ALIGNER] Reusing Phase-1 transcript: {phase1_path}")
                return data
        except Exception as e:
            logger.debug(f"[LYRIC_ALIGNER] Could not read Phase-1 transcript: {e}")

    whisper_json_path = os.path.join(cache_dir, f"{audio_basename}_whisper.json")
    if os.path.exists(whisper_json_path):
        try:
            with open(whisper_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("has_speech") and (data.get("words") or data.get("sentences")):
                logger.info(f"[LYRIC_ALIGNER] Reusing cached whisper pass: {whisper_json_path}")
                return data
        except Exception as e:
            logger.debug(f"[LYRIC_ALIGNER] Could not read whisper cache: {e}")

    try:
        from Audio_Modules.speech_boundary_detector import extract_speech_boundaries
        logger.info(f"[LYRIC_ALIGNER] No prior transcript found — running faster-whisper (model={WHISPER_MODEL_SIZE})")
        result = extract_speech_boundaries(
            audio_path, output_json_path=whisper_json_path, model_size=WHISPER_MODEL_SIZE
        )
        return result or {}
    except Exception as e:
        logger.warning(f"[LYRIC_ALIGNER] Whisper transcription failed: {e}")
        return {}


def _build_transcript_context(whisper_data: Dict[str, Any], max_words: int = 4000) -> str:
    """
    Builds the ASR grounding-context block appended to the prompt.
    """
    if not whisper_data or not whisper_data.get("has_speech"):
        return "\n\n### ASR TRANSCRIPT REFERENCE: none (instrumental or no speech detected)\n"

    words = [w for w in (_normalize_word(w) for w in whisper_data.get("words", [])) if w]

    if words:
        if len(words) > max_words:
            logger.warning(
                f"[LYRIC_ALIGNER] Transcript has {len(words)} words — truncating to {max_words}."
            )
        header = (
            "\n\n### FASTER-WHISPER WORD-LEVEL TRANSCRIPT (AUTHORITATIVE ASR TIMING — "
            f"{len(words)} words, model={WHISPER_MODEL_SIZE}):\n"
        )
        body = " ".join(f'[{w["start"]:.2f}-{w["end"]:.2f}] {w["text"]}' for w in words[:max_words])
    else:
        sentences = whisper_data.get("sentences", [])
        header = "\n\n### FASTER-WHISPER SENTENCE-LEVEL TRANSCRIPT (AUTHORITATIVE ASR TIMING — word-level unavailable):\n"
        body = "\n".join(
            f'[{s.get("start", 0.0):.2f}s - {s.get("end", 0.0):.2f}s]: "{s.get("text", "")}"'
            for s in sentences
        )

    instructions = (
        "\n\n### TIMESTAMP LOCK INSTRUCTIONS (READ CAREFULLY BEFORE BUILDING lyrics[]):\n"
        "Each [start-end] pair above is a PCM-measured timestamp from faster-whisper. "
        "Because timestamp estimation from raw audio is a known LLM weakness, Faster-Whisper timestamps "
        "are your absolute GROUND TRUTH for timing accuracy. Follow these rules without exception:\n"
        "  1. COPY every [start] and [end] timestamp EXACTLY into lyrics[].time and lyrics[].end. "
        "     Do not round, shift, snap to bar, or approximate. Keep timestamps locked.\n"
        "  2. Focus your intelligence on fixing any misheard word(s) in lyrics[].text by listening to "
        "     the raw audio track, ensuring 100% complete word-exact accuracy in transcript without missing any speech.\n"
        "  3. Fill asr_raw with the original whisper word(s) verbatim (before correction).\n"
        "  4. Set asr_confidence=high if whisper word sounds correct, medium if plausible, "
        "     low if you had to significantly change it.\n"
        "  5. If you hear a clear vocal that has no ASR entry at all, add a new lyrics entry using "
        "     the nearest beat timestamp from the BEAT/TEMPO DATA block, set asr_confidence=low, "
        "     and asr_raw=\"\" to flag it as inferred.\n"
        "  6. Do NOT skip, merge, or drop any ASR entry — every word above must appear as a lyrics entry.\n"
        "  7. Translate regional/Hinglish emotion into English when picking emotion_tag "
        "     (e.g. 'muskurana' -> face_closeup, 'naach' -> wide_energetic)."
    )
    return header + body + instructions


def _build_math_context(fast_report: Dict[str, Any]) -> str:
    """
    Serializes the machine-computed BeatEngine/pool math into a compact JSON
    block appended to the prompt.
    """
    if not fast_report or not fast_report.get("tempo_bpm"):
        return "\n\n### BEAT/TEMPO DATA: unavailable\n"

    def _safe_float(val: Any) -> Optional[float]:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            t = val.get("time")
            if isinstance(t, (int, float)):
                return float(t)
            if isinstance(t, str):
                try:
                    return float(t)
                except ValueError:
                    pass
        elif isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                pass
        return None

    tension_pts = fast_report.get("tension_arc") or []
    beat_times = [
        round(t, 2)
        for t in (_safe_float(p) for p in tension_pts[:300])
        if t is not None
    ]
    drops = [
        round(t, 2)
        for t in (_safe_float(p) for p in (fast_report.get("emotional_peak_moments") or []))
        if t is not None
    ]

    payload = {
        "tempo_bpm": fast_report.get("tempo_bpm"),
        "bar_duration_sec": fast_report.get("bar_duration_sec"),
        "energy_profile": fast_report.get("energy_profile"),
        "beat_timestamps_sec": beat_times,
        "drop_timestamps_sec": drops,
    }
    return (
        "\n\n### MACHINE-COMPUTED BEAT/TEMPO DATA (ground truth from waveform analysis — "
        "cross-check your tension_arc and recommended_cut_pace against this, don't contradict it "
        "without strong audible evidence):\n" + json.dumps(payload) +
        "\nUse the drop timestamps as strong candidates for emotional_peak_moments and beat "
        "timestamps to sanity-check whether a lyric phrase timestamp lands where vocal energy "
        "actually is versus an instrumental gap."
    )


def _recognize_audio_track(audio_path: str) -> Dict[str, Any]:
    """
    Identifies audio track title, artist, and genre using shazamio fingerprinting.
    Runs asynchronously with a 4.0s timeout so execution never hangs.
    """
    if not audio_path or not os.path.exists(audio_path):
        return {}
    try:
        import asyncio
        from shazamio import Shazam

        async def _async_recognize():
            shazam = Shazam()
            out = await shazam.recognize(audio_path)
            track = out.get("track", {})
            return {
                "title": track.get("title", ""),
                "artist": track.get("subtitle", ""),
                "genre": track.get("genres", {}).get("primary", ""),
            }

        loop = asyncio.new_event_loop()
        try:
            res = loop.run_until_complete(asyncio.wait_for(_async_recognize(), timeout=4.0))
            if res and res.get("title"):
                logger.info(f"🎵 [SHAZAM FINGERPRINT] Recognized track: '{res['title']}' by {res['artist']} ({res['genre']})")
                return res
        finally:
            loop.close()
    except Exception as e:
        logger.debug(f"[LYRIC_ALIGNER] Shazam audio recognition notice: {e}")
    return {}


def _build_shazam_context(track_info: Dict[str, Any]) -> str:
    if not track_info or not track_info.get("title"):
        return "\n\n### AUDIO TRACK IDENTIFICATION (SHAZAM FINGERPRINT): Unknown / Unindexed\n"
    return (
        f"\n\n### AUDIO TRACK IDENTIFICATION (SHAZAM AUDIO FINGERPRINT):\n"
        f"  - Track Title: '{track_info.get('title')}'\n"
        f"  - Artist: '{track_info.get('artist')}'\n"
        f"  - Primary Genre: '{track_info.get('genre')}'\n"
        "USE THIS KNOWN SONG IDENTIFICATION to ensure 100% exact lyric words in `transcript` and `lyrics[].text`. "
        "Correct any misheard ASR words (e.g. Whisper mistranslating song lyrics to English phonetics) using the official lyrics of this song!\n"
    )


# ─── Main API ─────────────────────────────────────────────────────────────────

def analyze_music(audio_path: str) -> Dict[str, Any]:
    """
    Run the full Musical Intelligence Report on `audio_path`.
    """
    if not ENABLE_LYRIC_SYNC:
        logger.info("[LYRIC_ALIGNER] ENABLE_LYRIC_SYNC=false — skipping.")
        return _empty_report()

    if not audio_path or not os.path.exists(audio_path):
        logger.warning(f"[LYRIC_ALIGNER] Audio file not found: {audio_path}")
        return _empty_report()

    file_size = os.path.getsize(audio_path)
    if file_size < _MIN_AUDIO_BYTES:
        logger.warning(f"[LYRIC_ALIGNER] Audio too small ({file_size}B) — skipping.")
        return _empty_report()

    # Resolve persistent cache file path (Original_audio/beats/<clip_folder>_<basename>_lyric.json)
    parent_dir = os.path.basename(os.path.dirname(audio_path))
    raw_basename = os.path.splitext(os.path.basename(audio_path))[0]
    audio_basename = f"{parent_dir}_{raw_basename}" if parent_dir else raw_basename
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(repo_root, "Original_audio", "beats")
    os.makedirs(cache_dir, exist_ok=True)
    cache_json_path = os.path.join(cache_dir, f"{audio_basename}_lyric.json")

    # 💾 CACHE HIT GUARD: only trust cache generated by the CURRENT prompt/fusion version
    if os.path.exists(cache_json_path) and os.path.getsize(cache_json_path) > 50:
        try:
            with open(cache_json_path, "r", encoding="utf-8") as f:
                cached_report = json.load(f)
            if (
                isinstance(cached_report, dict)
                and cached_report.get("_prompt_version") == _PROMPT_VERSION
                and (cached_report.get("_source") == "gemini" or cached_report.get("lyrics") or cached_report.get("shot_directives"))
            ):
                cached_report["_source"] = "cache_hit"
                logger.info(f"[LYRIC_ALIGNER] 💾 Persistent Cache Hit for '{audio_basename}' -> Loaded from {cache_json_path}")
                return cached_report
            elif isinstance(cached_report, dict):
                logger.info(f"[LYRIC_ALIGNER] Cache for '{audio_basename}' is from an older prompt version — regenerating.")
        except Exception as _ce:
            logger.warning(f"[LYRIC_ALIGNER] Cache read fallback for '{audio_basename}': {_ce}")

    # ── Lookup pool_metadata.json by name (0.000s single source of truth lookup) ────
    pool_meta = None
    try:
        from Audio_Modules.audio_pool_manager import AudioPoolManager
        audio_filename = os.path.basename(audio_path)
        pool_meta = AudioPoolManager().get_track_intelligence(audio_filename)
        if pool_meta and isinstance(pool_meta, dict):
            if pool_meta.get("_prompt_version") == _PROMPT_VERSION and (pool_meta.get("lyrics") or pool_meta.get("shot_directives")):
                logger.info(f"[LYRIC_ALIGNER] ⚡ pool_metadata.json Hit for '{audio_filename}' -> Loaded full semantic intel (0.000s).")
                pool_meta["_source"] = "pool_metadata_hit"
                return pool_meta
            elif pool_meta.get("lyrics") or pool_meta.get("shot_directives"):
                logger.info(f"[LYRIC_ALIGNER] pool_metadata.json intel for '{audio_filename}' is from an older prompt version — regenerating.")
    except Exception as _pme:
        logger.debug(f"[LYRIC_ALIGNER] pool_metadata lookup notice: {_pme}")

    # ⚡ STEP 1: Compute or Reuse BeatEngine Rhythm Math (BPM, Beats, Drops, Tension Arc)
    fast_report = _empty_report()
    if pool_meta and pool_meta.get("tempo_bpm"):
        bpm = float(pool_meta.get("tempo_bpm", 120.0))
        beats = pool_meta.get("beats", [])
        drops = pool_meta.get("drops", [])
        bar_dur = float(pool_meta.get("bar_duration_sec", round(4 * 60.0 / (bpm or 120.0), 3)))
        fast_report = {
            "has_vocals": pool_meta.get("has_vocals", False),
            "language": pool_meta.get("language", "Instrumental"),
            "tempo_bpm": bpm,
            "bar_duration_sec": bar_dur,
            "dominant_emotion": pool_meta.get("dominant_emotion", "hype" if len(drops) > 0 else "joy"),
            "energy_profile": pool_meta.get("energy_profile", "medium"),
            "sections": pool_meta.get("sections", []),
            "tension_arc": pool_meta.get("tension_arc", []),
            "lyrics": pool_meta.get("lyrics", []),
            "emotional_peak_moments": drops[:5] if drops else pool_meta.get("emotional_peak_moments", []),
            "shot_directives": pool_meta.get("shot_directives", []),
            "vibe_tags": pool_meta.get("vibe_tags", []),
            "instrumental_sections": [],
            "is_unusable": False,
            "unusable_reason": "",
            "transcript": str(pool_meta.get("transcript", "")),
            "_source": "pool_metadata_fastpath"
        }
        logger.info(f"⚡ [LYRIC_ALIGNER] Reused pre-computed rhythm math from pool_metadata.json: bpm={bpm:.1f}")
    else:
        try:
            from Audio_Modules.beat_engine import BeatEngine
            be_res = BeatEngine().analyze_beats_with_drops(audio_path)
            bpm = be_res.get("bpm", 120.0)
            beats = be_res.get("beats", [])
            drops = be_res.get("drops", [])
            bar_dur = round(4 * 60.0 / (bpm or 120.0), 3)
            fast_report = {
                "has_vocals": False,
                "language": "Instrumental",
                "tempo_bpm": bpm,
                "bar_duration_sec": bar_dur,
                "dominant_emotion": "hype" if len(drops) > 0 else "joy",
                "energy_profile": "high" if len(drops) > 0 else "medium",
                "sections": [
                    {"start": 0.0, "end": 15.0, "type": "chorus" if len(drops) > 0 else "verse", "energy": 0.85, "mood": "energetic", "recommended_cut_pace": "fast"}
                ],
                "tension_arc": [{"time": b, "tension": 0.8 if b in drops else 0.5} for b in beats[:20]],
                "lyrics": [],
                "emotional_peak_moments": drops[:5],
                "shot_directives": [],
                "vibe_tags": [be_res.get("vibe", "groove")],
                "instrumental_sections": [],
                "is_unusable": False,
                "unusable_reason": "",
                "transcript": "",
                "_source": "beat_engine_fastpath"
            }
            logger.info(f"⚡ [LYRIC_ALIGNER] BeatEngine computed rhythm math: bpm={bpm:.1f}, drops={len(drops)}")
        except Exception as _fe:
            logger.warning(f"Beat engine computation notice: {_fe}")

    # 🎵 STEP 1b: Shazam Audio Track Fingerprinting (Title & Artist Recognition)
    shazam_info = _recognize_audio_track(audio_path)
    shazam_context = _build_shazam_context(shazam_info)

    # 🎙️ STEP 2: Load or run word-level ASR
    whisper_data = _load_or_run_whisper(audio_path, cache_dir, audio_basename, pool_meta=pool_meta)
    transcript_context = _build_transcript_context(whisper_data)

    # 🧮 STEP 2b: Machine-computed rhythm math, serialized as grounding context.
    math_context = _build_math_context(fast_report)

    # 🧠 STEP 3: Gemini Multimodal Audio Call (Semantic Lyric & Emotion Extraction)
    gemini_router = None
    try:
        from Gemini_Modules.gemini_router_module import gemini_router
    except Exception as _e1:
        try:
            from Gemini_Modules.gemini_router_module.gemini_governor import gemini_router
        except Exception as _e2:
            try:
                from gemini_governor import gemini_router
            except Exception as _e3:
                logger.warning(f"[LYRIC_ALIGNER] gemini_router import failed: {_e1} | {_e2} | {_e3}")
                return fast_report if fast_report.get("tempo_bpm", 0) > 0 else _empty_report()

    analysis_res = []
    _uploaded_file = None
    _client_holder: Dict[str, Any] = {}

    def _worker():
        nonlocal _uploaded_file
        try:
            from google import genai as _genai_client_mod
            _api_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
            _client = _genai_client_mod.Client(api_key=_api_key)
            _client_holder["client"] = _client
            print(f"  └─ 📤 [LYRIC_ALIGNER] Uploading {os.path.basename(audio_path)} to Gemini API...", flush=True)
            uf = _client.files.upload(file=audio_path)
            _uploaded_file = uf
            _wait = 0
            while getattr(getattr(uf, "state", None), "name", "ACTIVE") == "PROCESSING" and _wait < 10:
                time.sleep(1)
                uf = _client.files.get(name=uf.name)
                _wait += 1
            if getattr(getattr(uf, "state", None), "name", "ACTIVE") != "ACTIVE":
                analysis_res.append(RuntimeError(f"Gemini file upload never reached ACTIVE state: {getattr(uf, 'state', None)}"))
                return
            print(f"  └─ ⏳ [LYRIC_ALIGNER] Audio ready. Requesting Musical Intelligence Report (math+ASR+Shazam fused)...", flush=True)
            t_start = time.time()
            full_prompt = _PROMPT + shazam_context + math_context + transcript_context
            raw_response = gemini_router.generate(
                task_type="analysis",
                prompt=[uf, full_prompt],
                module_name="lyric_rhythm_aligner",
                gen_config={"temperature": 0.2, "max_output_tokens": 8192},
            )
            latency = time.time() - t_start
            print(f"  └─ ✅ [LYRIC_ALIGNER] Musical Intelligence received in {latency:.1f}s.", flush=True)
            analysis_res.append(raw_response)
        except Exception as _ex:
            analysis_res.append(_ex)

    import threading
    a_thread = threading.Thread(target=_worker, daemon=True)
    a_thread.start()
    a_thread.join(timeout=GEMINI_AUDIO_CALL_TIMEOUT_SEC)

    if a_thread.is_alive():
        logger.warning(
            f"[LYRIC_ALIGNER] Gemini call still running past the {GEMINI_AUDIO_CALL_TIMEOUT_SEC:.0f}s "
            f"budget for '{os.path.basename(audio_path)}' — falling back to BeatEngine-only output."
        )
        def _late_cleanup():
            a_thread.join(timeout=GEMINI_AUDIO_CALL_TIMEOUT_SEC * 2)
            _client = _client_holder.get("client")
            if _client and _uploaded_file:
                try:
                    _client.files.delete(name=_uploaded_file.name)
                    logger.debug(f"[LYRIC_ALIGNER] Late cleanup deleted uploaded file: {_uploaded_file.name}")
                except Exception:
                    pass
        threading.Thread(target=_late_cleanup, daemon=True).start()

        try:
            with open(cache_json_path, "w", encoding="utf-8") as cf:
                json.dump(fast_report, cf, indent=2)
        except Exception:
            pass
        return fast_report if fast_report.get("tempo_bpm", 0) > 0 else _empty_report()

    if not analysis_res or isinstance(analysis_res[0], Exception):
        _err = analysis_res[0] if analysis_res else "no result returned"
        logger.warning(f"[LYRIC_ALIGNER] Gemini Audio Call failed for '{os.path.basename(audio_path)}': {_err} — using BeatEngine fallback.")
        try:
            with open(cache_json_path, "w", encoding="utf-8") as cf:
                json.dump(fast_report, cf, indent=2)
        except Exception:
            pass
        return fast_report if fast_report.get("tempo_bpm", 0) > 0 else _empty_report()

    raw_response = analysis_res[0]

    if not raw_response or len(str(raw_response).strip()) < 10:
        logger.warning("[LYRIC_ALIGNER] Empty response from Gemini — using BeatEngine fallback.")
        return fast_report if fast_report.get("tempo_bpm", 0) > 0 else _empty_report()

    try:
        cleaned = _clean_json(str(raw_response))
        report  = json.loads(cleaned)

        if not isinstance(report, dict):
            logger.warning("[LYRIC_ALIGNER] Gemini returned non-dict JSON — using BeatEngine fallback.")
            return fast_report if fast_report.get("tempo_bpm", 0) > 0 else _empty_report()

        if fast_report.get("tempo_bpm", 0) > 0:
            if not report.get("tempo_bpm"):
                report["tempo_bpm"] = fast_report["tempo_bpm"]
            if not report.get("bar_duration_sec"):
                report["bar_duration_sec"] = fast_report["bar_duration_sec"]
            if not report.get("tension_arc"):
                report["tension_arc"] = fast_report["tension_arc"]
            if not report.get("emotional_peak_moments"):
                report["emotional_peak_moments"] = fast_report["emotional_peak_moments"]

        report = _validate_and_enrich(report)
        report["_source"] = "gemini"
        report["_prompt_version"] = _PROMPT_VERSION

        try:
            temp_cache = cache_json_path + ".tmp"
            with open(temp_cache, "w", encoding="utf-8") as cf:
                json.dump(report, cf, indent=2)
            os.replace(temp_cache, cache_json_path)
            logger.info(f"[LYRIC_ALIGNER] 💾 Persisted lyric intelligence to disk: {cache_json_path}")
            try:
                from Audio_Modules.audio_pool_manager import AudioPoolManager
                _ext = os.path.splitext(str(audio_path))[1].lower()
                if _ext not in (".mp4", ".mkv", ".mov", ".avi", ".webm") and os.path.basename(str(audio_path)).lower() not in ("video.mp4", "raw.mp4"):
                    AudioPoolManager().merge_lyric_into_pool(
                        audio_path, report
                    )
            except Exception as _pm:
                logger.debug(f"[LYRIC_ALIGNER] pool merge (gemini_path): {_pm}")
        except Exception as _swe:
            logger.warning(f"[LYRIC_ALIGNER] Failed to persist lyric cache: {_swe}")

        n_sec  = len(report.get("sections", []))
        n_lyr  = len(report.get("lyrics", []))
        n_dir  = len(report.get("shot_directives", []))
        n_arc  = len(report.get("tension_arc", []))
        n_peak = len(report.get("emotional_peak_moments", []))
        logger.info(
            f"[LYRIC_ALIGNER] 🎶 FUSED Musical Intelligence Report | "
            f"vocals={report.get('has_vocals')} lang={report.get('language')} "
            f"bpm={report.get('tempo_bpm')} emotion={report.get('dominant_emotion')} "
            f"sections={n_sec} lyrics={n_lyr} directives={n_dir} "
            f"tension_arc={n_arc}pts peaks={n_peak}"
        )
        return report

    except json.JSONDecodeError as _jde:
        logger.error(f"[LYRIC_ALIGNER] JSON parse error: {_jde}")
        return fast_report if fast_report.get("tempo_bpm", 0) > 0 else _empty_report()
    except Exception as _e:
        logger.warning(f"[LYRIC_ALIGNER] Gemini processing exception: {_e}")
        return fast_report if fast_report.get("tempo_bpm", 0) > 0 else _empty_report()

    finally:
        _client = _client_holder.get("client")
        if _client and _uploaded_file:
            try:
                _client.files.delete(name=_uploaded_file.name)
                logger.debug(f"[LYRIC_ALIGNER] Cleaned up uploaded file: {_uploaded_file.name}")
            except Exception as _cde:
                logger.debug(f"[LYRIC_ALIGNER] Uploaded file cleanup failed (non-fatal): {_cde}")


def get_tension_at(tension_arc: List[Dict], time_sec: float) -> float:
    """
    Interpolate tension score at a specific timestamp from the tension arc.
    Returns 0.5 (neutral) if the arc is empty.
    """
    if not tension_arc:
        return 0.5
    arc = sorted(tension_arc, key=lambda x: x.get("time", 0.0))
    if time_sec <= arc[0].get("time", 0.0):
        return float(arc[0].get("tension", 0.5))
    if time_sec >= arc[-1].get("time", 0.0):
        return float(arc[-1].get("tension", 0.5))
    for i in range(len(arc) - 1):
        t0 = float(arc[i].get("time", 0.0))
        t1 = float(arc[i + 1].get("time", 0.0))
        if t0 <= time_sec <= t1:
            if t1 == t0:
                return float(arc[i].get("tension", 0.5))
            alpha = (time_sec - t0) / (t1 - t0)
            v0 = float(arc[i].get("tension", 0.5))
            v1 = float(arc[i + 1].get("tension", 0.5))
            return round(v0 + alpha * (v1 - v0), 3)
    return 0.5


def get_section_at(sections: List[Dict], time_sec: float) -> Optional[Dict]:
    """
    Return the section dict that contains `time_sec`, or None.
    """
    for sec in sections:
        if float(sec.get("start", 0)) <= time_sec < float(sec.get("end", 0)):
            return sec
    return None


def get_directive_at(directives: List[Dict], time_sec: float) -> Optional[Dict]:
    """
    Return the highest-priority shot directive active at `time_sec`, or None.
    """
    active = [
        d for d in directives
        if float(d.get("time", 0)) <= time_sec < float(d.get("time", 0)) + float(d.get("duration", 2.0))
    ]
    if not active:
        return None
    return max(active, key=lambda x: int(x.get("priority", 1)))


# ── Universal Creative Archetypes & 4D Scoring ──────────────────────────────

CREATIVE_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "TALKING_HEAD_PODCAST": {
        "name": "TALKING_HEAD_PODCAST",
        "description": "On-camera dialogue, interview, monologue, or educational talking head.",
        "preferred_vibes": ["chill", "lofi", "ambient", "acoustic", "subtle", "warm", "minimal"],
        "target_bpm": 90.0,
        "max_energy": 0.45,
        "prefer_instrumental": True,
        "fatal_genres": ["phonk", "hardstyle", "club", "edm", "heavy_bass", "trap", "drill", "aggressive"],
        "directive": (
            "This video features on-camera speech or interview dialogue. Vocal clarity is paramount. "
            "Select an unobtrusive, subtle, low-energy lofi or ambient background track (preferably instrumental). "
            "Loud dance music, heavy bass drops, phonk, club beats, and aggressive vocals are STRICTLY FORBIDDEN as they clash with spoken speech."
        )
    },
    "GLAMOUR_STRUT_PAPARAZZI": {
        "name": "GLAMOUR_STRUT_PAPARAZZI",
        "description": "Candid celebrity/model walk, paparazzi flashes, fashion runway, luxury strut, red carpet arrival.",
        "preferred_vibes": ["swagger", "bass", "trap", "hip_hop", "luxury", "strut", "phonk", "deep_house", "energetic", "hype"],
        "target_bpm": 125.0,
        "min_energy": 0.60,
        "prefer_instrumental": False,
        "fatal_genres": ["ambient", "meditation", "chatter", "crowd", "soft_acoustic", "slow_piano", "classical"],
        "directive": (
            "This video is a glamorous candid celebrity strut, runway walk, or paparazzi entrance. "
            "Camera shutter clicks and background chatter will be muted. Select a heavy-bass, swagger-filled, "
            "head-nodding hip-hop, luxury house, or phonk beat (BPM 115-135, high energy) that gives the subject an irresistible, iconic walk. "
            "Soft acoustic music, slow piano, and ambient crowd chatter are STRICTLY FORBIDDEN."
        )
    },
    "ADRENALINE_ACTION": {
        "name": "ADRENALINE_ACTION",
        "description": "Fitness, gym training, sports, athletics, martial arts, extreme sports, high-velocity movement.",
        "preferred_vibes": ["phonk", "hardstyle", "aggressive", "motivational", "trap", "hype", "explosive", "fast"],
        "target_bpm": 135.0,
        "min_energy": 0.70,
        "prefer_instrumental": False,
        "fatal_genres": ["lofi", "chill", "soft_acoustic", "ambient", "meditation", "slow"],
        "directive": (
            "This video is an explosive workout or athletic performance. Select a high-octane, adrenaline-pumping "
            "track with heavy 808 drops, driving rhythm, and peak energy (Brazilian phonk, hardstyle, or aggressive trap). "
            "Slow, relaxing, or mellow lofi music is STRICTLY FORBIDDEN."
        )
    },
    "ATMOSPHERIC_CINEMATIC": {
        "name": "ATMOSPHERIC_CINEMATIC",
        "description": "Travel, landscape, nature, drone vistas, architectural luxury, aesthetic B-roll, cinematic mood.",
        "preferred_vibes": ["cinematic", "atmospheric", "melodic", "deep", "euphoric", "inspiring", "electronic", "ambient"],
        "target_bpm": 115.0,
        "min_energy": 0.35,
        "prefer_instrumental": True,
        "fatal_genres": ["phonk", "aggressive_trap", "hardstyle", "screaming", "slapstick", "comedy"],
        "directive": (
            "This video is an aesthetic visual journey (travel, nature, or cinematic B-roll). Select a rich, "
            "atmospheric, melodic, or evocative electronic/cinematic track with emotional depth and sweeping soundscapes. "
            "Harsh distorted phonk and slapstick comedic music are STRICTLY FORBIDDEN."
        )
    },
    "PLAYFUL_RHYTHMIC": {
        "name": "PLAYFUL_RHYTHMIC",
        "description": "Comedy, meme, viral trend, playful lifestyle, dynamic skit, upbeat casual vlog.",
        "preferred_vibes": ["playful", "funk", "upbeat", "bouncy", "catchy", "pop", "happy", "quirky", "rhythmic"],
        "target_bpm": 120.0,
        "min_energy": 0.50,
        "prefer_instrumental": False,
        "fatal_genres": ["dark_phonk", "depressing", "melancholic", "horror", "dirge", "sad"],
        "directive": (
            "This video is upbeat, playful, humorous, or lifestyle-driven. Select a catchy, bouncy, rhythmic "
            "groove (funk, upbeat pop, playful bounce) that keeps the viewer engaged and smiling. "
            "Dark, aggressive, or depressing tracks are STRICTLY FORBIDDEN."
        )
    },
}


def classify_video_archetype(
    visual_ctx: Dict[str, Any],
    current_audio: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Classifies a clip into one of 5 Universal Creative Archetypes based on
    visual context, caption, hashtags, detected entities, and audio speech intelligence.
    """
    meta = metadata or {}
    cd = visual_ctx.get("content_director", {})

    intent = str(visual_ctx.get("intent") or cd.get("intent") or "viral_reel").lower()
    visual_event = str(visual_ctx.get("visual_event") or cd.get("visual_event") or "").lower()
    tone = str(visual_ctx.get("tone") or cd.get("tone") or "aspirational").lower()
    caption = str(meta.get("caption") or visual_ctx.get("caption") or "").lower()

    tags_list = meta.get("hashtags") or visual_ctx.get("hashtags") or []
    if isinstance(tags_list, list):
        hashtags = " ".join(str(h).lower() for h in tags_list)
    else:
        hashtags = str(tags_list).lower()

    entities_list = visual_ctx.get("detected_entities") or cd.get("detected_entities") or []
    entities = " ".join(str(e).lower() for e in entities_list)

    speech_intel = visual_ctx.get("speech_intelligence") or current_audio.get("context", {})
    speech_mode = str(speech_intel.get("speech_mode", "")).lower()
    is_talking = bool(
        speech_intel.get("is_talking_visually")
        or visual_ctx.get("is_talking_on_camera")
        or cd.get("is_talking_on_camera")
        or intent == "talking_head"
    )

    combined_text = f"{intent} {visual_event} {tone} {caption} {hashtags} {entities}".lower()

    def _has_kw(text: str, kws: List[str]) -> bool:
        for kw in kws:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return True
        return False

    # Priority 1: Explicit high-confidence intent matches
    if intent in ("comedy", "meme", "skit"):
        return CREATIVE_ARCHETYPES["PLAYFUL_RHYTHMIC"]
    if intent in ("fitness", "gym", "workout", "sports", "action", "boxing"):
        return CREATIVE_ARCHETYPES["ADRENALINE_ACTION"]
    if intent in ("candid_walk", "paparazzi", "fashion", "runway", "model", "glamour", "street_style"):
        return CREATIVE_ARCHETYPES["GLAMOUR_STRUT_PAPARAZZI"]
    if intent in ("talking_head", "podcast", "interview", "monologue"):
        return CREATIVE_ARCHETYPES["TALKING_HEAD_PODCAST"]
    if intent in ("travel", "nature", "landscape", "architecture", "cinematic"):
        return CREATIVE_ARCHETYPES["ATMOSPHERIC_CINEMATIC"]

    # Priority 2: Talking Head / Podcast / Interview
    podcast_kws = ["podcast", "interview", "talking", "monologue", "speech", "dialogue", "host", "explaining", "ted talk", "qa", "conversation"]
    if speech_mode in ("on_camera_dialogue", "voiceover_narration") or is_talking or _has_kw(combined_text, podcast_kws):
        paparazzi_kws = ["paparazzi", "strut", "runway", "model", "fashion show", "candid walk", "red carpet", "arrival"]
        if not _has_kw(combined_text, paparazzi_kws):
            return CREATIVE_ARCHETYPES["TALKING_HEAD_PODCAST"]

    # Priority 3: Comedy / Meme / Skit
    comedy_kws = ["comedy", "funny", "meme", "skit", "joke", "prank", "humor", "laugh", "challenge", "lol", "relatable"]
    if _has_kw(combined_text, comedy_kws):
        return CREATIVE_ARCHETYPES["PLAYFUL_RHYTHMIC"]

    # Priority 4: Glamour / Paparazzi / Model Strut
    paparazzi_kws = ["paparazzi", "strut", "walk", "candid", "runway", "model", "fashion", "arrival", "red carpet", "street style", "spotted", "outfit", "glamour", "celebrity walk"]
    if _has_kw(combined_text, paparazzi_kws):
        return CREATIVE_ARCHETYPES["GLAMOUR_STRUT_PAPARAZZI"]

    # Priority 5: Adrenaline / Action / Workout
    action_kws = ["gym", "workout", "fitness", "bodybuilding", "boxing", "training", "exercise", "bicep", "squat", "deadlift", "action", "athletic", "crossfit", "sprint", "mma"]
    if _has_kw(combined_text, action_kws):
        return CREATIVE_ARCHETYPES["ADRENALINE_ACTION"]

    # Priority 6: Atmospheric / Cinematic / Travel
    cinematic_kws = ["travel", "landscape", "nature", "drone", "vlog", "architecture", "aesthetic", "sunset", "ocean", "mountains", "scenic", "wanderlust", "cityscape"]
    if _has_kw(combined_text, cinematic_kws):
        return CREATIVE_ARCHETYPES["ATMOSPHERIC_CINEMATIC"]

    # Default fallback
    if speech_mode == "on_camera_dialogue":
        return CREATIVE_ARCHETYPES["TALKING_HEAD_PODCAST"]
    return CREATIVE_ARCHETYPES["PLAYFUL_RHYTHMIC"]


def compute_4d_audio_score(
    archetype: Dict[str, Any],
    candidate_meta: Dict[str, Any],
    candidate_filename: str,
    visual_ctx: Dict[str, Any],
    current_audio: Dict[str, Any],
    clip_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, float, float, float]:
    """
    Computes 4-Dimensional Hybrid Score:
    Score = S_archetype * (0.40 * S_semantic + 0.35 * S_rhythm + 0.25 * S_emotion) * S_fatigue

    Returns (total_score, s_archetype, s_semantic, s_rhythm, s_fatigue)
    """
    fn_lower = candidate_filename.lower()
    c_genre = str(candidate_meta.get("gemini_genre") or candidate_meta.get("genre") or "music").lower()
    vibes = candidate_meta.get("vibe_tags") or [candidate_meta.get("vibe", "energetic")]
    c_vibes = [str(v).lower() for v in (vibes if isinstance(vibes, list) else [vibes])]
    vibe_str = " ".join(c_vibes)
    c_energy = float(candidate_meta.get("energy") or candidate_meta.get("avg_energy") or 0.6)
    c_bpm = float(candidate_meta.get("tempo_bpm") or candidate_meta.get("bpm") or 120.0)
    c_vocals = bool(candidate_meta.get("has_vocals", False))
    c_emotion = str(candidate_meta.get("dominant_emotion", "hype")).lower()

    arch_name = archetype.get("name", "PLAYFUL_RHYTHMIC")
    fatal_genres = [g.lower() for g in archetype.get("fatal_genres", [])]

    # ── 1. S_archetype (0.0 to 1.0) ──────────────────────────────────────────
    s_archetype = 0.5  # Neutral default

    # Fatal genre check
    for fg in fatal_genres:
        if fg in c_genre or fg in vibe_str or fg in fn_lower:
            s_archetype = 0.0
            break

    if s_archetype > 0.0:
        if arch_name == "TALKING_HEAD_PODCAST":
            if c_energy > archetype.get("max_energy", 0.45):
                s_archetype = 0.0
            elif c_vocals:
                s_archetype = 0.1
            elif any(v in vibe_str for v in archetype.get("preferred_vibes", [])):
                s_archetype = 1.0
            else:
                s_archetype = 0.6

        elif arch_name == "GLAMOUR_STRUT_PAPARAZZI":
            if c_energy < archetype.get("min_energy", 0.60):
                s_archetype = 0.0
            elif any(v in vibe_str for v in ["swagger", "bass", "strut", "hip_hop", "trap", "phonk", "luxury"]):
                s_archetype = 1.0
            elif 115.0 <= c_bpm <= 135.0:
                s_archetype = 0.9
            else:
                s_archetype = 0.7

        elif arch_name == "ADRENALINE_ACTION":
            if c_energy < archetype.get("min_energy", 0.70):
                s_archetype = 0.0
            elif any(v in vibe_str for v in ["phonk", "hardstyle", "aggressive", "motivational", "trap"]):
                s_archetype = 1.0
            elif c_bpm >= 125.0:
                s_archetype = 0.85
            else:
                s_archetype = 0.6

        elif arch_name == "ATMOSPHERIC_CINEMATIC":
            if any(v in vibe_str for v in ["cinematic", "atmospheric", "melodic", "deep", "euphoric", "inspiring"]):
                s_archetype = 1.0
            else:
                s_archetype = 0.7

        elif arch_name == "PLAYFUL_RHYTHMIC":
            if any(v in vibe_str for v in ["playful", "funk", "upbeat", "bouncy", "pop", "happy"]):
                s_archetype = 1.0
            else:
                s_archetype = 0.75

    # If S_archetype is 0.0, instant return with 0.0! (Hard Disqualification)
    if s_archetype <= 0.0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    # ── 2. S_semantic (0.1 to 1.0) ───────────────────────────────────────────
    cd = visual_ctx.get("content_director", {})
    clip_text = f"{visual_ctx.get('intent', '')} {visual_ctx.get('tone', '')} {cd.get('visual_event', '')} {visual_ctx.get('visual_event', '')}".lower()
    if clip_metadata:
        clip_text += f" {clip_metadata.get('caption', '')} {' '.join(str(h) for h in clip_metadata.get('hashtags', []))}".lower()

    cand_transcript = str(candidate_meta.get("transcript") or candidate_meta.get("lyrics") or "").lower()
    cand_caption = str(candidate_meta.get("caption") or "").lower()
    cand_tags = " ".join(str(h) for h in (candidate_meta.get("hashtags") or [])).lower()
    cand_lang = str(candidate_meta.get("language") or "").lower()

    track_text = f"{c_genre} {vibe_str} {fn_lower} {c_emotion} {cand_lang} {cand_caption} {cand_tags} {cand_transcript}".lower()

    stopwords = {
        "this", "that", "with", "from", "have", "video", "reel", "post",
        "instagram", "tiktok", "edit", "editor", "your", "them", "they",
        "will", "been", "were", "what", "when", "where", "which", "there",
        "their", "about", "into", "more", "some", "such", "than", "then", "very"
    }
    clip_words = {w for w in re.findall(r"\w{3,}", clip_text) if w not in stopwords}
    track_words = {w for w in re.findall(r"\w{3,}", track_text) if w not in stopwords}
    exact_overlap = len(clip_words.intersection(track_words))

    # Partial substring overlap for morphological variants (e.g. dance / dancing, romantic / romance)
    partial_matches = 0
    for cw in clip_words:
        if len(cw) >= 4:
            if any((cw in tw or tw in cw) for tw in track_words if len(tw) >= 4 and cw != tw):
                partial_matches += 1

    # Cultural & language synergy boost
    indian_kws = ["bollywood", "hindi", "desi", "punjabi", "tamil", "telugu", "india", "mumbai", "saree", "lehenga", "bhangra", "garba", "tollywood", "kollywood", "actor", "actress", "shaadi"]
    is_clip_indian = any(kw in clip_text for kw in indian_kws)
    is_track_indian = any(kw in track_text for kw in indian_kws) or cand_lang in ["hi", "hindi", "pa", "punjabi", "ta", "tamil", "te", "telugu"]

    cultural_bonus = 0.0
    if is_clip_indian and is_track_indian:
        cultural_bonus = 0.25
    elif is_clip_indian and not is_track_indian and any(bad in track_text for bad in ["phonk", "drill", "death_metal"]):
        cultural_bonus = -0.20

    # Base semantic score starts at 0.35, scaled by overlap, partial match, and cultural synergy
    s_semantic = max(0.10, min(1.0, 0.35 + (exact_overlap * 0.15) + (min(3, partial_matches) * 0.08) + cultural_bonus))

    # ── 3. S_rhythm (0.1 to 1.0) ─────────────────────────────────────────────
    clip_math_bpm = float(current_audio.get("math", {}).get("tempo_bpm", 0.0))
    target_bpm = clip_math_bpm if clip_math_bpm > 30.0 else float(archetype.get("target_bpm", 120.0))
    bpm_diff = abs(target_bpm - c_bpm)
    s_rhythm = max(0.1, 1.0 - (bpm_diff / 75.0))

    # ── 4. S_emotion (0.1 to 1.0) ────────────────────────────────────────────
    target_tone = str(visual_ctx.get("tone") or cd.get("tone") or "aspirational").lower()
    emotion_match = 1.0 if (c_emotion in target_tone or target_tone in c_emotion) else 0.5
    energy_diff = abs((archetype.get("max_energy", 0.7) if arch_name == "TALKING_HEAD_PODCAST" else 0.8) - c_energy)
    s_emotion = min(1.0, max(0.1, (emotion_match * 0.6) + ((1.0 - min(1.0, energy_diff)) * 0.4)))

    # ── 5. S_fatigue (0.05 to 1.0) ───────────────────────────────────────────
    now = time.time()
    last_used = float(candidate_meta.get("last_used", 0) or 0)
    usage_count = int(candidate_meta.get("usage_count", 0) or 0)

    hrs_since_used = (now - last_used) / 3600.0 if last_used > 0 else 999.0
    recency_factor = min(1.0, hrs_since_used / 12.0) if last_used > 0 else 1.0
    usage_factor = 1.0 / (1.0 + (float(usage_count) ** 2) * 1.2) if usage_count > 0 else 1.0
    s_fatigue = max(0.08, recency_factor * usage_factor)

    # ── Final 4D Hybrid Formula ──────────────────────────────────────────────
    core_score = (0.40 * s_semantic) + (0.35 * s_rhythm) + (0.25 * s_emotion)
    final_score = s_archetype * core_score * s_fatigue

    return (final_score, s_archetype, s_semantic, s_rhythm, s_fatigue)


def select_best_audio_for_clip(
    clip_id: str,
    clip_folder: Optional[str] = None,
    audio_dir: Optional[str] = None,
    exclude_filenames: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Gemini Call 2 — Universal Creative Director BGM Selector.
    Classifies video creative archetype, filters out speech/noise, executes 4D hybrid scoring
    (S_archetype * (0.40 * S_semantic + 0.35 * S_rhythm + 0.25 * S_emotion) * S_fatigue),
    and grounds Gemini Call 2 with active visual narrative and director directives.
    """
    from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore

    store = ClipIntelligenceStore()
    clip_data = store.load(clip_id, clip_folder) or store.create_blank(clip_id, clip_folder or "")

    visual_ctx = clip_data.get("visual_context", {})
    current_audio = clip_data.get("audio_data", {})

    local_meta = {}
    if clip_folder and os.path.isdir(clip_folder):
        meta_path = os.path.join(clip_folder, "metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    local_meta = json.load(f)
            except Exception:
                pass

    # 1. Classify Creative Archetype
    archetype = classify_video_archetype(visual_ctx, current_audio, metadata=local_meta)
    arch_name = archetype.get("name", "PLAYFUL_RHYTHMIC")
    logger.info(f"🎭 [BGM ARCHETYPE] Clip '{clip_id}' classified as: {arch_name} — {archetype.get('description')}")

    # 2. Retrieve Candidate Audio from Telegram Storage Vault
    vault_audio_pool = {}
    try:
        from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
        vault = TelegramVaultIndexer()
        vault_audio_pool = vault.get_vault_audio_pool(current_clip_id=clip_id)
        if vault_audio_pool:
            logger.info(f"🏛️ [BGM SELECTOR - PRIMARY] Loaded {len(vault_audio_pool)} candidate audio track(s) from Telegram Storage Vault index.")
    except Exception as _tve:
        logger.debug(f"[BGM SELECTOR] Vault audio index lookup notice: {_tve}")

    try:
        from Audio_Modules.rejected_audio_blacklist import is_blacklisted as _is_bl
    except ImportError:
        def _is_bl(*args, **kwargs): return False

    from Audio_Modules.audio_pool_manager import AudioPoolManager
    _pm = AudioPoolManager()
    local_pool_files = _pm.get_files_index()

    pool_files = dict(vault_audio_pool)
    for fname, meta in local_pool_files.items():
        m_fid = meta.get("file_id") or meta.get("telegram_file_id") if isinstance(meta, dict) else None
        m_sc = meta.get("shortcode") if isinstance(meta, dict) else None
        if _is_bl(audio_filename=fname, audio_shortcode=m_sc, telegram_file_id=m_fid):
            logger.debug(f"🚫 [BGM POOL] Excluding blacklisted track '{fname}' from pool index.")
            continue
        if fname in pool_files:
            was_source_extract = pool_files[fname].get("is_source_extract", False)
            pool_files[fname].update(meta)
            if was_source_extract:
                pool_files[fname]["is_source_extract"] = True
        else:
            pool_files[fname] = meta

    if not pool_files:
        logger.warning("🎶 [BGM Selector] Telegram Vault index has no candidate tracks.")
        return {"selected_audio_track": None, "alignment_score": 0.0, "reasoning": "Empty Telegram Vault index."}

    previous_bgm = current_audio.get("selected_bgm_track") or current_audio.get("selected_audio_track")
    disqualified_tracks = set()
    if previous_bgm:
        disqualified_tracks.add(previous_bgm.lower())
        disqualified_tracks.add(os.path.basename(previous_bgm).lower())
    if exclude_filenames:
        for ef in exclude_filenames:
            if ef:
                disqualified_tracks.add(str(ef).lower())
                disqualified_tracks.add(os.path.basename(str(ef)).lower())

    try:
        from Telegram_Storage_Modules.telegram_http import is_mtproto_configured as _is_mtproto_ok
    except ImportError:
        def _is_mtproto_ok(): return False

    mtproto_available = _is_mtproto_ok()

    try:
        from Audio_Modules.audio_pool_manager import _is_pipeline_artifact
    except ImportError:
        def _is_pipeline_artifact(f):
            fl = f.lower()
            if fl.startswith("vault_bgm_") or fl.startswith("bgm_"):
                return False
            for bad in ["_master", "_proxy", "_stripped", "_clean", "mask_", "tmp_", "step_", "audio_ducked"]:
                if bad in fl:
                    return True
            return False

    def _is_noisy_or_unusable(fname, meta):
        if not isinstance(meta, dict):
            return False
        if meta.get("is_unusable", False) or meta.get("is_speech_only", False):
            return True
        reason = str(meta.get("unusable_reason", "")).lower()
        vibe = str(meta.get("vibe_tags", [])).lower()
        noise_kws = ("crowd", "babble", "screaming", "car_sound", "traffic", "pollution", "horn", "shouting", "chatter", "mic_static")
        if any(kw in reason for kw in noise_kws) or any(kw in vibe for kw in noise_kws):
            return True
        return False

    def _is_disqualified_by_size_or_blacklist(fname, meta):
        fid = meta.get("file_id") or meta.get("telegram_file_id")
        sc = meta.get("shortcode")
        if _is_bl(audio_filename=fname, audio_shortcode=sc, telegram_file_id=fid):
            logger.info(f"🚫 [BGM Selector] Track '{fname}' is in rejected_audio_blacklist — excluding.")
            return True
        fsize = meta.get("file_size") or 0
        try:
            fsize_f = float(fsize)
            if fsize_f > 20 * 1024 * 1024 and not mtproto_available:
                logger.info(f"🚫 [BGM Selector] Track '{fname}' is {fsize_f/(1024*1024):.1f}MB (>20MB) and MTProto is unconfigured — excluding.")
                return True
        except (ValueError, TypeError):
            pass
        return False

    def _is_too_short(fname, meta):
        if not isinstance(meta, dict):
            return False
        dur = meta.get("duration") or meta.get("duration_sec") or meta.get("audio_duration", 0.0)
        try:
            dur_f = float(dur)
            if dur_f > 0:
                return dur_f < 10.0
        except (ValueError, TypeError):
            pass
        return False

    clip_stem = (clip_id or "").lower().strip()
    folder_stem = os.path.basename(clip_folder or "").lower().strip()

    def _is_self_extracted(fname, meta):
        fn_l = fname.lower()
        if clip_stem and (
            fn_l == f"bgm_{clip_stem}.wav"
            or fn_l.startswith(f"bgm_{clip_stem}")
            or (f"manual_{clip_stem}" in fn_l)
            or (clip_stem in fn_l and "extracted" in fn_l)
        ):
            return True
        if folder_stem and folder_stem in fn_l:
            return True
        sess_val = str(meta.get("session_id", "")).lower()
        if clip_stem and clip_stem in sess_val:
            return True
        return False

    # Filter candidate tracks
    all_candidates = [
        fname for fname, meta in pool_files.items()
        if isinstance(meta, dict)
        and not _is_noisy_or_unusable(fname, meta)
        and not _is_pipeline_artifact(fname)
        and not _is_too_short(fname, meta)
        and not _is_disqualified_by_size_or_blacklist(fname, meta)
        and not _is_self_extracted(fname, meta)
        and fname.lower().endswith((".mp3", ".wav", ".m4a"))
    ]

    if not all_candidates:
        clean_sc = (clip_id or os.path.basename(clip_folder or "")).replace("manual_", "").strip() or "clip"
        from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
        vault = TelegramVaultIndexer()
        retrieved_audio = vault.hydrate_extracted_audio_from_vault(clean_sc, dest_dir=clip_folder)
        if not retrieved_audio and clip_id:
            retrieved_audio = vault.hydrate_extracted_audio_from_vault(clip_id, dest_dir=clip_folder)

        if retrieved_audio and os.path.exists(retrieved_audio) and os.path.getsize(retrieved_audio) > 1024:
            track_name = os.path.basename(retrieved_audio)
            if _is_bl(audio_filename=track_name, audio_shortcode=clean_sc):
                logger.warning(f"🚫 [BGM Selector] Vault extracted audio '{track_name}' is blacklisted — refusing fallback.")
            else:
                logger.info(f"🎙️ [BGM Selector] Retrieved clip's own extracted audio from Telegram Vault: {retrieved_audio}")
                return {
                    "selected_audio_track": track_name,
                    "alignment_score": 0.90,
                    "reasoning": f"Retrieved clip's own continuous audio track ({track_name}) from Telegram Vault.",
                    "physical_path": retrieved_audio,
                }
        logger.warning("🎶 [BGM Selector] No valid musical candidates found in merged pool index and vault has no audio.")
        return {"selected_audio_track": None, "alignment_score": 0.0, "reasoning": "No valid clean BGM tracks in pool."}

    # 3. 6-Hour Cooldown Enforcement
    cooldown_hours = float(os.getenv("AUDIO_COOLDOWN_HOURS", "6.0"))
    cooldown_sec = cooldown_hours * 3600.0
    now = time.time()

    cooling_down_tracks = set()
    for fname in all_candidates:
        meta = pool_files.get(fname, {})
        last_used = float(meta.get("last_used", 0) or 0)
        if last_used > 0 and (now - last_used) < cooldown_sec:
            cooling_down_tracks.add(fname.lower())
            cooling_down_tracks.add(os.path.basename(fname).lower())
            hrs_ago = (now - last_used) / 3600.0
            logger.info(
                f"⏳ [BGM COOLDOWN] Track '{fname}' was used {hrs_ago:.1f}h ago "
                f"(< {cooldown_hours:.1f}h cooldown limit) — disqualifying from candidate pool."
            )

    effective_disqualified = set(disqualified_tracks).union(cooling_down_tracks)

    fresh_candidates = [
        c for c in all_candidates
        if c.lower() not in effective_disqualified and os.path.basename(c).lower() not in effective_disqualified
    ]

    if not fresh_candidates and all_candidates:
        logger.warning(
            f"⚠️ [BGM COOLDOWN] All {len(all_candidates)} candidate tracks were used within the last {cooldown_hours:.1f}h! "
            f"Falling back to least-recently-used track to avoid pipeline failure."
        )
        sorted_by_lru = sorted(
            all_candidates,
            key=lambda c: float(pool_files.get(c, {}).get("last_used", 0) or 0)
        )
        lru_fresh = [c for c in sorted_by_lru if c.lower() not in disqualified_tracks and os.path.basename(c).lower() not in disqualified_tracks]
        available_candidates = lru_fresh if lru_fresh else sorted_by_lru
    else:
        available_candidates = fresh_candidates if fresh_candidates else all_candidates

    # 4. Compute 4D Hybrid Scores & Hard Disqualification
    scored_candidates = []
    disqualified_by_archetype = []

    for c_file in available_candidates:
        meta = pool_files.get(c_file, {})
        final_score, s_arch, s_sem, s_rhythm, s_fatigue = compute_4d_audio_score(
            archetype=archetype,
            candidate_meta=meta,
            candidate_filename=c_file,
            visual_ctx=visual_ctx,
            current_audio=current_audio,
            clip_metadata=local_meta,
        )
        c_fid = str(meta.get("file_id") or meta.get("telegram_file_id") or "")

        if s_arch <= 0.0:
            disqualified_by_archetype.append(c_file)
            logger.debug(f"🚫 [ARCHETYPE FATAL CLASH] Track '{c_file}' disqualified (S_archetype=0.0) for {arch_name}")
            continue

        scored_candidates.append({
            "filename": c_file,
            "file_id": c_fid,
            "score": final_score,
            "s_arch": s_arch,
            "s_sem": s_sem,
            "s_rhythm": s_rhythm,
            "s_fatigue": s_fatigue,
            "meta": meta,
        })

    # Defensive fallback if all available candidates had fatal archetype clashes
    if not scored_candidates and available_candidates:
        logger.warning(f"⚠️ [BGM ARCHETYPE] All {len(available_candidates)} candidates clashed with {arch_name}. Relaxing fatal constraints.")
        for c_file in available_candidates:
            meta = pool_files.get(c_file, {})
            c_fid = str(meta.get("file_id") or meta.get("telegram_file_id") or "")
            scored_candidates.append({
                "filename": c_file,
                "file_id": c_fid,
                "score": 0.50,
                "s_arch": 0.5,
                "s_sem": 0.5,
                "s_rhythm": 0.5,
                "s_fatigue": 0.5,
                "meta": meta,
            })

    # Sort descending by 4D score with tiny jitter for near-ties
    import random
    scored_candidates.sort(key=lambda x: x["score"] + random.uniform(0.0, 0.02), reverse=True)
    clean_scored_candidates = [
        c for c in scored_candidates
        if not _is_bl(c["filename"], audio_shortcode=c.get("meta", {}).get("shortcode"), telegram_file_id=c.get("file_id"))
    ]
    top_candidates = clean_scored_candidates[:15]

    if not top_candidates:
        logger.warning(f"🚫 [BGM Selector] Zero clean candidates remaining for clip '{clip_id}'. Refusing blacklisted audio.")
        return {"selected_audio_track": None, "alignment_score": 0.0, "reasoning": "All candidate tracks are blacklisted."}

    best_math_cand = top_candidates[0]
    selected_track = best_math_cand["filename"]
    selected_file_id = best_math_cand["file_id"]
    alignment_score = float(best_math_cand["score"])
    reasoning = f"Universal 4D Audio Engine Match: Archetype={arch_name} (score={alignment_score:.2f})."

    # 5. Build Hollywood Creative Director Prompt for Gemini Call 2
    cd = visual_ctx.get("content_director", {})
    main_subject = visual_ctx.get("main_subject") or cd.get("main_subject") or ""
    visual_event = visual_ctx.get("visual_event") or cd.get("visual_event") or ""
    clip_tone = visual_ctx.get("tone") or cd.get("tone") or "aspirational"
    editing_style = visual_ctx.get("editing_style") or "rhythm_driven"
    caption = local_meta.get("caption") or visual_ctx.get("caption") or ""
    tags = local_meta.get("hashtags") or visual_ctx.get("hashtags") or []
    hashtags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
    speech_mode = current_audio.get("context", {}).get("speech_mode", "silent_broll")

    top_lines = []
    for rank, cand in enumerate(top_candidates, start=1):
        c_file = cand["filename"]
        meta = cand["meta"]
        c_bpm = float(meta.get("tempo_bpm") or meta.get("bpm") or 120.0)
        c_energy = float(meta.get("energy") or meta.get("avg_energy") or 0.6)
        c_genre = str(meta.get("gemini_genre") or meta.get("genre") or "music")
        vibes = meta.get("vibe_tags") or [meta.get("vibe", "energetic")]
        vibe_str = ", ".join(vibes) if isinstance(vibes, list) else str(vibes)
        c_fid = cand["file_id"]
        fid_tag = f", telegram_file_id='{c_fid}'" if c_fid else ""
        last_used = float(meta.get("last_used", 0) or 0)
        hrs_ago = (now - last_used) / 3600.0 if last_used > 0 else 999.0

        c_lang = str(meta.get("language") or "").strip()
        c_emotion = str(meta.get("dominant_emotion") or meta.get("emotion") or "").strip()
        c_trans = str(meta.get("lyrics") or meta.get("transcript") or "").strip()
        c_cap = str(meta.get("caption") or "").strip()
        c_hook = float(meta.get("hook_time") or 0.0)

        info_parts = [
            f"genre='{c_genre}'",
            f"bpm={c_bpm:.1f}",
            f"energy={c_energy:.2f}",
            f"vibes='{vibe_str}'"
        ]
        if c_lang and c_lang != "unknown":
            info_parts.append(f"lang='{c_lang}'")
        if c_emotion:
            info_parts.append(f"emotion='{c_emotion}'")
        if c_hook > 0:
            info_parts.append(f"hook_drop={c_hook:.1f}s")
        if c_cap:
            cap_preview = (c_cap[:90] + "...") if len(c_cap) > 90 else c_cap
            info_parts.append(f"origin_caption='{cap_preview}'")
        if c_trans:
            trans_preview = (c_trans[:120] + "...") if len(c_trans) > 120 else c_trans
            info_parts.append(f"lyrics='{trans_preview}'")

        info_parts.append(f"archetype_fit={cand['s_arch']:.2f}")
        info_parts.append(f"4d_score={cand['score']:.3f}")
        info_parts.append(f"last_used={hrs_ago:.1f}h_ago")

        top_lines.append(f"#{rank} '{c_file}'{fid_tag}: " + ", ".join(info_parts))
    candidates_str = "\n".join(top_lines)
    all_forbidden = sorted(effective_disqualified.union(set(disqualified_by_archetype)))
    forbidden_str = ", ".join([f"'{t}'" for t in all_forbidden[:30]]) or "None"

    prompt = f"""You are the Lead Music Supervisor and Creative Director for viral short-form video reels.

MISSION:
Select the SINGLE BEST background music track from the candidates below that elevates this video into a viral masterpiece.

=== CREATIVE ARCHETYPE & EDITING DIRECTIVE ===
ARCHETYPE: {arch_name} — {archetype.get('description')}
DIRECTOR DIRECTIVE:
{archetype.get('directive')}

=== ACTIVE VIDEO SCENE BREAKDOWN ===
- Main Subject: {main_subject or 'Unknown Creator / Subject'}
- Visual Event: {visual_event or 'Action sequence'}
- Visual Style / Pacing: {editing_style}
- Emotional Tone: {clip_tone}
- Post Caption: {caption or 'N/A'}
- Hashtags: {hashtags_str or 'N/A'}
- Speech Mode: {speech_mode}

=== CRITICAL RULES ===
1. NEVER pick a track from the FORBIDDEN list.
2. STRICT SPEECH & NOISE BAN: Never pick crowd babble, microphone static, ambient street chatter, or spoken dialogue as music.
3. ARCHETYPE ALIGNMENT: Obey the Director Directive above. Match the energy, rhythm, and genre strictly to the archetype.
4. SEMANTIC & CULTURAL RESONANCE (CRITICAL):
   - You MUST match the cultural, linguistic, and emotional world of the video.
   - For Bollywood, Hindi, Indian, or Desi creators/scenes: prioritize tracks with matching Indian/Bollywood vibes, Hindi lyrics, or culturally authentic rhythm. NEVER pick an incongruous Western phonk, trap, or unrelated podcast track.
   - For fashion/glamour struts: pick tracks with swagger, luxury, beat drop, or high-fashion elegance.
   - Match the lyrics and mood to what is visually occurring (e.g. romance, celebration, fitness, comedy).
5. Return ONLY valid JSON with 'selected_audio_track', 'telegram_file_id', 'alignment_score', and 'reasoning'.

[FORBIDDEN TRACKS — DO NOT SELECT]
{forbidden_str}

[TOP CANDIDATE MUSIC PROFILES (Ranked by 4D Intelligence with Lyrics & Origin Context)]
{candidates_str}

Return ONLY valid JSON:
{{
  "selected_audio_track": "exact_candidate_filename.wav",
  "telegram_file_id": "file_id_if_available",
  "alignment_score": 0.95,
  "reasoning": "Creative director explanation: why this track's lyrics/subject/beat match the video narrative and cultural feel."
}}
"""

    # 6. Query Gemini Router
    try:
        try:
            from Gemini_Modules.gemini_router_module.gemini_governor import gemini_router
        except Exception:
            try:
                from Intelligence_Modules.gemini_governor import gemini_router
            except Exception:
                from gemini_governor import gemini_router

        if gemini_router:
            logger.info("🎶 [BGM Selector] Querying Gemini Router across model vanguard...")
            raw_response = gemini_router.generate(
                task_type="analysis",
                prompt=prompt,
                module_name="bgm_selector",
                gen_config={"temperature": 0.3}
            )
            if raw_response:
                data = json.loads(_clean_json(raw_response))
                win_track = data.get("selected_audio_track")
                win_fid = data.get("telegram_file_id")

                all_valid_names = set(c["filename"].lower() for c in top_candidates)
                if win_track and win_track.lower() in all_valid_names:
                    # ZERO-TRUST SECURITY GATE: Veto if win_track or win_fid is blacklisted
                    if _is_bl(win_track, telegram_file_id=win_fid):
                        logger.warning(
                            f"🚫 [BGM Selector - ZERO TRUST VETO] Gemini returned blacklisted track '{win_track}'! "
                            f"Vetoing and retaining verified clean 4D winner '{selected_track}'."
                        )
                    else:
                        selected_track = win_track
                        reasoning = data.get("reasoning", reasoning)
                        alignment_score = float(data.get("alignment_score", 0.92))
                        if win_fid:
                            selected_file_id = win_fid
                        else:
                            selected_file_id = pool_files.get(selected_track, {}).get("file_id", selected_file_id)
                        logger.info(f"🎶 [BGM Selector - Gemini Call 2] Winner: '{selected_track}' (score={alignment_score:.2f})")
                else:
                    logger.warning(f"🎶 [BGM Selector] Gemini returned unknown track '{win_track}' — forcing top 4D math winner '{selected_track}'.")
    except Exception as e:
        logger.warning(f"🎶 [BGM Selector - Gemini Call 2] Router fallback to top 4D math winner: {e}")

    # Final zero-trust verification on selected_track
    if selected_track and _is_bl(selected_track, telegram_file_id=selected_file_id):
        logger.error(f"🚫 [BGM Selector FATAL VETO] Selected track '{selected_track}' failed final zero-trust blacklist check! Selecting next clean candidate.")
        selected_track = None
        for c in top_candidates:
            if not _is_bl(c["filename"], telegram_file_id=c.get("file_id")):
                selected_track = c["filename"]
                selected_file_id = c.get("file_id")
                alignment_score = float(c["score"])
                break

    # Compute backup candidates for resilient Step 04 failover (strictly non-blacklisted)
    backup_candidates = [
        c["filename"] for c in top_candidates
        if c["filename"] != selected_track
        and not _is_bl(c["filename"], telegram_file_id=c.get("file_id"))
    ][:10]

    # Map candidate filenames to their Telegram file_ids for direct hydration
    cand_fids = {}
    for c in top_candidates:
        fn = c.get("filename")
        if fn:
            fid = pool_files.get(fn, {}).get("file_id") or pool_files.get(fn, {}).get("telegram_file_id")
            if fid:
                cand_fids[fn] = fid

    store.patch_bgm_selection(clip_data, selected_track, reasoning, alignment_score)
    store.save(clip_id, clip_data, clip_folder)

    win_meta = pool_files.get(selected_track, {}) if selected_track else {}
    return {
        "selected_audio_track": selected_track,
        "telegram_file_id": selected_file_id,
        "alignment_score": alignment_score,
        "reasoning": reasoning,
        "backup_candidates": backup_candidates,
        "candidate_file_ids": cand_fids,
        "archetype": arch_name,
        "hook_time": float(win_meta.get("hook_time") or 0.0),
        "lyrics": str(win_meta.get("lyrics") or win_meta.get("transcript") or ""),
        "dominant_emotion": str(win_meta.get("dominant_emotion") or ""),
    }


def compute_routing_parameters(
    lyric_intel: Optional[Dict[str, Any]],
    forensic_context: Optional[Dict[str, Any]],
    selected_bgm_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Translates lyric & music intelligence + visual context into mathematical FFmpeg synthesis parameters.
    Replaces standalone audio_rhythm_router.py by integrating semantic routing directly.
    """
    emotion = str((lyric_intel or {}).get("dominant_emotion", "hype")).lower().strip()
    energy  = str((lyric_intel or {}).get("energy_profile", "medium")).lower().strip()

    v_ctx = forensic_context or {}
    intent  = str(v_ctx.get("intent", "viral_reel")).lower()
    tone    = str(v_ctx.get("tone", "aspirational")).lower()

    if emotion in ("sadness", "melancholic", "nostalgia") or tone in ("dramatic", "melancholic"):
        strategy_name = "CINEMATIC_EMOTIONAL_SLOWMO"
        speed_factor  = 0.75
        transition_type = "dissolve"
        transition_duration = 0.8
        ducking_db = -3.0
        target_cut_interval = 2.8
    elif intent in ("podcast_speech", "speech") or emotion == "conversational":
        strategy_name = "VOICEOVER_PODCAST_DUCKING"
        speed_factor  = 1.0
        transition_type = "fade"
        transition_duration = 0.4
        ducking_db = -18.0
        target_cut_interval = 3.5
    elif energy in ("explosive", "high") or emotion in ("hype", "power", "euphoria", "joy", "celebration"):
        strategy_name = "BEAT_SNAPPED_HIGH_ENERGY"
        speed_factor  = 1.1
        transition_type = "glitch"
        transition_duration = 0.2
        ducking_db = -6.0
        target_cut_interval = 0.8
    else:
        strategy_name = "DYNAMIC_CONTRAST_REEL"
        speed_factor  = 1.0
        transition_type = "whip_pan"
        transition_duration = 0.3
        ducking_db = -6.0
        target_cut_interval = 1.5

    return {
        "strategy_name": strategy_name,
        "speed_factor": speed_factor,
        "transition_type": transition_type,
        "transition_duration": transition_duration,
        "bgm_ducking_db": ducking_db,
        "music_volume": 0.85 if strategy_name != "VOICEOVER_PODCAST_DUCKING" else 0.15,
        "target_cut_interval": target_cut_interval,
        "selected_audio_path": selected_bgm_path,
        "dominant_emotion": emotion,
        "energy_profile": energy,
        "recommended_editing_mode": strategy_name
    }
