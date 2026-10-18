"""
Gemini_Modules/forensic_analyzer.py
------------------------------------
Forensic Video Analyzer — video + corrected transcript -> timestamped explanation.

Pipeline (one Gemini video call):

    source video ──► proxy_encoder.encode_for_gemini()  (compressed, Gemini-tuned)
    clip audio   ──► whisper_transcriber ──► transcript_corrector  (final transcript)
                                   │
    compressed video + final transcript ──► Gemini ──► structured analysis

What Gemini returns, and what the CODE adds:
  Gemini : overview, visual timeline (scenes), per-transcript-line visual alignment,
           key moments, transcript conflicts, classification, watermarks, director block.
  Code   : attaches the EXACT transcript text/timestamps to every timeline scene
           (never trusts Gemini to copy speech), normalizes/clamps all timestamps,
           converts watermark boxes to source-pixel coordinates, enforces safety
           invariants, derives speech_intelligence and visual_vectors from real data.

Backward compatible: still returns intent / feature_flags / safety / editing_style /
content_strategy / content_director / creative_possibilities / selected_audio_track /
watermarks(x,y,w,h) / speech_intelligence / visual_vectors / clip_intelligence.
New: result["video_analysis"], result["transcript_meta"], result["video_proxy"].

Use render_explanation(result) for the human-readable timestamped report.
"""

import copy
import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from Gemini_Modules.gemini_router_module.gemini_governor import gemini_router
except Exception:
    try:
        from Gemini_Modules.gemini_governor import gemini_router
    except Exception:
        try:
            from gemini_governor import gemini_router
        except Exception:
            gemini_router = None

from dotenv import load_dotenv

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(os.path.join(_REPO_ROOT, ".env")):
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=True)
elif os.path.exists(os.path.join(_REPO_ROOT, "Credentials", ".env")):
    load_dotenv(os.path.join(_REPO_ROOT, "Credentials", ".env"), override=True)
else:
    load_dotenv(".env", override=True)

logger = logging.getLogger("forensic_analyzer")

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")      # only used if router is unavailable
_FPS_ENV = os.getenv("FORENSIC_VIDEO_FPS", "auto").strip().lower()  # "auto" or a number, e.g. "1"
AUTO_TRANSCRIBE = os.getenv("FORENSIC_AUTO_TRANSCRIBE", "true").lower() in ("true", "1", "yes")
GEMINI_TIMEOUT_SEC = float(os.getenv("FORENSIC_GEMINI_TIMEOUT", "180"))
FILE_ACTIVE_TIMEOUT_SEC = 120.0
MAX_ATTEMPTS = 2                      # one retry, only for unparseable output
_PROMPT_VERSION = 1
_MAX_TRANSCRIPT_CHARS = 14000
_MAX_WATERMARK_AREA = 0.35            # reject boxes covering >35% of the frame

ALLOWED_INTENTS = {
    "educational_explainer", "kids_animation", "tech_review", "fitness_action", "nature_travel",
    "fashion_lifestyle", "meme_viral", "food_cooking", "talking_head", "music_performance",
    "general_content",
}
ALLOWED_STYLES = {
    "educational_tutorial", "playful_animation", "fast_social", "cinematic", "documentary",
    "product_review", "fashion_showcase", "vlog", "news", "podcast_clip",
}
SHOT_TYPES = {"closeup", "medium", "wide", "aerial", "screen_recording", "text_card", "other"}
CAMERA_MOTIONS = {"static", "pan", "zoom", "handheld", "tracking", "mixed"}
SPEECH_RELATIONS = {
    "no_speech", "speaker_on_camera", "voiceover_matches_visual", "voiceover_b_roll",
    "lip_sync_to_audio", "singing_performance", "speech_unrelated_to_visual", "off_screen_speaker",
}
_SPOKEN_RELATIONS = {
    "speaker_on_camera", "voiceover_matches_visual", "voiceover_b_roll",
    "speech_unrelated_to_visual", "off_screen_speaker",
}
ALIGN_RELATIONS = {"matches_visual", "b_roll", "speaker_on_camera", "mismatch", "unclear"}
MOMENT_TYPES = {
    "hook", "punchline", "reveal", "peak", "cta", "transition", "product_shown",
    "text_overlay", "emotional_beat", "other",
}
FLAG_NAMES = (
    "enable_price_tags", "enable_fashion_caption", "enable_cinematic_zoom",
    "enable_speed_ramps", "enable_fast_pacing", "enable_voiceover",
)

# ── Prompt ────────────────────────────────────────────────────────────────────

FORENSIC_PROMPT = """You are a Professional Short-Form Content Director and forensic video analyst.

INPUTS
  (1) VIDEO — a compressed proxy of the clip: <<W>>x<<H>>, <<FPS>> frame(s) per second, <<DUR>>s long, original audio included.
  (2) TRANSCRIPT — the speech in that video. It was produced by Whisper and then corrected by a second
      model that listened to the audio. It is given below with timestamps.

YOUR JOB
Write a detailed, timestamped explanation of the video that CONNECTS what is seen with what is said,
then classify it for editing.

Your output has two consumers:
  A. A second-stage AI editor (Call 3) reads your JSON as raw text and builds an FFmpeg edit plan. It never
     sees the video — your fields are its ONLY visual understanding. Vague output degrades its plan.
  B. Your output is stored permanently as retrieval/training signal. Precision and honesty matter more than
     confident-sounding guesses; a wrong confident answer pollutes that store.

TIMESTAMP RULES
- Every timestamp is in SECONDS (float) from the very start of the video (t=0). Never MM:SS.
- Transcript timestamps are ground truth for WHEN speech happens. Anchor speech-related events to them;
  do not re-estimate speech timing from the video.
- Your visual timing comes from ~<<FPS>> frame(s)/second, so visual boundaries are only good to about
  +/-<<PREC>>s. Do not claim finer precision.

GROUNDING RULES
- Report only what is visible or audible. Do not infer off-screen events, intent, or context you cannot observe.
- Do NOT invent speech. Speech content comes from the transcript. If the transcript says there is no speech
  or is unavailable, describe the audio you hear (music, effects) and put any speech claim in "uncertainties".
- If the video is ambiguous, too short, or too sparse, say so in "uncertainties" and lower "confidence".
  "general_content" is a valid, useful answer — do not stretch a category.
- Mixed content: classify by dominant visual weight and name the secondary type in the overview.
- Never identify real people from their faces. Describe them neutrally ("a woman in a red jacket"). You may
  use a name that is SPOKEN in the transcript or SHOWN in on-screen text, and say which one it came from.

TASK 1 — DETAILED EXPLANATION
1. overview.summary: 5-8 sentences. What happens from start to finish, who/what is in it, where, why it
   exists, and how the speech and visuals work together.
2. timeline: contiguous scenes covering t=0 to the end with no gaps or overlaps. Start a new scene when the
   shot, subject, location, on-screen text, or the speech/visual relationship meaningfully changes.
   For each scene describe concretely: who/what, action, framing, camera movement, lighting/colour, on-screen
   text VERBATIM. Do NOT copy the transcript into scenes — the system attaches exact speech afterwards.
   Instead state speech_relation and, in speech_visual_note, how the words relate to what is shown
   (e.g. says "this jacket" while the jacket is on screen).
3. transcript_alignment: one entry for EVERY transcript line id: a short visual_at_time (max 20 words),
   whether the speaker is visible, and the relation.
4. key_moments: the moments an editor must know about (hook, punchline, reveal, peak, cta, product shown,
   text overlay, emotional beat), with importance 1-5.
5. transcript_conflicts: ONLY where the video gives concrete evidence that a transcript word is wrong
   (on-screen subtitle, name card, product label, clear lip movement mismatch). Do not "fix" the transcript
   from audio guesses — that was already done. Usually empty.
6. speaker_analysis and uncertainties: be explicit about what you could not determine.

TASK 2 — CLASSIFICATION
`intent` — single best fit:
  educational_explainer (tutorials, how-tos) | kids_animation | tech_review | fitness_action |
  nature_travel | fashion_lifestyle (outfits, beauty, celebrity highlights) | meme_viral |
  food_cooking | talking_head (podcast, interview, commentary, news) |
  music_performance (singing, instruments, dance, concerts) | general_content (no clear fit).

Feature flag defaults (override when the evidence contradicts them):
- Educational / Tutorial / Talking Head -> enable_voiceover=true, enable_speed_ramps=false.
- Animation -> enable_cinematic_zoom=true.  Fashion -> enable_fashion_caption=true, enable_cinematic_zoom=true;
  enable_price_tags only if products/apparel are visibly the focus.
- Fitness / Action -> enable_speed_ramps=true, enable_fast_pacing=true.
- Nature / Travel -> enable_cinematic_zoom=true, enable_fast_pacing=false.
- Music / Performance -> enable_speed_ramps=true if rhythm/choreography-driven, else cinematic zoom.
- Food -> enable_cinematic_zoom=true; enable_voiceover=true if narration is present in the transcript.
- meme_viral / general_content -> enable_fast_pacing=true unless the clip is slow and narrative.
Voiceover legibility beats fast pacing when they conflict.

`confidence` (0.0-1.0) reflects evidence quality, and Call 3 reads it as a trust signal:
  0.8-1.0 unambiguous across the clip | 0.5-0.79 likely but mixed/low-detail | <0.5 too sparse or
  contradictory — prefer general_content.

`safety.classification`:
- "safe": brand-safe, monetizable.
- "risky": borderline — mild innuendo, aggressive language (check the transcript too), alcohol/tobacco visible
  but not the focus, intense non-graphic action, unverified claims.
- "blocked": nudity/sexual content, graphic violence/gore, weapons used to threaten, visible drug use,
  hate symbols or hateful speech, content endangering minors, self-harm depiction, or copyrighted broadcast
  footage (sports leagues, films, TV) as the primary subject.
Any "blocked" trigger forces monetization_safe=false.



TASK 4 — CONTENT DIRECTOR (brief a human editor who has never seen the footage)
detected_entities (only visible ones, e.g. "person:female", "environment:indoor"), visual_event (1-2 concrete
sentences), viewer_attention (what grabs the eye in the first second), internet_context (recognizable
references, neutral wording), possible_narratives and recommended_narrative from
[educational_guide, playful_story, tech_breakdown, fitness_motivation, scenic_journey, fashion_moment,
celebrity_highlight, humor_reaction, recipe_walkthrough, commentary_talk, music_showcase],
tone from [educational, playful, aspirational, dramatic, humorous, calm_ambient, hype, informative, neutral],
editing_style from the style list below, engagement_hook (one concrete sentence for what should appear/be said
in the first 3 seconds, grounded in THIS clip), feature_commands echoing feature_flags.

Return ONLY one valid JSON object. No markdown, no commentary.

{
  "overview": {
    "summary": "",
    "setting": "",
    "subjects": [],
    "spoken_language": "",
    "narrative_arc": "<hook -> development -> payoff, 1-2 sentences>",
    "target_audience": "",
    "on_screen_text_summary": ""
  },
  "timeline": [
    {
      "start": <float>, "end": <float>, "scene_label": "",
      "visual": "<detailed, concrete description>",
      "on_screen_text": [],
      "shot_type": "closeup|medium|wide|aerial|screen_recording|text_card|other",
      "camera_motion": "static|pan|zoom|handheld|tracking|mixed",
      "non_speech_audio": "<music/effects heard, or empty>",
      "mood": "",
      "speech_relation": "no_speech|speaker_on_camera|voiceover_matches_visual|voiceover_b_roll|lip_sync_to_audio|singing_performance|speech_unrelated_to_visual|off_screen_speaker",
      "speech_visual_note": "",
      "edit_notes": "<what an editor should know: cut-safe, must-keep, trim candidate>"
    }
  ],
  "transcript_alignment": [
    {"id": <int transcript line id>, "visual_at_time": "", "speaker_on_screen": true|false|null,
     "relation": "matches_visual|b_roll|speaker_on_camera|mismatch|unclear"}
  ],
  "key_moments": [
    {"time": <float>, "type": "hook|punchline|reveal|peak|cta|transition|product_shown|text_overlay|emotional_beat|other",
     "description": "", "importance": <int 1-5>}
  ],
  "transcript_conflicts": [
    {"id": <int>, "video_evidence": "", "suggested_text": ""}
  ],
  "speaker_analysis": {
    "visible_people": <int>,
    "is_talking_on_camera": <bool>,
    "voice_matches_visible_speaker": true|false|null,
    "notes": ""
  },
  "uncertainties": [],

  "intent": "<intent>",
  "confidence": <float 0.0-1.0>,
  "editing_style": "educational_tutorial|playful_animation|fast_social|cinematic|documentary|product_review|fashion_showcase|vlog|news|podcast_clip",
  "feature_flags": {
    "enable_price_tags": <bool>, "enable_fashion_caption": <bool>, "enable_cinematic_zoom": <bool>,
    "enable_speed_ramps": <bool>, "enable_fast_pacing": <bool>, "enable_voiceover": <bool>
  },
  "platform_priority": ["youtube_shorts", "instagram_reels", "facebook_reels"],
  "safety": {"classification": "safe|risky|blocked", "monetization_safe": <bool>},
  "content_director": {
    "detected_entities": [], "visual_event": "", "viewer_attention": "", "internet_context": [],
    "possible_narratives": [], "recommended_narrative": "", "tone": "", "editing_style": "",
    "is_talking_on_camera": <bool>, "engagement_hook": "",
    "feature_commands": {
      "enable_fast_pacing": <bool>, "enable_cinematic_zoom": <bool>, "enable_speed_ramps": <bool>,
      "enable_voiceover": <bool>, "enable_price_tags": <bool>, "enable_fashion_caption": <bool>
    }
  }
}

<<TRANSCRIPT_BLOCK>>
Creator Handle Hint: '<<CREATOR>>'
"""

_BGM_TASK = """
---

TASK 5 — BGM AUDIO SELECTION & CREATIVE POSSIBILITIES

[Candidate Background Music Tracks Available]
<<AUDIO_TABLE>>

Judge the video (and its speech, per the transcript) against the candidate tracks:
1. "selected_audio_track": the SINGLE best matching track filename (e.g. 'Fit_girl_2.mp3').
2. "creative_possibilities": 2-3 edit options as objects:
   [{"rank": 1, "clip_label": "hero_hook", "target_duration_sec": 15, "editing_style": "rhythm_driven"},
    {"rank": 2, "clip_label": "action_cut", "target_duration_sec": 30, "editing_style": "fast_cuts"}]
Add both fields to the returned JSON object.
"""

# ── Defaults ──────────────────────────────────────────────────────────────────

def _default_result() -> Dict[str, Any]:
    flags = {
        "enable_price_tags": False, "enable_fashion_caption": False,
        "enable_cinematic_zoom": True, "enable_speed_ramps": False,
        "enable_fast_pacing": False, "enable_voiceover": True,
    }
    return {
        "watermarks": [],
        "intent": "unknown",
        "confidence": 0.0,
        "editing_style": "cinematic",
        "feature_flags": dict(flags),
        "platform_priority": ["youtube_shorts", "instagram_reels", "facebook_reels"],
        "safety": {"classification": "risky", "monetization_safe": False},
        "content_strategy": {
            "intent": "unknown", "confidence": 0.0,
            "recommended_editing_style": "cinematic",
            "feature_flags": dict(flags), "safety": "risky",
        },
    }


DEFAULT_RESULT = _default_result()


# ── Small helpers ─────────────────────────────────────────────────────────────

def _f(v: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _s(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _str_list(v: Any) -> List[str]:
    return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []


def _clamp01(v: Any, default: float = 0.5) -> float:
    return max(0.0, min(1.0, _f(v, default)))


def _enum(v: Any, allowed: set, default: str) -> str:
    v = _s(v).lower()
    return v if v in allowed else default


def _clean_json(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        text = m.group(1) if m else text.replace("```json", "").replace("```", "")
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    return re.sub(r",\s*([\]}])", r"\1", text.strip())


def _fmt_ts(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


# ── Normalizers (pure functions) ──────────────────────────────────────────────

def _normalize_timeline(raw: Any, duration: float) -> Tuple[List[Dict[str, Any]], float]:
    """Clamp, sort, de-overlap. Returns (timeline, coverage_ratio)."""
    items = []
    if isinstance(raw, list):
        for e in raw:
            if not isinstance(e, dict):
                continue
            s, en = _f(e.get("start"), None), _f(e.get("end"), None)
            if s is None or en is None:
                continue
            s = max(0.0, s)
            if duration:
                en = min(duration, en)
            if en > s:
                items.append((s, en, e))
    items.sort(key=lambda x: x[0])

    out, prev_end = [], 0.0
    for s, en, e in items:
        s = max(s, prev_end)
        if en - s < 0.05:
            continue
        out.append({
            "start": round(s, 2), "end": round(en, 2),
            "scene_label": _s(e.get("scene_label")),
            "visual": _s(e.get("visual")),
            "on_screen_text": _str_list(e.get("on_screen_text")),
            "shot_type": _enum(e.get("shot_type"), SHOT_TYPES, "other"),
            "camera_motion": _enum(e.get("camera_motion"), CAMERA_MOTIONS, "static"),
            "non_speech_audio": _s(e.get("non_speech_audio")),
            "mood": _s(e.get("mood")),
            "speech_relation": _enum(e.get("speech_relation"), SPEECH_RELATIONS, "no_speech"),
            "speech_visual_note": _s(e.get("speech_visual_note")),
            "edit_notes": _s(e.get("edit_notes")),
        })
        prev_end = en
    covered = sum(x["end"] - x["start"] for x in out)
    return out, round(covered / duration, 3) if duration else 0.0


def _attach_transcript(
    timeline: List[Dict[str, Any]],
    alignment_raw: Any,
    conflicts_raw: Any,
    transcript: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    segs = (transcript or {}).get("segments") or []
    by_id = {s["id"]: s for s in segs}

    covered_ids = set()
    for e in timeline:
        ov = [s for s in segs if s["start"] < e["end"] and s["end"] > e["start"]]
        covered_ids.update(s["id"] for s in ov)
        e["speech_segments"] = [{"id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"]} for s in ov]
        e["speech_text"] = " ".join(s["text"] for s in ov)
        if ov and e["speech_relation"] == "no_speech":
            e["speech_relation"] = "unspecified"
        if not ov and segs and e["speech_relation"] in _SPOKEN_RELATIONS:
            e["speech_relation"] = "no_speech"

    gem_align: Dict[int, Dict[str, Any]] = {}
    if isinstance(alignment_raw, list):
        for a in alignment_raw:
            try:
                gem_align[int(a.get("id"))] = a
            except (TypeError, ValueError, AttributeError):
                continue
    alignment = []
    for s in segs:
        a = gem_align.get(s["id"], {})
        on_screen = a.get("speaker_on_screen")
        alignment.append({
            "id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"],
            "visual_at_time": _s(a.get("visual_at_time")),
            "speaker_on_screen": on_screen if isinstance(on_screen, bool) else None,
            "relation": _enum(a.get("relation"), ALIGN_RELATIONS, "unclear"),
        })

    conflicts = []
    if isinstance(conflicts_raw, list):
        for c in conflicts_raw:
            try:
                cid = int(c.get("id"))
            except (TypeError, ValueError, AttributeError):
                continue
            if cid in by_id and _s(c.get("suggested_text")):
                conflicts.append({
                    "id": cid, "transcript_text": by_id[cid]["text"],
                    "video_evidence": _s(c.get("video_evidence")),
                    "suggested_text": _s(c.get("suggested_text")),
                })
    uncovered = sorted(i for i in by_id if i not in covered_ids)
    return timeline, alignment, conflicts, uncovered


def _normalize_key_moments(raw: Any, duration: float) -> List[Dict[str, Any]]:
    out = []
    if isinstance(raw, list):
        for m in raw:
            if not isinstance(m, dict):
                continue
            t = _f(m.get("time"), None)
            if t is None or t < 0 or (duration and t > duration + 0.5):
                continue
            out.append({
                "time": round(min(t, duration) if duration else t, 2),
                "type": _enum(m.get("type"), MOMENT_TYPES, "other"),
                "description": _s(m.get("description")),
                "importance": int(max(1, min(5, _f(m.get("importance"), 3)))),
            })
    return sorted(out, key=lambda x: x["time"])


def _convert_watermarks(raw: Any, src_w: int, src_h: int, duration: float) -> List[Dict[str, Any]]:
    """Gemini box_2d (0-1000 normalized) -> source-pixel x,y,w,h, with sanity rejection."""
    out = []
    if not isinstance(raw, list) or src_w <= 0 or src_h <= 0:
        return out
    for wm in raw:
        box = wm.get("box_2d") if isinstance(wm, dict) else None
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        try:
            ymin, xmin, ymax, xmax = [max(0.0, min(1000.0, float(v))) for v in box]
        except (TypeError, ValueError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        if (xmax - xmin) * (ymax - ymin) / 1e6 > _MAX_WATERMARK_AREA:
            logger.info("🔬 Dropped implausibly large watermark box (likely scene content).")
            continue
        x, y = int(round(xmin / 1000 * src_w)), int(round(ymin / 1000 * src_h))
        w, h = int(round((xmax - xmin) / 1000 * src_w)), int(round((ymax - ymin) / 1000 * src_h))
        if w < 2 or h < 2:
            continue
        out.append({
            "x": x, "y": y, "w": w, "h": h,
            "norm_box": [round(ymin), round(xmin), round(ymax), round(xmax)],
            "label": _s(wm.get("label")),
            "first_seen": round(max(0.0, _f(wm.get("first_seen"), 0.0)), 2),
            "last_seen": round(min(duration, _f(wm.get("last_seen"), duration)) if duration else _f(wm.get("last_seen"), 0.0), 2),
            "confidence": round(_clamp01(wm.get("confidence"), 0.5), 2),
            "coord_space": f"source_{src_w}x{src_h}",
        })
    return out


def _derive_speech_intel(is_talking: bool, transcript: Optional[Dict[str, Any]], p1_audio: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    has_audio = bool(p1_audio.get("has_audio", True)) if p1_audio else True
    if transcript is not None:
        has_vocals = bool(transcript.get("has_speech"))
        is_speech_vocal = has_vocals and not bool(transcript.get("is_singing"))
    elif p1_audio:
        has_vocals = bool(p1_audio.get("has_vocals", False))
        is_speech_vocal = bool(p1_audio.get("is_speech_vocal", False))
    else:
        has_vocals, is_speech_vocal = is_talking, is_talking

    if not has_vocals:
        mode, action = "silent_broll", "audio_replace_full_bgm"
    elif is_talking and is_speech_vocal:
        mode, action = "on_camera_dialogue", "preserve_voice_duck_bgm"
    elif is_talking and not is_speech_vocal:
        mode, action = "lip_sync_dub", "audio_replace_full_bgm"
    elif not is_talking and is_speech_vocal:
        mode, action = "voiceover_narration", "preserve_voice_duck_bgm"
    else:
        mode, action = "music_broll", "audio_replace_full_bgm"
    if not has_audio:
        mode, action = "silent_broll", "audio_replace_full_bgm"
    return {
        "speech_mode": mode,
        "is_talking_visually": is_talking,
        "has_spoken_vocal": has_vocals,
        "is_speech_vocal": is_speech_vocal,
        "recommended_audio_action": action,
    }


def _derive_visual_vectors(va: Dict[str, Any], duration: float) -> Dict[str, Any]:
    ts = {0.0}
    for e in va.get("timeline", []):
        ts.add(round(e["start"], 2))
        ts.add(round((e["start"] + e["end"]) / 2, 2))
    for m in va.get("key_moments", []):
        ts.add(m["time"])
    targeted = sorted(t for t in ts if 0 <= t <= (duration or 1e9))
    cuts = sorted({e["start"] for e in va.get("timeline", []) if e["start"] > 0.05})

    climax = None
    ranked = [m for m in va.get("key_moments", []) if m["type"] in ("peak", "reveal", "punchline")]
    if ranked:
        climax = max(ranked, key=lambda m: m["importance"])["time"]
    if climax is None:
        climax = round(duration * 0.75, 2) if duration else 10.0
    return {
        "targeted_timestamps_sec": targeted,
        "scene_cut_timestamps": cuts,
        "hook_zone_end_sec": round(min(5.0, duration), 2) if duration else 5.0,
        "climax_zone_start_sec": climax,
    }


# ── Report renderer ───────────────────────────────────────────────────────────

def render_explanation(result: Dict[str, Any]) -> str:
    """Human-readable, timestamped explanation: visuals and speech interleaved."""
    va = result.get("video_analysis") or {}
    ov = va.get("overview") or {}
    lines = []
    lines.append("VIDEO EXPLANATION")
    lines.append("=" * 60)
    if ov.get("summary"):
        lines += [ov["summary"], ""]
    for label, key in (("Setting", "setting"), ("Language", "spoken_language"),
                       ("Arc", "narrative_arc"), ("Audience", "target_audience"),
                       ("On-screen text", "on_screen_text_summary")):
        if ov.get(key):
            lines.append(f"{label}: {ov[key]}")
    if ov.get("subjects"):
        lines.append("Subjects: " + ", ".join(ov["subjects"]))
    lines.append(f"Intent: {result.get('intent')} (confidence {result.get('confidence')}) | "
                 f"Style: {result.get('editing_style')} | Safety: {result.get('safety', {}).get('classification')}")

    lines += ["", "TIMELINE", "-" * 60]
    for e in va.get("timeline", []):
        lines.append(f"[{_fmt_ts(e['start'])} - {_fmt_ts(e['end'])}] {e.get('scene_label') or 'scene'}  "
                     f"({e['shot_type']}, {e['camera_motion']}, {e['mood'] or 'neutral'})")
        lines.append(f"  VISUAL : {e['visual']}")
        if e.get("on_screen_text"):
            lines.append(f"  TEXT   : {' | '.join(e['on_screen_text'])}")
        if e.get("non_speech_audio"):
            lines.append(f"  AUDIO  : {e['non_speech_audio']}")
        for s in e.get("speech_segments", []):
            lines.append(f"  SPEECH : [{_fmt_ts(s['start'])}-{_fmt_ts(s['end'])}] \"{s['text']}\"")
        lines.append(f"  LINK   : {e['speech_relation']}" + (f" — {e['speech_visual_note']}" if e.get("speech_visual_note") else ""))
        if e.get("edit_notes"):
            lines.append(f"  EDIT   : {e['edit_notes']}")
        lines.append("")

    if va.get("key_moments"):
        lines += ["KEY MOMENTS", "-" * 60]
        for m in va["key_moments"]:
            lines.append(f"[{_fmt_ts(m['time'])}] {m['type']} (importance {m['importance']}): {m['description']}")
        lines.append("")
    if va.get("transcript_conflicts"):
        lines += ["TRANSCRIPT CONFLICTS (video evidence)", "-" * 60]
        for c in va["transcript_conflicts"]:
            lines.append(f"#{c['id']}: \"{c['transcript_text']}\" -> \"{c['suggested_text']}\"  [{c['video_evidence']}]")
        lines.append("")
    if va.get("uncertainties"):
        lines += ["UNCERTAINTIES", "-" * 60] + [f"- {u}" for u in va["uncertainties"]] + [""]
    return "\n".join(lines)


# ── Main class ────────────────────────────────────────────────────────────────

class ForensicVideoAnalyzer:
    """
    Usage:
        analyzer = ForensicVideoAnalyzer()
        result = analyzer.analyze(video_path)                       # auto Whisper -> corrector
        result = analyzer.analyze(video_path, transcript=my_final)  # bring your own transcript
        print(render_explanation(result))
    """

    def __init__(self):
        self.router = gemini_router
        self._api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._available = bool(self._api_key)
        if not self._available:
            logger.warning("🔬 ForensicAnalyzer: GEMINI_API_KEY not set — will return defaults")
        else:
            logger.info(f"🔬 ForensicAnalyzer: ACTIVE (video+transcript mode, router={'yes' if self.router else 'no'})")

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        video_path: str,
        frame_paths: Optional[List[str]] = None,       # accepted for compatibility; ignored (video mode)
        creator_name: Optional[str] = None,
        audio_candidates: Optional[List[dict]] = None,
        transcript: Optional[Dict[str, Any]] = None,   # output of transcript_corrector.correct_transcript()
        use_cache: bool = True,
    ) -> dict:
        scene_context: Dict[str, Any] = {}
        try:
            scene_context = self._scene_intel(video_path, creator_name)
            if frame_paths:
                logger.debug("🔬 frame_paths ignored — analyzer now sends the compressed video.")
            if not self._available:
                return self._fallback("analyzer_unavailable", scene_context)
            if not video_path or not os.path.isfile(video_path):
                return self._fallback("video_not_found", scene_context)

            clip_folder = os.path.dirname(os.path.abspath(video_path))
            clip_id = os.path.basename(clip_folder)
            p1_audio = self._load_p1_audio(clip_folder)

            # 1. Compressed video
            try:
                proxy = self._prepare_proxy(video_path, clip_folder)
            except Exception as pe:
                logger.error(f"🔬 Proxy encode failed: {pe}")
                return self._fallback(f"proxy_encode_failed: {pe}", scene_context)

            # 2. Final transcript (Whisper -> Gemini corrector), or the one supplied
            transcript = self._resolve_transcript(transcript, video_path, clip_folder, clip_id, proxy, p1_audio, creator_name)

            # 3. Gemini (cached on proxy + transcript + inputs)
            cache_path = os.path.join(clip_folder, "forensic_analysis.json")
            key = self._cache_key(proxy, transcript, creator_name, audio_candidates)
            result = self._load_cache(cache_path, key) if use_cache else None
            if result is None:
                prompt = self._build_prompt(proxy, transcript, creator_name, audio_candidates)
                try:
                    data = self._call_gemini(proxy, prompt)
                except Exception as ge:
                    logger.error(f"🔬 Gemini video analysis failed: {ge}")
                    return self._fallback(f"gemini_failed: {ge}", scene_context, transcript)
                result = self._build_result(data, proxy, transcript)
                self._save_cache(cache_path, key, result)

            result["scene_context"] = scene_context

            # 4. Clip intelligence store (speech mode, visual context, real visual vectors)
            self._update_clip_store(video_path, clip_folder, clip_id, result, transcript, p1_audio, proxy)
            return result

        except Exception as e:
            logger.error(f"🔬 ForensicAnalyzer: unexpected error — {e}")
            return self._fallback(f"unexpected: {e}", scene_context)

    # ── Fallback ───────────────────────────────────────────────────────────────

    def _fallback(self, reason: str, scene_context: Optional[dict] = None, transcript: Optional[dict] = None) -> dict:
        res = _default_result()
        res["scene_context"] = scene_context or {}
        res["_error"] = reason
        if transcript is not None:
            res["transcript_meta"] = self._transcript_meta(transcript)
        return res

    # ── Stage helpers ──────────────────────────────────────────────────────────

    def _scene_intel(self, video_path: str, creator_name: Optional[str]) -> dict:
        try:
            try:
                from Core_Modules.scene_intel import analyze_scene_pre_pipeline
            except ImportError:
                from Gemini_Modules.scene_intel import analyze_scene_pre_pipeline
            ctx = analyze_scene_pre_pipeline(video_path, creator_name=creator_name)
            logger.info(
                f"👤 SceneIntel: faces={ctx.get('num_detected_faces')} | "
                f"subjects={ctx.get('num_subjects')} | face_cache={ctx.get('face_cache_status')}"
            )
            return ctx or {}
        except Exception as sie:
            logger.debug(f"SceneIntel fallback: {sie}")
            return {}

    def _load_p1_audio(self, clip_folder: str) -> Optional[dict]:
        try:
            from Audio_Modules.audio_extractor import load_audio_analysis
            return load_audio_analysis(clip_folder)
        except Exception:
            return None

    def _prepare_proxy(self, video_path: str, clip_folder: str) -> Dict[str, Any]:
        try:
            from Main_Modules.proxy_encoder import encode_for_gemini
        except ImportError:
            from proxy_encoder import encode_for_gemini
        fps = None
        if _FPS_ENV not in ("", "auto"):
            try:
                fps = float(_FPS_ENV)
            except ValueError:
                fps = None
        return encode_for_gemini(video_path, output_dir=clip_folder, fps=fps)

    def _resolve_transcript(self, provided, video_path, clip_folder, clip_id, proxy, p1_audio, creator_name):
        if isinstance(provided, dict):
            return provided
        if not AUTO_TRANSCRIBE or not proxy.get("audio_present"):
            return None
        try:
            from Audio_Modules.whisper_transcriber import transcribe_audio
            from Audio_Modules.transcript_corrector import correct_transcript

            wav = None
            if p1_audio and p1_audio.get("wav_path") and os.path.isfile(p1_audio["wav_path"]):
                wav = p1_audio["wav_path"]
            else:
                wav = self._extract_speech_wav(video_path, clip_folder, clip_id)
            if not wav:
                return None
            hint = f"Creator/channel: {creator_name}" if creator_name else ""
            whisper = transcribe_audio(wav, output_json_path=os.path.join(clip_folder, "whisper_transcript.json"))
            return correct_transcript(wav, whisper, context_hint=hint, cache_dir=clip_folder)
        except Exception as te:
            logger.warning(f"🔬 Transcript pipeline failed ({te}) — analyzing video without a transcript.")
            return None

    def _extract_speech_wav(self, video_path: str, clip_folder: str, clip_id: str) -> Optional[str]:
        out = os.path.join(clip_folder, f"{clip_id}_speech_16k.wav")
        if os.path.isfile(out) and os.path.getsize(out) > 4096:
            return out
        try:
            try:
                from Main_Modules.proxy_encoder import _ffmpeg_bin
            except ImportError:
                from proxy_encoder import _ffmpeg_bin
            subprocess.run(
                [_ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-i", video_path,
                 "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out],
                check=True, capture_output=True, timeout=300,
            )
            return out if os.path.isfile(out) else None
        except Exception as e:
            logger.warning(f"🔬 Speech WAV extraction failed: {e}")
            return None

    # ── Prompt assembly ────────────────────────────────────────────────────────

    @staticmethod
    def _transcript_meta(tr: Optional[dict]) -> dict:
        if tr is None:
            return {"available": False}
        return {
            "available": True,
            "source": tr.get("_source", "provided"),
            "language": tr.get("language"),
            "has_speech": bool(tr.get("has_speech")),
            "is_singing": bool(tr.get("is_singing")),
            "segments": len(tr.get("segments") or []),
            "full_text": tr.get("full_text", ""),
            "stats": tr.get("stats", {}),
        }

    @staticmethod
    def _format_transcript_block(tr: Optional[dict]) -> str:
        if tr is None:
            return ("TRANSCRIPT: UNAVAILABLE. Speech content is unverified. You may describe audio you hear, "
                    "but state any speech claim as unverified in \"uncertainties\". Return an empty transcript_alignment.")
        segs = tr.get("segments") or []
        if not segs:
            return ("TRANSCRIPT: no speech was detected in this clip. Do not invent speech. "
                    "Return an empty transcript_alignment.")
        src = tr.get("_source", "provided")
        note = "corrected against the audio" if src in ("gemini_corrected", "cache_hit", "provided") else \
               "UNCORRECTED Whisper draft — treat wording with caution"
        lines = [f"TRANSCRIPT ({note}; language: {tr.get('language', 'unknown')}"
                 f"{'; singing detected' if tr.get('is_singing') else ''}).",
                 "Format: [id] start-end (seconds) | text"]
        for s in segs:
            low = " | LOW_CONFIDENCE" if (s.get("confidence") is not None and s["confidence"] < 0.6) else ""
            lines.append(f"[{s['id']}] {s['start']:.2f}-{s['end']:.2f} | {s['text']}{low}")
        cuts = (tr.get("clean_cut_timestamps") or [])[:20]
        if cuts:
            lines.append("Clean cut points (word/sentence ends): " + ", ".join(f"{c['timestamp_sec']}s" for c in cuts))
        block = "\n".join(lines)
        if len(block) > _MAX_TRANSCRIPT_CHARS:
            block = block[:_MAX_TRANSCRIPT_CHARS] + "\n[...transcript truncated...]"
        return block

    def _build_prompt(self, proxy, transcript, creator_name, audio_candidates) -> str:
        fps = float(proxy.get("fps", 1.0))
        prompt = (
            FORENSIC_PROMPT
            .replace("<<W>>", str(proxy.get("width", "?")))
            .replace("<<H>>", str(proxy.get("height", "?")))
            .replace("<<FPS>>", f"{fps:g}")
            .replace("<<DUR>>", f"{proxy.get('duration_sec', 0):.1f}")
            .replace("<<PREC>>", f"{max(0.5, 1.0 / fps):g}")
            .replace("<<TRANSCRIPT_BLOCK>>", self._format_transcript_block(transcript))
            .replace("<<CREATOR>>", creator_name or "unknown")
        )
        if audio_candidates:
            rows = [
                f"{i}. Track: '{c.get('track_name')}' | BPM: {c.get('bpm')} | Energy: {c.get('avg_energy')} | Drops: {c.get('drops')}"
                for i, c in enumerate(audio_candidates, start=1)
            ]
            prompt += _BGM_TASK.replace("<<AUDIO_TABLE>>", "\n".join(rows))
        return prompt

    def _cache_key(self, proxy, transcript, creator_name, audio_candidates) -> str:
        h = hashlib.sha1()
        h.update(str(proxy.get("gemini_ready_path")).encode())
        h.update(str(proxy.get("gemini_size_mb")).encode())
        h.update(json.dumps([(s["start"], s["end"], s["text"]) for s in (transcript or {}).get("segments", [])],
                            ensure_ascii=False).encode())
        h.update(str(creator_name).encode())
        h.update(json.dumps(audio_candidates or [], sort_keys=True, default=str).encode())
        h.update(f"{_PROMPT_VERSION}|{GEMINI_MODEL}|{proxy.get('fps')}".encode())
        return h.hexdigest()

    def _load_cache(self, path: str, key: str) -> Optional[dict]:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("_cache_key") == key and "video_analysis" in data:
                    logger.info("💾 ForensicAnalyzer: cache hit")
                    return data
        except Exception:
            pass
        return None

    def _save_cache(self, path: str, key: str, result: dict) -> None:
        try:
            payload = dict(result)
            payload["_cache_key"] = key
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning(f"🔬 Could not persist forensic analysis: {e}")

    # ── Gemini call ────────────────────────────────────────────────────────────

    def _generate(self, client, content_part, prompt: str) -> str:
        box: Dict[str, Any] = {}

        def _worker():
            try:
                if self.router is not None:
                    box["out"] = self.router.generate(
                        task_type="vision",
                        prompt=[content_part, prompt],
                        module_name="forensic_analyzer",
                        gen_config={"temperature": 0.2, "response_mime_type": "application/json"},
                    )
                else:
                    from google.genai import types
                    resp = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=[content_part, prompt],
                        config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json"),
                    )
                    box["out"] = resp.text
            except Exception as e:
                box["err"] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=GEMINI_TIMEOUT_SEC)
        if t.is_alive():
            raise TimeoutError(f"Gemini did not respond within {GEMINI_TIMEOUT_SEC:.0f}s")
        if "err" in box:
            raise box["err"]
        return str(box.get("out") or "")

    def _call_gemini(self, proxy: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Upload compressed video, ask Gemini, parse JSON. Raises on failure."""
        from google import genai

        client = genai.Client(api_key=self._api_key)
        uploaded = None
        try:
            path = proxy["gemini_ready_path"]
            logger.info(f"📤 ForensicAnalyzer: uploading {os.path.basename(path)} ({proxy.get('gemini_size_mb')}MB)...")
            uploaded = client.files.upload(file=path)

            deadline = time.time() + FILE_ACTIVE_TIMEOUT_SEC
            while getattr(getattr(uploaded, "state", None), "name", "ACTIVE") == "PROCESSING":
                if time.time() > deadline:
                    raise TimeoutError("Gemini video stayed in PROCESSING state")
                time.sleep(2)
                uploaded = client.files.get(name=uploaded.name)
            state = getattr(getattr(uploaded, "state", None), "name", "ACTIVE")
            if state != "ACTIVE":
                raise RuntimeError(f"Gemini file upload ended in state {state}")

            content_part: Any = uploaded
            fps = float(proxy.get("fps", 1.0))

            logger.info(f"🔬 ForensicAnalyzer: analyzing (~{proxy.get('gemini_token_est')} video tokens, fps={fps:g})...")
            last_err: Optional[Exception] = None
            for attempt in range(1, MAX_ATTEMPTS + 1):
                t0 = time.time()
                raw = self._generate(client, content_part, prompt)
                logger.info(f"✅ ForensicAnalyzer: Gemini replied in {time.time() - t0:.1f}s (attempt {attempt})")
                try:
                    data = json.loads(_clean_json(raw))
                    if isinstance(data, dict) and (data.get("timeline") is not None or data.get("intent")):
                        return data
                    last_err = ValueError("Gemini JSON missing expected fields")
                except json.JSONDecodeError as je:
                    last_err = je
                logger.warning(f"🔬 Unusable Gemini output on attempt {attempt}: {last_err}")
            raise RuntimeError(f"Gemini output unusable after {MAX_ATTEMPTS} attempts: {last_err}")
        finally:
            if uploaded is not None:
                try:
                    client.files.delete(name=uploaded.name)
                except Exception:
                    pass

    # ── Parse / assemble ───────────────────────────────────────────────────────

    def _build_result(self, data: Dict[str, Any], proxy: Dict[str, Any], transcript: Optional[dict]) -> Dict[str, Any]:
        duration = float(proxy.get("duration_sec") or proxy.get("source_duration_sec") or 0.0)

        # ── Detailed explanation (video <-> transcript join, done in code) ──
        timeline, coverage = _normalize_timeline(data.get("timeline"), duration)
        timeline, alignment, conflicts, uncovered = _attach_transcript(
            timeline, data.get("transcript_alignment"), data.get("transcript_conflicts"), transcript
        )
        ov_raw = data.get("overview") if isinstance(data.get("overview"), dict) else {}
        overview = {
            "summary": _s(ov_raw.get("summary")),
            "setting": _s(ov_raw.get("setting")),
            "subjects": _str_list(ov_raw.get("subjects")),
            "spoken_language": _s(ov_raw.get("spoken_language")) or _s((transcript or {}).get("language")),
            "narrative_arc": _s(ov_raw.get("narrative_arc")),
            "target_audience": _s(ov_raw.get("target_audience")),
            "on_screen_text_summary": _s(ov_raw.get("on_screen_text_summary")),
        }
        sp_raw = data.get("speaker_analysis") if isinstance(data.get("speaker_analysis"), dict) else {}
        vmatch = sp_raw.get("voice_matches_visible_speaker")
        speaker = {
            "visible_people": int(max(0, _f(sp_raw.get("visible_people"), 0))),
            "is_talking_on_camera": bool(sp_raw.get("is_talking_on_camera", False)),
            "voice_matches_visible_speaker": vmatch if isinstance(vmatch, bool) else None,
            "notes": _s(sp_raw.get("notes")),
        }
        uncertainties = _str_list(data.get("uncertainties"))
        if duration and coverage < 0.9:
            uncertainties.append(f"Timeline covers only {coverage:.0%} of the video; gaps are unanalyzed.")
        if uncovered:
            uncertainties.append(f"Transcript lines {uncovered} fall outside every timeline scene.")
        if abs(proxy.get("duration_drift_sec", 0.0)) > 0.75:
            uncertainties.append(
                f"Proxy duration differs from source by {proxy['duration_drift_sec']:+.2f}s; timestamps may be offset."
            )

        video_analysis = {
            "overview": overview,
            "timeline": timeline,
            "transcript_alignment": alignment,
            "key_moments": _normalize_key_moments(data.get("key_moments"), duration),
            "transcript_conflicts": conflicts,
            "speaker_analysis": speaker,
            "uncertainties": uncertainties,
            "coverage": coverage,
            "uncovered_transcript_ids": uncovered,
        }

        # ── Classification (legacy schema) ──
        intent_valid = _s(data.get("intent")).lower() in ALLOWED_INTENTS
        intent = _enum(data.get("intent"), ALLOWED_INTENTS, "general_content")
        confidence = _clamp01(data.get("confidence"), 0.5)
        if not intent_valid:
            confidence = min(confidence, 0.49)
        editing_style = _enum(data.get("editing_style"), ALLOWED_STYLES, "cinematic")
        flags_raw = data.get("feature_flags") if isinstance(data.get("feature_flags"), dict) else {}
        defaults = {"enable_cinematic_zoom": True, "enable_voiceover": True}
        feature_flags = {k: bool(flags_raw.get(k, defaults.get(k, False))) for k in FLAG_NAMES}

        safety_obj = data.get("safety")
        if isinstance(safety_obj, dict):
            safety_cls = _s(safety_obj.get("classification")).lower()
            mon_safe = bool(safety_obj.get("monetization_safe", safety_cls == "safe"))
        else:
            safety_cls = _s(safety_obj).lower()
            mon_safe = safety_cls == "safe"
        if safety_cls not in ("safe", "risky", "blocked"):
            safety_cls = "risky"
        if safety_cls != "safe":
            mon_safe = False

        platform_priority = data.get("platform_priority")
        if not (isinstance(platform_priority, list) and platform_priority):
            platform_priority = ["youtube_shorts", "instagram_reels", "facebook_reels"]

        cd_raw = data.get("content_director") if isinstance(data.get("content_director"), dict) else {}
        raw_cmds = cd_raw.get("feature_commands") if isinstance(cd_raw.get("feature_commands"), dict) else {}
        content_director = {
            "detected_entities": _str_list(cd_raw.get("detected_entities")),
            "visual_event": _s(cd_raw.get("visual_event")),
            "viewer_attention": _s(cd_raw.get("viewer_attention")),
            "internet_context": _str_list(cd_raw.get("internet_context")),
            "possible_narratives": _str_list(cd_raw.get("possible_narratives")),
            "recommended_narrative": _s(cd_raw.get("recommended_narrative")),
            "tone": _s(cd_raw.get("tone")),
            "editing_style": _s(cd_raw.get("editing_style")),
            "is_talking_on_camera": bool(cd_raw.get("is_talking_on_camera", speaker["is_talking_on_camera"])),
            "engagement_hook": _s(cd_raw.get("engagement_hook")),
            "feature_commands": {k: bool(raw_cmds.get(k, feature_flags[k])) for k in FLAG_NAMES},
        }

        creative = data.get("creative_possibilities")
        result = {
            "watermarks": [],
            "intent": intent,
            "confidence": round(confidence, 3),
            "editing_style": editing_style,
            "feature_flags": feature_flags,
            "platform_priority": platform_priority,
            "selected_audio_track": _s(data.get("selected_audio_track")),
            "creative_possibilities": creative if isinstance(creative, list) else [],
            "safety": {"classification": safety_cls, "monetization_safe": mon_safe},
            "content_strategy": {          # backward-compat wrapper read by orchestrator
                "intent": intent, "confidence": round(confidence, 3),
                "recommended_editing_style": editing_style,
                "feature_flags": dict(feature_flags), "safety": safety_cls,
            },
            "content_director": content_director,
            "video_analysis": video_analysis,
            "transcript_meta": self._transcript_meta(transcript),
            "video_proxy": {k: proxy.get(k) for k in (
                "gemini_ready_path", "fps", "width", "height", "source_width", "source_height",
                "duration_sec", "duration_drift_sec", "gemini_size_mb", "size_reduction_pct", "gemini_token_est")},
        }
        logger.info(
            f"🎬 ForensicAnalyzer: intent={intent} conf={confidence:.2f} safety={safety_cls} | "
            f"{len(timeline)} scenes ({coverage:.0%} coverage), {len(alignment)} transcript lines aligned, "
            f"{len(video_analysis['key_moments'])} key moments, {len(result['watermarks'])} watermark(s), "
            f"{len(conflicts)} transcript conflict(s)"
        )
        return result

    # ── Clip intelligence store ────────────────────────────────────────────────

    def _update_clip_store(self, video_path, clip_folder, clip_id, result, transcript, p1_audio, proxy) -> None:
        try:
            from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore

            store = ClipIntelligenceStore()
            clip_data = store.load(clip_id, clip_folder) or store.create_blank(clip_id, clip_folder)
            cd = result.get("content_director", {})
            va = result.get("video_analysis", {})
            duration = float(proxy.get("duration_sec") or 0.0)

            # 1. audio_data.math from Phase-1 DSP
            if p1_audio:
                store.patch_audio_math(clip_data, {
                    "tempo_bpm": p1_audio.get("tempo_bpm", 120.0),
                    "beat_timestamps": p1_audio.get("beats", []),
                    "drop_timestamps": p1_audio.get("drops", []),
                    "avg_energy": p1_audio.get("avg_energy", 0.5),
                    "vibe": p1_audio.get("vibe", "unknown"),
                    "beat_count": p1_audio.get("beat_count", len(p1_audio.get("beats", []))),
                    "drop_count": p1_audio.get("drop_count", len(p1_audio.get("drops", []))),
                    "wav_path": p1_audio.get("wav_path"),
                })

            # 2. speech intelligence (transcript is now the authority for vocals)
            is_talking = bool(cd.get("is_talking_on_camera") or va.get("speaker_analysis", {}).get("is_talking_on_camera")
                              or result.get("intent") == "talking_head")
            speech_intel = _derive_speech_intel(is_talking, transcript, p1_audio)
            result["speech_intelligence"] = speech_intel

            # 3. visual_context
            visual_ctx = {
                "intent": result.get("intent", "viral_reel"),
                "tone": cd.get("tone", "aspirational"),
                "editing_style": result.get("editing_style", "cinematic"),
                "engagement_hook": cd.get("engagement_hook", ""),
                "detected_entities": cd.get("detected_entities", []),
                "possible_narratives": cd.get("possible_narratives", []),
                "recommended_narrative": cd.get("recommended_narrative", ""),
                "feature_flags": result.get("feature_flags", {}),
                "safety": result.get("safety", {}),
                "creative_possibilities": result.get("creative_possibilities", []),
                "speech_intelligence": speech_intel,
                "video_summary": va.get("overview", {}).get("summary", ""),
            }
            store.patch_visual_context(clip_data, visual_ctx)

            # 4. audio_data.context
            store.patch_audio_context(clip_data, {
                "has_vocals": speech_intel["has_spoken_vocal"],
                "speech_mode": speech_intel["speech_mode"],
                "recommended_audio_action": speech_intel["recommended_audio_action"],
                "dominant_emotion": cd.get("tone", "hype"),
                "vibe": result.get("editing_style", "rhythm_driven"),
            })

            # 5. visual_vectors from the REAL timeline
            vectors = _derive_visual_vectors(va, duration)
            store.patch_visual_vectors(clip_data, vectors)

            store.save(clip_id, clip_data, clip_folder)
            result["clip_intelligence"] = clip_data
            result["visual_vectors"] = vectors
            logger.info(f"🧠 [ClipIntelligenceStore] '{clip_id}' -> speech_mode='{speech_intel['speech_mode']}' "
                        f"({speech_intel['recommended_audio_action']}).")
        except Exception as store_err:
            logger.warning(f"🧠 [ClipIntelligenceStore] Update warning: {store_err}")
            result.setdefault("speech_intelligence", _derive_speech_intel(
                bool(result.get("content_director", {}).get("is_talking_on_camera")), transcript, p1_audio))


# ── Module-level singleton ────────────────────────────────────────────────────

_analyzer: Optional[ForensicVideoAnalyzer] = None


def get_analyzer() -> ForensicVideoAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ForensicVideoAnalyzer()
    return _analyzer


def analyze_video(video_path: str, frame_paths: Optional[List[str]] = None, intelligence_cache=None, **kwargs) -> dict:
    """
    Main entry. kwargs pass through to analyze(): creator_name, audio_candidates,
    transcript, use_cache. frame_paths / intelligence_cache are accepted for
    compatibility and ignored.
    """
    return get_analyzer().analyze(video_path, frame_paths=frame_paths, **kwargs)


# Alias for legacy support
analyze = analyze_video


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if len(sys.argv) < 2:
        print("usage: python forensic_analyzer.py <video_file> [creator_name]")
        raise SystemExit(1)
    res = analyze_video(sys.argv[1], creator_name=sys.argv[2] if len(sys.argv) > 2 else None)
    if res.get("_error"):
        print(f"FAILED: {res['_error']}")
    else:
        print(render_explanation(res))
