"""
audio_strategy.py
-----------------
Decides WHAT to do with a clip's audio before any BGM is selected, and ranks
BGM candidates by semantic connection instead of rotating through them.

Pure functions. No external dependencies.

Pipeline:
    stats    = speech_stats(whisper_data, duration)
    strategy = decide_audio_strategy(visual_ctx, stats, audio_type=..., has_audio=...)
    if strategy["action"] == "replace":
        ranked = rank_candidates(cands, clip_text, ...)
"""
import re
import time
from typing import Any, Callable, Dict, List, Optional

_STOP = {"the", "and", "with", "for", "from", "this", "that", "video", "clip",
         "reel", "music", "track", "song", "unknown"}


def _tokens(text: str) -> set:
    # split on underscores too: "bollywood_celebrity_arrival" -> 3 tokens
    return {p for p in re.split(r"[^a-z0-9]+", str(text).lower())
            if len(p) >= 3 and p not in _STOP}


# ── 1. Evidence: how much of this audio is actually speech? ──────────────────

def speech_stats(whisper: Optional[dict], duration: float) -> Dict[str, Any]:
    """`known` is False when ASR never ran/failed; absence of data != absence of speech."""
    known = bool(whisper)
    raw_words = (whisper or {}).get("words", []) or []
    if not raw_words and (whisper or {}).get("segments"):
        for seg in whisper["segments"]:
            if isinstance(seg, dict):
                raw_words.extend(seg.get("words", [seg]))
    elif not raw_words and (whisper or {}).get("sentences"):
        raw_words = (whisper or {}).get("sentences", [])

    spans = []
    for w in raw_words:
        if not isinstance(w, dict):
            continue
        s = w.get("start", w.get("start_time", w.get("time")))
        e = w.get("end", w.get("end_time"))
        try:
            s = float(s)
            e = float(e) if e is not None else s + 0.3
        except (TypeError, ValueError):
            continue
        spans.append((s, max(s, e)))

    if not spans or duration <= 0:
        if not spans and (whisper or {}).get("text") and duration > 0:
            txt = str((whisper or {}).get("text", "")).strip()
            words = txt.split()
            if words:
                n_w = len(words)
                wpm = n_w / duration * 60.0
                approx_cov = min(duration, n_w * 0.4)
                return {
                    "speech_ratio": min(1.0, approx_cov / duration),
                    "wpm": wpm,
                    "n_words": n_w,
                    "known": known,
                }
        return {"speech_ratio": 0.0, "wpm": 0.0, "n_words": len(spans), "known": known}

    spans.sort()
    covered, cs, ce = 0.0, spans[0][0], spans[0][1]
    for s, e in spans[1:]:
        if s - ce <= 0.6:          # bridge natural pauses between words
            ce = max(ce, e)
        else:
            covered += ce - cs
            cs, ce = s, e
    covered += ce - cs

    return {
        "speech_ratio": min(1.0, covered / duration),
        "wpm": len(spans) / duration * 60.0,
        "n_words": len(spans),
        "known": known
    }


# ── 2. Decision: preserve, bed, or replace the original audio ────────────────

def decide_audio_strategy(visual_ctx: Dict[str, Any],
                          stats: Dict[str, Any],
                          audio_type: Optional[str] = None,
                          has_audio: Optional[bool] = True) -> Dict[str, Any]:
    """
    audio_type (from Gemini hearing the video): song | speech | speech_over_music |
                                                ambient | music | noise | silence
    Replacing original audio is destructive, so it needs POSITIVE evidence.
    Unknown / conflicting evidence -> preserve.
    """
    cd = visual_ctx.get("content_director", {}) or {}
    talking = bool(visual_ctx.get("is_talking_on_camera")
                   or cd.get("is_talking_on_camera")
                   or (visual_ctx.get("speech_intelligence") or {}).get("is_talking_visually"))
    atype = str(audio_type or "").lower().strip()
    ratio, wpm = stats.get("speech_ratio", 0.0), stats.get("wpm", 0.0)

    def out(action, mode, why, bed_db=None):
        return {"action": action, "mode": mode, "reason": why,
                "bgm_allowed": action != "preserve", "bed_db": bed_db}

    if has_audio is False or atype == "silence":
        return out("replace", "silent_broll", "no usable original audio")

    asr_says_speech = stats.get("known") and ratio >= 0.30 and wpm >= 60
    if atype in ("speech", "speech_over_music") or asr_says_speech:
        return out("preserve", "speech_primary",
                   f"speech evidence (asr ratio={ratio:.2f}, wpm={wpm:.0f}, gemini={atype or 'n/a'}, talking={talking})")

    if atype == "song" and talking:
        return out("preserve", "lip_sync_song", "on-camera performance of the original song")

    if atype == "ambient":
        return out("preserve", "ambient_natural", "natural sound is part of the content", bed_db=-16.0)

    if stats.get("known") and ratio < 0.10 and not talking and atype in ("", "music", "song", "noise"):
        return out("replace", "music_broll", f"no speech (ratio={ratio:.2f}), nobody talking")

    return out("preserve", "unknown_fail_safe", "insufficient evidence; not destroying original audio")


# ── 3. Connection: does this track MEAN something relevant to this clip? ─────

def clip_card(visual_ctx: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> str:
    m = meta or {}
    cd = visual_ctx.get("content_director", {}) or {}
    parts = [
        visual_ctx.get("main_subject") or cd.get("main_subject"),
        visual_ctx.get("intent"),
        cd.get("visual_event"),
        cd.get("recommended_narrative"),
        cd.get("tone"),
        m.get("caption"),
        " ".join(map(str, m.get("hashtags") or []))
    ]
    return " ".join(str(p) for p in parts if p)


def track_card(m: Dict[str, Any]) -> str:
    parts = [
        m.get("gemini_genre") or m.get("genre"),
        " ".join(map(str, m.get("vibe_tags") or [])),
        m.get("dominant_emotion"),
        m.get("language"),
        m.get("track_name") or m.get("title") or m.get("filename"),
        m.get("caption"),
        " ".join(map(str, m.get("hashtags") or [])),
        str(m.get("lyrics") or m.get("transcript") or "")[:400]   # lyrics/speech meaning, not just tags
    ]
    return " ".join(str(p) for p in parts if p)


def _cos(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return num / den if den else 0.0


def semantic_connection(clip_text: str, track_text: str,
                        embed: Optional[Callable[[str], List[float]]] = None) -> float:
    """0..1. Use `embed` (cache track vectors in pool_metadata); lexical is a weak fallback."""
    if embed:
        return max(0.0, min(1.0, _cos(embed(clip_text), embed(track_text))))
    a, b = _tokens(clip_text), _tokens(track_text)
    if not a or not b:
        return 0.0
    return min(1.0, len(a & b) / ((len(a) * len(b)) ** 0.5) * 2.0)


# ── 4. Ranking: connection first, diversity only as a bounded tie-breaker ────

def rank_candidates(cands: List[Dict[str, Any]], clip_text: str,
                    clip_energy: float = 0.6, clip_bpm: Optional[float] = None,
                    embed: Optional[Callable[[str], List[float]]] = None,
                    now: Optional[float] = None) -> List[Dict[str, Any]]:
    """cands: [{"filename", "meta", "archetype_fit": 0..1 (0 = hard clash)}] or candidate dicts directly."""
    now = now or time.time()
    scored = []
    for c in cands:
        m = c.get("meta") if isinstance(c.get("meta"), dict) else c
        fit = float(c.get("archetype_fit", 1.0))
        if fit <= 0.0:
            continue
        sem = semantic_connection(clip_text, track_card(m), embed)
        energy = float(m.get("energy") or m.get("avg_energy") or 0.5)
        mood = 1.0 - min(1.0, abs(energy - clip_energy))
        bpm = float(m.get("tempo_bpm") or m.get("bpm") or 0.0)
        rhythm = max(0.0, 1.0 - abs(clip_bpm - bpm) / 60.0) if (clip_bpm and bpm) else 0.5
        base = (0.60 * sem + 0.25 * mood + 0.15 * rhythm) * fit
        last = float(m.get("last_used") or 0)
        hrs = (now - last) / 3600.0 if last else 1e9
        fatigue = 0.10 * max(0.0, 1.0 - hrs / 24.0) + 0.05 * min(1.0, int(m.get("usage_count") or 0) / 10.0)
        scored.append({**c, "semantic": sem, "score": base - fatigue})   # penalty capped at 0.15

    scored.sort(key=lambda s: s["score"], reverse=True)
    if not scored:
        return scored

    top = scored[0]["score"]
    near = sorted((s for s in scored if top - s["score"] <= 0.04),
                  key=lambda s: float((s.get("meta") or {}).get("last_used") or 0))   # LRU only among near-ties
    rest = [s for s in scored if s not in near]
    return near + rest


# ── 5. Editing parameters driven by the CLIP's audio strategy ────────────────

def routing_for_strategy(strategy: Dict[str, Any], music_routing: Dict[str, Any]) -> Dict[str, Any]:
    """Speech/preserved audio must never inherit beat-driven params from a BGM's emotion."""
    if not strategy or strategy.get("action") == "replace":
        return {**music_routing, "beat_sync": True, "cut_on": "beats"}

    return {
        **music_routing,
        "strategy_name": "SPEECH_FOLLOW_CUTS" if strategy.get("mode") == "speech_primary" else "PRESERVE_ORIGINAL_AUDIO",
        "speed_factor": 1.0,              # never time-stretch speech
        "transition_type": "cut",
        "transition_duration": 0.0,
        "target_cut_interval": 4.0,
        "beat_sync": False,
        "cut_on": "speech_boundaries",
        "bgm_ducking_db": -20.0,
        "music_volume": 0.0 if not strategy.get("bgm_allowed") else (strategy.get("bed_db") or 0.12),
        "recommended_editing_mode": strategy.get("mode", "preserve_original")
    }
