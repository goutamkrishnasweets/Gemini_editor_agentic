"""
Phase_2 / 05_rhythm_timeline.py
===============================
Step 5: Rhythm & Micro-Shot Timeline Builder.
Computes psycho-acoustic beat routing parameters and builds 2.0s-3.5s human-scale jump-cut micro-shots.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("Phase2.Step05")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Gemini_Modules.lyric_rhythm_aligner import analyze_music, compute_routing_parameters
try:
    from Audio_Modules.rhythm_timeline_builder import RhythmTimelineBuilder
except ImportError:
    from rhythm_timeline_builder import RhythmTimelineBuilder


def _calculate_optimal_audio_window(
    lyric_intel: Dict[str, Any],
    bgm_duration: float,
    target_duration: float,
) -> float:
    """
    Finds the golden audio sliding window [T_start, T_start + target_duration]
    maximizing energy, chorus, or beat drop alignment instead of blindly starting at 00:00.
    """
    if bgm_duration <= target_duration or not lyric_intel:
        return 0.0

    max_offset = max(0.0, bgm_duration - target_duration)
    if max_offset <= 0.5:
        return 0.0

    # 1. Check sections for chorus, hook, or drop
    sections = lyric_intel.get("sections") or []
    for sec in sections:
        sec_type = str(sec.get("type", "")).lower()
        sec_start = float(sec.get("start", 0.0))
        if any(kw in sec_type for kw in ("chorus", "hook", "drop")) and sec_start <= max_offset:
            logger.info(f"🎵 [GOLDEN WINDOW] Selected section '{sec_type}' start at {sec_start:.2f}s")
            return round(sec_start, 2)

    # 2. Check beat drops / emotional peak moments
    drops = lyric_intel.get("drops") or lyric_intel.get("emotional_peak_moments") or []
    valid_drops = [float(d) for d in drops if float(d) <= bgm_duration]
    if valid_drops:
        # If there is a drop that can land inside our window (e.g. 2.0s to 3.0s build-up before drop)
        for d in valid_drops:
            lead_in = 2.5
            cand_offset = max(0.0, d - lead_in)
            if cand_offset <= max_offset:
                logger.info(f"💥 [GOLDEN WINDOW] Selected drop lead-in at {cand_offset:.2f}s (drop hits at {d:.2f}s)")
                return round(cand_offset, 2)

    # 3. Check tension arc for highest average energy window
    tension_arc = lyric_intel.get("tension_arc") or []
    if len(tension_arc) >= 3:
        best_offset = 0.0
        best_score = -1.0
        step = 1.0
        curr = 0.0
        while curr <= max_offset:
            window_end = curr + target_duration
            pts = [p.get("tension", 0.5) for p in tension_arc if curr <= p.get("time", 0.0) <= window_end]
            avg_score = sum(pts) / len(pts) if pts else 0.0
            if avg_score > best_score:
                best_score = avg_score
                best_offset = curr
            curr += step
        if best_score > 0.0:
            logger.info(f"⚡ [GOLDEN WINDOW] Selected highest tension arc window at {best_offset:.2f}s (score={best_score:.2f})")
            return round(best_offset, 2)

    return 0.0


def build_rhythm_timeline(
    video_path: str,
    selected_bgm_path: Optional[str] = None,
    forensic_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Builds micro-shots and routing parameters for editing with built-in
    Auto-Reject & Re-Selection Fallback Gate (max 3-4 attempts).
    """
    logger.info(f"🥁 [STEP 05] Building rhythm timeline for: {os.path.basename(video_path)}")

    builder = RhythmTimelineBuilder()
    active_bgm_path = selected_bgm_path
    lyric_intel = {}
    route_params = {}
    micro_shots = []

    # ── AUTO-REJECT & RE-SELECT FALLBACK LOOP (Max 3 to 4 attempts) ──────────
    MAX_REJECT_ATTEMPTS = 4
    current_attempt = 1
    disqualified_tracks = set()

    while current_attempt <= MAX_REJECT_ATTEMPTS:
        target_audio = active_bgm_path if (active_bgm_path and os.path.isfile(active_bgm_path)) else video_path
        lyric_intel = {}

        try:
            if os.path.isfile(target_audio):
                lyric_intel = analyze_music(target_audio)
        except Exception as le_err:
            logger.debug(f"Lyric intel fallback: {le_err}")

        # Check reject conditions only if a background music track was explicitly selected
        is_bad_audio = False
        reject_reason = ""

        is_user_custom = bool(
            (forensic_context or {}).get("is_user_custom_audio", False)
            or (active_bgm_path and "user_custom_audio" in os.path.basename(active_bgm_path).lower())
        )

        if active_bgm_path and os.path.isfile(active_bgm_path):
            bgm_dur = 0.0
            try:
                bgm_dur = builder._get_duration(active_bgm_path)
            except Exception:
                pass

            if lyric_intel.get("is_unusable", False) and not is_user_custom:
                is_bad_audio = True
                reject_reason = lyric_intel.get("unusable_reason") or "Audio is non-music noise / distortion"
            elif lyric_intel.get("is_speech_only", False) and not is_user_custom:
                is_bad_audio = True
                reject_reason = "Audio is spoken dialogue / chatter without musical backing"
            elif 0 < bgm_dur < 10.0 and not is_user_custom:
                is_bad_audio = True
                reject_reason = f"Audio snippet too short ({bgm_dur:.1f}s < 10s minimum)"

        # If audio is bad and we have remaining attempts, auto-reject & re-select
        if is_bad_audio and current_attempt < MAX_REJECT_ATTEMPTS:
            bad_track_name = os.path.basename(active_bgm_path)
            disqualified_tracks.add(bad_track_name)
            logger.warning(
                f"⚠️ [AUTO-REJECT BGM] Selected track '{bad_track_name}' is invalid ({reject_reason}). "
                f"Auto-selecting next best BGM candidate from pool (Attempt {current_attempt}/{MAX_REJECT_ATTEMPTS})..."
            )

            # Quarantine/flag the bad track in AudioPoolManager
            try:
                from Audio_Modules.audio_pool_manager import AudioPoolManager
                apm = AudioPoolManager()
                meta = apm._get_file_metadata(bad_track_name) or {}
                meta["is_unusable"] = True
                meta["unusable_reason"] = reject_reason
                apm._set_file_metadata(bad_track_name, meta)
                apm._save_metadata()
            except Exception as _q_err:
                logger.debug(f"Quarantine save notice: {_q_err}")

            # Auto-select next candidate from pool
            try:
                import importlib
                step04 = importlib.import_module("Phase_2.04_bgm_selector")
                _ctx = forensic_context or {}
                clip_id = _ctx.get("clip_id") or os.path.splitext(os.path.basename(video_path))[0]
                clip_folder = _ctx.get("clip_folder") or os.path.dirname(video_path)
                next_bgm_res = step04.select_clip_bgm(
                    clip_id=clip_id,
                    clip_folder=clip_folder,
                    exclude_filenames=disqualified_tracks
                )
                new_path = next_bgm_res.get("physical_path")
                if new_path and os.path.isfile(new_path) and new_path != active_bgm_path:
                    active_bgm_path = new_path
                    current_attempt += 1
                    logger.info(f"🔄 [AUTO-SWAP BGM] Swapped to candidate: '{os.path.basename(active_bgm_path)}'")
                    continue
            except Exception as _re_sel_err:
                logger.warning(f"Auto re-selection error: {_re_sel_err}")
                break

        # Clean/usable track verified or attempts exhausted -> break and build timeline
        break

    # Compute routing parameters with final verified audio
    route_params = compute_routing_parameters(lyric_intel, forensic_context, active_bgm_path)

    # Protect dialogue and apply audio strategy constraints
    try:
        from Audio_Modules.audio_strategy import routing_for_strategy
        _strategy = (forensic_context or {}).get("audio_strategy", {})
        if _strategy:
            route_params = routing_for_strategy(_strategy, route_params)
            logger.info(
                f"🛡️ [STEP 05 AUDIO STRATEGY APPLIED] mode='{_strategy.get('mode')}' | "
                f"speed={route_params.get('speed_factor')}x | "
                f"cut_on='{route_params.get('cut_on')}' | "
                f"music_vol={route_params.get('music_volume')}"
            )
    except Exception as _as_err:
        logger.warning(f"Audio strategy routing notice: {_as_err}")

    # Full psycho-acoustic timeline construction using RhythmTimelineBuilder
    v_dur = 0.0
    bgm_dur = 0.0
    try:
        v_dur = builder._get_duration(video_path)
    except Exception:
        pass
    if active_bgm_path and os.path.isfile(active_bgm_path):
        try:
            bgm_dur = builder._get_duration(active_bgm_path)
        except Exception:
            pass

    is_user_custom = bool(
        (forensic_context or {}).get("is_user_custom_audio", False)
        or (active_bgm_path and "user_custom_audio" in os.path.basename(active_bgm_path).lower())
    )

    # ── DYNAMIC DURATION & GOLDEN SLIDING WINDOW CALCULATION ──────────────────
    audio_start_offset = 0.0
    if bgm_dur > 0 and v_dur > 0:
        if bgm_dur < v_dur:
            # Scenario A: Audio < Video (e.g. Audio 9s, Video 24s) -> Condense video to audio duration
            target_duration = bgm_dur
            audio_start_offset = 0.0
            logger.info(
                f"⚡ [SCENARIO A: MICRO-REEL CONDENSATION] Audio ({bgm_dur:.2f}s) < Video ({v_dur:.2f}s). "
                f"Dynamically condensing video to {target_duration:.2f}s hyper-dense micro-reel (100-200% retention)."
            )
        else:
            # Scenario B: Audio >= Video (e.g. Audio 45s, Video 15s) -> Golden sliding window
            target_duration = min(v_dur, 15.0) if v_dur > 15.0 else v_dur
            audio_start_offset = _calculate_optimal_audio_window(
                lyric_intel=lyric_intel,
                bgm_duration=bgm_dur,
                target_duration=target_duration,
            )
            logger.info(
                f"🎧 [SCENARIO B: GOLDEN SLIDING WINDOW] Audio ({bgm_dur:.2f}s) >= Video ({v_dur:.2f}s). "
                f"Selected peak audio offset={audio_start_offset:.2f}s for target duration {target_duration:.2f}s."
            )
    elif bgm_dur > 0:
        target_duration = bgm_dur
    elif v_dur > 0:
        target_duration = min(v_dur, 15.0)
    else:
        target_duration = 15.0

    try:
        if v_dur > 0:
            builder.min_duration = min(2.0, target_duration / 3.0) if target_duration < 6.0 else 2.0
            builder.max_duration = min(4.0, target_duration)

            target_audio = active_bgm_path if (active_bgm_path and os.path.isfile(active_bgm_path)) else video_path
            bgm_beats = lyric_intel.get("emotional_peak_moments", [])
            if not bgm_beats and os.path.isfile(target_audio):
                bgm_beats = builder.analyze_beats(target_audio)

            raw_scenes = [{"clip_id": 0, "start": 0.0, "end": v_dur, "score": 0.85}]

            full_timeline = builder.build_timeline(
                scenes=raw_scenes,
                beat_grid=bgm_beats,
                vibe=route_params.get("recommended_editing_mode", "hype"),
                music_intelligence=lyric_intel if lyric_intel else None,
                target_duration_hint=target_duration,
            )
            micro_shots = full_timeline if full_timeline else []
    except Exception as rte:
        logger.warning(f"⚠️ [STEP 05] Rhythm timeline builder fallback notice: {rte}")

    logger.info(
        f"✓ [STEP 05 SUCCESS] Built {len(micro_shots)} human-scale micro-shots (target={target_duration:.1f}s, offset={audio_start_offset:.1f}s) | "
        f"speed={route_params.get('speed_factor', 1.0)}x | "
        f"ducking={route_params.get('bgm_ducking_db', -6.0)}dB | "
        f"verified_bgm='{os.path.basename(active_bgm_path) if active_bgm_path else 'none'}'"
    )

    return {
        "route_params": route_params,
        "lyric_intel": lyric_intel,
        "micro_shots": micro_shots,
        "selected_bgm_path": active_bgm_path,
        "target_duration": target_duration,
        "audio_start_offset": audio_start_offset,
        "is_user_custom_audio": is_user_custom,
    }
