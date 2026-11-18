"""
Phase_2 / 02_forensic_perception.py
====================================
Step 2: Gemini Call 1 — Multimodal Forensic Perception & Frame Vector Generator.
Analyzes 480p proxy video + WAV audio + Phase 1 DSP math.
Fills visual_context, audio_data.context, and visual_vectors in ClipIntelligenceStore.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("Phase2.Step02")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import json
import subprocess
from Gemini_Modules.forensic_analyzer import ForensicVideoAnalyzer
from Audio_Modules.audio_strategy import speech_stats, decide_audio_strategy


def _find_whisper_data(video_path: str) -> Optional[dict]:
    """Finds Whisper ASR transcript JSON if present near the video file."""
    clip_dir = os.path.dirname(os.path.abspath(video_path))
    clip_stem = os.path.splitext(os.path.basename(video_path))[0]

    for p in [
        os.path.join(clip_dir, f"{clip_stem}_whisper.json"),
        os.path.join(clip_dir, "whisper.json"),
        os.path.join(clip_dir, f"{clip_stem}_speech.json"),
        os.path.join(clip_dir, "speech.json"),
    ]:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    meta_path = os.path.join(clip_dir, "metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
                if isinstance(m, dict):
                    w = m.get("whisper_transcript") or m.get("whisper_data") or m.get("whisper")
                    if isinstance(w, dict) and w:
                        return w
        except Exception:
            pass

    ci_path = os.path.join(clip_dir, ".clip_intelligence.json")
    if os.path.isfile(ci_path):
        try:
            with open(ci_path, "r", encoding="utf-8") as f:
                ci = json.load(f)
                if isinstance(ci, dict):
                    w = (ci.get("audio_data") or {}).get("whisper_transcript")
                    if isinstance(w, dict) and w:
                        return w
        except Exception:
            pass

    return None


def _get_video_duration(video_path: str) -> float:
    """Probes video duration in seconds via ffprobe."""
    if not os.path.isfile(video_path):
        return 0.0
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", video_path
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        data = json.loads(out.stdout)
        return float(data.get("format", {}).get("duration", 0.0))
    except Exception:
        return 0.0


def run_forensic_perception(
    video_path: str,
    creator_name: Optional[str] = None,
    audio_candidates: Optional[List[dict]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes Gemini Call 1.
    Returns result dictionary containing visual_context, audio_data, visual_vectors,
    and determined audio_strategy (preserve / bed / replace).
    """
    logger.info(f"🔬 [STEP 02] Running Gemini Call 1 Forensic Perception for: {os.path.basename(video_path)}")

    # Auto-Input-Only Creative Brief lookup (None for Manual Input — manual mode stays untouched)
    creative_brief = None
    try:
        from Gemini_Modules.creative_planner import load_creative_brief
        creative_brief = load_creative_brief(video_path)
        if creative_brief:
            logger.info(
                "🧠 [STEP 02 CREATIVE BRIEF] Auto-Input brief loaded for '%s' (%d mandatory shots).",
                creative_brief.get("target_keyword", ""),
                len(creative_brief.get("mandatory_shots") or []),
            )
    except Exception as _cb_err:
        logger.debug(f"[STEP 02] creative_brief lookup notice: {_cb_err}")

    analyzer = ForensicVideoAnalyzer(api_key=api_key)
    res = analyzer.analyze(
        video_path=video_path,
        creator_name=creator_name,
        audio_candidates=audio_candidates,
        creative_brief=creative_brief,
    )

    # Merge Auto-Input Creative Planner decisions into forensic context for downstream enforcement
    if creative_brief and isinstance(res, dict):
        res["creative_brief"] = creative_brief
        enforced_flags = creative_brief.get("enforced_feature_flags") or {}
        if enforced_flags:
            res.setdefault("feature_flags", {}).update(enforced_flags)
            if isinstance(res.get("content_strategy"), dict):
                res["content_strategy"].setdefault("feature_flags", {}).update(enforced_flags)
        narration = (creative_brief.get("narration_script") or creative_brief.get("hook_line") or "").strip()
        if narration:
            cd = res.setdefault("content_director", {})
            cd["engagement_hook"] = narration
            if creative_brief.get("tone"):
                cd["tone"] = creative_brief["tone"]

    # ── Step 2b: Audio Strategy Decision Engine ──────────────────────────────
    try:
        whisper_data = _find_whisper_data(video_path)
        dur = float(res.get("duration") or 0.0)
        if dur <= 0:
            dur = _get_video_duration(video_path)
            if dur > 0:
                res["duration"] = dur

        stats = speech_stats(whisper_data, dur)
        strategy = decide_audio_strategy(
            visual_ctx=res,
            stats=stats,
            audio_type=res.get("audio_type"),
            has_audio=res.get("has_audio", True),
        )
        res["audio_strategy"] = strategy
        res["speech_stats"] = stats
        logger.info(
            f"✓ [STEP 02 AUDIO STRATEGY] action='{strategy['action']}' | "
            f"mode='{strategy['mode']}' | reason='{strategy['reason']}'"
        )
    except Exception as _strat_err:
        logger.warning(f"⚠️ [STEP 02] Audio strategy engine fallback notice: {_strat_err}")

    logger.info(
        f"✓ [STEP 02 SUCCESS] Forensic perception complete: "
        f"intent='{res.get('intent')}' | style='{res.get('editing_style')}'"
    )
    return res
