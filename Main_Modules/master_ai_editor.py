"""
master_ai_editor.py — End-to-End AI Video Editor Engine
======================================================
World-Class Master Video Editor Pipeline driven by Gemini & FFmpeg.

Architecture (2-Pass Perception & Tactical FFmpeg Master Synthesis):

  [Raw Input Video] + [Optional BGM Audio]
          │
          ▼
    1. Proxy Encoder (`proxy_encoder.py`)
       └─ Fast 480p H.264 proxy compression
          │
          ▼
    2. 2-Pass Adaptive Frame Sampler (`strategic_frame_sampler.py`)
       ├─ HSV Shot-Cut Transitions
       ├─ Hook Zone (0-5s) + Climax Zone + Motion Peaks
       └─ 256x256 High-Gradient Micro-Crops (watermarks/text)
          │
          ▼
    3. Pass 1: Forensic Perception Engine (`forensic_analyzer.py`)
       └─ Gemini Vision call: full visual context, intent, tone, entities, hook
          │
          ▼
    4. Audio Rhythm Engine (`beat_engine.py` / `rhythm_timeline_builder.py`)
       └─ Waveform beat grid extraction & BPM alignment
          │
          ▼
    5. Pass 2: Master Director Synthesis (`gemini_ffmpeg_synthesis.py`)
       └─ Gemini FFmpeg Master Director generates complex filtergraph recipe
       └─ Executes FFmpeg command pipeline (cuts, xfade transitions, speed ramps, BGM ducking)
          │
          ▼
  [Final Rendered Master Reel (.MP4)]

Usage:
    from Main_Modules.master_ai_editor import edit_video_master
    result = edit_video_master("source_clip.mp4", bgm_path="music.mp3", output_path="master_output.mp4")

CLI:
    python master_ai_editor.py source_clip.mp4 --bgm music.mp3 --output master_output.mp4
"""

import os
import sys
import json
import time
import logging
import tempfile
import shutil
import argparse
import signal
from typing import Dict, List, Optional, Any

# Instant Ctrl+C Termination Handler
def _instant_sigint_handler(signum, frame):
    print("\n🛑 [INSTANT EXIT] Ctrl+C detected. Terminating process immediately...", flush=True)
    os._exit(130)

try:
    signal.signal(signal.SIGINT, _instant_sigint_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _instant_sigint_handler)
except Exception:
    pass

# Configure Logging with unbuffered StreamHandler
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | [%(name)s] %(message)s", datefmt="%H:%M:%S"))

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not root_logger.handlers:
    root_logger.addHandler(handler)

logger = logging.getLogger("master_ai_editor")

# ── Imports from Local Workspace ──────────────────────────────────────────────
# The sample tree is intentionally kept behind the workspace root so that the
# canonical AMTCE modules are imported rather than the shadow copies living under
# `simpler update/`.
from Main_Modules.proxy_encoder import encode_proxy
from Main_Modules.strategic_frame_sampler import extract_strategic_frame_files, extract_high_gradient_crops
from Gemini_Modules.forensic_analyzer import ForensicVideoAnalyzer
from Gemini_Modules.gemini_ffmpeg_synthesis import GeminiFFmpegEngine
from Audio_Modules.audio_extractor import extract_audio
from Audio_Modules.beat_engine import BeatEngine
from Gemini_Modules.lyric_rhythm_aligner import analyze_music, compute_routing_parameters
try:
    from Audio_Modules.rhythm_timeline_builder import RhythmTimelineBuilder
except ImportError:
    from rhythm_timeline_builder import RhythmTimelineBuilder


class MasterAIEditor:
    """
    End-to-End Master AI Video Editor.
    Combines 2-Pass Vision Perception, Lyric & Rhythm Intelligence, Beat Sync, and FFmpeg Master Synthesis.
    """

    def __init__(self):
        self.forensic_analyzer       = ForensicVideoAnalyzer()
        self.rhythm_timeline_builder = RhythmTimelineBuilder()
        self.ffmpeg_engine           = GeminiFFmpegEngine()
        self.beat_engine             = BeatEngine()

    def process(
        self,
        video_path: str,
        bgm_path: Optional[str] = None,
        output_path: Optional[str] = None,
        target_duration: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Execute full Master AI Video Editing Pipeline.

        Args:
            video_path:      Path to raw input video.
            bgm_path:        Path to optional background music file.
            output_path:     Destination path for rendered video.
            target_duration: Target duration in seconds for output reel.

        Returns:
            Dict containing pipeline result, forensic context, audio route params, and output path.
        """
        start_time = time.time()
        video_path = os.path.abspath(video_path)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        if output_path is None:
            dir_name = os.path.dirname(video_path)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(dir_name, f"{base_name}_master_edit.mp4")
        output_path = os.path.abspath(output_path)

        # Ensure Original_audio directory exists at workspace root for audio extraction & rotation
        audio_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "Original_audio"))
        os.makedirs(audio_dir, exist_ok=True)

        tmp_dir = tempfile.mkdtemp(prefix="master_ai_edit_")
        logger.info(f"🚀 Starting Master AI Video Editing Pipeline for: {os.path.basename(video_path)}")

        try:
            # ── STEP 1: Proxy Compression (480p) ──────────────────────────────
            logger.info("📦 [Step 1/6] Encoding 480p H.264 Proxy for Vision Analysis...")
            proxy_path = encode_proxy(video_path, overwrite=False)
            logger.info(f"   ✓ Proxy Ready: {proxy_path}")

            # ── STEP 2: 2-Pass Adaptive Frame Extraction ─────────
            logger.info("👁️ [Step 2/6] Sampling Strategic Keyframe Images...")
            frame_files = extract_strategic_frame_files(proxy_path, tmp_dir, include_micro_crops=False)
            logger.info(f"   ✓ Extracted {len(frame_files)} strategic keyframes")

            # ── STEP 2.5: Candidate BGM Audio Metadata Scan ─────────────────
            logger.info("🎵 [Step 2.5/6] Scanning candidate BGM audio track metadata...")
            try:
                from Audio_Modules.beat_engine import get_candidate_audio_metadata
                audio_candidates = get_candidate_audio_metadata(audio_dir=audio_dir, max_tracks=5)
                logger.info(f"   ✓ Scanned metadata for {len(audio_candidates)} candidate BGM track(s)")
            except Exception as ace:
                logger.warning(f"   ⚠ Candidate audio metadata scan fallback: {ace}")
                audio_candidates = []

            # ── STEP 3: Multimodal Vision Perception & Audio Matching ───────
            parent_folder = os.path.basename(os.path.dirname(video_path))
            creator_name = parent_folder.split("_")[0] if "_" in parent_folder else "unknown"
            logger.info(f"🔬 [Step 3/6] Multimodal Vision & Audio Perception: Creator='{creator_name}'...")
            forensic_context = self.forensic_analyzer.analyze(
                proxy_path,
                frame_paths=frame_files,
                creator_name=creator_name,
                audio_candidates=audio_candidates
            )
            selected_track_name = forensic_context.get("selected_audio_track", None)
            logger.info(
                f"   ✓ Perception Complete: intent='{forensic_context.get('intent')}' | "
                f"style='{forensic_context.get('editing_style')}' | "
                f"selected_bgm='{selected_track_name}'"
            )

            # ── STEP 3.1: Audio Strategy Determination (Speech vs B-Roll vs Ambient) ──
            if "audio_strategy" not in forensic_context:
                try:
                    from Audio_Modules.audio_strategy import speech_stats, decide_audio_strategy
                    v_dur = 0.0
                    try:
                        v_dur = self.rhythm_timeline_builder._get_duration(video_path)
                    except Exception:
                        pass
                    whisper_data = None
                    clip_dir = os.path.dirname(video_path)
                    meta_file = os.path.join(clip_dir, "metadata.json")
                    if os.path.isfile(meta_file):
                        try:
                            with open(meta_file, "r", encoding="utf-8") as _mf:
                                _md = json.load(_mf)
                                whisper_data = _md.get("whisper_transcript") or _md.get("whisper")
                        except Exception:
                            pass
                    if not whisper_data:
                        for _wf in [f"{os.path.splitext(os.path.basename(video_path))[0]}_whisper.json", "whisper.json"]:
                            _wp = os.path.join(clip_dir, _wf)
                            if os.path.isfile(_wp):
                                try:
                                    with open(_wp, "r", encoding="utf-8") as _wf_f:
                                        whisper_data = json.load(_wf_f)
                                        break
                                except Exception:
                                    pass

                    _stats = speech_stats(whisper_data, v_dur)
                    _strategy = decide_audio_strategy(
                        visual_ctx=forensic_context,
                        stats=_stats,
                        audio_type=forensic_context.get("audio_type"),
                        has_audio=forensic_context.get("has_audio", True)
                    )
                    forensic_context["audio_strategy"] = _strategy
                    forensic_context["speech_stats"] = _stats
                    logger.info(
                        f"   ✓ Audio Strategy: action='{_strategy['action']}' | "
                        f"mode='{_strategy['mode']}' | reason='{_strategy['reason']}'"
                    )
                except Exception as _as_err:
                    logger.warning(f"Audio strategy determination notice: {_as_err}")

            # ── STEP 3.5: Resolve selected BGM to file path + pool rotation ──
            # 1. Match Gemini Vision's selected_audio_track from candidate list
            selected_bgm_path = None
            selected_bgm_beats = []
            selected_bgm_bpm   = 120.0

            if selected_track_name and audio_candidates:
                target_clean = str(selected_track_name).lower().strip().replace(".mp3", "").replace(".wav", "")
                for c in audio_candidates:
                    t_name = str(c.get("track_name", "")).lower().strip().replace(".mp3", "").replace(".wav", "")
                    fp = c.get("file_path", "")
                    if target_clean in t_name or t_name in target_clean:
                        if fp and os.path.isfile(fp):
                            selected_bgm_path = fp
                            selected_bgm_bpm  = c.get("bpm", 120.0)
                            selected_bgm_beats = c.get("drops", []) or c.get("beats", [])
                            logger.info(f"🎶 [GEMINI_AUDIO_MATCH] Matched Gemini pick '{selected_track_name}' -> '{os.path.basename(fp)}'")
                            break

            # 2. Multi-factor Semantic Ranking Fallback if Gemini Vision didn't pick a track from candidate list
            if not selected_bgm_path and audio_candidates:
                try:
                    from Audio_Modules.audio_strategy import clip_card, rank_candidates
                    clip_text = clip_card(forensic_context)
                    ranked_cands = rank_candidates(audio_candidates, clip_text=clip_text)
                    if ranked_cands:
                        top_c = ranked_cands[0]
                        fp = top_c.get("file_path", "")
                        if fp and os.path.isfile(fp):
                            selected_bgm_path = fp
                            selected_bgm_bpm = top_c.get("bpm", 120.0)
                            selected_bgm_beats = top_c.get("drops", []) or top_c.get("beats", [])
                            logger.info(
                                f"🎶 [SEMANTIC_RANKED_BGM] Winner '{top_c.get('track_name')}' "
                                f"(score={top_c.get('score', 0.0):.3f}, semantic={top_c.get('semantic', 0.0):.3f})"
                            )
                except Exception as _rank_err:
                    logger.debug(f"Semantic audio candidate ranking notice: {_rank_err}")

            # 3. Algorithmic Pool Selection Fallback if Gemini Vision & Semantic Ranking didn't pick a matching track
            if not selected_bgm_path:
                try:
                    from Audio_Modules.audio_pool_manager import AudioPoolManager
                    pool = AudioPoolManager(base_dir=audio_dir)
                    intent_cat = forensic_context.get("intent", "viral_reel")
                    pool_winner = pool.select_best_audio(content_category=intent_cat)
                    if pool_winner and os.path.isfile(pool_winner):
                        selected_bgm_path = pool_winner
                        logger.info(f"🎶 [POOL_BGM] Selected external BGM from pool: {selected_bgm_path.replace(os.sep, '/')}")
                        winner_name = os.path.basename(pool_winner)
                        cached = next((c for c in audio_candidates if c.get("track_name") == winner_name), None)
                        if cached:
                            selected_bgm_beats = cached.get("beats", []) or cached.get("drops", [])
                            selected_bgm_bpm   = cached.get("bpm", 120.0)
                            logger.info(f"   ✓ Using cached beat data for '{winner_name}' (BPM={selected_bgm_bpm:.1f}, beats={len(selected_bgm_beats)})")
                        else:
                            try:
                                from Audio_Modules.beat_engine import BeatEngine as _BE
                                _res = _BE().analyze_beats_with_drops(selected_bgm_path)
                                selected_bgm_beats = _res.get("beats", [])
                                selected_bgm_bpm   = _res.get("bpm", 120.0)
                            except Exception:
                                pass
                except Exception as apm_err:
                    logger.debug(f"Algorithmic audio pool fallback error: {apm_err}")

            # 3. Process Selected BGM Track
            if selected_bgm_path:
                logger.info(f"🏆 [BEST_AUDIO] Winner (External Pool): {selected_bgm_path.replace(os.sep, '/')} "
                            f"→ Beat sync will use extracted BGM grid: {os.path.basename(selected_bgm_path)} "
                            f"(BPM={selected_bgm_bpm:.1f}, drops={len(selected_bgm_beats)})")

                # Musical Intelligence Report
                try:
                    from Audio_Modules.music_intelligence import classify_music
                    genre, conf = classify_music(selected_bgm_path)
                    logger.info(f"🎵 [MUSIC_INTEL] Running Musical Intelligence Report on '{os.path.basename(selected_bgm_path)}': genre={genre} (confidence={conf:.2f})")
                except Exception as mie:
                    logger.debug(f"Music intelligence report fallback: {mie}")

                # Register usage metadata without moving to cooldown (rotation disabled per directive)
                try:
                    from Audio_Modules.audio_pool_manager import AudioPoolManager
                    pool = AudioPoolManager(base_dir=audio_dir)
                    pool.use_audio(selected_bgm_path)
                except Exception as pr_err:
                    logger.debug(f"Usage logging notice: {pr_err}")
            else:
                # Mandatory BGM Assignment: If Gemini didn't return a specific pick or video is muted, dynamically select best track from pool (LRU rotated)
                try:
                    from Audio_Modules.audio_pool_manager import AudioPoolManager
                    pool = AudioPoolManager(base_dir=audio_dir)
                    intent_cat = forensic_context.get("intent", "viral_reel")
                    selected_bgm_path = pool.select_best_audio(content_category=intent_cat)
                    if selected_bgm_path and os.path.isfile(selected_bgm_path):
                        winner_name = os.path.basename(selected_bgm_path)
                        cached = next((c for c in audio_candidates if c.get("track_name") == winner_name), None)
                        if cached:
                            selected_bgm_bpm = cached.get("bpm", 120.0)
                            selected_bgm_beats = cached.get("beats", []) or cached.get("drops", [])
                        logger.info(f"🎶 [MANDATORY_BGM_RESCUE] Selected LRU rotated BGM track from pool: {os.path.basename(selected_bgm_path)}")
                except Exception as rescue_err:
                    logger.debug(f"Mandatory BGM pool rescue fallback notice: {rescue_err}")

            # ── STEP 4: Audio Extraction, Semantic Analysis & Rhythm Routing ──
            # Primary beat grid source: selected BGM (if resolved)
            # Fallback: Phase 1 cached clip ambient audio
            clip_dir = os.path.dirname(video_path)
            _phase1_audio = None

            if selected_bgm_path and os.path.isfile(selected_bgm_path):
                # Use selected BGM track's beat grid
                beat_grid = selected_bgm_beats
                bpm       = selected_bgm_bpm
                if not beat_grid:
                    try:
                        beat_grid = self.beat_engine.analyze_beats(selected_bgm_path)
                    except Exception as _b_err:
                        logger.warning(f"   ⚠ BGM beat extraction fallback: {_b_err}")
                logger.info(
                    "♪ [Step 4/6] Beat grid from selected BGM (%s): beats=%d BPM=%.1f",
                    os.path.basename(selected_bgm_path), len(beat_grid), bpm,
                )
            else:
                # Fall back to Phase 1 clip ambient audio analysis
                try:
                    from Audio_Modules.audio_extractor import load_audio_analysis, run_phase1_audio_analysis
                    _phase1_audio = load_audio_analysis(clip_dir)
                    if _phase1_audio:
                        logger.info(
                            "♻️ [Step 4/6] Using Phase 1 pre-computed audio: "
                            "beats=%d drops=%d tempo=%.1fBPM vibe=%s",
                            _phase1_audio.get("beat_count", 0),
                            _phase1_audio.get("drop_count", 0),
                            _phase1_audio.get("tempo_bpm", 0.0),
                            _phase1_audio.get("vibe", "unknown"),
                        )
                        beat_grid = _phase1_audio.get("beats", [])
                        bpm = _phase1_audio.get("tempo_bpm", 120.0)
                    else:
                        logger.info("🎵 [Step 4/6] No Phase 1 audio cache — running live extraction...")
                        _phase1_audio = run_phase1_audio_analysis(video_path, clip_dir)
                        beat_grid = _phase1_audio.get("beats", [])
                        bpm = _phase1_audio.get("tempo_bpm", 120.0)
                except Exception as _ae:
                    logger.warning("⚠️ Phase 1 audio cache load failed (%s) — defaulting BPM=120.", _ae)
                    beat_grid = []
                    bpm = 120.0

            # Resolve target_audio for semantic analyzer
            # Priority: selected BGM → clip wav (Phase 1) → video itself
            if selected_bgm_path and os.path.isfile(selected_bgm_path):
                target_audio = selected_bgm_path
            elif bgm_path and os.path.isfile(bgm_path):
                target_audio = bgm_path
            elif _phase1_audio and _phase1_audio.get("wav_path") and os.path.isfile(_phase1_audio["wav_path"]):
                target_audio = _phase1_audio["wav_path"]
            else:
                target_audio = video_path  # last resort: feed raw video

            logger.info("🎵 [Step 4/6] Semantic Audio Engine: Analyzing Meaning & Mood of %s...",
                        os.path.basename(target_audio))
            
            # Load persistent lyric & rhythm intelligence (0.000s disk cache hit if previously analyzed)
            lyric_intel = {}
            try:
                lyric_intel = analyze_music(target_audio)
                if lyric_intel and lyric_intel.get("_source") != "fallback":
                    logger.info("   ✓ Lyric Intelligence Synced: emotion='%s' | vocals=%s | lang='%s' (_source=%s)",
                                lyric_intel.get("dominant_emotion"), lyric_intel.get("has_vocals"),
                                lyric_intel.get("language"), lyric_intel.get("_source"))
            except Exception as _le_err:
                logger.debug(f"Lyric rhythm aligner integration fallback: {_le_err}")

            if not beat_grid and os.path.isfile(target_audio):
                try:
                    beat_res  = self.beat_engine.analyze(target_audio)
                    beat_grid = beat_res.get("beats", [])
                    bpm       = beat_res.get("bpm", 120.0)
                except Exception as be_err:
                    logger.warning("   ⚠ Beat Engine fallback: %s", be_err)

            # Unified Semantic Understanding & Route Parameter Calculation
            route_params = compute_routing_parameters(lyric_intel, forensic_context, selected_bgm_path)

            # Apply Audio Strategy Constraints (Dialogue Protection & Speech Cut Boundaries)
            try:
                from Audio_Modules.audio_strategy import routing_for_strategy
                _strat = forensic_context.get("audio_strategy", {})
                if _strat:
                    route_params = routing_for_strategy(_strat, route_params)
            except Exception as _rfs_err:
                logger.warning(f"Audio strategy routing notice: {_rfs_err}")

            logger.info(
                f"   ✓ Unified Route Selected: '{route_params['strategy_name']}' | "
                f"speed={route_params['speed_factor']}x | "
                f"transition='{route_params['transition_type']}' | "
                f"ducking={route_params['bgm_ducking_db']}dB"
            )

            # ── STEP 4.5: Rhythm Timeline Construction (Human-Scale Takes 2.0s-3.5s) ──
            logger.info("🥁 [Step 4.5/6] Rhythm Timeline Engine: Building micro-shot timeline...")
            micro_shots = []
            rtb_timeline = []
            try:
                v_dur = self.rhythm_timeline_builder._get_duration(video_path)
                if v_dur > 6.0:
                    self.rhythm_timeline_builder.min_duration = 2.0
                    self.rhythm_timeline_builder.max_duration = 3.5
                    timeline_scenes = forensic_context.get("timeline") or []
                    if timeline_scenes and isinstance(timeline_scenes, list):
                        raw_scenes = [
                            {
                                "clip_id": idx,
                                "start": float(sc.get("start", 0.0)),
                                "end": float(sc.get("end", v_dur)),
                                "score": float(sc.get("importance", 0.85) / 5.0) if "importance" in sc else 0.85,
                                "scene_label": sc.get("scene_label", f"scene_{idx}"),
                                "shot_type": sc.get("shot_type", "medium"),
                            }
                            for idx, sc in enumerate(timeline_scenes)
                            if float(sc.get("end", 0.0)) > float(sc.get("start", 0.0))
                        ]
                    else:
                        raw_scenes = [{"clip_id": 0, "start": 0.0, "end": v_dur, "score": 0.85}]
                    mi_info = {
                        "bar_duration_sec": 60.0 / (bpm or 120.0),
                        "sections": lyric_intel.get("sections", [{"start": 0.0, "end": v_dur, "type": "verse"}]),
                        "vibe": route_params.get("recommended_editing_mode", "hype")
                    }
                    all_shots = self.rhythm_timeline_builder._extract_micro_shots(
                        scenes=raw_scenes,
                        vibe=route_params.get("recommended_editing_mode", "hype"),
                        music_intelligence=mi_info
                    )
                    try:
                        target_hint = min(target_duration, v_dur) if v_dur > 0 else target_duration
                        rtb_timeline = self.rhythm_timeline_builder.build_timeline(
                            scenes=raw_scenes,
                            beat_grid=beat_grid,
                            vibe=route_params.get("recommended_editing_mode", "hype"),
                            music_intelligence=lyric_intel if lyric_intel else mi_info,
                            target_duration_hint=target_hint,
                        )
                    except Exception as _tb_err:
                        logger.debug(f"build_timeline fallback: {_tb_err}")

                    # Human-Scale Jump-Cutting: Select 2.0s-3.5s takes (Hook, Action, Climax)
                    if len(all_shots) >= 4:
                        selected = [all_shots[0]]
                        step_gap = max(2, len(all_shots) // 3)
                        for idx in range(step_gap, len(all_shots) - 1, step_gap):
                            selected.append(all_shots[idx])
                        if all_shots[-1] not in selected:
                            selected.append(all_shots[-1])
                        micro_shots = selected
                        logger.info(f"   ✓ Human-Scale Jump-Cutting: Selected {len(micro_shots)} dynamic sub-shots (2.0s-3.5s takes) out of {len(all_shots)} candidates.")
                    else:
                        micro_shots = all_shots
            except Exception as rte:
                logger.warning(f"   ⚠ Rhythm timeline construction fallback: {rte}")

            # ── STEP 5: Pass 2 Master Director Synthesis & FFmpeg Execution ────
            logger.info("🎬 [Step 5/6] Pass 2 Master Director: Synthesizing FFmpeg Filtergraph & Rendering Reel...")
            user_req = (
                f"Master edit for video intent '{forensic_context.get('intent', 'viral_reel')}', "
                f"visual tone '{forensic_context.get('tone', 'aspirational')}', "
                f"audio mood '{route_params.get('dominant_emotion', 'hype')}', "
                f"editing route '{route_params['strategy_name']}', "
                f"speed factor '{route_params['speed_factor']}x', "
                f"transition '{route_params['transition_type']}'. "
                f"Target duration: {target_duration}s."
            )
            # Pass selected BGM from mathematical audio router (preferred), then pool/bgm_path, then clip's clean extracted audio
            _final_bgm = route_params.get("selected_audio_path") or selected_bgm_path or bgm_path
            if not _final_bgm or not os.path.exists(_final_bgm):
                clip_extracted = os.path.join(os.path.dirname(video_path), "video_extracted.wav")
                if os.path.isfile(clip_extracted):
                    _final_bgm = clip_extracted
                    logger.info(f"🎙️ [CONTINUOUS AUDIO ADOPTED] No external pool BGM — adopting clip's clean extracted audio: {os.path.basename(_final_bgm)}")
                elif _phase1_audio and _phase1_audio.get("wav_path") and os.path.isfile(_phase1_audio["wav_path"]):
                    _final_bgm = _phase1_audio["wav_path"]
                    logger.info(f"🎙️ [CONTINUOUS AUDIO ADOPTED] Using Phase 1 extracted wav: {os.path.basename(_final_bgm)}")

            if _final_bgm and os.path.exists(_final_bgm):
                logger.info(f"🎶 [AUDIO TRACK VERIFIED] Using continuous audio: {os.path.basename(_final_bgm)}")
            extra_inputs = {
                "micro_shots": micro_shots,
                "rtb_timeline": rtb_timeline if rtb_timeline else micro_shots,
            }
            if lyric_intel:
                extra_inputs["lyric_intel"] = lyric_intel
            synthesis_result = self.ffmpeg_engine.run_full_pipeline(
                user_request=user_req,
                input_video_path=video_path,
                output_video_path=output_path,
                audio_path=_final_bgm,
                forensic_context=forensic_context,
                extra_inputs=extra_inputs
            )

            elapsed = time.time() - start_time
            logger.info(f"✅ Master Reel Successfully Rendered in {elapsed:.1f}s -> {output_path}")

            return {
                "success": True,
                "output_video": output_path,
                "forensic_context": forensic_context,
                "synthesis_result": synthesis_result,
                "elapsed_seconds": round(elapsed, 2),
            }

        except Exception as exc:
            logger.error(f"❌ Master AI Editor Failed: {exc}", exc_info=True)
            return {
                "success": False,
                "error": str(exc),
                "output_video": None,
            }
        finally:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Module API Function ───────────────────────────────────────────────────────

_editor_singleton: Optional[MasterAIEditor] = None

def edit_video_master(
    video_path: str,
    bgm_path: Optional[str] = None,
    output_path: Optional[str] = None,
    target_duration: float = 15.0,
) -> Dict[str, Any]:
    """Module function to execute Master AI Video Editing Pipeline."""
    global _editor_singleton
    if _editor_singleton is None:
        _editor_singleton = MasterAIEditor()
    return _editor_singleton.process(
        video_path=video_path,
        bgm_path=bgm_path,
        output_path=output_path,
        target_duration=target_duration,
    )


# ── CLI Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMTCE Master AI Video Editor (Gemini + FFmpeg)")
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("--bgm", type=str, default=None, help="Path to optional background music file")
    parser.add_argument("--output", type=str, default=None, help="Path to output rendered video")
    parser.add_argument("--duration", type=float, default=15.0, help="Target duration in seconds (default 15.0)")

    args = parser.parse_args()
    res = edit_video_master(
        args.video,
        bgm_path=args.bgm,
        output_path=args.output,
        target_duration=args.duration,
    )
    if res.get("success"):
        try:
            print(f"\n🎉 MASTER REEL COMPLETE: {res['output_video']}")
        except UnicodeEncodeError:
            print(f"\n[+] MASTER REEL COMPLETE: {res['output_video']}")
    else:
        try:
            print(f"\n💥 EDIT FAILED: {res.get('error')}")
        except UnicodeEncodeError:
            print(f"\n[-] EDIT FAILED: {res.get('error')}")
        sys.exit(1)
