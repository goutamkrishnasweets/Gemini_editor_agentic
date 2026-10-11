"""
Intelligence_Modules/forensic_analyzer.py
------------------------------------------
Forensic Video Analyzer — Vision-AI based frame inspection.

Performs TWO tasks on extracted video frames via Gemini Vision:

  TASK 1 — CONTENT ANALYSIS
    Classifies content intent, confidence, recommended editing feature flags,
    and monetization safety rating.

Output format (strict JSON):
{
  "watermarks": [{"x":0,"y":0,"w":0,"h":0}],
  "content_strategy": {
    "intent": "...",
    "confidence": 0.0-1.0,
    "feature_flags": {
        "enable_price_tags": true/false,
        "enable_fashion_caption": true/false,
        "enable_cinematic_zoom": true/false,
        "enable_speed_ramps": true/false,
        "enable_fast_pacing": true/false,
        "enable_voiceover": true/false
    },
    "recommended_editing_style": "...",
    "safety": "safe|risky|blocked"
  }
}
"""

import os
import json
import logging
import re
import subprocess
import tempfile
import shutil
import copy
from typing import List, Optional, Any, Dict
try:
    from Gemini_Modules.gemini_router_module.gemini_governor import gemini_router
except ImportError:
    try:
        from gemini_governor import gemini_router
    except ImportError:
        gemini_router = None

from dotenv import load_dotenv

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env first, then Credentials/.env with override=False so real env vars win,
# and .env takes precedence over Credentials/.env
for env_candidate in [
    os.path.join(_REPO_ROOT, ".env"),
    os.path.join(_REPO_ROOT, "Credentials", ".env"),
]:
    if os.path.exists(env_candidate):
        load_dotenv(env_candidate, override=False)

logger = logging.getLogger("forensic_analyzer")

# ── Prompt ────────────────────────────────────────────────────────────────────

FORENSIC_PROMPT = """
You are a Professional Short-Form Content Director analyzing a sparse set of sampled video frames.
You do not have the full video — only {frame_count} frames at {width}x{height}. Treat every frame as
partial evidence, not the whole story.

Your output has two consumers, both of which matter:
1. A second-stage AI editor (Gemini Call 3) will read your JSON as raw text and reason over it to build
   an actual FFmpeg edit plan. Your fields are its only source of visual understanding — it never sees
   the frames itself. Vague or generic output here directly degrades what it produces.
2. Your output is stored permanently as training/retrieval signal for a future recommendation system.
   Precision and honesty matter more than confident-sounding guesses — a wrong but confident answer
   pollutes that store in a way that compounds over time.

---

GROUNDING RULES (apply to everything below)

- Only report what is directly visible in the frames provided. Do not infer off-screen action, audio
  content, or context you cannot see. If you would need information from outside these frames to be
  sure, say so in `visual_event` rather than stating it as fact.
- If frames are ambiguous, contradictory, or too sparse to classify confidently, say so — do not force
  a confident answer to satisfy the schema. Use lower confidence and describe uncertainty in `visual_event`.
  A well-flagged "uncertain" is more useful to Call 3 than a wrong "certain."
- If the video appears to mix multiple content types, classify by whichever type occupies the most visual
  weight across sampled frames, and name the secondary type explicitly inside `visual_event`.

---

TASK 1 — CONTENT CLASSIFICATION

Examine all frames and classify this video by what you directly observe.

`intent` — Write a short, specific content label that accurately describes what this video IS, derived
entirely from what you see. Be specific and descriptive rather than generic.
Examples of good outputs: "bollywood_celebrity_arrival", "gym_workout_reel", "product_unboxing_review",
"street_food_vendor", "standup_comedy_clip", "corporate_tech_keynote", "wildlife_drone_footage".
Bad outputs: "general", "video", "content". If frames are too ambiguous to classify, write "unclassifiable"
and explain in `visual_event`.

`feature_flags` — Decide each flag independently from what you observe in the frames. Do not apply
category-based rules. Ask yourself for each flag: does the visual evidence in these frames directly
support enabling this feature for better viewer engagement?
- enable_cinematic_zoom: would slow push-in or pull-out zooms enhance the visual subject on screen?
- enable_speed_ramps: is there physical motion, action, or rhythm that would benefit from speed variation?
- enable_fast_pacing: are there multiple distinct shots or scenes that suggest rapid cut editing style?
- enable_voiceover: is there visible on-camera speech, narration, or commentary that needs audio treatment?
- enable_fashion_caption: are specific garments, accessories, or fashion items the visual focus of frames?
- enable_price_tags: are specific purchasable products, items, or merchandise visibly featured?

`confidence` (0.0–1.0) reflects evidence quality, not category enthusiasm:
- 0.8–1.0: classification is unambiguous across most/all sampled frames.
- 0.5–0.79: likely correct, but some frames are inconclusive, low-detail, or mixed-content.
- 0.0–0.49: too sparse/blurry/static/contradictory to classify confidently.

Safety Rules (`safety.classification`) — these are policy rules, always apply them:
- "safe": brand-safe, no policy-sensitive content, suitable for ad monetization.
- "risky": borderline — mild innuendo, implied aggressive language, alcohol/tobacco visible but not the
  focus, intense-but-non-graphic action, unverified claims-style content.
- "blocked": nudity/sexual content, graphic violence/gore, weapons used to threaten, visible drug use,
  hate symbols or hateful gestures, content endangering minors, self-harm depiction, or copyrighted
  broadcast footage (sports leagues, films, TV) used as the primary subject.
If any "blocked" trigger is present, `monetization_safe` must be false regardless of other flags.

---

TASK 2 — CONTENT DIRECTOR (Human Editor Intelligence)

This block is the richest signal Call 3 receives — write it like you're briefing a human editor who
has never seen the footage, not like you're filling out a form. All fields are free-form — write
exactly what you observe, not what fits a predefined list.

1. main_subject: the primary human, celebrity, object, or hero entity — use their real name if
   identifiable (e.g. "Kareena Kapoor", "Virat Kohli", "Porsche 911 GT3"), otherwise describe
   accurately (e.g. "Fitness model in olive activewear", "Street food chef at outdoor stall").
2. detected_entities: specific people, objects, brands, garments, environments, or cultural references
   visible. Be specific and granular. Free-form list of strings.
3. visual_event: 1-2 concrete, vivid sentences describing what is actually happening across sampled
   frames. Ground every claim in what is visible. Name specific subjects and actions.
4. viewer_attention: the single object or subject most likely to grab a viewer's attention in the
   first frame.
5. internet_context: recognizable cultural, celebrity, fashion, or viral references visible in the
   frames. List any events, trends, or contexts you can identify from visual evidence alone.
6. possible_narratives: list 2-4 specific narrative angles this footage could support as short-form
   content. Write them as editor briefs, not category labels.
7. recommended_narrative: the single best narrative angle from your list above. One sentence.
8. tone: the dominant emotional register of this footage as you observe it. Free-form word or phrase.
9. editing_style: the editing approach that best fits what you see in the frames. Free-form descriptor.
10. engagement_hook: one concrete sentence describing what should appear or be said in the first 3
    seconds — grounded in what is actually in the frames, naming the subject and visual action.
    Call 3 will act on this directly, so specificity has outsized value.
11. feature_commands: echo the same feature flags you set in Task 1 for consistency.

---

Return ONLY valid JSON. No markdown wrappers, no commentary. Just the JSON object.

Required JSON format:
{{
  "intent": "<free-form content label derived from visual observation>",
  "confidence": <float 0.0-1.0>,
  "editing_style": "<free-form editing style descriptor>",
  "feature_flags": {{
    "enable_price_tags": <bool>,
    "enable_fashion_caption": <bool>,
    "enable_cinematic_zoom": <bool>,
    "enable_speed_ramps": <bool>,
    "enable_fast_pacing": <bool>,
    "enable_voiceover": <bool>
  }},
  "platform_priority": ["<primary platform>", "<secondary platform>"],
  "safety": {{
    "classification": "safe|risky|blocked",
    "monetization_safe": <bool>
  }},
  "content_director": {{
    "main_subject": "<full name or accurate visual description>",
    "detected_entities": [],
    "visual_event": "",
    "viewer_attention": "",
    "internet_context": [],
    "possible_narratives": [],
    "recommended_narrative": "",
    "tone": "",
    "editing_style": "",
    "is_talking_on_camera": <bool>,
    "engagement_hook": "",
    "feature_commands": {{
      "enable_fast_pacing": <bool>,
      "enable_cinematic_zoom": <bool>,
      "enable_speed_ramps": <bool>,
      "enable_voiceover": <bool>,
      "enable_price_tags": <bool>,
      "enable_fashion_caption": <bool>
    }}
  }}
}}

Frame dimensions: {width}x{height} pixels.
Number of frames provided: {frame_count}
"""


# ── Default fallback ──────────────────────────────────────────────────────────

DEFAULT_RESULT = {
    "_failed":                   True,
    "watermarks":                [],
    "intent":                    "unknown",
    "confidence":                0.0,
    "editing_style":             "cinematic",
    "main_subject":              "",
    "is_talking_on_camera":      False,
    "content_director":          {},
    "feature_flags": {
        "enable_price_tags":      False,
        "enable_fashion_caption": False,
        "enable_cinematic_zoom":  True,
        "enable_speed_ramps":     False,
        "enable_fast_pacing":     False,
        "enable_voiceover":       True,
    },
    "platform_priority": ["youtube_shorts", "instagram_reels", "facebook_reels"],
    "safety": {
        "classification":    "risky",
        "monetization_safe": False,
    },
    # Backward-compat wrapper so orchestrator.get("content_strategy") still works
    "content_strategy": {
        "intent":                    "unknown",
        "confidence":                0.0,
        "recommended_editing_style": "cinematic",
        "feature_flags": {
            "enable_price_tags":      False,
            "enable_fashion_caption": False,
            "enable_cinematic_zoom":  True,
            "enable_speed_ramps":     False,
            "enable_fast_pacing":     False,
            "enable_voiceover":       True,
        },
        "safety": "risky",
    },
}


# ── Main class ────────────────────────────────────────────────────────────────

class ForensicVideoAnalyzer:
    """
    Extracts frames from a video and sends them to Gemini Vision for forensic analysis.

    Usage:
        analyzer = ForensicVideoAnalyzer()
        result = analyzer.analyze(video_path)
        # result is a dict matching the JSON schema above
    """

    # How many frames to sample from the video (spread evenly)
    FRAME_COUNT = 5
    # Target resolution for frames sent to Gemini (keeps token count low)
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 360

    def __init__(self):
        self.router = gemini_router
        self._available = bool(self.router is not None)
        if not self._available:
            logger.warning("🔬 ForensicAnalyzer: gemini_router unavailable — will return defaults")
            return

        model_name = os.getenv("GEMINI_MODEL")
        logger.info(f"🔬 ForensicAnalyzer: ACTIVE (model={model_name})")

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, video_path: str,
                frame_paths: Optional[List[str]] = None,
                creator_name: Optional[str] = None,
                audio_candidates: Optional[List[dict]] = None) -> dict:
        """
        Perform forensic & scene analysis on a video.

        Args:
            video_path:       Path to source video file.
            frame_paths:      Optional list of pre-extracted frame image paths.
            creator_name:     Optional creator handle/title hint for face cache RAG.
            audio_candidates: Optional list of candidate BGM audio metadata dicts.
        """
        clip_folder = os.path.dirname(os.path.abspath(video_path))
        clip_id = os.path.basename(clip_folder)
        scene_context = {}
        tmp_dir = None
        own_frames = False

        try:
            # ── Pre-Pipeline Scene & Face Intelligence ─────────────────────────
            try:
                try:
                    from Core_Modules.scene_intel import analyze_scene_pre_pipeline
                except ImportError:
                    from Gemini_Modules.scene_intel import analyze_scene_pre_pipeline
                scene_context = analyze_scene_pre_pipeline(video_path, creator_name=creator_name)
                logger.info(
                    f"👤 SceneIntel: faces={scene_context.get('num_detected_faces')} | "
                    f"subjects={scene_context.get('num_subjects')} | "
                    f"face_cache={scene_context.get('face_cache_status')}"
                )
            except Exception as sie:
                logger.debug(f"SceneIntel fallback: {sie}")

            if not self._available:
                logger.info("🔬 ForensicAnalyzer skipped (unavailable)")
                res = copy.deepcopy(DEFAULT_RESULT)
                res["scene_context"] = scene_context
                res.pop("_failed", None)
                return res

            # ── Step 0: Ensure 480p Proxy Video exists for Gemini & Sampling ──
            proxy_video_path = None
            try:
                from Main_Modules.proxy_encoder import ensure_proxy
                proxy_video_path = ensure_proxy(video_path)
                logger.info(f"🔬 ForensicAnalyzer: Using 480p proxy for analysis: {proxy_video_path}")
            except Exception as pe:
                logger.debug(f"🔬 ForensicAnalyzer: Proxy encode notice ({pe}); falling back to source video")
                proxy_video_path = video_path

            # ── Step 1: Extract frames (Hook-Dense Strategic Sampler) ─────────
            sampling_context = None
            if frame_paths and all(os.path.exists(p) for p in frame_paths):
                frames = frame_paths
            else:
                tmp_dir = tempfile.mkdtemp(prefix="forensic_frames_")
                own_frames = True
                sampling_target = proxy_video_path if (proxy_video_path and os.path.exists(proxy_video_path)) else video_path
                try:
                    try:
                        from Main_Modules.strategic_frame_sampler import extract_strategic_frame_files
                    except ImportError:
                        from Core_Modules.scene_intel import extract_strategic_frame_files
                    res_frames = extract_strategic_frame_files(sampling_target, tmp_dir, return_meta=True)
                    if isinstance(res_frames, tuple):
                        frames, sample_meta = res_frames
                        sampling_context = self._build_sampling_note(sample_meta)
                    else:
                        frames = res_frames
                    logger.info(f"🔬 ForensicAnalyzer: loaded {len(frames)} strategic hook-dense frames")
                except Exception as sse:
                    logger.debug(f"Strategic frame extraction notice ({sse}); using ffmpeg frame extraction")
                    frames = self._extract_frames(sampling_target, tmp_dir)
                    sampling_context = None

            if not frames:
                logger.warning("🔬 ForensicAnalyzer: no frames extracted — returning default")
                res = copy.deepcopy(DEFAULT_RESULT)
                res["scene_context"] = scene_context
                res.pop("_failed", None)
                return res

            # ── Step 2: Build Gemini payload with audio candidate table + proxy video ───────
            result = self._call_gemini_with_audio(
                frames,
                creator_name=creator_name,
                clip_dir=clip_folder,
                audio_candidates=audio_candidates,
                sampling_context=sampling_context,
                video_proxy_path=proxy_video_path
            )
            result["scene_context"] = scene_context

            # Skip persistence if analysis failed to avoid store pollution
            is_failed = bool(result.pop("_failed", False)) or (result.get("confidence", 0.0) == 0.0 and result.get("intent") == "unknown")
            if is_failed:
                logger.warning(f"🔬 ForensicAnalyzer: Analysis flagged as failed — skipping persistence for '{clip_id}'.")
                return result

            # ── Step 2.5: Save to Master ClipIntelligenceStore (Schema v3) ──
            try:
                from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore
                from Audio_Modules.audio_extractor import load_audio_analysis

                store = ClipIntelligenceStore()
                clip_data = store.load(clip_id, clip_folder) or store.create_blank(clip_id, clip_folder)

                # 1. Fill audio_data.math FIRST from Phase 1 DSP audio_analysis.json
                p1_audio = load_audio_analysis(clip_folder)
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

                # 2. Fill visual_context SECOND (Gemini visual semantic context)
                cd_block = result.get("content_director", {})
                discovered_subject = cd_block.get("main_subject") or result.get("main_subject") or ""

                visual_ctx = {
                    "main_subject": discovered_subject,
                    "person_name": discovered_subject,
                    "visual_event": cd_block.get("visual_event", ""),
                    "intent": result.get("intent", "viral_reel"),
                    "tone": cd_block.get("tone", "aspirational"),
                    "editing_style": result.get("editing_style", "cinematic"),
                    "engagement_hook": cd_block.get("engagement_hook", ""),
                    "detected_entities": cd_block.get("detected_entities", []),
                    "possible_narratives": cd_block.get("possible_narratives", []),
                    "recommended_narrative": cd_block.get("recommended_narrative", ""),
                    "feature_flags": result.get("feature_flags", {}),
                    "safety": result.get("safety", {}),
                    "creative_possibilities": result.get("creative_possibilities", []),
                    "content_director": cd_block,
                }
                # Calculate 3-Signal speech_intelligence (Vision Talking + Audio Vocals + Speech Formants)
                is_talking = bool(result.get("is_talking_on_camera") or cd_block.get("is_talking_on_camera", False) or result.get("intent") == "talking_head")
                has_audio = bool(p1_audio.get("has_audio", True)) if p1_audio else True
                has_vocals = bool(p1_audio.get("has_vocals", True)) if p1_audio else is_talking
                is_speech_vocal = bool(p1_audio.get("is_speech_vocal", True)) if p1_audio else is_talking

                if is_talking and has_vocals and is_speech_vocal:
                    speech_mode = "on_camera_dialogue"
                    rec_action = "preserve_voice_duck_bgm"
                elif is_talking and has_vocals and not is_speech_vocal:
                    # On-camera musical performance / lip-sync: preserve original song sync
                    speech_mode = "lip_sync_dub"
                    rec_action = "preserve_original_audio"
                elif not is_talking and has_vocals and is_speech_vocal:
                    speech_mode = "voiceover_narration"
                    rec_action = "preserve_voice_duck_bgm"
                elif not is_talking and has_vocals and not is_speech_vocal:
                    speech_mode = "music_broll"
                    rec_action = "audio_replace_full_bgm"
                elif not has_vocals:
                    speech_mode = "silent_broll"
                    rec_action = "audio_replace_full_bgm"
                else:
                    # SAFE DEFAULT: Default to preserving voice if audio is present!
                    speech_mode = "on_camera_dialogue" if has_audio else "silent_broll"
                    rec_action = "preserve_voice_duck_bgm" if has_audio else "audio_replace_full_bgm"

                speech_intel = {
                    "speech_mode": speech_mode,
                    "is_talking_visually": is_talking,
                    "has_spoken_vocal": has_vocals,
                    "is_speech_vocal": is_speech_vocal,
                    "recommended_audio_action": rec_action
                }
                result["speech_intelligence"] = speech_intel
                visual_ctx["speech_intelligence"] = speech_intel

                store.patch_visual_context(clip_data, visual_ctx)

                # 3. Fill audio_data.context THIRD (Gemini audio semantic context into same audio_data block)
                audio_ctx = {
                    "has_vocals": has_vocals,
                    "speech_mode": speech_mode,
                    "recommended_audio_action": rec_action,
                    "dominant_emotion": result.get("content_director", {}).get("tone", "hype"),
                    "vibe": result.get("editing_style", "rhythm_driven"),
                }
                store.patch_audio_context(clip_data, audio_ctx)

                # 4. Fill visual_vectors FOURTH (Mathematical targeting timestamps for OpenCV)
                # Compute strategic targeting timestamps
                targeted_ts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
                if isinstance(result.get("creative_possibilities"), list):
                    for cp in result["creative_possibilities"]:
                        if isinstance(cp, dict) and "target_duration_sec" in cp:
                            dur = float(cp["target_duration_sec"])
                            targeted_ts.extend([round(dur * 0.25, 2), round(dur * 0.5, 2), round(dur * 0.75, 2), round(dur, 2)])
                targeted_ts = sorted(list(set(targeted_ts)))

                vectors = {
                    "targeted_timestamps_sec": targeted_ts,
                    "scene_cut_timestamps": [t for t in targeted_ts if t > 2.0],
                    "hook_zone_end_sec": 5.0,
                    "climax_zone_start_sec": max(targeted_ts) if targeted_ts else 10.0,
                }
                store.patch_visual_vectors(clip_data, vectors)

                # Save updated clip intelligence JSON to disk and pool_metadata.json
                store.save(clip_id, clip_data, clip_folder)
                result["clip_intelligence"] = clip_data
                result["visual_vectors"] = vectors
                logger.info(f"🧠 [ClipIntelligenceStore] Updated clip '{clip_id}' -> speech_mode='{speech_mode}' ({rec_action}).")
            except Exception as store_err:
                logger.warning(f"🧠 [ClipIntelligenceStore] Update warning: {store_err}")

            # ── Step 2.6: Write full forensic result to pool_metadata.json ────
            # Key: gemini_semantic_visual_intelligence
            # Written & saved by AudioPoolManager so Telegram vault sync is automatic.
            try:
                from Audio_Modules.audio_pool_manager import AudioPoolManager
                _pool_mgr = AudioPoolManager()
                _semantic_payload = {
                    "intent":                result.get("intent", "unknown"),
                    "confidence":            result.get("confidence", 0.0),
                    "editing_style":         result.get("editing_style", "cinematic"),
                    "feature_flags":         result.get("feature_flags", {}),
                    "platform_priority":     result.get("platform_priority", []),
                    "safety":                result.get("safety", {}),
                    "speech_intelligence":   result.get("speech_intelligence", {}),
                    "content_director":      result.get("content_director", {}),
                    "selected_audio_track":  result.get("selected_audio_track", ""),
                    "creative_possibilities":result.get("creative_possibilities", []),
                    "scene_context":         result.get("scene_context", {}),
                    "updated_at":            __import__("time").time(),
                }
                # Remove root level key if present
                _pool_mgr.metadata.pop("gemini_semantic_visual_intelligence", None)

                # Also inject into matching entry inside files["social_media_id"]
                files_dict = _pool_mgr.metadata.get("files", {})
                social_entries = files_dict.get("social_media_id", {})
                if isinstance(social_entries, dict):
                    for url_key, entry_dict in social_entries.items():
                        if isinstance(entry_dict, dict):
                            sc = str(entry_dict.get("shortcode", "")).strip()
                            # 1. Exact non-empty shortcode match against clip_id
                            sc_match = bool(sc and sc == clip_id)
                            # 2. Exact URL path segment match (prevents 1-2 char substring collision)
                            url_segments = [p for p in re.split(r'[/_\\?&=]', url_key) if p]
                            url_match = bool(clip_id and clip_id in url_segments)

                            if sc_match or url_match:
                                entry_dict["gemini_semantic_visual_intelligence"] = _semantic_payload
                                logger.info(f"📊 [ForensicAnalyzer] Injected gemini_semantic_visual_intelligence into files['social_media_id']['{url_key}']")

                _pool_mgr._save_metadata(sync_to_vault=True)
                logger.info(
                    "📊 [ForensicAnalyzer] Saved gemini_semantic_visual_intelligence -> pool_metadata.json "
                    f"(intent={_semantic_payload['intent']}, safety={_semantic_payload.get('safety', {}).get('classification', '?')})"
                )
            except Exception as _pm_err:
                logger.warning(f"📊 [ForensicAnalyzer] pool_metadata write warning: {_pm_err}")

            return result

        except Exception as e:
            import traceback
            logger.error(f"🔬 ForensicAnalyzer: unexpected error — {e}\n{traceback.format_exc()}")
            res = copy.deepcopy(DEFAULT_RESULT)
            res["scene_context"] = scene_context
            res.pop("_failed", None)
            return res

        finally:
            # ── Step 3: Cleanup temporary extracted frames ────────────────────
            if own_frames and tmp_dir and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _build_sampling_note(self, sample_meta: dict) -> str:
        """Build a factual sampling-zone note from actual sampler output."""
        if not sample_meta or not isinstance(sample_meta, dict):
            return ""
        hook = sample_meta.get("hook_count")
        body = sample_meta.get("body_count")
        climax = sample_meta.get("climax_count")
        motion = sample_meta.get("motion_count")
        if hook is None or body is None:
            return ""
        return (
            f"[FRAME SAMPLING CONTEXT] {hook} hook frames (0-5s, dense), "
            f"{body} body frames (sparse, evenly spaced)"
            + (f", {climax} climax frames (last 10s, dense)" if climax else "")
            + (f", {motion} peak-motion frames (optical flow)" if motion else "")
            + ". Hook zone is intentionally over-represented — weight body/climax frames "
            "at least equally when judging overall tone, don't let the hook dominate the classification."
        )

    def _extract_frames(self, video_path: str, out_dir: str) -> List[str]:
        """
        Extract FRAME_COUNT frames evenly spread across the video using FFmpeg.
        Returns list of absolute paths to extracted JPEG files.
        """
        if not os.path.exists(video_path):
            logger.warning(f"🔬 Frame extraction: video not found — {video_path}")
            return []

        try:
            # Get duration via ffprobe (checking format=duration and stream=duration for v:0)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "format=duration:stream=duration",
                 "-of", "json", video_path],
                capture_output=True, text=True, timeout=15
            )
            dur_data = json.loads(probe.stdout)
            fmt_dur = dur_data.get("format", {}).get("duration")
            streams = dur_data.get("streams", [])
            stream_dur = streams[0].get("duration") if streams else None
            try:
                duration = float(fmt_dur or stream_dur or 10.0)
            except (ValueError, TypeError):
                duration = 10.0
        except Exception as pe:
            logger.warning(f"🔬 Frame extraction: duration probe failed — {pe}")
            duration = 10.0

        n = self.FRAME_COUNT
        interval = max(0.5, duration / (n + 1))
        frame_paths = []

        for i in range(1, n + 1):
            ts = round(i * interval, 2)
            out_path = os.path.join(out_dir, f"frame_{i:02d}.jpg")
            cmd = [
                "ffmpeg", "-y", "-ss", str(ts),
                "-i", video_path,
                "-vframes", "1",
                "-vf", f"scale={self.FRAME_WIDTH}:{self.FRAME_HEIGHT}",
                "-q:v", "3",
                out_path
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=30)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
                    frame_paths.append(out_path)
            except Exception as fe:
                logger.debug(f"🔬 Frame {i} extraction failed: {fe}")

        logger.info(f"🔬 Extracted {len(frame_paths)}/{n} forensic frames")
        return frame_paths

    def _cleanup_uploaded_file(self, uploaded_file: Any) -> None:
        """Deletes uploaded media file from the Gemini File API to prevent resource leakage."""
        if not uploaded_file:
            return
        try:
            fname = getattr(uploaded_file, "name", None)
            if not fname:
                return
            if hasattr(self.router, "delete_file"):
                self.router.delete_file(fname)
            else:
                try:
                    import google.genai as genai_mod
                    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                    if api_key and hasattr(genai_mod, "Client"):
                        client = genai_mod.Client(api_key=api_key)
                        client.files.delete(name=fname)
                        logger.debug(f"🗑️ Deleted uploaded file {fname} from Gemini File API.")
                        return
                except Exception:
                    pass
                try:
                    import google.generativeai as genai_legacy
                    genai_legacy.delete_file(fname)
                    logger.debug(f"🗑️ (Legacy) Deleted uploaded file {fname} from Gemini File API.")
                except Exception:
                    pass
        except Exception as cle:
            logger.debug(f"File API cleanup notice: {cle}")

    def _call_gemini_with_audio(self, frame_paths: List[str],
                                creator_name: Optional[str] = None,
                                clip_dir: Optional[str] = None,
                                audio_candidates: Optional[List[dict]] = None,
                                sampling_context: Optional[str] = None,
                                video_proxy_path: Optional[str] = None) -> dict:
        """
        Send keyframes + candidate BGM audio metadata table to Gemini 2.5 Flash Vision.
        Gemini selects matching audio track and generates structured creative_possibilities edit plan.
        """
        audio_table_str = ""
        if audio_candidates:
            table_lines = []
            for idx, c in enumerate(audio_candidates, start=1):
                table_lines.append(
                    f"{idx}. Track: '{c.get('track_name')}' | BPM: {c.get('bpm')} | Energy: {c.get('avg_energy')} | Drops: {c.get('drops')}"
                )
            audio_table_str = "\n".join(table_lines)

        global_frames = [p for p in frame_paths if "detail_crop" not in p]
        prompt_text = FORENSIC_PROMPT.format(
            width=self.FRAME_WIDTH,
            height=self.FRAME_HEIGHT,
            frame_count=len(global_frames)
        )
        if sampling_context:
            prompt_text += f"\n\n{sampling_context}\n"
        prompt_text += f"\nCreator Handle Hint: '{creator_name or 'unknown'}'\n"

        if video_proxy_path and os.path.isfile(video_proxy_path):
            prompt_text += (
                "\n\n[MULTIMODAL VIDEO ATTACHMENT NOTICE]\n"
                "A 480p proxy of the complete video with audio is attached to this request alongside the sampled frames. "
                "Use the video and its audio as primary evidence for scene pacing, speech detection, and rhythm, "
                "and treat the sampled frames as high-fidelity visual reference checkpoints.\n"
            )

        if audio_table_str:
            prompt_text += f"""
---

TASK 4 — BGM AUDIO SELECTION & CREATIVE POSSIBILITIES

[Candidate Background Music Tracks Available]
{audio_table_str}

Analyze the visual content against the candidate BGM audio tracks listed above:
1. `selected_audio_track`: Choose the SINGLE best matching track filename (e.g. 'Fit_girl_2.mp3').
2. `creative_possibilities`: Provide 2-3 structured edit options as objects:
   [
     {{"rank": 1, "clip_label": "hero_hook", "target_duration_sec": 15, "editing_style": "rhythm_driven"}},
     {{"rank": 2, "clip_label": "action_cut", "target_duration_sec": 30, "editing_style": "fast_cuts"}}
   ]

Include `selected_audio_track` and `creative_possibilities` in your returned JSON object.
"""

        # ── Targeted Clip Ground Truth Metadata & Full CBM Injection ──────────
        target_clip_meta_block = ""
        cbm_block = ""

        if clip_dir and os.path.isdir(clip_dir):
            folder_name = os.path.basename(clip_dir)

            # 1. Load local clip metadata.json
            local_meta = {}
            local_meta_path = os.path.join(clip_dir, "metadata.json")
            if os.path.exists(local_meta_path):
                try:
                    with open(local_meta_path, "r", encoding="utf-8") as _lmf:
                        local_meta = json.load(_lmf)
                except Exception as _lme:
                    logger.debug(f"🔬 Could not load local metadata.json: {_lme}")

            # 2. Load pool_metadata.json
            pool_data = None
            possible_pm_paths = [
                os.path.join(clip_dir, "pool_metadata.json"),
                os.path.join(_REPO_ROOT, "Original_audio", "pool_metadata.json"),
            ]
            for pmp in possible_pm_paths:
                if os.path.exists(pmp):
                    try:
                        with open(pmp, "r", encoding="utf-8") as f:
                            pool_data = json.load(f)
                        break
                    except Exception as _pm_e:
                        logger.debug(f"🔬 Could not load pool_metadata ({pmp}): {_pm_e}")

            # Safe shortcode resolution order:
            # 1) metadata.json shortcode
            # 2) exact match against known shortcode values in pool_metadata
            # 3) suffix after underscore if in pool_metadata or as fallback
            clean_sc = str(local_meta.get("shortcode", "")).strip()

            known_pool_shortcodes = set()
            if pool_data and isinstance(pool_data, dict):
                social_entries = pool_data.get("files", {}).get("social_media_id", {})
                if isinstance(social_entries, dict):
                    for k, v in social_entries.items():
                        if isinstance(v, dict) and v.get("shortcode"):
                            known_pool_shortcodes.add(str(v["shortcode"]).strip())

            if not clean_sc:
                base_name = folder_name[len("manual_"):] if folder_name.startswith("manual_") else folder_name
                if folder_name in known_pool_shortcodes:
                    clean_sc = folder_name
                elif base_name in known_pool_shortcodes:
                    clean_sc = base_name
                elif "_" in base_name and base_name.split("_", 1)[1] in known_pool_shortcodes:
                    clean_sc = base_name.split("_", 1)[1]
                elif "_" in base_name and len(base_name.split("_", 1)[1]) >= 5:
                    clean_sc = base_name.split("_", 1)[1]
                else:
                    clean_sc = base_name

            # 3. Locate the specific clip entry in pool_metadata
            clip_entry = {}
            if pool_data and isinstance(pool_data, dict):
                social_entries = pool_data.get("files", {}).get("social_media_id", {})
                if isinstance(social_entries, dict):
                    for k, v in social_entries.items():
                        if isinstance(v, dict):
                            sc = str(v.get("shortcode", "")).strip()
                            sc_match = bool(clean_sc and sc and clean_sc == sc)
                            folder_match = bool(folder_name and folder_name in k)
                            if sc_match or folder_match:
                                clip_entry = v
                                break

            # Merge fields between local_meta and clip_entry
            owner_user = local_meta.get("ownerUsername") or clip_entry.get("ownerUsername") or local_meta.get("uploader") or ""
            if owner_user == "manual":
                owner_user = ""
            caption = local_meta.get("caption") or clip_entry.get("caption") or ""
            hashtags = local_meta.get("hashtags") or clip_entry.get("hashtags") or []
            tagged_users = local_meta.get("taggedUsers") or clip_entry.get("taggedUsers") or []

            # 4. Construct authoritative targeted grounding block
            meta_lines = [
                "### ACTIVE CLIP GROUND TRUTH METADATA (AUTHORITATIVE)",
                f"- Clip ID / Shortcode: {clean_sc}",
            ]
            if owner_user:
                meta_lines.append(f"- Creator / Account Username: @{owner_user}")
            if caption:
                meta_lines.append(f"- Post Caption: {caption}")
            if hashtags:
                tag_str = ", ".join(hashtags) if isinstance(hashtags, list) else str(hashtags)
                meta_lines.append(f"- Hashtags: {tag_str}")
            if tagged_users:
                tu_str = ", ".join(tagged_users) if isinstance(tagged_users, list) else str(tagged_users)
                meta_lines.append(f"- Tagged Accounts: {tu_str}")

            meta_lines.extend([
                "\nCRITICAL GROUNDING DIRECTIVE FOR `main_subject` & ENTITIES:",
                "1. Treat the above caption, hashtags, and username as authoritative ground truth for identifying the celebrity or creator.",
                "2. If the hashtags or caption identify the person, you MUST set `main_subject` to that exact person's name.",
                "3. NEVER substitute, guess, or hallucinate a different celebrity (e.g. guessing someone else based on visual resemblance alone) when the hashtags or caption already identify the subject.",
            ])
            target_clip_meta_block = "\n".join(meta_lines)

            # 5. Extract Full CBM (Creator Behavior Model) with ZERO character/token truncation
            cbm_data = clip_entry.get("creator_behavior_model")
            if not cbm_data and pool_data and isinstance(pool_data, dict):
                cbm_data = pool_data.get("creator_behavior_model") or pool_data.get("cbm")
                if not cbm_data:
                    for entry in pool_data.get("files", {}).get("social_media_id", {}).values():
                        if isinstance(entry, dict) and entry.get("creator_behavior_model"):
                            cbm_data = entry["creator_behavior_model"]
                            break

            if cbm_data:
                cbm_block = (
                    f"\n---\n\n### CREATOR BEHAVIOR MODEL (CBM) — FULL VERIFIED PROFILE\n"
                    f"```json\n{json.dumps(cbm_data, indent=2, default=str)}\n```\n"
                    f"DIRECTOR DIRECTIVE: Align visual rhythm, camera pacing, tone, and audience hook tactics "
                    f"to this creator's verified behavioral DNA.\n"
                )

        if target_clip_meta_block:
            prompt_text += f"\n---\n\n{target_clip_meta_block}\n"
        if cbm_block:
            prompt_text += f"{cbm_block}\n"

        payload = [prompt_text]
        uploaded_file = None
        pil_images = []

        try:
            # Attach uploaded proxy video if available
            if video_proxy_path and os.path.isfile(video_proxy_path):
                try:
                    if hasattr(self.router, "upload_file"):
                        uploaded_file = self.router.upload_file(video_proxy_path)
                        if uploaded_file:
                            payload.append(uploaded_file)
                            logger.info(f"🔬 ForensicAnalyzer: Attached multimodal proxy video to Gemini payload -> {os.path.basename(video_proxy_path)}")
                except Exception as ve:
                    logger.warning(f"🔬 ForensicAnalyzer: Video proxy upload notice ({ve}); proceeding with keyframes only")

            try:
                from PIL import Image
                for p in frame_paths:
                    try:
                        img = Image.open(p)
                        pil_images.append(img)
                    except Exception as ie:
                        logger.debug(f"🔬 Could not open frame {p}: {ie}")
                payload.extend(pil_images)
            except ImportError:
                logger.warning("🔬 PIL not available — sending text-only prompt")

            attempt_err = None
            try:
                res_txt = self.router.generate(
                    task_type="vision",
                    prompt=payload,
                    module_name="forensic_analyzer",
                    gen_config={"temperature": 0.2, "response_mime_type": "application/json"}
                )
                if res_txt:
                    parsed = self._parse_response(res_txt)
                    if parsed:
                        return parsed
            except Exception as e:
                attempt_err = e
                logger.warning(f"🔬 Multimodal video perception error: {e}")

            # Check if error is quota / rate limit — DO NOT retry on quota exhaustion
            is_quota_error = False
            if attempt_err:
                err_msg = str(attempt_err).lower()
                status_code = getattr(attempt_err, "code", None) or getattr(attempt_err, "status_code", None)
                if status_code == 429 or re.search(r'\b429\b', err_msg) or any(term in err_msg for term in ["quota", "resource_exhausted", "rate_limit", "rate limit", "credits"]):
                    logger.warning("🔬 API Quota/Rate-limit encountered — skipping keyframe retry to preserve quota.")
                    is_quota_error = True

            # Attempt 2: If proxy video was attached AND router raised an exception that is NOT quota,
            # retry ONCE with keyframes only (omitting video proxy attachment)
            if uploaded_file and pil_images and attempt_err is not None and not is_quota_error:
                logger.info("🔬 Retrying perception with keyframes only (omitting video proxy attachment)...")
                try:
                    retry_payload = [prompt_text] + pil_images
                    res_txt = self.router.generate(
                        task_type="vision",
                        prompt=retry_payload,
                        module_name="forensic_analyzer",
                        gen_config={"temperature": 0.2, "response_mime_type": "application/json"}
                    )
                    if res_txt:
                        parsed = self._parse_response(res_txt)
                        if parsed:
                            return parsed
                except Exception as retry_err:
                    logger.error(f"🔬 Keyframe fallback attempt also failed: {retry_err}")

            return copy.deepcopy(DEFAULT_RESULT)

        finally:
            if uploaded_file:
                self._cleanup_uploaded_file(uploaded_file)
    def _parse_response(self, raw: str) -> Optional[dict]:
        """
        Parse and validate Gemini JSON response.
        Handles BOTH schemas:
          - New (flat): intent/feature_flags/safety.classification at root level
          - Old (nested): content_strategy.intent / content_strategy.safety string
        Always returns both formats so orchestrator.py backward-compat is maintained.
        """
        try:
            match = re.search(r'(\{.*\})', raw, re.DOTALL)
            if not match:
                logger.warning("🔬 Forensic parse: no JSON object found in response")
                return None

            data = json.loads(match.group(1))

            # ── Watermarks ────────────────────────────────────────────────────
            # Dedicated detection is handled by gemini_enhance_for_watermark.py
            watermarks = []

            # ── Detect which schema the model returned ────────────────────────
            # New schema: feature_flags at root level
            # Old schema: feature_flags inside content_strategy
            if "feature_flags" in data:
                # NEW flat schema (director prompt)
                flags_raw = data.get("feature_flags", {})
                intent    = str(data.get("intent", "unknown"))
                confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
                editing_style = str(data.get("editing_style", "cinematic"))
                platform_priority = data.get("platform_priority",
                                    ["youtube_shorts", "instagram_reels", "facebook_reels"])

                # safety is now a nested object
                safety_obj = data.get("safety", {})
                if isinstance(safety_obj, dict):
                    safety_cls   = str(safety_obj.get("classification", "risky")).lower()
                    mon_safe     = bool(safety_obj.get("monetization_safe", safety_cls == "safe"))
                else:
                    # Model returned a string instead of object — tolerate it
                    safety_cls   = str(safety_obj).lower().strip()
                    mon_safe     = safety_cls == "safe"

            else:
                # OLD nested content_strategy schema (backward compat)
                cs = data.get("content_strategy", {})
                if not isinstance(cs, dict):
                    cs = {}
                flags_raw      = cs.get("feature_flags", {})
                intent         = str(cs.get("intent", "unknown"))
                confidence     = max(0.0, min(1.0, float(cs.get("confidence", 0.5))))
                editing_style  = str(cs.get("recommended_editing_style", "cinematic"))
                platform_priority = ["youtube_shorts", "instagram_reels", "facebook_reels"]
                safety_cls     = str(cs.get("safety", "risky")).lower().strip()
                mon_safe       = safety_cls == "safe"

            if not isinstance(flags_raw, dict):
                flags_raw = {}

            safety_cls = safety_cls if safety_cls in ("safe", "risky", "blocked") else "risky"

            feature_flags = {
                "enable_price_tags":      bool(flags_raw.get("enable_price_tags",      False)),
                "enable_fashion_caption": bool(flags_raw.get("enable_fashion_caption", False)),
                "enable_cinematic_zoom":  bool(flags_raw.get("enable_cinematic_zoom",  True)),
                "enable_speed_ramps":     bool(flags_raw.get("enable_speed_ramps",     False)),
                "enable_fast_pacing":     bool(flags_raw.get("enable_fast_pacing",     False)),
                "enable_voiceover":       bool(flags_raw.get("enable_voiceover",       True)),
            }

            # Extract BGM selection fields if present
            selected_audio_track = str(data.get("selected_audio_track", ""))
            creative_possibilities = data.get("creative_possibilities", [])
            if not isinstance(creative_possibilities, list):
                creative_possibilities = []

            result = {
                # ── New flat schema ────────────────────────────────────────────
                "watermarks":             watermarks,
                "intent":                 intent,
                "confidence":             round(confidence, 3),
                "editing_style":          editing_style,
                "feature_flags":          feature_flags,
                "platform_priority":      platform_priority,
                "selected_audio_track":   selected_audio_track,
                "creative_possibilities": creative_possibilities,
                "safety": {
                    "classification":    safety_cls,
                    "monetization_safe": mon_safe,
                },
                # ── Backward-compat content_strategy wrapper ───────────────────
                # orchestrator.py reads .get("content_strategy", {}) for flags/safety/intent
                "content_strategy": {
                    "intent":                    intent,
                    "confidence":                round(confidence, 3),
                    "recommended_editing_style": editing_style,
                    "feature_flags":             feature_flags,
                    "safety":                    safety_cls,   # str — matches old code
                },
            }

            # ── Content Director block (new in this version) ───────────────────
            # Extract and validate. If missing, embed empty defaults (non-breaking).
            try:
                cd_raw = data.get("content_director", {})
                if isinstance(cd_raw, dict) and cd_raw:
                    def _str(k): return str(cd_raw.get(k, ""))
                    def _lst(k):
                        v = cd_raw.get(k, [])
                        return [str(x) for x in v] if isinstance(v, list) else []

                    allowed_flags = {
                        "enable_fast_pacing", "enable_cinematic_zoom",
                        "enable_speed_ramps", "enable_voiceover",
                        "enable_price_tags",  "enable_fashion_caption",
                    }
                    raw_cmds = cd_raw.get("feature_commands", {})
                    feature_commands = {
                        k: bool(v)
                        for k, v in (raw_cmds.items() if isinstance(raw_cmds, dict) else [])
                        if k in allowed_flags
                    }
                    for f in allowed_flags:
                        feature_commands.setdefault(f, False)

                    main_subj = _str("main_subject")
                    is_talking = bool(cd_raw.get("is_talking_on_camera", False))

                    result["content_director"] = {
                        "main_subject":          main_subj,
                        "detected_entities":     _lst("detected_entities"),
                        "visual_event":          _str("visual_event"),
                        "viewer_attention":      _str("viewer_attention"),
                        "internet_context":      _lst("internet_context"),
                        "possible_narratives":   _lst("possible_narratives"),
                        "recommended_narrative": _str("recommended_narrative"),
                        "tone":                  _str("tone"),
                        "editing_style":         _str("editing_style"),
                        "is_talking_on_camera":  is_talking,
                        "engagement_hook":       _str("engagement_hook"),
                        "feature_commands":      feature_commands,
                    }
                    result["main_subject"] = main_subj
                    result["is_talking_on_camera"] = is_talking
                    result["_failed"] = False
                    logger.info(
                        f"🎬 ContentDirector: subject='{main_subj}' "
                        f"narrative={result['content_director']['recommended_narrative']} "
                        f"style={result['content_director']['editing_style']} "
                        f"tone={result['content_director']['tone']} "
                        f"hook='{result['content_director']['engagement_hook'][:60]}'"
                    )
                else:
                    logger.info("🎬 ContentDirector: no block in Gemini response — using defaults")
                    result["content_director"] = {}
                    result["main_subject"] = ""
                    result["is_talking_on_camera"] = False
                    result["_failed"] = False
            except Exception as _cde:
                logger.warning(f"🎬 ContentDirector parse error (non-critical): {_cde}")
                result["content_director"] = {}
                result["main_subject"] = ""
                result["is_talking_on_camera"] = False
                result["_failed"] = False

            logger.info(
                f"🔬 Forensic result: intent={intent} "
                f"style={editing_style} safety={safety_cls} "
                f"watermarks={len(watermarks)} confidence={confidence:.2f} "
                f"monetizable={mon_safe}"
            )
            active_flags = [k.replace("enable_", "") for k, v in feature_flags.items() if v]
            logger.info(
                f"🔬 ForensicFlags: intent={intent}, style={editing_style}, "
                f"flags=[{', '.join(active_flags) or 'none'}]"
            )
            return result

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"🔬 Forensic parse error: {e}")
            return None


# ── Module-level singleton ────────────────────────────────────────────────────

_analyzer: Optional[ForensicVideoAnalyzer] = None


def get_analyzer() -> ForensicVideoAnalyzer:
    """Return the module-level singleton, creating it on first call."""
    global _analyzer
    if _analyzer is None:
        _analyzer = ForensicVideoAnalyzer()
    return _analyzer


def analyze_video(video_path: str,
                  frame_paths: Optional[List[str]] = None,
                  creator_name: Optional[str] = None,
                  audio_candidates: Optional[List[dict]] = None,
                  intelligence_cache=None) -> dict:
    """
    Main Orchestrator for Forensic Analysis.
    Auto-extracts strategic frames if frame_paths is None.

    Args:
        video_path:         Path to source video file.
        frame_paths:        Optional pre-extracted frame image paths.
        creator_name:       Optional creator handle/title hint for face cache RAG.
        audio_candidates:   Optional list of candidate BGM audio metadata dicts.
        intelligence_cache: Deprecated: unused legacy parameter preserved for caller backward-compatibility.
    """
    return get_analyzer().analyze(
        video_path,
        frame_paths=frame_paths,
        creator_name=creator_name,
        audio_candidates=audio_candidates
    )

# Alias for legacy support
analyze = analyze_video
