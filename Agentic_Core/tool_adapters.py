"""
Agentic_Core / tool_adapters.py
===============================
Standardized Gemini Tool Adapters for Gemini Editor.
Wraps existing Phase 1, Phase 2, Phase 3, and Auditor modules into
clean, exception-safe functions with Google GenAI function-calling schemas.
"""

import os
import sys
import logging
import traceback
from typing import Dict, Any, List, Optional

logger = logging.getLogger("AgenticCore.ToolAdapters")

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Google GenAI imports
try:
    from google import genai
    from google.genai import types
    _HAS_GENAI_TYPES = True
except ImportError:
    _HAS_GENAI_TYPES = False
    logger.warning("⚠️ google.genai not found. Tool schema definitions may be unavailable.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. TOOL: Ingest Source (Phase 1 Wrapper)
# ─────────────────────────────────────────────────────────────────────────────
def tool_ingest_source(
    source: str,
    limit: int = 1,
    platform: str = "instagram"
) -> Dict[str, Any]:
    """
    Ingest video sources: accepts direct reel/post URL or creator account handle.
    Executes Phase 1: deduplication, video download, proxy encode, audio extraction,
    and beat analysis.
    """
    logger.info(f"📥 [TOOL: INGEST] Ingesting source='{source}' (limit={limit}, platform='{platform}')")
    try:
        from Phase_1.phase1_orchestrator import run_phase1_pipeline

        source_clean = source.strip().strip("'").strip('"')

        # Local upload (file or clip directory): Phase 1 is NOT needed — Phase 2 accepts raw clips directly
        # (same behaviour as the legacy direct upload flow). Never treat a filesystem path as a creator handle.
        _looks_like_path = (os.sep in source_clean) or source_clean.lower().endswith((".mp4", ".mov", ".mkv", ".webm"))
        if _looks_like_path and not source_clean.lower().startswith("http"):
            local_abs = os.path.abspath(source_clean)
            if os.path.isdir(local_abs):
                vids = [os.path.join(local_abs, f) for f in os.listdir(local_abs) if f.lower().endswith((".mp4", ".mov", ".mkv", ".webm"))]
                if not vids:
                    return {"status": "failed", "message": f"No video file found in directory '{local_abs}'.", "source": source_clean}
                local_abs_dir, local_files = local_abs, [vids[0]]
            elif os.path.isfile(local_abs):
                local_abs_dir, local_files = os.path.dirname(local_abs), [local_abs]
            else:
                return {"status": "failed", "message": f"Local path does not exist: '{local_abs}'.", "source": source_clean}
            return {
                "status": "success",
                "message": f"Local upload detected. Skip scraping. Render with clip_dir='{local_abs_dir}'.",
                "source": source_clean,
                "count": 1,
                "clip_dir": local_abs_dir,
                "clip_dirs": [local_abs_dir],
                "downloaded_files": local_files,
            }

        is_url = source_clean.startswith("http://") or source_clean.startswith("https://")

        if is_url:
            res = run_phase1_pipeline(
                mode="manual",
                url=source_clean,
                platform=platform
            )
        else:
            handle = source_clean.lstrip("@")
            res = run_phase1_pipeline(
                mode="auto",
                target_accounts=[handle],
                limit_per_account=limit,
                platform=platform
            )

        if not res.get("success", False) and not res.get("downloaded_files"):
            return {
                "status": "failed",
                "message": res.get("error") or "Failed to download or ingest clip.",
                "source": source_clean,
                "downloaded_files": []
            }

        files = res.get("downloaded_files", [])
        clip_dirs = []
        for f in files:
            d = os.path.dirname(f)
            if d and d not in clip_dirs:
                clip_dirs.append(os.path.abspath(d))

        if res.get("clip_dir") and os.path.abspath(res["clip_dir"]) not in clip_dirs:
            clip_dirs.append(os.path.abspath(res["clip_dir"]))

        primary_dir = clip_dirs[0] if clip_dirs else ""

        return {
            "status": "success",
            "message": (
                f"Successfully ingested {len(files)} clip(s). "
                + (f"Render EACH of these clip_dirs one at a time: {clip_dirs}." if len(clip_dirs) > 1
                   else f"For rendering, use clip_dir='{primary_dir}'.")
            ),
            "source": source_clean,
            "count": len(files),
            "clip_dir": primary_dir,
            "clip_dirs": clip_dirs,
            "downloaded_files": [os.path.abspath(f) for f in files]
        }

    except Exception as e:
        logger.error(f"❌ [TOOL: INGEST] Error: {e}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "message": f"Ingestion exception: {str(e)}",
            "source": source
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOOL: Render Edit (Phase 2 Master Wrapper)
# ─────────────────────────────────────────────────────────────────────────────
def tool_render_edit(
    clip_dir: str,
    edit_directive: str = "",
    skip_existing: bool = False
) -> Dict[str, Any]:
    """
    Executes Phase 2 master editing pipeline on a single clip directory:
    Forensic scene perception, watermark inpainting, BGM selection, rhythm sync,
    and FFmpeg synthesis.
    """
    logger.info(f"🎬 [TOOL: RENDER] Rendering clip_dir='{clip_dir}' | directive='{edit_directive}'")
    try:
        from Phase_2.phase2_orchestrator import run_phase2_pipeline

        raw_dir = clip_dir.strip().strip("'").strip('"') if clip_dir else ""
        clip_dir_abs = os.path.abspath(raw_dir) if raw_dir else ""

        # Smart directory resolution: resolve raw shortcode or relative name against downloads/
        if not clip_dir_abs or not os.path.isdir(clip_dir_abs):
            candidates = [
                os.path.join(_REPO_ROOT, "downloads", raw_dir),
                os.path.join(_REPO_ROOT, "downloads", f"manual_{raw_dir}"),
                os.path.join(_REPO_ROOT, "downloads", f"actress_{raw_dir}"),
            ]
            downloads_base = os.path.join(_REPO_ROOT, "downloads")
            if os.path.exists(downloads_base):
                subdirs = [os.path.join(downloads_base, d) for d in os.listdir(downloads_base) if os.path.isdir(os.path.join(downloads_base, d))]
                # Sort subdirs by modification time (most recent first)
                subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                for sd in subdirs:
                    bname = os.path.basename(sd)
                    if raw_dir and (raw_dir in bname or bname.endswith(f"_{raw_dir}")):
                        candidates.insert(0, sd)
                # If no directory specified, fallback to most recent downloaded directory
                if not raw_dir and subdirs:
                    candidates.append(subdirs[0])

            for cand in candidates:
                if cand and os.path.isdir(cand):
                    clip_dir_abs = os.path.abspath(cand)
                    logger.info(f"📂 [TOOL: RENDER] Resolved clip_dir '{raw_dir}' ➔ '{clip_dir_abs}'")
                    break

        if not clip_dir_abs or not os.path.isdir(clip_dir_abs):
            return {
                "status": "error",
                "message": f"Clip directory does not exist: '{clip_dir_abs or raw_dir}'. Please verify Phase 1 downloaded the clip."
            }

        res = run_phase2_pipeline(
            target_dirs=[clip_dir_abs],
            user_edit_directive=edit_directive if edit_directive else None,
            skip_existing=skip_existing,
            limit=1
        )

        rendered = res.get("rendered_files", [])
        if not rendered:
            return {
                "status": "failed",
                "message": res.get("error") or "Phase 2 render produced no output files.",
                "clip_dir": clip_dir_abs
            }

        master_video = os.path.abspath(rendered[0])
        clip_id = os.path.basename(clip_dir_abs)

        # Retrieve intelligence summary from store
        intel_summary = {}
        try:
            from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore
            store = ClipIntelligenceStore()
            intel = store.load(clip_id, clip_dir_abs)
            if intel:
                vc = intel.get("visual_context", {})
                intel_summary = {
                    "intent": vc.get("intent", "general"),
                    "dominant_emotion": intel.get("audio_data", {}).get("context", {}).get("dominant_emotion", "energetic"),
                    "detected_entities": vc.get("detected_entities", [])[:5]
                }
        except Exception:
            pass

        return {
            "status": "success",
            "message": f"Master reel rendered successfully: {os.path.basename(master_video)}",
            "rendered_video_path": master_video,
            "clip_dir": clip_dir_abs,
            "clip_id": clip_id,
            "intelligence_summary": intel_summary
        }

    except Exception as e:
        logger.error(f"❌ [TOOL: RENDER] Error: {e}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "message": f"Render pipeline exception: {str(e)}",
            "clip_dir": clip_dir
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. TOOL: Audit Clip (Gemini Clip Auditor Wrapper)
# ─────────────────────────────────────────────────────────────────────────────
def tool_audit_clip(
    video_path: str,
    clip_dir: str = "",
    niche: str = "fashion_lifestyle"
) -> Dict[str, Any]:
    """
    Audits a rendered video: validates duration (>=5s), watermark removal quality,
    engagement retention, and generates viral SEO metadata.
    """
    logger.info(f"🔍 [TOOL: AUDIT] Auditing video='{video_path}' (clip_dir='{clip_dir}')")
    try:
        from Gemini_Modules.gemini_clip_auditor import run_clip_audit_and_seo
        from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore

        video_path_abs = os.path.abspath(video_path.strip().strip("'").strip('"'))
        if not os.path.isfile(video_path_abs):
            return {
                "status": "error",
                "message": f"Video file not found: {video_path_abs}"
            }

        clip_id = ""
        clip_folder = clip_dir.strip().strip("'").strip('"') if clip_dir else ""
        if clip_folder and os.path.isdir(clip_folder):
            clip_id = os.path.basename(clip_folder)
        else:
            clip_id = os.path.splitext(os.path.basename(video_path_abs))[0].replace("_master", "")

        store = ClipIntelligenceStore()
        intel = store.load(clip_id, clip_folder or None) or {}

        audit_res = run_clip_audit_and_seo(
            video_path=video_path_abs,
            niche=niche,
            cache=intel,
            metadata=intel.get("metadata")
        )

        passed = audit_res.get("audit_passed", False)
        rejection = audit_res.get("rejection_reason", "")
        duration = audit_res.get("duration", 0.0)
        seo_meta = audit_res.get("seo_metadata", {})

        return {
            "status": "success",
            "audit_passed": passed,
            "rejection_reason": rejection if not passed else None,
            "duration_sec": duration,
            "video_path": video_path_abs,
            "seo_title": seo_meta.get("viral_seo_title") or f"Trending Reel ({niche})",
            "recommendation": "PROCEED_TO_PUBLISH" if passed else "RETRY_RENDER_WITH_DIRECTIVE"
        }

    except Exception as e:
        logger.error(f"❌ [TOOL: AUDIT] Error: {e}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "message": f"Auditor exception: {str(e)}",
            "video_path": video_path
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOOL: Publish Clip (Phase 3 Master Wrapper)
# ─────────────────────────────────────────────────────────────────────────────
def tool_publish_clip(
    video_path: str,
    clip_dir: str = "",
    platforms: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Executes Phase 3 multi-platform distribution: monetization gate, viral title/caption
    synthesis, publishing to YouTube Shorts, Meta (Instagram/Facebook), and TikTok,
    and commits final results to RAG memory.
    """
    logger.info(f"🚀 [TOOL: PUBLISH] Publishing video='{video_path}'")
    try:
        from Phase_3.phase3_orchestrator import run_phase3_orchestration
        from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore

        video_path_abs = os.path.abspath(video_path.strip().strip("'").strip('"'))
        if not os.path.isfile(video_path_abs):
            return {
                "status": "error",
                "message": f"Video file not found: {video_path_abs}"
            }

        clip_id = ""
        clip_folder = clip_dir.strip().strip("'").strip('"') if clip_dir else ""
        if clip_folder and os.path.isdir(clip_folder):
            clip_id = os.path.basename(clip_folder)
        else:
            clip_id = os.path.splitext(os.path.basename(video_path_abs))[0].replace("_master", "")

        store = ClipIntelligenceStore()
        intel = store.load(clip_id, clip_folder or None) or {}

        pub_res = run_phase3_orchestration(video_path_abs, intelligence=intel, platforms=platforms)

        return {
            "status": pub_res.get("status", "success"),
            "clip_id": pub_res.get("clip_id", clip_id),
            "video_path": video_path_abs,
            "publish_results": pub_res.get("publish_results", {}),
            "execution_time_sec": pub_res.get("execution_time_sec", 0.0)
        }

    except Exception as e:
        logger.error(f"❌ [TOOL: PUBLISH] Error: {e}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "message": f"Publishing pipeline exception: {str(e)}",
            "video_path": video_path
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. TOOL: Get Active Sessions (Session Discovery)
# ─────────────────────────────────────────────────────────────────────────────
def tool_get_active_sessions(limit: int = 5) -> Dict[str, Any]:
    """
    Returns recent video editing sessions from TelegramSessionManager.
    Useful for conversational re-edit requests (e.g. 're-edit the last video' or 'make it faster')
    to find the active session_id, clip_id, video path, and current retry count.
    """
    logger.info(f"📋 [TOOL: SESSIONS] Fetching recent active sessions (limit={limit})")
    try:
        from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
        sm = TelegramSessionManager()
        sess_list = list(sm.sessions.values())
        sess_list.sort(key=lambda s: s.get("updated_at", 0), reverse=True)
        recent = sess_list[:limit]

        summaries = []
        for s in recent:
            summaries.append({
                "session_id": s.get("session_id"),
                "clip_id": s.get("clip_id"),
                "creator": s.get("creator"),
                "status": s.get("status"),
                "retry_count": s.get("retry_count", 0),
                "video_path": s.get("video_path"),
                "has_video": bool(s.get("video_path") and os.path.exists(s.get("video_path"))),
                "created_at": s.get("created_at")
            })

        return {
            "status": "success",
            "count": len(summaries),
            "sessions": summaries
        }
    except Exception as e:
        logger.error(f"❌ [TOOL: SESSIONS] Error: {e}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "message": f"Failed to retrieve active sessions: {str(e)}"
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. TOOL: Re-Edit Session (Phase 2 Master Re-Edit Wrapper)
# ─────────────────────────────────────────────────────────────────────────────
def tool_reedit_session(
    session_id: str,
    edit_directive: str = ""
) -> Dict[str, Any]:
    """
    Re-edits an existing video session based on user feedback or creative directives.
    Retrieves source clips from session history or Telegram Storage Vault,
    guards against double-mixing, blacklists rejected audio tracks, and runs
    Phase 2 rhythm synthesis with the user's directive.
    """
    import time
    logger.info(f"🔄 [TOOL: RE-EDIT] Re-editing session '{session_id}' with directive: '{edit_directive}'")
    try:
        from Telegram_Storage_Modules.telegram_session_manager import TelegramSessionManager
        from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
        from Phase_2.phase2_orchestrator import run_phase2_pipeline

        sm = TelegramSessionManager()
        sess = sm.get_session(session_id)
        if not sess:
            return {
                "status": "error",
                "message": f"Session '{session_id}' not found in TelegramSessionManager."
            }

        clip_id = sess.get("clip_id") or ""
        raw_input = sess.get("raw_video_path")
        if not raw_input and clip_id and clip_id != "Processed Shorts":
            cand = os.path.join(_REPO_ROOT, "downloads", clip_id, "video.mp4")
            if os.path.exists(cand):
                raw_input = cand

        video_path = sess.get("video_path")
        target_input = raw_input
        if not target_input and video_path and "_master.mp4" not in video_path and os.path.exists(video_path):
            target_input = video_path

        # Double-mix prevention: never re-edit _master.mp4
        if target_input and "_master.mp4" in target_input:
            bname = os.path.basename(target_input).replace("_master.mp4", "")
            d_dir = os.path.join(_REPO_ROOT, "downloads")
            if os.path.isdir(d_dir):
                for d in os.listdir(d_dir):
                    if bname in d or d in bname:
                        cand = os.path.join(d_dir, d, "video.mp4")
                        if os.path.exists(cand):
                            target_input = cand
                            break

        # Telegram Vault recovery if source is missing on disk
        if not target_input or not os.path.exists(target_input):
            try:
                vi = TelegramVaultIndexer()
                clean_sc = clip_id.replace("manual_", "").strip() if clip_id else ""
                v_entry = vi.find_entry_by_shortcode(clean_sc) or vi.find_entry_by_shortcode(clip_id)
                if v_entry:
                    recovery_dir = os.path.join(_REPO_ROOT, "downloads", clip_id or f"recovered_{session_id}")
                    os.makedirs(recovery_dir, exist_ok=True)
                    hydrated = vi.hydrate_raw_video_from_vault(clean_sc or clip_id, dest_dir=recovery_dir)
                    if hydrated and os.path.exists(hydrated):
                        target_input = hydrated
            except Exception as _ve:
                logger.warning(f"Vault recovery warning in tool_reedit_session: {_ve}")

        if not target_input or not os.path.exists(target_input):
            return {
                "status": "failed",
                "message": f"Source video could not be recovered on disk or vault for session '{session_id}'."
            }

        # Check audio directive & blacklist rejected audio
        is_music = any(kw in edit_directive.lower() for kw in ["music", "bgm", "song", "track", "rhythm", "soundtrack", "audio", "beat"])
        curr_audio = sess.get("selected_audio")
        if is_music and curr_audio:
            try:
                from Audio_Modules.rejected_audio_blacklist import add as blacklist_add, clean_track_identifier
                audio_fn = curr_audio
                if isinstance(curr_audio, dict):
                    audio_fn = curr_audio.get("selected_audio_track") or curr_audio.get("filename") or ""
                audio_fn_str = str(audio_fn).strip()
                audio_sc = clean_track_identifier(audio_fn_str)
                blacklist_add(
                    audio_filename=audio_fn_str,
                    audio_shortcode=audio_sc or None,
                    reason="user_requested_music_change"
                )
                logger.info(f"🚫 [TOOL: RE-EDIT] Added '{audio_fn_str}' (sc: '{audio_sc}') to rejected_audio_blacklist.")
            except Exception as _bl_err:
                logger.debug(f"[TOOL: RE-EDIT] Blacklist add notice: {_bl_err}")

        clip_dir = os.path.dirname(target_input)
        if not clip_dir or not os.path.isdir(clip_dir):
            clip_dir = os.path.join(_REPO_ROOT, "downloads", clip_id)

        # Register retry attempt
        req_chat = sess.get("requestor_chat_id")
        should_retry, count = sm.register_retry(req_chat or session_id)

        # Run Phase 2 master editing pipeline
        p2_res = run_phase2_pipeline(
            target_dirs=[clip_dir],
            user_edit_directive=edit_directive if edit_directive else None,
            skip_existing=False,
            limit=1
        )

        rendered = p2_res.get("rendered_files", [])
        if not rendered:
            return {
                "status": "failed",
                "message": p2_res.get("error") or "Phase 2 render produced no output during re-edit.",
                "session_id": session_id
            }

        new_master = os.path.abspath(rendered[0])
        # Update session attempt history and active video path
        sm.record_rendered_attempt(session_id, new_master)
        sess["video_path"] = new_master
        sess["updated_at"] = time.time()
        sm._save_sessions()

        return {
            "status": "success",
            "message": f"Successfully re-rendered session '{session_id}' with directive '{edit_directive}' (Attempt #{count})",
            "session_id": session_id,
            "clip_id": clip_id,
            "retry_count": count,
            "rendered_video_path": new_master,
            "new_master_video": new_master,
            "requestor_chat_id": req_chat
        }

    except Exception as e:
        logger.error(f"❌ [TOOL: RE-EDIT] Error: {e}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "message": f"Re-edit exception: {str(e)}",
            "session_id": session_id
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tool Dispatch Registry
# ─────────────────────────────────────────────────────────────────────────────
TOOL_DISPATCH_MAP = {
    "tool_ingest_source": tool_ingest_source,
    "tool_render_edit": tool_render_edit,
    "tool_audit_clip": tool_audit_clip,
    "tool_publish_clip": tool_publish_clip,
    "tool_get_active_sessions": tool_get_active_sessions,
    "tool_reedit_session": tool_reedit_session,
}


def get_agent_tools() -> List[Any]:
    """
    Returns official google.genai types.Tool definitions for all macro tools.
    """
    if not _HAS_GENAI_TYPES:
        raise RuntimeError("google.genai SDK is required to build types.Tool schemas.")

    declarations = [
        types.FunctionDeclaration(
            name="tool_ingest_source",
            description="Ingest video sources: accepts a direct Instagram/YouTube URL or creator account handle (e.g. '@cristiano'). Executes Phase 1 deduplication, downloading, proxy encoding, and rhythm beat extraction. Returns clip_dirs and file paths.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "source": types.Schema(
                        type=types.Type.STRING,
                        description="Target source URL (e.g. 'https://instagram.com/reel/...') or account handle (e.g. '@nike')."
                    ),
                    "limit": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum number of reels/clips to scrape per account (default 1)."
                    ),
                    "platform": types.Schema(
                        type=types.Type.STRING,
                        description="Target social platform: 'instagram' or 'youtube' (default 'instagram')."
                    )
                },
                required=["source"]
            )
        ),
        types.FunctionDeclaration(
            name="tool_render_edit",
            description="Executes Phase 2 master editing on a specific clip directory: runs scene perception, watermark inpainting, BGM selection, and FFmpeg rhythm synthesis. Returns rendered_video_path.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "clip_dir": types.Schema(
                        type=types.Type.STRING,
                        description="Absolute path to target clip directory from tool_ingest_source."
                    ),
                    "edit_directive": types.Schema(
                        type=types.Type.STRING,
                        description="Optional creative directive (e.g., 'focus on high-energy drop', 'cinematic color grading')."
                    ),
                    "skip_existing": types.Schema(
                        type=types.Type.BOOLEAN,
                        description="Whether to skip if master edit already exists (default False for fresh renders)."
                    )
                },
                required=["clip_dir"]
            )
        ),
        types.FunctionDeclaration(
            name="tool_audit_clip",
            description="Audits a rendered video file: verifies duration (>=5s), watermark removal quality, engagement retention score, and viral SEO metadata. Returns audit_passed (bool) and recommendation.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "video_path": types.Schema(
                        type=types.Type.STRING,
                        description="Absolute path to rendered master .mp4 video."
                    ),
                    "clip_dir": types.Schema(
                        type=types.Type.STRING,
                        description="Optional clip directory path for context hydration."
                    ),
                    "niche": types.Schema(
                        type=types.Type.STRING,
                        description="Target content niche (default 'fashion_lifestyle')."
                    )
                },
                required=["video_path"]
            )
        ),
        types.FunctionDeclaration(
            name="tool_publish_clip",
            description="Executes Phase 3 multi-platform distribution: checks monetization gate, generates viral platform SEO, uploads to YouTube Shorts, Meta, and TikTok, and commits to RAG memory.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "video_path": types.Schema(
                        type=types.Type.STRING,
                        description="Absolute path to rendered video to publish."
                    ),
                    "clip_dir": types.Schema(
                        type=types.Type.STRING,
                        description="Optional clip directory path for intelligence hydration."
                    ),
                    "platforms": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="Target platforms: ['youtube', 'meta', 'tiktok']."
                    )
                },
                required=["video_path"]
            )
        ),
        types.FunctionDeclaration(
            name="tool_get_active_sessions",
            description="Lists recent active video editing sessions. Useful when user wants to revise, adjust, or re-edit the last video without explicitly specifying a session ID.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "limit": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of recent sessions to inspect (default 5)."
                    )
                }
            )
        ),
        types.FunctionDeclaration(
            name="tool_reedit_session",
            description="Re-edits an existing video session based on a creative directive (e.g. 'faster cuts', 'change music', 'hook 2s shorter'). Recovers source from disk or Telegram Vault, avoids double-mix, and renders new attempt.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "session_id": types.Schema(
                        type=types.Type.STRING,
                        description="Unique session ID (e.g. 'sess_177...')."
                    ),
                    "edit_directive": types.Schema(
                        type=types.Type.STRING,
                        description="User's creative directive or feedback for the re-edit."
                    )
                },
                required=["session_id"]
            )
        )
    ]

    return [types.Tool(function_declarations=declarations)]
