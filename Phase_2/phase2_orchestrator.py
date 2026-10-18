"""
Phase_2 / phase2_orchestrator.py
================================
Master Orchestrator for Phase 2 (AI Perception, BGM Selection & Master FFmpeg Rendering).
Coordinates sequential steps 01 -> 07 and supports real-time event callbacks for the live web tracker.
"""

import json
import os
import sys
import time
import logging
import importlib
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Callable, Union, Set

logger = logging.getLogger("Phase2.Orchestrator")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Dynamic import of indexed step modules
step01 = importlib.import_module("Phase_2.01_folder_scanner")
step02 = importlib.import_module("Phase_2.02_forensic_perception")
step03 = importlib.import_module("Phase_2.03_vector_frame_extractor")
step04 = importlib.import_module("Phase_2.04_bgm_selector")
step05 = importlib.import_module("Phase_2.05_rhythm_timeline")
step06 = importlib.import_module("Phase_2.06_ffmpeg_synthesis")
step07 = importlib.import_module("Phase_2.07_master_render")

try:
    from Import_Modules.tracker_notifier import notify_tracker
except ImportError:
    notify_tracker = None


def run_phase2_pipeline(
    input_path: Optional[str] = None,
    downloads_dir: Optional[str] = None,
    master_edits_dir: Optional[str] = None,
    limit: Optional[int] = None,
    target_dirs: Optional[List[str]] = None,
    skip_existing: bool = False,
    on_rendered_callback: Optional[Callable[[str], None]] = None,
    event_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    user_edit_directive: Optional[str] = None,
    excluded_audio: Optional[Union[str, List[str], Set[str]]] = None,
) -> Dict[str, Any]:
    """
    Executes Phase 2 Pipeline through indexed steps 01 -> 07.
    """
    if event_callback is None and notify_tracker is not None:
        event_callback = notify_tracker

    def _emit(step_id: str, status: str, payload: Dict[str, Any]):
        if event_callback:
            try:
                event_callback(step_id, status, payload)
            except Exception as _cb_err:
                logger.debug(f"Event callback warning: {_cb_err}")

    if master_edits_dir is None:
        master_edits_dir = os.path.join(_REPO_ROOT, "Processed Shorts")
    os.makedirs(master_edits_dir, exist_ok=True)

    logger.info(f"🚀 [PHASE 2 ORCHESTRATOR] Starting Master AI Editing Pipeline -> Output: {master_edits_dir}")
    _emit("step_01", "running", {"message": "Scanning downloads/ clip targets..."})

    # Step 1: Scan Targets
    targets = step01.scan_clip_targets(
        input_path=input_path,
        downloads_dir=downloads_dir,
        target_dirs=target_dirs,
        limit=limit,
    )
    _emit("step_01", "success", {"message": f"Verified {len(targets)} clip target(s)."})

    rendered_files = []
    skipped_count = 0
    failed_count = 0
    batch_used_bgms = set()

    for idx, target in enumerate(targets, start=1):
        folder_name = target["folder_name"]
        video_path = target["video_path"]
        clip_dir = target["dir"]

        target_output = os.path.join(master_edits_dir, f"{folder_name}_master.mp4")

        if skip_existing and os.path.isfile(target_output) and os.path.getsize(target_output) > 1024:
            logger.info(f"♻️ [SKIP EXPLICITLY REQUESTED] '{folder_name}_master.mp4' already exists.")
            rendered_files.append(os.path.abspath(target_output))
            continue
        elif not skip_existing and os.path.isfile(target_output):
            try:
                os.remove(target_output)
                logger.info(f"🧹 Removed stale master reel before fresh re-edit: {os.path.basename(target_output)}")
            except Exception as _rm_err:
                logger.debug(f"Notice removing pre-existing master reel {target_output}: {_rm_err}")

        logger.info(f"\n🚀 [PHASE 2 CLIP {idx}/{len(targets)}] Ingesting folder: {folder_name}")
        tmp_dir = tempfile.mkdtemp(prefix="phase2_frames_")

        try:
            # ─────────────────────────────────────────────────────────────────
            # STEP 2A: UPFRONT WATERMARK DETECTION & INPAINTING
            # Erase foreign watermarks first so Scene Intel, Forensic Perception,
            # and FFmpeg single-pass synthesis operate on clean video.
            # ─────────────────────────────────────────────────────────────────
            _emit("step_02_watermark", "running", {"message": f"Checking/inpainting watermarks for {folder_name}..."})
            is_auto = folder_name.startswith("auto_")
            folder_prefix = "auto_" if is_auto else "manual_"
            prefix = "auto_clean_" if is_auto else "manual_clean_"
            shortcode_stem = folder_name.replace("manual_", "").replace("auto_", "").strip()
            clean_filename = f"{prefix}{shortcode_stem}.mp4"
            clean_raw_path = os.path.join(clip_dir, clean_filename)
            working_video_path = video_path
            watermark_boxes = []

            # Check if user explicitly requested inpainting in directive (typo-tolerant)
            try:
                from Watermark_and_Inpainting.watermark_directive_helper import is_inpaint_directive, extract_watermark_keywords
                is_inpaint_requested = is_inpaint_directive(user_edit_directive)
            except Exception as _dir_err:
                logger.debug(f"watermark_directive_helper import notice: {_dir_err}")
                is_inpaint_requested = bool(
                    user_edit_directive and any(
                        kw in user_edit_directive.lower()
                        for kw in ["inpaint", "inapint", "watermark", "clean", "logo", "remove", "erase", "filmygyan", "flimygyan"]
                    )
                )

            # Only genuine clean/inpainted outputs belong in candidate_clean
            candidate_clean = [
                clean_raw_path,
                os.path.join(clip_dir, f"manual_clean_{shortcode_stem}.mp4"),
                os.path.join(clip_dir, f"auto_clean_{shortcode_stem}.mp4"),
                os.path.join(clip_dir, "video_inpainted_clean.mp4"),
            ]

            if is_inpaint_requested:
                logger.info(f"🧼 [INPAINT DIRECTIVE OVERRIDE] User requested inpainting: '{user_edit_directive}'. Invalidating clean video cache.")
                for c_cand in candidate_clean:
                    if os.path.exists(c_cand):
                        try:
                            os.remove(c_cand)
                            coords_f = c_cand + ".coords.json"
                            if os.path.exists(coords_f):
                                os.remove(coords_f)
                        except Exception as _rm_err:
                            logger.debug(f"Cache purge notice: {_rm_err}")
                clean_raw_path = os.path.join(clip_dir, clean_filename)

                # Ensure video_path is a true raw source (not a deleted or stale clean video)
                if not os.path.exists(video_path) or "_clean.mp4" in video_path.lower():
                    raw_candidates = [
                        os.path.join(clip_dir, "video.mp4"),
                        os.path.join(clip_dir, f"manual_raw_{shortcode_stem}.mp4"),
                        os.path.join(clip_dir, f"auto_raw_{shortcode_stem}.mp4"),
                        os.path.join(clip_dir, f"manual_{shortcode_stem}.mp4"),
                        os.path.join(clip_dir, f"auto_{shortcode_stem}.mp4"),
                    ]
                    found_raw = None
                    for rc in raw_candidates:
                        if os.path.exists(rc) and os.path.getsize(rc) > 1024 and "_clean" not in rc.lower():
                            found_raw = rc
                            break
                    if not found_raw:
                        try:
                            from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
                            _v_indexer = TelegramVaultIndexer()
                            _raw_hydrated = _v_indexer.hydrate_raw_video_from_vault(folder_name, dest_dir=clip_dir, check_clean_first=False)
                            if _raw_hydrated and os.path.exists(_raw_hydrated) and os.path.getsize(_raw_hydrated) > 1024:
                                found_raw = _raw_hydrated
                        except Exception as _vr_err:
                            logger.warning(f"Vault raw recovery notice during inpaint override: {_vr_err}")
                    if found_raw:
                        video_path = found_raw
                        working_video_path = found_raw
            else:
                for c_cand in candidate_clean:
                    if "_master.mp4" in c_cand.lower():
                        continue
                    if os.path.exists(c_cand) and os.path.getsize(c_cand) > 1024:
                        clean_raw_path = c_cand
                        break

            # 1. Check Telegram Storage Vault for pre-cleaned video (wm_clean_file_id) if not on local disk
            if not is_inpaint_requested and (not os.path.exists(clean_raw_path) or os.path.getsize(clean_raw_path) <= 1024):
                try:
                    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
                    _v_indexer = TelegramVaultIndexer()
                    _clean_hydrated = _v_indexer.hydrate_clean_video_from_vault(folder_name, clip_dir)
                    if _clean_hydrated and os.path.exists(_clean_hydrated) and os.path.getsize(_clean_hydrated) > 1024:
                        clean_raw_path = _clean_hydrated
                        logger.info(f"⚡ [VAULT INPAINTING CACHE HIT] Hydrated pre-cleaned video from Telegram Vault: {os.path.basename(clean_raw_path)} — skipping OpenCV inpainting!")
                except Exception as _vi_err:
                    logger.debug(f"[UPFRONT INPAINTING] Vault clean check notice: {_vi_err}")

            # 2. If clean video already exists, reuse it and load sidecar coordinates
            if not is_inpaint_requested and os.path.exists(clean_raw_path) and os.path.getsize(clean_raw_path) > 1024 and os.path.abspath(clean_raw_path) != os.path.abspath(target_output):
                working_video_path = clean_raw_path
                logger.info(f"⚡ [UPFRONT INPAINTING CACHE] Reusing clean inpainted raw video: {os.path.basename(working_video_path)}")
                coords_sidecar = clean_raw_path + ".coords.json"
                if os.path.exists(coords_sidecar):
                    try:
                        with open(coords_sidecar, "r", encoding="utf-8") as _csf:
                            _cdata = json.load(_csf)
                        watermark_boxes = _cdata.get("watermark_boxes") or []
                        if watermark_boxes:
                            logger.info(f"📍 [WATERMARK ALIGNMENT] Loaded {len(watermark_boxes)} inpaint coordinate box(es) from sidecar.")
                    except Exception as _ce:
                        logger.warning(f"⚠️ [WATERMARK ALIGNMENT] Failed to load coords sidecar: {_ce}")
            elif is_inpaint_requested:
                # 3. Detect and Inpaint Watermarks Upfront ONLY when explicitly requested
                try:
                    from Watermark_and_Inpainting.gemini_enhance_for_watermark import detect_watermark_from_video
                    from Watermark_and_Inpainting.watermark_main import run_watermark_removal

                    logger.info(f"🧼 [UPFRONT INPAINTING] User requested inpainting. Running detection on: {os.path.basename(video_path)}")
                    detect_kw = ""
                    try:
                        from Watermark_and_Inpainting.watermark_directive_helper import extract_watermark_keywords
                        detect_kw = extract_watermark_keywords(user_edit_directive)
                    except Exception:
                        detect_kw = user_edit_directive or ""
                    items, _ = detect_watermark_from_video(video_path=video_path, keywords=detect_kw)
                    if items:
                        watermark_boxes = items
                        logger.info(f"💎 [STEP 02 WATERMARK DETECTED] Found {len(items)} watermark bounding box vector(s).")

                    inpainted_path, _ = run_watermark_removal(
                        input_path=video_path,
                        output_path=clean_raw_path,
                        predetected_watermarks=items if items else None,
                        keywords=detect_kw,
                        retry_level=1
                    )
                    if inpainted_path and os.path.exists(inpainted_path) and os.path.getsize(inpainted_path) > 1024:
                        working_video_path = inpainted_path
                        logger.info(f"✅ [UPFRONT INPAINTING SUCCESS] Raw video cleaned: {os.path.basename(working_video_path)}")

                        # Load sidecar coordinates
                        coords_sidecar = clean_raw_path + ".coords.json"
                        if os.path.exists(coords_sidecar):
                            try:
                                with open(coords_sidecar, "r", encoding="utf-8") as _csf:
                                    _cdata = json.load(_csf)
                                _wm_b = _cdata.get("watermark_boxes")
                                if _wm_b:
                                    watermark_boxes = _wm_b
                                    logger.info(f"📍 [WATERMARK ALIGNMENT] Loaded {len(watermark_boxes)} inpaint coordinate box(es) for brand overlay masking.")
                            except Exception as _ce:
                                logger.debug(f"Coords sidecar read notice: {_ce}")

                        # Upload clean inpainted source to Telegram Vault ONLY if watermarks were actually detected and cleaned
                        if watermark_boxes:
                            try:
                                from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
                                vault_idx = TelegramVaultIndexer()
                                clean_fid = vault_idx.update_inpainted_clean_source_in_vault(
                                    clean_video_path=clean_raw_path,
                                    clip_folder_name=folder_name
                                )
                                if clean_fid:
                                    logger.info(f"☁️ [CLEAN VAULT UPLOAD] Uploaded clean video to vault (file_id: {clean_fid[:15]}...)")
                            except Exception as _clean_up_err:
                                logger.warning(f"⚠️ [UPFRONT INPAINTING] Clean vault upload notice: {_clean_up_err}")
                except Exception as _inp_err:
                    logger.warning(f"⚠️ [UPFRONT INPAINTING] Inpainting notice: {_inp_err}")
                    working_video_path = video_path
            else:
                logger.info(f"⏭️ [UPFRONT INPAINTING] Skipped — no inpaint directive requested by user (0 Gemini calls).")
                working_video_path = video_path

            # ─────────────────────────────────────────────────────────────────
            # STEP 2B: SCENE INTEL & GEMINI CALL 1 FORENSIC PERCEPTION
            # Analyzes the CLEAN video (working_video_path) to determine visual
            # intent, tone, style, narrative, faces/subjects, and speech mode.
            # ─────────────────────────────────────────────────────────────────
            _emit("step_02", "running", {"message": f"Running Gemini Call 1 Forensic Perception on clean video for {folder_name}..."})

            # Pre-hydrate pool metadata with this clip's data before Gemini runs
            try:
                from Audio_Modules.audio_pool_manager import AudioPoolManager
                AudioPoolManager().hydrate_harvested_clip_metadata()
            except Exception as _ph_err:
                logger.debug(f"[STEP 02] Pre-hydration notice: {_ph_err}")

            creator_hint = folder_name.split("_")[0] if "_" in folder_name else "unknown"
            if creator_hint == "manual":
                _meta_file = os.path.join(clip_dir, "metadata.json")
                if os.path.exists(_meta_file):
                    try:
                        with open(_meta_file, "r", encoding="utf-8") as _mf:
                            _m_data = json.load(_mf)
                            _real_owner = _m_data.get("ownerUsername") or _m_data.get("uploader")
                            if _real_owner and _real_owner != "manual":
                                creator_hint = _real_owner
                    except Exception:
                        pass
                if creator_hint == "manual":
                    clean_sc = folder_name.replace("manual_", "").strip()
                    if clean_sc:
                        creator_hint = clean_sc

            forensic_res = step02.run_forensic_perception(
                video_path=working_video_path,
                creator_name=creator_hint,
            )
            forensic_res["working_video_path"] = working_video_path
            forensic_res["watermark_boxes"] = watermark_boxes
            forensic_res["inpainted_upfront"] = (working_video_path == clean_raw_path)
            _emit("step_02", "success", {"message": f"Perception complete: intent='{forensic_res.get('intent')}'."})


            # Step 3: OpenCV Vector-Guided Frame Extraction
            _emit("step_03", "running", {"message": "Extracting vector-guided keyframes..."})
            visual_vectors = forensic_res.get("visual_vectors", {})
            frame_paths = step03.extract_targeted_frames(
                video_path=working_video_path,
                tmp_dir=tmp_dir,
                visual_vectors=visual_vectors,
            )
            _emit("step_03", "success", {"message": f"Extracted {len(frame_paths)} targeted keyframes."})

            # Step 4: Gemini Call 2 BGM Selector
            _emit("step_04", "running", {"message": f"Selecting optimal BGM track for clip '{folder_name}'..."})

            exclude_bgm = set(batch_used_bgms)
            force_new_music = False

            # Ingest explicitly passed excluded_audio
            if excluded_audio:
                force_new_music = True
                if isinstance(excluded_audio, (list, set, tuple)):
                    for ea in excluded_audio:
                        if ea:
                            exclude_bgm.add(str(ea))
                            exclude_bgm.add(os.path.basename(str(ea)))
                elif isinstance(excluded_audio, str) and excluded_audio.strip():
                    exclude_bgm.add(excluded_audio.strip())
                    exclude_bgm.add(os.path.basename(excluded_audio.strip()))
                logger.info(f"🚫 [STEP 04] Applied explicit excluded_audio: {exclude_bgm}")

            if user_edit_directive:
                d_lower = user_edit_directive.lower()
                if any(kw in d_lower for kw in ["music", "bgm", "song", "track", "rhythm", "soundtrack", "audio", "beat"]):
                    force_new_music = True
                    try:
                        from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore
                        _st = ClipIntelligenceStore(clip_id=folder_name, clip_folder=clip_dir)
                        _prev_aud = _st.get("audio_data") or {}
                        _prev_track = _prev_aud.get("selected_bgm_track") or _prev_aud.get("selected_audio_track")
                        if _prev_track:
                            exclude_bgm.add(_prev_track)
                            exclude_bgm.add(os.path.basename(_prev_track))
                            logger.info(f"🚫 [STEP 04 RE-EDIT] User requested music change! Excluding previous BGM: '{_prev_track}'")
                    except Exception as _ex_err:
                        logger.debug(f"Exclusion lookup notice: {_ex_err}")

            bgm_res = step04.select_clip_bgm(
                clip_id=folder_name,
                clip_folder=clip_dir,
                intent_vector={"preserve_music": not force_new_music},
                exclude_filenames=exclude_bgm if exclude_bgm else None,
            )
            selected_bgm_path = bgm_res.get("physical_path")
            selected_track_name = bgm_res.get("selected_audio_track")
            if selected_track_name:
                batch_used_bgms.add(selected_track_name)
                batch_used_bgms.add(os.path.basename(selected_track_name))
            if selected_bgm_path:
                batch_used_bgms.add(os.path.basename(selected_bgm_path))
            _emit("step_04", "success", {"message": f"BGM selected: '{selected_track_name}'."})

            # Step 5: Rhythm & Micro-Shot Timeline Builder
            _emit("step_05", "running", {"message": "Building rhythm & micro-shot timeline..."})
            forensic_res["route_params"] = forensic_res.get("route_params", {})
            timeline_res = step05.build_rhythm_timeline(
                video_path=video_path,
                selected_bgm_path=selected_bgm_path,
                forensic_context=forensic_res,
            )
            micro_shots = timeline_res.get("micro_shots", [])
            forensic_res["lyric_intel"] = timeline_res.get("lyric_intel", {})
            if timeline_res.get("route_params"):
                forensic_res["route_params"] = timeline_res["route_params"]
            if timeline_res.get("selected_bgm_path"):
                selected_bgm_path = timeline_res["selected_bgm_path"]
            _emit("step_05", "success", {"message": f"Built {len(micro_shots)} micro-shot takes (2.0s-3.5s)."})

            # Step 6: Gemini Call 3 FFmpeg Synthesis
            _emit("step_06", "running", {"message": "Synthesizing FFmpeg Master Filtergraph Editing Plan..."})
            synthesis_res = step06.synthesize_editing_plan(
                video_path=working_video_path,
                output_path=target_output,
                selected_bgm_path=selected_bgm_path,
                forensic_context=forensic_res,
                micro_shots=micro_shots,
                target_duration=15.0,
                user_edit_directive=user_edit_directive,
            )
            if synthesis_res is None:
                synthesis_res = {"status": "failed", "error": "synthesize_editing_plan returned None"}
            _emit("step_06", "success", {"message": f"FFmpeg synthesis status: {synthesis_res.get('status')}."})

            # Step 7: Master Render Verification & QA Gate
            _emit("step_07", "running", {"message": f"Verifying master render for {folder_name}..."})
            render_res = step07.verify_master_render(
                output_path=target_output,
                synthesis_result=synthesis_res,
            )
            if render_res is None:
                render_res = {"success": False, "error": "verify_master_render returned None"}

            if render_res.get("success"):
                out_path = render_res["output_video"]
                rendered_files.append(out_path)
                _emit("step_07", "success", {"message": f"Master reel ready: {os.path.basename(out_path)}."})

                if on_rendered_callback:
                    try:
                        on_rendered_callback(out_path)
                    except Exception as cb_err:
                        logger.warning(f"Render callback notice: {cb_err}")
            else:
                failed_count += 1
                _emit("step_07", "failed", {"message": f"Master render verification failed: {render_res.get('error')}."})

        except Exception as clip_err:
            failed_count += 1
            logger.error(f"❌ Exception editing '{folder_name}': {clip_err}", exc_info=True)
            _emit("step_07", "failed", {"message": f"Clip edit exception: {clip_err}"})
        finally:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(
        f"\n🎉 PHASE 2 COMPLETE: {len(rendered_files)} master reel(s) rendered | "
        f"{skipped_count} skipped | {failed_count} failed."
    )
    return {
        "success": True,
        "rendered_count": len(rendered_files),
        "rendered_files": rendered_files,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "master_edits_dir": master_edits_dir,
    }
