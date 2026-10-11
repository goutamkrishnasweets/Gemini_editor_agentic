"""
Phase_2 / 04_bgm_selector.py
============================
Step 4: Gemini Call 2 — BGM Selector.
Cross-matches clip's visual_context + audio_data vs ALL pooled clip audio records from ClipIntelligenceStore.
Selects single best BGM track and saves decision to clip intelligence JSON.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("Phase2.Step04")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Gemini_Modules.lyric_rhythm_aligner import select_best_audio_for_clip
from Audio_Modules.audio_pool_manager import AudioPoolManager


def select_clip_bgm(
    clip_id: str,
    clip_folder: Optional[str] = None,
    audio_dir: Optional[str] = None,
    intent_vector: Optional[Dict[str, Any]] = None,
    exclude_filenames: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Executes Gemini Call 2 BGM Selector.

    Stage 2 Cache Lock:
        If intent_vector.preserve_music == True AND a cached BGM track is found
        in ClipIntelligenceStore, the cached track is returned IMMEDIATELY and
        Gemini Call 2 is skipped entirely. This guarantees 0 unintended music
        changes on re-edits and saves ~2000 tokens per run.

    Returns dict containing selected_audio_track, full physical path, and alignment score.
    """
    _intent = intent_vector or {}
    preserve_music = _intent.get("preserve_music", False)

    # ── STATEFUL CACHE LOCK (Stage 2 Cache-First BGM) ────────────────────────
    try:
        from Audio_Modules.rejected_audio_blacklist import is_blacklisted as _is_bl
    except ImportError:
        def _is_bl(*args, **kwargs): return False

    if preserve_music:
        cached_bgm = _try_load_cached_bgm(clip_id, clip_folder, audio_dir)
        cached_track = cached_bgm.get("selected_audio_track") if cached_bgm else None
        cached_fid = cached_bgm.get("telegram_file_id") if cached_bgm else None

        is_bl = bool(cached_track and _is_bl(cached_track, telegram_file_id=cached_fid))

        is_excluded = False
        if cached_track and exclude_filenames:
            ct_lower = cached_track.lower()
            ct_base = os.path.basename(cached_track).lower()
            for ef in exclude_filenames:
                ef_lower = str(ef).lower()
                if ct_lower == ef_lower or ct_base == os.path.basename(ef_lower):
                    is_excluded = True
                    break

        if cached_bgm and not is_excluded and not is_bl:
            logger.info(
                f"🔒 [STEP 04] BGM CACHE LOCK — Skipping Gemini Call 2. "
                f"Reusing: '{cached_track}' "
                f"(intent: preserve_music=True)"
            )
            return cached_bgm
        elif is_bl:
            logger.warning(
                f"🚫 [STEP 04] Cached BGM '{cached_track}' is in REJECTED_AUDIO_BLACKLIST! "
                "Busting cache lock and running Gemini Call 2 fresh for alternative BGM."
            )
        elif is_excluded:
            logger.info(
                f"🚫 [STEP 04] Cached BGM '{cached_track}' is in exclude_filenames. "
                "Running Gemini Call 2 fresh for alternative BGM."
            )
        else:
            logger.info(
                "[STEP 04] preserve_music=True but no cached BGM found. "
                "Running Gemini Call 2 fresh."
            )

    # ── STANDARD GEMINI CALL 2 ────────────────────────────────────────────────
    logger.info(f"🎶 [STEP 04] Running Gemini Call 2 BGM Selector for clip '{clip_id}'...")

    res = select_best_audio_for_clip(
        clip_id=clip_id,
        clip_folder=clip_folder,
        audio_dir=audio_dir,
        exclude_filenames=exclude_filenames
    )
    selected_track_name = res.get("selected_audio_track")
    backup_candidates = res.get("backup_candidates") or []

    # Resolve physical path
    resolved_path = None
    target_dest = clip_folder if (clip_folder and os.path.exists(clip_folder)) else os.path.join(_REPO_ROOT, "data", "runtime_audio")
    os.makedirs(target_dest, exist_ok=True)

    candidates_to_try = []
    if selected_track_name and not _is_bl(selected_track_name, telegram_file_id=res.get("telegram_file_id")):
        candidates_to_try.append(selected_track_name)
    for c in backup_candidates:
        if c and c not in candidates_to_try and not _is_bl(c):
            candidates_to_try.append(c)

    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
    vault = TelegramVaultIndexer()

    # Pre-map file IDs from lyric_rhythm_aligner
    cand_fids = dict(res.get("candidate_file_ids") or {})
    if selected_track_name and "telegram_file_id" in res and selected_track_name not in cand_fids:
        cand_fids[selected_track_name] = res["telegram_file_id"]

    # Supplement candidate list from vault audio pool if pool is small
    try:
        vault_pool = vault.get_vault_audio_pool(current_clip_id=clip_id) or {}
        for vtrack, vmeta in vault_pool.items():
            vfid = vmeta.get("file_id") or vmeta.get("telegram_file_id") if isinstance(vmeta, dict) else None
            if _is_bl(vtrack, telegram_file_id=vfid):
                continue
            if vtrack not in candidates_to_try and (not exclude_filenames or vtrack not in exclude_filenames):
                candidates_to_try.append(vtrack)
                if vtrack not in cand_fids and vfid:
                    cand_fids[vtrack] = vfid
    except Exception as _vp_err:
        logger.debug(f"[STEP 04] Notice fetching vault audio pool candidates: {_vp_err}")

    failed_candidates = set()
    for cand_name in candidates_to_try:
        cand_fid = cand_fids.get(cand_name)
        if _is_bl(cand_name, telegram_file_id=cand_fid):
            logger.warning(f"🚫 [STEP 04] Skipping blacklisted candidate '{cand_name}'.")
            failed_candidates.add(cand_name)
            continue
        try:
            resolved_path = vault.hydrate_bgm_track_from_vault(
                cand_name,
                target_dest,
                file_id=cand_fid
            )
            if resolved_path and os.path.isfile(resolved_path) and os.path.getsize(resolved_path) > 1024:
                selected_track_name = cand_name
                res["selected_audio_track"] = selected_track_name
                logger.info(f"✓ [STEP 04] BGM track retrieved and verified: '{selected_track_name}' -> {resolved_path}")
                break
            else:
                logger.warning(f"⚠️ [STEP 04] Failed to hydrate BGM candidate '{cand_name}'. Trying next candidate...")
                failed_candidates.add(cand_name)
                resolved_path = None
        except Exception as _vh_err:
            logger.debug(f"[STEP 04] Vault BGM track hydration notice for '{cand_name}': {_vh_err}")
            failed_candidates.add(cand_name)
            resolved_path = None

    # Fallback to pool manager if Gemini's primary and backup candidates failed to hydrate
    if not resolved_path:
        logger.warning("⚠️ [STEP 04] Primary and backup candidates unavailable. Resorting to pool manager emergency fallback.")
        try:
            pool = AudioPoolManager(base_dir=audio_dir)
            all_excludes = set(exclude_filenames or set()) | failed_candidates
            resolved_path = pool.select_best_audio(exclude_filenames=all_excludes)
            if not resolved_path and failed_candidates:
                # If everything was excluded, retry excluding only failed candidates
                resolved_path = pool.select_best_audio(exclude_filenames=failed_candidates)
            if resolved_path and os.path.isfile(resolved_path) and os.path.getsize(resolved_path) > 1024:
                selected_track_name = os.path.basename(resolved_path)
                res["selected_audio_track"] = selected_track_name
                res["alignment_score"] = res.get("alignment_score") or 0.80
                logger.info(f"✓ [STEP 04] Pool manager fallback succeeded: '{selected_track_name}' -> {resolved_path}")
            else:
                resolved_path = None
        except Exception as pool_err:
            logger.warning(f"⚠️ [STEP 04] BGM pool manager fallback notice: {pool_err}")

    # If Telegram Vault hydration failed for all candidates and pool manager has no tracks
    if not resolved_path:
        err_msg = (
            f"❌ [STEP 04 ERROR] Telegram Vault BGM hydration failed for clip '{clip_id}'. "
            f"Candidates attempted from Telegram Vault: {candidates_to_try}. "
            f"Unable to retrieve any valid BGM track from Telegram Storage Group. "
            f"Please verify that audio tracks in your Telegram Storage Group are accessible "
            f"and under 20MB (or configure TELEGRAM_API_ID / TELEGRAM_API_HASH for Pyrogram MTProto)."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    res["physical_path"] = resolved_path

    # Save selected BGM track choice to ClipIntelligenceStore & update usage count
    if resolved_path and selected_track_name:
        try:
            from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore
            store = ClipIntelligenceStore(clip_id=clip_id, clip_folder=clip_folder)
            audio_data = store.get("audio_data") or {}
            audio_data["selected_bgm_track"] = selected_track_name
            audio_data["selected_audio_track"] = selected_track_name
            audio_data["alignment_score"] = res.get("alignment_score", 0.85)
            store.set("audio_data", audio_data)

            pm = AudioPoolManager()
            pm.record_selected_audio(
                social_url_or_shortcode=clip_id or "",
                selected_track_name=selected_track_name,
                telegram_file_id=res.get("telegram_file_id"),
                alignment_score=res.get("alignment_score", 0.85),
                reasoning=res.get("reasoning", "")
            )
            # Only increment usage counter if track physically exists on disk and is not harvested audio
            if not selected_track_name.startswith("extracted_") and not selected_track_name.startswith("telegram_"):
                pm.use_audio(selected_track_name)
        except Exception as _st_err:
            logger.warning(f"[STEP 04] ⚠️ Failed to record BGM selection in ClipIntelligenceStore / AudioPoolManager: {_st_err}")

    logger.info(
        f"✓ [STEP 04 SUCCESS] BGM Selected: '{selected_track_name}' "
        f"(score={res.get('alignment_score', 0.0):.2f}) -> {resolved_path}"
    )
    return res


def _try_load_cached_bgm(
    clip_id: str,
    clip_folder: Optional[str],
    audio_dir: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Attempts to load a previously selected BGM track from ClipIntelligenceStore.
    Returns a fully resolved BGM result dict if a cache hit is found, else None.
    """
    try:
        from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore
        store = ClipIntelligenceStore(clip_id=clip_id, clip_folder=clip_folder)
        audio_data = store.get("audio_data") or {}
        cached_track = audio_data.get("selected_bgm_track") or audio_data.get("selected_audio_track")

        if not cached_track:
            return None

        # Resolve physical path from disk
        if audio_dir is None:
            audio_dir = os.path.join(_REPO_ROOT, "Original_audio")

        resolved_path = None
        for candidate_dir in [
            os.path.join(audio_dir, "active"),
            audio_dir,
            os.path.join(audio_dir, "cooldown"),
        ]:
            if os.path.isdir(candidate_dir):
                cpath = os.path.join(candidate_dir, cached_track)
                if os.path.isfile(cpath):
                    resolved_path = cpath
                    break

        if not resolved_path:
            logger.warning(f"[STEP 04] Cached BGM '{cached_track}' not found on disk. Will re-select.")
            return None

        return {
            "selected_audio_track": cached_track,
            "physical_path": resolved_path,
            "alignment_score": audio_data.get("alignment_score", 1.0),
            "_source": "cache_lock",
        }

    except Exception as e:
        logger.debug(f"[STEP 04] Cache load attempt failed: {e}")
        return None
