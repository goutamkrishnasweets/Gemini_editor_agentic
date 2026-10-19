"""
Gemini_Modules/clip_intelligence_store.py
==========================================
Master Clip Intelligence Store.

Manages per-clip intelligence JSON files and pools all records into
Original_audio/pool_metadata.json.

Per-clip file: downloads/{clip_id}/.clip_intelligence.json
Pool file:     Original_audio/pool_metadata.json → "clips" section

JSON Structure (schema_version=3):
{
  "clip_id": "manual_1785614074",
  "clip_folder": "...",
  "schema_version": 3,
  "created_at": <timestamp>,

  "phase1": { source_url, platform, shortcode, video_path, proxy_path, wav_path },

  "visual_context": {
    "_source": "gemini_call_1_forensic",
    intent, tone, editing_style, detected_entities, feature_flags, safety, ...
  },

  "audio_data": {
    "_description": "Unified audio block. math fills first (Phase 1 DSP), context fills second (Gemini Call 1).",
    "math": {
      "_source": "phase1_dsp", tempo_bpm, beat_timestamps, drop_timestamps, avg_energy, vibe, ...
    },
    "context": {
      "_source": "gemini_call_1_forensic", has_vocals, language, dominant_emotion, sections, tension_arc, ...
    },
    "selected_bgm_track": null,
    "bgm_selection_reasoning": null,
    "bgm_alignment_score": null
  },

  "visual_vectors": {
    "_source": "gemini_call_1_forensic",
    targeted_timestamps_sec, scene_cut_timestamps, hook_zone_end_sec, recommended_speed_ramp_zones
  },

  "editing_plan": { ... },   ← written by Gemini Call 3 (gemini_ffmpeg_synthesis)
  "output": { ... }          ← written after FFmpeg render
}
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ClipIntelligenceStore")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_POOL_FILE = os.path.join(_REPO_ROOT, "Original_audio", "pool_metadata.json")
_INTEL_FILENAME = ".clip_intelligence.json"
SCHEMA_VERSION = 3


class ClipIntelligenceStore:
    """
    Reads and writes per-clip intelligence JSON records.
    Pools all clip records into pool_metadata.json → "clips" section.
    """

    def __init__(self, clip_id: Optional[str] = None, clip_folder: Optional[str] = None):
        """
        Optional constructor. When clip_id / clip_folder are provided the instance is
        bound to that clip so callers can use the convenience .get() / .set() API
        (used by 04_bgm_selector.py and similar callers).
        """
        self._bound_clip_id: Optional[str] = clip_id
        self._bound_clip_folder: Optional[str] = clip_folder
        # Lazily loaded data dict for the bound clip
        self._data: Optional[Dict[str, Any]] = None

    # ── Convenience API (bound-clip mode) ───────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Return a top-level key from the bound clip's intelligence record."""
        if self._data is None and self._bound_clip_id:
            self._data = self.load(self._bound_clip_id, self._bound_clip_folder) or {}
        return (self._data or {}).get(key, default)

    def set(self, key: str, value: Any) -> bool:
        """Write a top-level key into the bound clip's intelligence record and persist."""
        if not self._bound_clip_id:
            logger.warning("[ClipStore] .set() called without a bound clip_id — no-op.")
            return False
        if self._data is None:
            self._data = self.load(self._bound_clip_id, self._bound_clip_folder) or {}
        if not self._data:
            self._data = self.create_blank(self._bound_clip_id, self._bound_clip_folder or "")
        self._data[key] = value
        return self.save(self._bound_clip_id, self._data, self._bound_clip_folder)

    # ── Load / Save per-clip ─────────────────────────────────────────────────

    def load(self, clip_id: str, clip_folder: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load clip intelligence JSON from disk or pool. Returns None if not found."""
        path = self._intel_path(clip_id, clip_folder)
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[ClipStore] Failed to load {path}: {e}")

        # Secondary fallback: Look up in master pool_metadata.json → "clips" section
        try:
            pool = self._pool_read()
            clips = pool.get("clips", {})
            if clip_id in clips:
                return clips[clip_id]
            for cid, cdata in clips.items():
                if clip_id in cid or cid in clip_id:
                    return cdata
            # Tier 3: pool v3 real data lives in files.social_media_id
            social = pool.get("files", {}).get("social_media_id", {})
            for url_key, entry in social.items():
                if not isinstance(entry, dict):
                    continue
                sc = str(entry.get("shortcode", ""))
                sm = str(entry.get("social_media_id", ""))
                if (clip_id == sc
                        or clip_id in url_key
                        or clip_id in sm
                        or (sc and sc in clip_id)
                        or url_key.rstrip("/").endswith(f"/{clip_id}")):
                    logger.debug(f"[ClipStore] Tier-3 hit: clip_id={clip_id} matched url_key={url_key[:60]}")
                    return entry
        except Exception:
            pass

        return None

    def save(self, clip_id: str, data: Dict[str, Any], clip_folder: Optional[str] = None) -> bool:
        """Save clip intelligence JSON to disk and update pool."""
        path = self._intel_path(clip_id, clip_folder or data.get("clip_folder"))
        if not path:
            logger.warning(f"[ClipStore] Cannot determine intel path for clip_id={clip_id}")
            return False
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"[ClipStore] ✓ Saved clip intelligence → {path}")
            self._pool_write(clip_id, data)
            return True
        except Exception as e:
            logger.error(f"[ClipStore] Save failed for {clip_id}: {e}")
            return False

    # ── Initialize a blank clip record ───────────────────────────────────────

    def create_blank(self, clip_id: str, clip_folder: str) -> Dict[str, Any]:
        """Returns a blank clip intelligence record with correct schema."""
        return {
            "clip_id": clip_id,
            "clip_folder": clip_folder,
            "schema_version": SCHEMA_VERSION,
            "created_at": time.time(),
            "phase1": {},
            "visual_context": {"_source": "pending", "_description": "Gemini Call 1 visual semantic context"},
            "audio_data": {
                "_description": "Unified audio block. math fills first (Phase 1 DSP), context fills second (Gemini Call 1).",
                "math": {"_source": "phase1_dsp", "_filled_at": None},
                "context": {"_source": "pending_gemini_call_1", "_filled_at": None},
                "selected_bgm_track": None,
                "bgm_selection_reasoning": None,
                "bgm_alignment_score": None,
                "_bgm_selected_at": None,
            },
            "visual_vectors": {"_source": "pending_gemini_call_1", "targeted_timestamps_sec": []},
            "editing_plan": {"_source": "pending_gemini_call_3"},
            "output": {"render_success": False},
        }

    # ── Patch individual blocks ───────────────────────────────────────────────

    def patch_phase1(self, data: Dict[str, Any], phase1: Dict[str, Any]) -> Dict[str, Any]:
        data["phase1"] = phase1
        return data

    def patch_audio_math(self, data: Dict[str, Any], math: Dict[str, Any]) -> Dict[str, Any]:
        """Fill audio_data.math block from Phase 1 DSP audio_analysis.json."""
        data["audio_data"]["math"] = {
            "_source": "phase1_dsp",
            "_filled_at": "step1_before_gemini",
            **math,
        }
        return data

    def patch_visual_context(self, data: Dict[str, Any], visual_context: Dict[str, Any]) -> Dict[str, Any]:
        data["visual_context"] = {"_source": "gemini_call_1_forensic", "_filled_at": time.time(), **visual_context}
        return data

    def patch_audio_context(self, data: Dict[str, Any], audio_context: Dict[str, Any]) -> Dict[str, Any]:
        """Fill audio_data.context block from Gemini Call 1 semantic audio output."""
        data["audio_data"]["context"] = {
            "_source": "gemini_call_1_forensic",
            "_filled_at": time.time(),
            **audio_context,
        }
        return data

    def patch_visual_vectors(self, data: Dict[str, Any], vectors: Dict[str, Any]) -> Dict[str, Any]:
        data["visual_vectors"] = {"_source": "gemini_call_1_forensic", "_filled_at": time.time(), **vectors}
        return data

    def patch_bgm_selection(
        self,
        data: Dict[str, Any],
        selected_track: str,
        reasoning: str,
        alignment_score: float,
    ) -> Dict[str, Any]:
        """Fill BGM selection result from Gemini Call 2 (lyric_rhythm_aligner)."""
        data["audio_data"]["selected_bgm_track"] = selected_track
        data["audio_data"]["bgm_selection_reasoning"] = reasoning
        data["audio_data"]["bgm_alignment_score"] = alignment_score
        data["audio_data"]["_bgm_selected_at"] = time.time()
        return data

    def patch_editing_plan(self, data: Dict[str, Any], editing_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Fill editing_plan from Gemini Call 3 (gemini_ffmpeg_synthesis)."""
        data["editing_plan"] = {"_source": "gemini_call_3_ffmpeg_synthesis", "_filled_at": time.time(), **editing_plan}
        return data

    def patch_output(self, data: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
        """Fill output block after FFmpeg render completes."""
        data["output"] = {"_filled_at": time.time(), **output}
        return data

    # ── Cross-clip Pool Queries ───────────────────────────────────────────────

    def get_all_audio_data(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Returns audio_data blocks from all pooled clips.
        Used by Gemini Call 2 (lyric_rhythm_aligner) to select best BGM
        by cross-matching against historical clip audio behaviors.
        """
        pool = self._pool_read()
        clips = pool.get("clips", {})
        results = []
        for clip_id, clip_data in list(clips.items())[-limit:]:
            audio_data = clip_data.get("audio_data", {})
            if audio_data:
                results.append({
                    "clip_id": clip_id,
                    "audio_data": audio_data,
                    "visual_context_summary": {
                        "intent": clip_data.get("visual_context", {}).get("intent"),
                        "tone": clip_data.get("visual_context", {}).get("tone"),
                        "editing_style": clip_data.get("visual_context", {}).get("editing_style"),
                    },
                })
        return results

    def get_all_clips_for_rag(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Returns all clip intelligence records structured for RAG ingestion.
        Each record contains visual_context + audio_data + editing_plan + output.
        This is the feed for the Creator Brain RAG builder.
        """
        pool = self._pool_read()
        clips = pool.get("clips", {})
        rag_records = []
        for clip_id, clip_data in list(clips.items())[-limit:]:
            rag_records.append({
                "clip_id": clip_id,
                "visual_context": clip_data.get("visual_context", {}),
                "audio_data": clip_data.get("audio_data", {}),
                "editing_plan": clip_data.get("editing_plan", {}),
                "output": clip_data.get("output", {}),
            })
        return rag_records

    # ── Pool management ───────────────────────────────────────────────────────

    def _pool_read(self) -> Dict[str, Any]:
        if os.path.isfile(_POOL_FILE):
            try:
                with open(_POOL_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"version": SCHEMA_VERSION, "files": {}, "clips": {}}

    def _pool_write(self, clip_id: str, data: Dict[str, Any]):
        """Upsert clip record into pool_metadata.json → "clips" section (sorted alphabetically by clip_id)."""
        pool = self._pool_read()
        if "clips" not in pool:
            pool["clips"] = {}
        pool["clips"][clip_id] = data
        # Sort dictionary by clip_id key for clean indexed ordering
        pool["clips"] = dict(sorted(pool["clips"].items(), key=lambda x: x[0]))
        try:
            with open(_POOL_FILE, "w", encoding="utf-8") as f:
                json.dump(pool, f, indent=2, ensure_ascii=False)
            # Automatic sync to visual_pool_metadata.json
            try:
                from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
                TelegramVaultIndexer().sync_visual_pool_metadata(clip_id, data)
            except Exception as _vp_err:
                logger.debug(f"[ClipStore] Visual pool metadata sync notice: {_vp_err}")
            try:
                from Audio_Modules.audio_pool_manager import AudioPoolManager
                AudioPoolManager()._sync_to_telegram_vault()
            except Exception as _pm_err:
                logger.debug(f"[ClipStore] Pool metadata vault sync notice: {_pm_err}")
        except Exception as e:
            logger.warning(f"[ClipStore] Pool write failed: {e}")

    def purge_clip(self, clip_id: str, clip_folder: Optional[str] = None):
        """
        Completely purges a clip's metadata:
        1. Removes clip_id entry from pool_metadata.json -> "clips" section.
        2. Deletes .clip_intelligence.json file if present.
        """
        pool = self._pool_read()
        clips = pool.get("clips", {})
        
        # Remove from pool_metadata.json
        keys_to_remove = [k for k in clips if k == clip_id or clip_id in k or k in clip_id]
        for k in keys_to_remove:
            clips.pop(k, None)
            logger.info(f"🗑️ [ClipStore] Purged clip index '{k}' from master pool_metadata.json")

        pool["clips"] = clips
        try:
            with open(_POOL_FILE, "w", encoding="utf-8") as f:
                json.dump(pool, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[ClipStore] Pool purge write failed: {e}")

        # Delete local .clip_intelligence.json if it exists
        if clip_folder and os.path.exists(clip_folder):
            intel_file = os.path.join(clip_folder, _INTEL_FILENAME)
            if os.path.exists(intel_file):
                try:
                    os.remove(intel_file)
                    logger.info(f"🗑️ [ClipStore] Deleted local intelligence file: {intel_file}")
                except Exception as _e:
                    logger.warning(f"⚠️ Failed to delete local intelligence file: {_e}")

    def _intel_path(self, clip_id: str, clip_folder: Optional[str]) -> Optional[str]:
        # If clip_folder contains an existing intelligence file, return it
        if clip_folder:
            candidate = os.path.join(clip_folder, _INTEL_FILENAME)
            if os.path.isfile(candidate):
                return candidate

        # Standard canonical location: downloads/{clip_id}/.clip_intelligence.json
        downloads = os.path.join(_REPO_ROOT, "downloads")
        candidate = os.path.join(downloads, clip_id, _INTEL_FILENAME)
        if os.path.isfile(candidate):
            return candidate

        # Fallback for new saves: avoid writing into Processed Shorts
        if clip_folder and "processed shorts" not in str(clip_folder).lower():
            os.makedirs(clip_folder, exist_ok=True)
            return os.path.join(clip_folder, _INTEL_FILENAME)

        os.makedirs(os.path.dirname(candidate), exist_ok=True)
        return candidate
