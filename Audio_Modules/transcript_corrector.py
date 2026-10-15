"""
Audio_Modules/transcript_corrector.py
=====================================
STAGE 2 of the transcript pipeline (replaces the old lyric_rhythm_aligner for dialogue).

    raw audio  +  Whisper draft transcript  ->  Gemini  ->  corrected transcript

Gemini LISTENS to the audio and judges the Whisper draft against it. It returns
per-segment decisions (keep / edit / delete) plus any speech Whisper missed.

Design rules (why it's built this way):
  1. Gemini fixes TEXT. It does not own TIMESTAMPS. LLM audio timestamps drift;
     Whisper's are far better. So Gemini never rewrites existing timings. We
     re-align corrected words onto Whisper's word timings with a diff, and only
     interpolate timings for words that were genuinely new.
  2. Segment ids are preserved. Gemini cannot merge/split, so nothing silently
     disappears. Deleted segments are kept in `removed_segments` for audit.
  3. Gemini's edits are not trusted blindly. A rewrite that shares almost nothing
     with a high-confidence Whisper segment is rejected as a likely paraphrase.
  4. Never raises. On any failure you get the Whisper transcript back unchanged,
     tagged `_source: "whisper_only"`.

Controlled by: ENABLE_TRANSCRIPT_CORRECTION=true (default)
"""

import difflib
import hashlib
import json
import logging
import os
import re
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    if os.path.exists(".env"):
        load_dotenv(".env", override=False)
    _cred_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Credentials", ".env")
    if os.path.exists(_cred_env):
        load_dotenv(_cred_env, override=False)
except Exception:
    pass

logger = logging.getLogger("transcript_corrector")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Config ───────────────────────────────────────────────────────────────────
ENABLE_TRANSCRIPT_CORRECTION = os.getenv("ENABLE_TRANSCRIPT_CORRECTION", "true").lower() in ("true", "1", "yes")
GEMINI_MODEL = os.getenv("TRANSCRIPT_CORRECTOR_MODEL", "gemini-2.5-flash")  # only used if router is unavailable
GEMINI_TIMEOUT_SEC = float(os.getenv("TRANSCRIPT_CORRECTOR_TIMEOUT", "90"))
FILE_ACTIVE_TIMEOUT_SEC = 30.0
MAX_ATTEMPTS = 2                       # 1 retry, only for unparseable JSON

_MIN_AUDIO_BYTES = 32_768
_PROMPT_VERSION = 1

# Edit guard: reject an "edit" this dissimilar to a segment Whisper was confident about
_MIN_EDIT_SIMILARITY = 0.30
_CONFIDENT_SEGMENT = 0.80

# Inserted segment must not overlap an existing kept segment by more than this fraction
_MAX_INSERT_OVERLAP = 0.50

# Gap between words that counts as a natural cut point
_CUT_GAP_SEC = 0.25

VALID_ISSUES = {
    "mishearing", "wrong_language", "wrong_script", "missed_words", "hallucination",
    "proper_noun", "punctuation", "repeated_text", "unintelligible",
}

# ─── Prompt ───────────────────────────────────────────────────────────────────

_PROMPT = """You are a transcript error-correction engine for speech and song lyrics.

You are given:
  (1) an AUDIO file, and
  (2) a DRAFT transcript of that audio from Whisper ASR, with timestamps.

The AUDIO is the ground truth. The DRAFT is often wrong: misheard words, wrong language or
script, missed words, wrong names, bad punctuation, and invented text (Whisper hallucinates
phrases like "thank you for watching" or subtitle credits over music, noise or silence).

Listen to the audio and correct the draft.

RULES
1. Fix only what is audibly wrong. If a segment is already correct, return action "keep".
   Do NOT paraphrase, polish grammar, summarize, censor, or translate.
2. Keep every language as spoken. Keep the draft's script (e.g. romanized Hinglish stays
   romanized) unless the draft's script is clearly wrong for the spoken language.
3. Return one decision for EVERY draft segment id. Never merge or split segments and never
   change an existing segment's timing.
4. If a draft segment is not speech or singing (music, noise, silence, or hallucinated
   text), use action "delete".
5. If clearly audible speech or singing is MISSING from the draft, add it under "inserted",
   with start/end in seconds from the very beginning of the audio. Insert only if confident.
6. Words marked [?] are low-confidence in Whisper. Check them first, but verify everything.
7. Never invent text you cannot hear. If a word is unintelligible, keep the draft's word and
   add the issue "unintelligible".
8. Return ONLY one strict JSON object. No markdown, no commentary.

JSON schema:
{
  "language": "<main spoken/sung language>",
  "has_speech": true | false,
  "is_singing": true | false,
  "segments": [
    {
      "id": <int — the draft segment id>,
      "action": "keep" | "edit" | "delete",
      "text": "<full corrected text of the segment; identical to draft if keep; empty if delete>",
      "issues": ["mishearing" | "wrong_language" | "wrong_script" | "missed_words" | "hallucination" | "proper_noun" | "punctuation" | "repeated_text" | "unintelligible"],
      "confidence": <float 0.0-1.0 — your confidence in the final text>
    }
  ],
  "inserted": [
    {"start": <float>, "end": <float>, "text": "<text>", "issues": [], "confidence": <float 0.0-1.0>}
  ]
}
<<CONTEXT>>
DRAFT TRANSCRIPT (detected language: <<LANG>>, audio duration: <<DUR>>s)
Format: [id] start-end | conf=<segment confidence> | SUSPECT? | text
<<TRANSCRIPT>>
"""

# ─── Text helpers ─────────────────────────────────────────────────────────────

def _norm(token: str) -> str:
    """Normalize for comparison. Drops punctuation/symbols/spaces but KEEPS letters,
    combining marks (Devanagari matras etc.) and digits."""
    return "".join(ch for ch in token if unicodedata.category(ch)[0] not in "PSZC").lower()


def _norm_text(text: str) -> str:
    return " ".join(filter(None, (_norm(t) for t in text.split())))


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm_text(a), _norm_text(b), autojunk=False).ratio()


def _clean_json(text: str) -> str:
    """Strip markdown fences, extract the outer object, drop trailing commas."""
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


# ─── Word re-alignment ────────────────────────────────────────────────────────

def _spread(tokens: List[str], t0: float, t1: float) -> List[Dict[str, Any]]:
    """Distribute tokens across [t0, t1] proportional to character length."""
    if not tokens:
        return []
    t1 = max(t1, t0)
    weights = [max(1, len(_norm(t))) for t in tokens]
    total = float(sum(weights))
    out, cursor = [], t0
    for tok, w in zip(tokens, weights):
        end = cursor + (t1 - t0) * (w / total)
        out.append({
            "word": tok, "start": round(cursor, 3), "end": round(end, 3),
            "confidence": None, "source": "interpolated",
        })
        cursor = end
    return out


def _realign_words(
    orig_words: List[Dict[str, Any]],
    new_text: str,
    seg_start: float,
    seg_end: float,
) -> List[Dict[str, Any]]:
    """
    Map corrected text back onto Whisper's word timings.
      - unchanged words  -> keep Whisper's exact timing + confidence
      - replaced words   -> share the time span of the words they replaced
      - inserted words   -> interpolated into the gap between neighbours
      - removed words    -> dropped
    """
    new_tokens = new_text.split()
    if not new_tokens:
        return []
    if not orig_words:
        return _spread(new_tokens, seg_start, seg_end)

    sm = difflib.SequenceMatcher(
        None,
        [_norm(w["word"]) for w in orig_words],
        [_norm(t) for t in new_tokens],
        autojunk=False,
    )
    out: List[Dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ow = orig_words[i1 + k]
                out.append({
                    "word": new_tokens[j1 + k],
                    "start": ow["start"], "end": ow["end"],
                    "confidence": ow.get("confidence"), "source": "whisper",
                })
        elif tag in ("replace", "insert"):
            if i2 > i1:
                t0, t1 = orig_words[i1]["start"], orig_words[i2 - 1]["end"]
            else:
                t0 = orig_words[i1 - 1]["end"] if i1 > 0 else seg_start
                t1 = orig_words[i1]["start"] if i1 < len(orig_words) else seg_end
            out.extend(_spread(new_tokens[j1:j2], t0, t1))
    return out


# ─── Gemini plumbing ──────────────────────────────────────────────────────────

def _get_router():
    try:
        from Gemini_Modules.gemini_router_module import gemini_router
        return gemini_router
    except Exception:
        pass
    try:
        from Gemini_Modules.gemini_router_module.gemini_governor import gemini_router
        return gemini_router
    except Exception:
        pass
    try:
        from gemini_governor import gemini_router
        return gemini_router
    except Exception:
        return None


def _format_draft(whisper: Dict[str, Any]) -> str:
    lines = []
    for seg in whisper.get("segments", []):
        words = seg.get("words") or []
        if words:
            text = " ".join(
                f"{w['word']}[?]" if (w.get("confidence") is not None and w["confidence"] < 0.5) else w["word"]
                for w in words
            )
        else:
            text = seg.get("text", "")
        flag = " | SUSPECT" if seg.get("suspect") else ""
        lines.append(
            f"[{seg['id']}] {seg['start']:.2f}-{seg['end']:.2f} | conf={seg.get('confidence', 0):.2f}{flag} | {text}"
        )
    return "\n".join(lines) if lines else "(Whisper produced no segments — listen for any missed speech and use \"inserted\".)"


def _build_prompt(whisper: Dict[str, Any], context_hint: str) -> str:
    ctx = f"\nCONTEXT FROM THE USER (use it to resolve names/spellings, do not treat as audio): {context_hint.strip()}\n" if context_hint.strip() else ""
    return (
        _PROMPT
        .replace("<<CONTEXT>>", ctx)
        .replace("<<LANG>>", str(whisper.get("language", "unknown")))
        .replace("<<DUR>>", str(whisper.get("duration_sec", 0)))
        .replace("<<TRANSCRIPT>>", _format_draft(whisper))
    )


def _generate(client, uploaded_file, prompt: str, timeout: float) -> str:
    box: Dict[str, Any] = {}

    def _worker():
        try:
            router = _get_router()
            if router is not None:
                box["out"] = router.generate(
                    task_type="analysis",
                    prompt=[uploaded_file, prompt],
                    module_name="transcript_corrector",
                    gen_config={"temperature": 0.0},
                )
            else:
                from google.genai import types
                resp = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[uploaded_file, prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.0, response_mime_type="application/json"
                    ),
                )
                box["out"] = resp.text
        except Exception as e:
            box["err"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"Gemini did not respond within {timeout:.0f}s")
    if "err" in box:
        raise box["err"]
    return str(box.get("out") or "")


def _query_gemini(audio_path: str, prompt: str) -> Dict[str, Any]:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY / GOOGLE_API_KEY set")

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        logger.info(f"📤 [CORRECTOR] Uploading {os.path.basename(audio_path)}...")
        uploaded = client.files.upload(file=audio_path)

        deadline = time.time() + FILE_ACTIVE_TIMEOUT_SEC
        while getattr(getattr(uploaded, "state", None), "name", "ACTIVE") == "PROCESSING":
            if time.time() > deadline:
                raise TimeoutError("Gemini file stayed in PROCESSING state")
            time.sleep(1)
            uploaded = client.files.get(name=uploaded.name)
        state = getattr(getattr(uploaded, "state", None), "name", "ACTIVE")
        if state != "ACTIVE":
            raise RuntimeError(f"Gemini file upload ended in state {state}")

        last_err: Optional[Exception] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            t0 = time.time()
            raw = _generate(client, uploaded, prompt, GEMINI_TIMEOUT_SEC)
            logger.info(f"✅ [CORRECTOR] Gemini replied in {time.time() - t0:.1f}s (attempt {attempt})")
            try:
                data = json.loads(_clean_json(raw))
                if isinstance(data, dict) and isinstance(data.get("segments"), list):
                    return data
                last_err = ValueError("Gemini JSON missing 'segments' list")
            except json.JSONDecodeError as e:
                last_err = e
            logger.warning(f"[CORRECTOR] Unusable Gemini output on attempt {attempt}: {last_err}")
        raise RuntimeError(f"Gemini output unusable after {MAX_ATTEMPTS} attempts: {last_err}")
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


# ─── Merge: Whisper draft + Gemini decisions -> final transcript ──────────────

def _clean_issues(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    return [str(i) for i in raw if str(i) in VALID_ISSUES]


def _clamp01(v: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _derive_cut_points(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean cut timestamps (word ends at punctuation / pauses / segment ends)."""
    cuts: Dict[float, Dict[str, Any]] = {}
    for seg in segments:
        words = seg.get("words", [])
        for idx, w in enumerate(words):
            ts = round(w["end"], 3)
            if re.search(r"[.?!,।]$", w["word"]):
                cuts.setdefault(ts, {"timestamp_sec": ts, "word_after_which_to_cut": w["word"], "reason": "sentence_punctuation"})
            elif idx + 1 < len(words) and words[idx + 1]["start"] - w["end"] >= _CUT_GAP_SEC:
                cuts.setdefault(ts, {"timestamp_sec": ts, "word_after_which_to_cut": w["word"], "reason": "natural_pause"})
        if words:
            ts = round(seg["end"], 3)
            cuts.setdefault(ts, {"timestamp_sec": ts, "word_after_which_to_cut": words[-1]["word"], "reason": "segment_end"})
    return [cuts[k] for k in sorted(cuts)]


def _apply_corrections(whisper: Dict[str, Any], gem: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    decisions: Dict[int, Dict[str, Any]] = {}
    if gem:
        for d in gem.get("segments", []):
            try:
                decisions[int(d.get("id"))] = d
            except (TypeError, ValueError):
                continue

    duration = float(whisper.get("duration_sec", 0.0))
    out_segments: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    stats = {"total": 0, "kept": 0, "edited": 0, "removed": 0, "inserted": 0, "rejected_edits": 0}

    for seg in whisper.get("segments", []):
        stats["total"] += 1
        sid = seg["id"]
        orig_text = seg["text"]
        orig_words = seg.get("words") or []
        d = decisions.get(sid)
        action = str(d.get("action", "keep")).lower() if d else "keep"
        issues = _clean_issues(d.get("issues")) if d else []
        g_conf = _clamp01(d.get("confidence"), seg.get("confidence", 0.5)) if d else seg.get("confidence", 0.5)

        if action == "delete":
            stats["removed"] += 1
            removed.append({
                "source_id": sid, "start": seg["start"], "end": seg["end"],
                "original_text": orig_text, "issues": issues,
                "whisper_suspect": seg.get("suspect", False),
            })
            continue

        new_text = orig_text
        if action == "edit":
            candidate = str(d.get("text", "")).strip()
            if not candidate:
                action = "keep"
            else:
                sim = _similarity(orig_text, candidate)
                if sim < _MIN_EDIT_SIMILARITY and seg.get("confidence", 0) >= _CONFIDENT_SEGMENT and not seg.get("suspect"):
                    stats["rejected_edits"] += 1
                    rejected.append({
                        "source_id": sid, "original_text": orig_text,
                        "proposed_text": candidate, "similarity": round(sim, 3),
                        "reason": "rewrite_of_confident_segment",
                    })
                    action = "keep"
                else:
                    new_text = candidate

        changed = action == "edit" and new_text != orig_text
        if changed:
            stats["edited"] += 1
            words = _realign_words(orig_words, new_text, seg["start"], seg["end"])
        else:
            stats["kept"] += 1
            changed = False
            words = [dict(w, source="whisper") for w in orig_words] or _spread(orig_text.split(), seg["start"], seg["end"])

        out_segments.append({
            "source_id": sid,
            "start": seg["start"], "end": seg["end"],
            "text": new_text, "original_text": orig_text,
            "changed": changed,
            "issues": issues,
            "confidence": round(g_conf if gem else seg.get("confidence", 0.5), 3),
            "whisper_confidence": seg.get("confidence"),
            "origin": "whisper",
            "words": words,
        })

    # Speech Whisper missed
    if gem:
        for ins in gem.get("inserted", []) or []:
            try:
                s, e = float(ins.get("start")), float(ins.get("end"))
                text = str(ins.get("text", "")).strip()
            except (TypeError, ValueError):
                continue
            if not text or e <= s or s < 0 or (duration and s > duration + 0.5):
                continue
            if duration:
                e = min(e, duration)
            dur = e - s
            overlap = sum(
                max(0.0, min(e, o["end"]) - max(s, o["start"])) for o in out_segments
            )
            if dur <= 0 or overlap / dur > _MAX_INSERT_OVERLAP:
                continue
            stats["inserted"] += 1
            out_segments.append({
                "source_id": None, "start": round(s, 3), "end": round(e, 3),
                "text": text, "original_text": "", "changed": True,
                "issues": _clean_issues(ins.get("issues")) or ["missed_words"],
                "confidence": round(_clamp01(ins.get("confidence"), 0.5), 3),
                "whisper_confidence": None, "origin": "gemini_inserted",
                "words": _spread(text.split(), s, e),
            })

    out_segments.sort(key=lambda x: x["start"])
    for idx, seg in enumerate(out_segments):
        seg["id"] = idx

    flat_words = [w for seg in out_segments for w in seg["words"]]
    language = (gem or {}).get("language") or whisper.get("language", "unknown")

    return {
        "has_speech": len(out_segments) > 0,
        "language": language,
        "is_singing": bool((gem or {}).get("is_singing", False)),
        "duration_sec": duration,
        "full_text": " ".join(s["text"] for s in out_segments),
        "segments": out_segments,
        "words": flat_words,
        "removed_segments": removed,
        "rejected_edits": rejected,
        "clean_cut_timestamps": _derive_cut_points(out_segments),
        "stats": stats,
        "whisper_model": whisper.get("model_size"),
    }


# ─── Main API ─────────────────────────────────────────────────────────────────

def _cache_path(audio_path: str, cache_dir: Optional[str]) -> str:
    cache_dir = cache_dir or os.path.join(_REPO_ROOT, "Original_audio", "beats")
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(cache_dir, f"{base}_transcript.json")


def _cache_key(whisper: Dict[str, Any], context_hint: str) -> str:
    h = hashlib.sha1()
    h.update(str(whisper.get("audio_sha1", "")).encode())
    h.update(json.dumps([(s["start"], s["end"], s["text"]) for s in whisper.get("segments", [])], ensure_ascii=False).encode())
    h.update(context_hint.encode())
    h.update(f"{_PROMPT_VERSION}|{GEMINI_MODEL}".encode())
    return h.hexdigest()


def _log_changes(result: Dict[str, Any]) -> None:
    for seg in result.get("segments", []):
        if seg.get("changed"):
            logger.info(f"✏️ [CORRECTOR] #{seg['id']} {seg['start']:.1f}s: '{seg['original_text'][:60]}' -> '{seg['text'][:60]}' {seg.get('issues')}")
    for r in result.get("removed_segments", []):
        logger.info(f"🗑️ [CORRECTOR] removed {r['start']:.1f}s: '{r['original_text'][:60]}' {r.get('issues')}")


def correct_transcript(
    audio_path: str,
    whisper_result: Optional[Dict[str, Any]] = None,
    *,
    context_hint: str = "",
    cache_dir: Optional[str] = None,
    run_on_empty: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Correct a Whisper transcript by having Gemini listen to the audio.
    NEVER raises.
    """
    try:
        if whisper_result is None:
            from Audio_Modules.whisper_transcriber import transcribe_audio
            whisper_result = transcribe_audio(audio_path)

        if not isinstance(whisper_result, dict):
            whisper_result = {"segments": []}

        def _passthrough(reason: str) -> Dict[str, Any]:
            res = _apply_corrections(whisper_result, None)
            res["_source"] = "whisper_only"
            res["_reason"] = reason
            return res

        if not ENABLE_TRANSCRIPT_CORRECTION:
            return _passthrough("disabled")
        if not audio_path or not os.path.exists(audio_path):
            return _passthrough("audio_not_found")
        if os.path.getsize(audio_path) < _MIN_AUDIO_BYTES:
            return _passthrough("audio_too_small")
        if not whisper_result.get("segments") and not run_on_empty:
            return _passthrough("no_speech_detected")

        cache_file = _cache_path(audio_path, cache_dir)
        key = _cache_key(whisper_result, context_hint)
        if not force and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("_cache_key") == key:
                    cached["_source"] = "cache_hit"
                    logger.info(f"💾 [CORRECTOR] Cache hit: {cache_file}")
                    return cached
            except Exception:
                pass

        prompt = _build_prompt(whisper_result, context_hint)
        try:
            gem = _query_gemini(audio_path, prompt)
        except Exception as ge:
            logger.warning(f"[CORRECTOR] Gemini failed ({ge}) — returning Whisper transcript unchanged.")
            return _passthrough(f"gemini_failed: {ge}")

        result = _apply_corrections(whisper_result, gem)
        result["_source"] = "gemini_corrected"
        result["_cache_key"] = key

        try:
            tmp = cache_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            os.replace(tmp, cache_file)
        except Exception as we:
            logger.warning(f"[CORRECTOR] Failed to persist transcript: {we}")

        s = result["stats"]
        logger.info(
            f"🎯 [CORRECTOR] {os.path.basename(audio_path)}: {s['total']} segs | "
            f"edited={s['edited']} removed={s['removed']} inserted={s['inserted']} "
            f"rejected_edits={s['rejected_edits']}"
        )
        _log_changes(result)
        return result

    except Exception as e:
        logger.error(f"❌ [CORRECTOR] Unexpected failure: {e}")
        try:
            res = _apply_corrections(whisper_result or {"segments": []}, None)
        except Exception:
            res = {"has_speech": False, "segments": [], "words": [], "full_text": "", "clean_cut_timestamps": []}
        res["_source"] = "whisper_only"
        res["_reason"] = f"unexpected: {e}"
        return res


def transcribe_and_correct(
    audio_path: str,
    *,
    language: Optional[str] = None,
    context_hint: str = "",
    force: bool = False,
) -> Dict[str, Any]:
    """Convenience: Stage 1 (Whisper) -> Stage 2 (Gemini correction)."""
    from Audio_Modules.whisper_transcriber import transcribe_audio
    whisper = transcribe_audio(audio_path, language=language, initial_prompt=context_hint or None, force=force)
    return correct_transcript(audio_path, whisper, context_hint=context_hint, force=force)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("usage: python transcript_corrector.py <audio_file> [language] [context hint...]")
        raise SystemExit(1)
    lang = sys.argv[1 + 1] if len(sys.argv) > 2 else None
    hint = " ".join(sys.argv[3:])
    print(json.dumps(transcribe_and_correct(sys.argv[1], language=lang, context_hint=hint), ensure_ascii=False, indent=2))
