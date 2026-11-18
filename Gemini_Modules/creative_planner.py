"""
Gemini_Modules / creative_planner.py
====================================
Auto-Input-Only Gemini Creative Content Planner.

Runs BEFORE downloading video files or running FFmpeg synthesis:
  1. Consumes verified news/trend items + zero-download candidate video metadata
     (and optional transcript context).
  2. Makes ONE structured Gemini call to produce a monetization-safe Creative Brief:
     - Scroll-stopping 0-3s hook + commentary narration script (fed to Gemini TTS)
     - Which 1-2 candidate video URLs to actually download (skips the other 18)
     - Mandatory visual shots for ForensicVideoAnalyzer to locate & timestamp
     - Code-enforced feature flags (voiceover, pacing, cinematic zoom, speed ramps)
  3. Persists `creative_brief.json` ONLY inside Auto Input clip folders so
     Manual Input mode remains 100% untouched.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("creative_planner")

CREATIVE_BRIEF_FILENAME = "creative_brief.json"

try:
    from Gemini_Modules.gemini_router_module.gemini_router import gemini_router
    _HAS_ROUTER = True
except ImportError:
    gemini_router = None
    _HAS_ROUTER = False


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        from json_repair import repair_json
        repaired = repair_json(cleaned, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _normalize_brief(
    raw: Dict[str, Any],
    target_keyword: str,
    news_items: List[Dict[str, Any]],
    candidate_videos: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validates and clamps Creative Brief fields so downstream code can enforce them safely."""
    valid_urls = [c.get("url") for c in candidate_videos if c.get("url")]
    valid_url_set = set(valid_urls)

    raw_selected = raw.get("selected_video_urls") or []
    selected_urls = [u for u in raw_selected if isinstance(u, str) and u in valid_url_set]
    videos_needed = max(1, min(2, int(raw.get("videos_needed") or 1)))
    if not selected_urls and valid_urls:
        selected_urls = valid_urls[:videos_needed]
    else:
        selected_urls = selected_urls[:videos_needed]

    raw_shots = raw.get("mandatory_shots") or []
    mandatory_shots: List[Dict[str, Any]] = []
    for idx, sh in enumerate(raw_shots):
        if not isinstance(sh, dict):
            continue
        desc = str(sh.get("description") or "").strip()
        if not desc:
            continue
        role = str(sh.get("role") or ("hook" if idx == 0 else "body")).lower().strip()
        min_s = max(0.8, min(5.0, float(sh.get("min_s") or 1.5)))
        max_s = max(min_s + 0.5, min(6.0, float(sh.get("max_s") or 3.0)))
        mandatory_shots.append({
            "role": role,
            "description": desc,
            "min_s": round(min_s, 2),
            "max_s": round(max_s, 2),
        })

    if not mandatory_shots:
        mandatory_shots = [
            {
                "role": "hook",
                "description": f"High-impact close-up or expressive moment featuring {target_keyword}",
                "min_s": 1.5,
                "max_s": 3.0,
            },
            {
                "role": "context",
                "description": f"Clear medium/wide shot establishing {target_keyword} and surroundings",
                "min_s": 2.0,
                "max_s": 4.0,
            },
        ]

    raw_flags = raw.get("enforced_feature_flags") or {}
    if not isinstance(raw_flags, dict):
        raw_flags = {}

    hook_line = str(raw.get("hook_line") or "").strip()
    narration_script = str(raw.get("narration_script") or "").strip()
    if not narration_script and hook_line:
        narration_script = hook_line
    if not hook_line and narration_script:
        hook_line = narration_script.split(".")[0].strip()

    has_narration = bool(narration_script)
    enforced_flags = {
        "enable_voiceover": bool(raw_flags.get("enable_voiceover", has_narration)),
        "enable_fast_pacing": bool(raw_flags.get("enable_fast_pacing", True)),
        "enable_cinematic_zoom": bool(raw_flags.get("enable_cinematic_zoom", True)),
        "enable_speed_ramps": bool(raw_flags.get("enable_speed_ramps", False)),
    }

    return {
        "mode": "auto_creative_planner",
        "target_keyword": target_keyword,
        "story_angle": str(raw.get("story_angle") or f"Latest verified update on {target_keyword}").strip(),
        "hook_line": hook_line,
        "narration_script": narration_script,
        "first_3s_visual_directive": str(
            raw.get("first_3s_visual_directive")
            or mandatory_shots[0]["description"]
        ).strip(),
        "videos_needed": len(selected_urls) if selected_urls else videos_needed,
        "selected_video_urls": selected_urls,
        "mandatory_shots": mandatory_shots,
        "enforced_feature_flags": enforced_flags,
        "tone": str(raw.get("tone") or "engaging_commentary").strip(),
        "bgm_mood": str(raw.get("bgm_mood") or "upbeat_subtle").strip(),
        "news_sources": [
            {"title": n.get("title", ""), "source": n.get("source", ""), "provider": n.get("provider", "")}
            for n in news_items[:5]
        ],
    }


def _fallback_brief(
    target_keyword: str,
    news_items: List[Dict[str, Any]],
    candidate_videos: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Deterministic fallback brief when Gemini API is unavailable or rate-limited."""
    top_news = news_items[0] if news_items else {}
    headline = (top_news.get("title") or "").strip()
    summary = (top_news.get("summary") or "").strip()
    source = (top_news.get("source") or "latest reports").strip()

    if headline:
        hook = f"Big update on {target_keyword} that everyone is talking about right now."
        script = f"{hook} According to {source}, {headline}."
        if summary and summary.lower() not in headline.lower():
            script = f"{script} {summary[:160].rstrip('.')}."
    else:
        hook = f"Watch this standout moment from {target_keyword}."
        script = f"{hook} Here is why fans cannot stop replaying this clip."

    raw = {
        "story_angle": headline or f"Trending spotlight on {target_keyword}",
        "hook_line": hook,
        "narration_script": script,
        "first_3s_visual_directive": f"Sharp close-up or peak motion moment of {target_keyword}",
        "videos_needed": 1,
        "selected_video_urls": [c["url"] for c in candidate_videos[:1] if c.get("url")],
        "mandatory_shots": [
            {"role": "hook", "description": f"Immediate scroll-stopping close-up of {target_keyword}", "min_s": 1.5, "max_s": 3.0},
            {"role": "context", "description": f"Clear action or expression shot of {target_keyword}", "min_s": 2.0, "max_s": 4.0},
            {"role": "climax", "description": f"Memorable closing visual of {target_keyword}", "min_s": 2.0, "max_s": 3.5},
        ],
        "enforced_feature_flags": {
            "enable_voiceover": True,
            "enable_fast_pacing": True,
            "enable_cinematic_zoom": True,
            "enable_speed_ramps": False,
        },
        "tone": "engaging_commentary",
        "bgm_mood": "upbeat_subtle",
    }
    return _normalize_brief(raw, target_keyword, news_items, candidate_videos)


class GeminiCreativePlanner:
    """Builds a monetization-safe Creative Brief before downloading/synthesizing Auto Input videos."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def _build_prompt(
        self,
        target_keyword: str,
        news_items: List[Dict[str, Any]],
        candidate_videos: List[Dict[str, Any]],
        transcript: Optional[str] = None,
    ) -> str:
        news_block = json.dumps(news_items[:6], ensure_ascii=False, indent=2) if news_items else "[]"
        cand_slim = [
            {
                "url": c.get("url", ""),
                "title": c.get("title", ""),
                "description": (c.get("description") or "")[:200],
                "duration": c.get("duration", 0),
                "keyword_score": c.get("keyword_score", 0),
                "matched_keywords": c.get("matched_keywords", []),
            }
            for c in candidate_videos[:20]
        ]
        cand_block = json.dumps(cand_slim, ensure_ascii=False, indent=2) if cand_slim else "[]"
        tr_block = f"\nEXISTING TRANSCRIPT CONTEXT:\n{transcript[:800]}\n" if transcript else ""

        return (
            "You are an elite Viral Short-Form Creative Director planning a MONETIZABLE 15-25s YouTube Short / Reel.\n"
            f"TARGET KEYWORD / SUBJECT: {target_keyword}\n\n"
            "VERIFIED RECENT NEWS / UPDATES:\n"
            f"{news_block}\n\n"
            "CANDIDATE VIDEOS (METADATA ONLY — NOT DOWNLOADED YET):\n"
            f"{cand_block}\n"
            f"{tr_block}\n"
            "RULES:\n"
            "1. MONETIZATION SAFETY: Base commentary STRICTLY on the verified news facts provided above. "
            "Never fabricate unverified rumors or defamatory claims. Frame commentary as engaging, transformative reporting.\n"
            "2. SCROLL-STOP HOOK: The first 3 seconds (`hook_line` + `first_3s_visual_directive`) must immediately arrest attention.\n"
            "3. FOOTAGE SELECTION: Pick ONLY 1-2 best URLs from CANDIDATE VIDEOS (`selected_video_urls`) that best match the story.\n"
            "4. MANDATORY SHOTS: Specify 2-4 concrete visual shots (`mandatory_shots`) for our Forensic Video Analyzer to locate "
            "and timestamp inside the downloaded clip (always put role='hook' first for 0-3s).\n"
            "5. TOOL / FEATURE FLAGS: Set `enable_voiceover: true` whenever spoken commentary adds transformative value.\n\n"
            "RETURN STRICT JSON ONLY with these exact keys:\n"
            "{\n"
            '  "story_angle": "string",\n'
            '  "hook_line": "punchy 0-3s opening sentence",\n'
            '  "narration_script": "complete 25-55 word spoken commentary script",\n'
            '  "first_3s_visual_directive": "exact visual needed in first 3 seconds",\n'
            '  "videos_needed": 1,\n'
            '  "selected_video_urls": ["url_from_candidates"],\n'
            '  "mandatory_shots": [\n'
            '    {"role": "hook", "description": "what forensic analyzer must find", "min_s": 1.5, "max_s": 3.0}\n'
            "  ],\n"
            '  "enforced_feature_flags": {\n'
            '    "enable_voiceover": true,\n'
            '    "enable_fast_pacing": true,\n'
            '    "enable_cinematic_zoom": true,\n'
            '    "enable_speed_ramps": false\n'
            "  },\n"
            '  "tone": "energetic_commentary",\n'
            '  "bgm_mood": "upbeat_subtle"\n'
            "}"
        )

    def plan(
        self,
        target_keyword: str,
        news_items: Optional[List[Dict[str, Any]]] = None,
        candidate_videos: Optional[List[Dict[str, Any]]] = None,
        transcript: Optional[str] = None,
    ) -> Dict[str, Any]:
        news_list = list(news_items or [])
        cand_list = list(candidate_videos or [])
        prompt = self._build_prompt(target_keyword, news_list, cand_list, transcript=transcript)

        raw_text = ""
        try:
            if _HAS_ROUTER and gemini_router:
                resp = gemini_router.generate(
                    task_type="creative",
                    prompt=[prompt],
                    module_name="creative_planner",
                    gen_config={"temperature": 0.4, "response_mime_type": "application/json"},
                    api_key=self.api_key,
                )
                raw_text = getattr(resp, "text", None) or (resp if isinstance(resp, str) else "")
            elif self.api_key:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=self.api_key)
                resp = client.models.generate_content(
                    model=os.getenv("GEMINI_CREATIVE_PLANNER_MODEL", "gemini-2.5-flash"),
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.4,
                        response_mime_type="application/json",
                    ),
                )
                raw_text = getattr(resp, "text", "") or ""
        except Exception as exc:
            logger.warning("🧠 [CREATIVE PLANNER] Gemini call failed (%s) — using deterministic fallback brief.", exc)

        parsed = _extract_json_object(raw_text)
        if parsed:
            brief = _normalize_brief(parsed, target_keyword, news_list, cand_list)
            logger.info(
                "🧠 [CREATIVE PLANNER] Brief ready for '%s' | selected=%d video(s) | shots=%d | voiceover=%s",
                target_keyword,
                len(brief["selected_video_urls"]),
                len(brief["mandatory_shots"]),
                brief["enforced_feature_flags"].get("enable_voiceover"),
            )
            return brief

        logger.info("🧠 [CREATIVE PLANNER] Using deterministic fallback brief for '%s'.", target_keyword)
        return _fallback_brief(target_keyword, news_list, cand_list)


def save_creative_brief(brief: Dict[str, Any], clip_dir: str) -> str:
    """Saves creative_brief.json locally inside an Auto Input clip folder."""
    os.makedirs(clip_dir, exist_ok=True)
    path = os.path.join(clip_dir, CREATIVE_BRIEF_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(brief, f, indent=2, ensure_ascii=False)
    return path


def load_creative_brief(clip_dir_or_video_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads creative_brief.json if present for this clip (Auto Input only).
    Returns None for Manual Input clips so manual mode stays 100% untouched.
    """
    if not clip_dir_or_video_path:
        return None
    base_dir = (
        clip_dir_or_video_path
        if os.path.isdir(clip_dir_or_video_path)
        else os.path.dirname(os.path.abspath(clip_dir_or_video_path))
    )
    path = os.path.join(base_dir, CREATIVE_BRIEF_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("mode") == "auto_creative_planner":
            return data
    except Exception as exc:
        logger.debug("creative_brief load notice (%s): %s", path, exc)
    return None
