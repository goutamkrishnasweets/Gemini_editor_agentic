"""
Gemini_Modules/stickman_director.py
===================================
Autonomous Stickman Video Director Engine.

Transforms topics, harvested internet drama, transcripts, and Manga/Comic PDFs into
customizable-length animated stick-figure videos (in 10-second multiples: 30s, 60s, 90s).

Based on the 5-Stage High-Completion Heartbeat Engine:
  1. Golden Hook: Instant curiosity trigger / visual paradox (0-3s).
  2. Disrupting Assumptions: Shatters what everyone assumes is true.
  3. Unveiling Insider Secrets: Pulls back the curtain on hidden mechanics / drama.
  4. Ultimate Truth Revelation: Delivers maximum cognitive payoff or comedic punchline.
  5. Elevation & High-Engagement Discussion: Closes with a comment-section debate spark.

Supports:
  - Style 1 (Classic Minimalist: Light or Dark canvas)
  - Style 2A (Modern Studio Tech: Beanie Zeke on grid)
  - Style 2B (Cinematic Story: full-color 2D narrative)
  - Style 3 (Prehistoric Hand-Drawn / Dark Comedy: animal-hide, satirical battle)
  - Style 4 (Cozy Storytelling: expressive minimalist cartoon)

Outputs:
  - Storyboard Proposal (Phase A: English VO script ~20-25 words per 10s, visual narrative arc)
  - Production Prompt Package (Phase B: N standalone 10-second prompt contracts for Veo/Gemini Omni Flash)
"""

import os
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

try:
    from Gemini_Modules.gemini_router_module.gemini_governor import gemini_router
except Exception:
    try:
        from gemini_governor import gemini_router
    except Exception:
        gemini_router = None

logger = logging.getLogger("stickman_director")

# ─── Style Catalogs & Continuity Anchors ──────────────────────────────────────

STYLE_CATALOG = {
    "Style 1": {
        "name": "Style 1 (Classic Minimalist)",
        "char_anchor": (
            "A minimalist black line stick figure on a clean digitally pure-white background. "
            "Uniform medium line weight, circular hollow head, simple proportions. "
            "Two arms, two legs, sharp fluid kinetic animation."
        ),
        "env_light": "flat, uniform, digitally pure-white canvas with no textures, gradients, or 3D depth",
        "env_dark": "flat, uniform, pitch-black canvas with pure white line art and no 3D depth",
        "negative": "NO 3D SHADING, NO GRADIENTS, NO REALISTIC FACIAL FEATURES, NO NOSE, NO LIPS."
    },
    "Style 2A": {
        "name": "Style 2A (Modern Studio Tech)",
        "char_anchor": (
            "A minimalist 2D animated stick figure wearing a bright red beanie (smooth knit, no pom-pom) "
            "and a yellow t-shirt, with simple black stick limbs and shorts. "
            "Simple black lines, vibrant colors, smooth 2D animation style."
        ),
        "env": (
            "in a modern bright white studio space with subtle light-gray perspective grid lines on the floor plane. "
            "High-key studio lighting, clean white negative space, sleek glowing cyan and electric blue glass holographic UI elements."
        ),
        "negative": "STRICTLY MINIMALIST, NO CIRCUIT BOARD TEXTURES, NO SCI-FI WALL PANELS, NO CRACKED CONCRETE."
    },
    "Style 2B": {
        "name": "Style 2B (Cinematic Story)",
        "char_anchor": (
            "A minimalist 2D animated stick figure wearing a bright red beanie (smooth knit, no pom-pom) "
            "and a yellow t-shirt, with simple black stick limbs and shorts. "
            "Simple black lines, vibrant colors, smooth 2D animation style."
        ),
        "env": "full-color cinematic 2D environment with volumetric lighting and depth of field",
        "negative": "NO PHOTOREALISTIC 3D RENDERING, NO UNCANNY VALLEY PUPILS, NO REALISTIC CLOTHING TEXTURES."
    },
    "Style 3": {
        "name": "Style 3 (Prehistoric Hand-Drawn / Dark Comedy)",
        "char_anchor": (
            "Expressive 2D hand-drawn stick figure characters with solid white filled heads, black stick limbs, "
            "and rugged animal-hide costumes. Distinctive expressive cartoon eyes and mouths, satirical dark comedy energy."
        ),
        "env": "earthy prehistoric landscape, primitive stone and wooden tools, warm amber and bone-white accents",
        "negative": "NO HIGH-TECH ELEMENTS, NO MODERN UI, NO ULTRA-REALISTIC GORE, KEEP CARTOON SATIRE."
    },
    "Style 4": {
        "name": "Style 4 (Cozy Storytelling)",
        "char_anchor": (
            "Expressive minimalist characters with white filled heads and torsos, dark stick limbs, and gentle cartoon facial features. "
            "Soft, warm, storybook aesthetic."
        ),
        "env": "rich full-color 2D cartoon storybook backgrounds, warm cozy ambient lighting, cinematic atmospheric framing",
        "negative": "NO COLD SCI-FI ELEMENTS, NO HARSH CONTRASTS, NO AGGRESSIVE ACTION."
    }
}

VOICE_CONTINUITY_PROFILES = {
    "default_male": (
        "Narrator: Engaging, thoughtful male voice, neutral American accent, warm conversational baritone, "
        "crisp studio close-mic recording, steady confident pacing (approx. 130 WPM), optimistic and intellectually curious delivery."
    ),
    "satirical_hype": (
        "Narrator: Fast-paced, witty, satirical narrator, dynamic comedic cadence, crisp dry studio acoustics, "
        "punchy comedic timing with sudden ironic pauses."
    ),
    "prehistoric_comic": (
        "Narrator: Deadpan, dry comedic baritone narrator delivering absurd stone-age observations with absolute solemn seriousness."
    )
}

# ─── System Prompt ────────────────────────────────────────────────────────────

DIRECTOR_SYSTEM_PROMPT = """You are the World-Class Executive Director of Animated Stickman Videos.

YOUR CORE CONTRACT:
Transform the given source material (topic, manga chapter summary, or internet drama) into an unforgettable, high-retention short-form video in exact 10-second clip units.
Total Target Duration: <<DUR>> seconds (<<NUM_CLIPS>> clips of 10 seconds each).
Aspect Ratio: <<ASPECT>>
Visual Style: <<STYLE_NAME>>

THE 5-STAGE HIGH-COMPLETION HEARTBEAT ENGINE:
1. Golden Hook (Clip 1, 0-10s): Instant curiosity trigger via a counter-intuitive question or visual paradox. Hook the audience in the first 2-3 seconds.
2. Disrupting Assumptions (Clip 2, 10-20s): State what everyone assumes is true, then shatter it to build cognitive conflict.
3. Unveiling Insider Secrets (Clips 3-4, 20-40s): Pull back the curtain on hidden mechanics, crazy drama, or the ridiculous core truth.
4. Ultimate Truth Revelation (Clip 5, 40-50s): Deliver the underlying insight or comedic climax with maximum emotional/comedic payoff.
5. Elevation & High-Engagement Discussion (Clip 6, 50-60s): Conclude with a memorable punchline and an irresistible debate prompt for the comment section.
(If duration is 30s: compress into 3 clips. If 60s: standard 6 clips).

WORD COUNT & DIALOGUE RULES:
- Exactly 20-25 words of English voiceover per 10-second clip (approx. 2-2.5 words per second).
- Audio voiceover ONLY: strictly NO speech bubbles, NO dialogue boxes in the visuals.
- Timed visual beats per clip: `[0-3s]`, `[3-7s]`, and `[7-10s]`.

STRICT JSON OUTPUT:
Return ONLY a valid JSON object matching the requested schema. No markdown wrapping outside, no chatty conversational filler.
"""

PROPOSAL_SCHEMA = """{
  "title": "<Catchy Viral Title>",
  "target_duration_sec": <<DUR>>,
  "aspect_ratio": "<<ASPECT>>",
  "visual_style": "<<STYLE_NAME>>",
  "logline": "<One sentence premise>",
  "voice_profile": "<Selected voice profile>",
  "clips": [
    {
      "clip_index": 1,
      "duration_sec": 10,
      "stage": "Golden Hook | Disrupting Assumptions | Unveiling Insider Secrets | Ultimate Truth Revelation | Elevation & Discussion",
      "voiceover_dialogue": "<20-25 words of English speech>",
      "visual_beats": {
        "0_to_3s": "<Establish premise / action>",
        "3_to_7s": "<Escalate / transform>",
        "7_to_10s": "<Climax / transition to next clip>"
      },
      "veo_generation_prompt": "<Comprehensive standalone generation prompt matching the production contract>"
    }
  ]
}"""


def _clean_json(text: str) -> str:
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


# ─── StickmanDirector Class ───────────────────────────────────────────────────

class StickmanDirector:
    """
    Directs Stickman short-form animated video packages from:
      - Raw topics & notes
      - Manga/Comic chapter storyboards
      - Video transcripts & viral news
    """

    def __init__(self, model_name: Optional[str] = None):
        self.router = gemini_router
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def _call_gemini(self, system_instruction: str, prompt: str) -> str:
        if self.router is not None:
            return self.router.generate(
                task_type="creative_director",
                prompt=f"{system_instruction}\n\n{prompt}",
                module_name="stickman_director",
                gen_config={"temperature": 0.3, "response_mime_type": "application/json"},
            )

        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self._api_key)
        resp = client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
                response_mime_type="application/json"
            )
        )
        return resp.text or ""

    def _resolve_style_key(self, style_input: str) -> str:
        style_input = (style_input or "").lower()
        if "3" in style_input or "prehistoric" in style_input or "comedy" in style_input or "satir" in style_input:
            return "Style 3"
        if "4" in style_input or "cozy" in style_input or "story" in style_input:
            return "Style 4"
        if "2a" in style_input or "tech" in style_input or "studio" in style_input:
            return "Style 2A"
        if "2b" in style_input or "beanie" in style_input or "manga" in style_input or "anime" in style_input:
            return "Style 2B"
        return "Style 1"

    def direct_topic(
        self,
        topic_or_copy: str,
        style: str = "Style 3 (Prehistoric Dark Comedy)",
        duration_sec: int = 60,
        aspect_ratio: str = "9:16",
        voice_profile: str = "satirical_hype"
    ) -> Dict[str, Any]:
        """
        Directs a stickman animated production package from raw text or a viral topic.
        """
        # Duration clamped to nearest 10s
        duration_sec = max(30, min(180, int(round(duration_sec / 10.0) * 10)))
        num_clips = duration_sec // 10

        style_key = self._resolve_style_key(style)
        style_meta = STYLE_CATALOG.get(style_key, STYLE_CATALOG["Style 3"])
        voice_block = VOICE_CONTINUITY_PROFILES.get(voice_profile, VOICE_CONTINUITY_PROFILES["satirical_hype"])

        sys_prompt = (
            DIRECTOR_SYSTEM_PROMPT
            .replace("<<DUR>>", str(duration_sec))
            .replace("<<NUM_CLIPS>>", str(num_clips))
            .replace("<<ASPECT>>", aspect_ratio)
            .replace("<<STYLE_NAME>>", style_meta["name"])
        )

        user_prompt = f"""
SOURCE MATERIAL / TOPIC:
\"\"\"{topic_or_copy}\"\"\"

STYLE SPECIFICATIONS:
- Name: {style_meta['name']}
- Character Anchor: {style_meta['char_anchor']}
- Environment: {style_meta.get('env', style_meta.get('env_light'))}
- Negative Constraints: {style_meta.get('negative', '')}

VOICE CONTINUITY CONTRACT (repeat in every clip):
{voice_block}

Produce the complete JSON production proposal with {num_clips} clips (10s each).
Schema:
{PROPOSAL_SCHEMA.replace('<<DUR>>', str(duration_sec)).replace('<<ASPECT>>', aspect_ratio).replace('<<STYLE_NAME>>', style_meta['name'])}
"""

        logger.info(f"🎬 [StickmanDirector] Directing {duration_sec}s short in {style_meta['name']}...")
        t0 = time.time()
        raw = self._call_gemini(sys_prompt, user_prompt)
        elapsed = round(time.time() - t0, 2)

        try:
            package = json.loads(_clean_json(raw))
            package["status"] = "success"
            package["execution_time_sec"] = elapsed
            package["source_topic"] = topic_or_copy[:200]
            logger.info(f"✅ [StickmanDirector] Generated {len(package.get('clips', []))} clips in {elapsed}s.")
            return package
        except Exception as e:
            logger.error(f"❌ [StickmanDirector] JSON parse failed ({e}):\n{raw[:500]}")
            return {
                "status": "error",
                "error": str(e),
                "raw_response": raw[:1000]
            }

    def direct_from_manga(
        self,
        manga_storyboard: Dict[str, Any],
        style: str = "Style 3 (Prehistoric Dark Comedy)",
        duration_sec: int = 60,
        aspect_ratio: str = "9:16"
    ) -> Dict[str, Any]:
        """
        Directs an animated stickman recap from Manga/Comic Phase 1 storyboard data.
        """
        title = manga_storyboard.get("input_source") or "Manga Chapter"
        scenes = manga_storyboard.get("scenes") or []
        dialogues = [s.get("dialogue_summary", "") for s in scenes if s.get("dialogue_summary")]
        all_dialogue = " \n".join(dialogues[:15])

        manga_context = (
            f"Manga/Comic Chapter: {os.path.basename(title)}\n"
            f"Total Scenes Extracted: {len(scenes)} | Total Pages: {manga_storyboard.get('total_pages')}\n\n"
            f"Key Scene Dialogues & Plot Progression:\n{all_dialogue}\n"
        )

        package = self.direct_topic(
            topic_or_copy=manga_context,
            style=style,
            duration_sec=duration_sec,
            aspect_ratio=aspect_ratio,
            voice_profile="satirical_hype"
        )
        if package.get("status") == "success":
            package["source_manga"] = title
            package["manga_scenes_referenced"] = len(scenes)
        return package

    def direct_from_clip(
        self,
        clip_id: str,
        clip_folder: Optional[str] = None,
        style: str = "Style 2B (Cinematic Story)",
        duration_sec: int = 60,
        aspect_ratio: str = "9:16"
    ) -> Dict[str, Any]:
        """
        Directs a stickman sketch from an existing clip's ClipIntelligenceStore metadata.
        """
        try:
            from Gemini_Modules.clip_intelligence_store import ClipIntelligenceStore
            store = ClipIntelligenceStore()
            clip_data = store.load(clip_id, clip_folder) or {}
            visual_ctx = clip_data.get("visual_context") or {}
            transcript = (clip_data.get("audio_data") or {}).get("whisper_transcript") or {}

            topic = (
                f"Clip: {clip_id}\n"
                f"Event: {visual_ctx.get('visual_event')}\n"
                f"Hook: {visual_ctx.get('engagement_hook')}\n"
                f"Transcript: {transcript.get('full_text', '')}\n"
            )
            return self.direct_topic(topic, style=style, duration_sec=duration_sec, aspect_ratio=aspect_ratio)
        except Exception as e:
            logger.error(f"❌ [StickmanDirector] direct_from_clip failed: {e}")
            return {"status": "error", "error": str(e)}

    def export_package(self, package: Dict[str, Any], output_path: str) -> str:
        """Saves production package to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(package, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 [StickmanDirector] Production package exported to: {output_path}")
        return output_path


# ── Module-level helper ───────────────────────────────────────────────────────

_global_director: Optional[StickmanDirector] = None


def get_stickman_director() -> StickmanDirector:
    global _global_director
    if _global_director is None:
        _global_director = StickmanDirector()
    return _global_director


def direct_stickman(topic_or_source: Any, **kwargs) -> Dict[str, Any]:
    director = get_stickman_director()
    if isinstance(topic_or_source, dict) and "scenes" in topic_or_source:
        return director.direct_from_manga(topic_or_source, **kwargs)
    return director.direct_topic(str(topic_or_source), **kwargs)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    test_topic = sys.argv[1] if len(sys.argv) > 1 else "Why every superhero movie villain has the same ridiculous backstory."
    res = direct_stickman(test_topic, style="Style 3")
    print(json.dumps(res, indent=2))
