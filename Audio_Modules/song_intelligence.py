"""
Audio_Modules/song_intelligence.py
===================================
Instant Song Intelligence & Cultural Entity Enrichment for Audio Tracks.

Pipeline:
  1. Shazam Audio Math Fingerprint: Identifies track title, artist, genre from raw WAV
     using shazamio with clean socket teardown (force_close=True, zero lingering SSL transports).
  2. Decoupled Deep Entity Enrichment: Instant text-only Gemini query (~300ms, zero audio upload)
     retrieving official movie/album, music video cast actors, singers, director, and keywords.
  3. Direct Persistence: Saves shazam_info & song_entities directly to audio_analysis.json
     and pool_metadata.json during Phase 1 audio extraction.
"""

import os
import sys
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("AMTE.song_intelligence")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def recognize_track_with_shazam(wav_path: str, timeout_sec: float = 7.0) -> Dict[str, Any]:
    """
    Identifies track title, artist, and primary genre using shazamio audio fingerprinting.
    Uses a clean TCPConnector with force_close=True to prevent lingering keepalive SSL sockets.
    """
    if not wav_path or not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1024:
        return {}

    try:
        import asyncio
        import aiohttp
        from shazamio import Shazam
        from shazamio.interfaces.client import HTTPClientInterface
        from shazamio.utils import validate_json

        class CleanShazamHTTPClient(HTTPClientInterface):
            def __init__(self, session: aiohttp.ClientSession):
                self._session = session

            async def request(self, method: str, url: str, *args, **kwargs) -> Any:
                for attempt in range(2):
                    try:
                        if method.upper() == "POST":
                            async with self._session.post(url, **kwargs) as resp:
                                return await validate_json(resp, *args)
                        elif method.upper() == "GET":
                            async with self._session.get(url, **kwargs) as resp:
                                return await validate_json(resp, *args)
                        else:
                            raise ValueError(f"Unsupported method: {method}")
                    except Exception as exc:
                        if attempt == 0:
                            await asyncio.sleep(0.3)
                            continue
                        raise exc

        async def _async_recognize():
            connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
            timeout = aiohttp.ClientTimeout(total=max(3.0, timeout_sec - 0.5), connect=3.0)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                custom_client = CleanShazamHTTPClient(session)
                shazam = Shazam(http_client=custom_client)
                out = await shazam.recognize(wav_path)
                track = out.get("track", {}) if isinstance(out, dict) else {}
                title = track.get("title", "").strip()
                artist = track.get("subtitle", "").strip()
                genre = track.get("genres", {}).get("primary", "").strip() if isinstance(track.get("genres"), dict) else ""
                shazam_url = track.get("url", "").strip()

                return {
                    "title": title,
                    "artist": artist,
                    "genre": genre,
                    "shazam_url": shazam_url,
                    "is_recognized": bool(title),
                }

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(asyncio.wait_for(_async_recognize(), timeout=timeout_sec))
            if res and res.get("title"):
                logger.info(f"🎵 [SHAZAM FINGERPRINT] Recognized track: '{res['title']}' by {res['artist']} ({res.get('genre') or 'Music'})")
                return res
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug(f"[SHAZAM] Recognition completed or notice: {e}")
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
                if hasattr(loop, "shutdown_default_executor"):
                    loop.run_until_complete(loop.shutdown_default_executor())
                loop.run_until_complete(asyncio.sleep(0.1))
            except Exception:
                pass
            finally:
                loop.set_exception_handler(lambda _l, _c: None)
                try:
                    asyncio.set_event_loop(None)
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass

    except Exception as e:
        logger.debug(f"[SHAZAM] Recognition bypassed/notice: {e}")

    return {}


def fetch_song_deep_entities(shazam_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decoupled text-only factual lookup for song cultural entities:
    - movie_or_album: official source film/show/album
    - cast_members: actors who appeared in the music video / visual song
    - singers: vocalists
    - music_director / composers
    - director: video or movie director
    - search_keywords: aliases, actor names, movie titles, viral tags
    """
    title = (shazam_info.get("title") or "").strip()
    artist = (shazam_info.get("artist") or "").strip()
    genre = (shazam_info.get("genre") or "").strip()

    if not title:
        return {}

    prompt = f"""You are a musical and cinematic knowledge engine.
Given this officially recognized song:
  Track Title: "{title}"
  Artist / Composer: "{artist}"
  Primary Genre: "{genre}"

Provide deep, factual information about the song and its visual music video:
1. "movie_or_album": Official movie, web series, album, or original music video title.
2. "cast_members": List of prominent actors/actresses who performed or appeared in the official visual music video / movie song (e.g. Kartik Aaryan, Sunny Singh, Sakshi Malik).
3. "singers": List of main vocalists who sang the track.
4. "music_director": List of music composers / producers.
5. "director": Director of the movie or music video.
6. "visual_theme": Concise phrase describing the visual vibe (e.g. "Club party dance high energy", "Romantic rainy street walk", "Wedding celebration").
7. "search_keywords": List of all actor names, aliases, movie titles, and viral phrases associated with this track.

Return ONLY a valid JSON object matching this schema with NO commentary or markdown codeblocks:
{{
  "title": "{title}",
  "movie_or_album": "",
  "cast_members": [],
  "singers": [],
  "music_director": [],
  "director": "",
  "visual_theme": "",
  "search_keywords": []
}}
"""

    entities: Dict[str, Any] = {
        "title": title,
        "movie_or_album": "",
        "cast_members": [],
        "singers": [artist] if artist else [],
        "music_director": [],
        "director": "",
        "visual_theme": "",
        "search_keywords": [title, artist] if artist else [title],
    }

    try:
        raw_text = ""
        # 1. Try gemini_router first
        try:
            from Gemini_Modules.gemini_router_module import gemini_router
            raw_text = gemini_router.generate(
                task_type="creative",
                prompt=prompt,
                module_name="song_intelligence",
                gen_config={"temperature": 0.1, "max_output_tokens": 800},
            )
        except Exception:
            pass

        # 2. Fallback to direct Google GenAI Client
        if not raw_text:
            api_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
            if api_key:
                try:
                    from google import genai
                    from google.genai import types
                    client = genai.Client(api_key=api_key)
                    resp = client.models.generate_content(
                        model=os.getenv("FAST_GEMINI_MODEL", "gemini-2.5-flash"),
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            response_mime_type="application/json",
                        ),
                    )
                    raw_text = resp.text
                except Exception:
                    pass

        # 3. Fallback to legacy google.generativeai
        if not raw_text:
            try:
                import google.generativeai as gai
                gai_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
                if gai_key:
                    gai.configure(api_key=gai_key)
                    model = gai.GenerativeModel("gemini-2.5-flash")
                    resp = model.generate_content(prompt)
                    raw_text = resp.text
            except Exception:
                pass

        if raw_text:
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                entities.update({
                    "title": parsed.get("title") or title,
                    "movie_or_album": str(parsed.get("movie_or_album") or "").strip(),
                    "cast_members": [str(c).strip() for c in parsed.get("cast_members", []) if c],
                    "singers": [str(s).strip() for s in parsed.get("singers", []) if s] or entities["singers"],
                    "music_director": [str(m).strip() for m in parsed.get("music_director", []) if m],
                    "director": str(parsed.get("director") or "").strip(),
                    "visual_theme": str(parsed.get("visual_theme") or "").strip(),
                    "search_keywords": list(dict.fromkeys(
                        [w.strip() for w in parsed.get("search_keywords", []) if w] + entities["search_keywords"]
                    )),
                })
                logger.info(
                    f"🎬 [SONG ENTITIES] '{title}' ➔ Movie: '{entities['movie_or_album']}' | "
                    f"Cast: {entities['cast_members'][:4]} | Singers: {entities['singers'][:3]}"
                )

    except Exception as e:
        logger.debug(f"[SONG ENTITIES] Notice during entity enrichment: {e}")

    return entities


def enrich_audio_with_song_intelligence(wav_path: str, clip_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Complete Phase 1 Audio Enrichment:
      1. Shazam fingerprint
      2. Decoupled deep entity query (cast, movie, director)
      3. Persist to audio_analysis.json and pool_metadata.json
    """
    if not wav_path or not os.path.exists(wav_path):
        return {}

    shazam_info = recognize_track_with_shazam(wav_path)
    song_entities = fetch_song_deep_entities(shazam_info) if shazam_info.get("title") else {}

    result = {
        "shazam_info": shazam_info,
        "song_entities": song_entities,
    }

    if not clip_dir or not os.path.isdir(clip_dir):
        return result

    # 1. Update audio_analysis.json
    analysis_path = os.path.join(clip_dir, "audio_analysis.json")
    if os.path.exists(analysis_path):
        try:
            with open(analysis_path, "r", encoding="utf-8") as f:
                adata = json.load(f)
            adata["shazam_info"] = shazam_info
            adata["song_entities"] = song_entities
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(adata, f, indent=2, ensure_ascii=False)
            logger.info("💾 [SONG INTEL] Updated audio_analysis.json with song entities.")
        except Exception as _ae:
            logger.debug(f"[SONG INTEL] Notice updating audio_analysis.json: {_ae}")

    # 2. Sync into pool_metadata.json
    try:
        from Audio_Modules.audio_pool_manager import AudioPoolManager
        from Telegram_Storage_Modules.telegram_vault_indexer import extract_clean_shortcode
        pm = AudioPoolManager()
        folder_name = os.path.basename(clip_dir)
        sc = extract_clean_shortcode(folder_name) or folder_name.replace("manual_", "").strip()

        with pm.lock:
            files_root = pm.metadata.setdefault("files", {})
            clips = files_root.setdefault("social_media_id", {})
            updated = False
            for k, entry in clips.items():
                if isinstance(entry, dict):
                    entry_sc = str(entry.get("shortcode") or "")
                    if (sc and entry_sc.lower() == sc.lower()) or (sc and sc.lower() in k.lower()):
                        a_data = entry.setdefault("audio_data", {})
                        a_data["shazam_info"] = shazam_info
                        a_data["song_entities"] = song_entities
                        entry["shazam_info"] = shazam_info
                        entry["song_entities"] = song_entities
                        updated = True
            if updated:
                pm._save_metadata()
                logger.info(f"📦 [SONG INTEL] Synced song intelligence to pool_metadata.json for shortcode '{sc}'")
    except Exception as _pme:
        logger.debug(f"[SONG INTEL] Notice syncing to pool_metadata: {_pme}")

    return result
