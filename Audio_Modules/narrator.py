# narrator.py - Gemini TTS (primary) + edge-tts (fallback) with Audio Normalization
import sys
import os
import asyncio
import base64
import subprocess
import wave
from typing import Optional

import logging as _logging
_logging.getLogger("asyncio").setLevel(_logging.WARNING)  # silence IocpProactor spam

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    edge_tts = None
import html
import re
try:
    from config import TEMP_DIR
except ImportError:
    TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
    os.makedirs(TEMP_DIR, exist_ok=True)


class TTSEngine:
    # Map of tricky words to their phonetic spellings for edge-tts
    PRONUNCIATION_MAP = {
        "Dasei": "Dah-say",
        "York": "Yoh-rk",
        "Saturn": "Sat-urn",
        "Warcury": "War-cure-ee",
        "Nusjuro": "Nus-ju-roh",
        "Ju Peter": "Ju-Peter",
        "Mars": "Mar-ss",
        "Five Elders": "Five Elderrs",
        "Gorosei": "Goh-roh-say",
        "Vegapunk": "Vay-ga-punk",
        "Egghead": "Egg-head",
        "Haki": "Hah-kee"
    }

    def __init__(self):

        pass

    def _sanitize_text(self, text: str) -> str:
        """Sanitizes text for edge-tts (SSML) and removes illegal characters."""
        import unicodedata
        if not text:
            return ""

        # 0a. Unicode NFC normalisation — Malayalam has multiple code-point forms
        #     for visually identical glyphs; NFC gives TTS the canonical form.
        text = unicodedata.normalize('NFC', text)

        # 0b. Strip ALL bracketed stage-directions that must not be spoken:
        #     [PAUSE], [PAUSE 1.2s], [IMPACT SFX], [WARNING], [SFX:...], [COUNTDOWN], etc.
        text = re.sub(r'\[[^\]]{0,60}\]', '', text, flags=re.IGNORECASE).strip()

        # 0c. Strip bare countdown digits ("3", "2", "1") on their own
        text = re.sub(r'(?<![\w\u0D00-\u0D7F])\d{1,2}(?![\w\u0D00-\u0D7F])', '', text).strip()

        # 1. Strip non-printable characters
        text = "".join(c for c in text if c.isprintable())

        # 2. Unescape any pre-existing HTML entities (e.g. &#x27;, &quot;, &amp;)
        # CRITICAL: DO NOT use html.escape() here! edge_tts.Communicate takes plain text.
        # Calling html.escape() turns quotes into '&#x27;', which Edge-TTS literally speaks
        # as 'nna irupathiezhu' (Malayalam for 27).
        text = html.unescape(text)
        # Strip remaining quotes and XML entity remnants
        text = re.sub(r'&#[xX]?[0-9a-fA-F]+;', '', text)
        text = re.sub(r'[\'\"`“”‘’]', '', text)

        # 3. Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # 4. Guard: must contain at least one real letter or digit
        if not any(c.isalnum() for c in text):
            return ""

        # 5. Apply Pronunciation Map (character names, proper nouns)
        for word, phonetic in self.PRONUNCIATION_MAP.items():
            text = re.sub(rf'\b{re.escape(word)}\b', phonetic, text, flags=re.IGNORECASE)

        return text

    async def _amake_voiceover(self, text: str, voice: str, output_path: str, rate: str = "+0%", pitch: str = "+0Hz"):
        # Ensure rate/pitch have sign
        if not (rate.startswith('+') or rate.startswith('-')):
            rate = f"+{rate}"
        if not (pitch.startswith('+') or pitch.startswith('-')):
            pitch = f"+{pitch}"
            
        # Auto-detect Malayalam script and assign native Malayalam voice if default voice is non-Malayalam
        if re.search(r'[\u0D00-\u0D7F]', text) and not voice.startswith("ml-IN"):
            voice = "ml-IN-MidhunNeural"

        # Sanitization
        text = self._sanitize_text(text)
        if not text or len(text.strip()) < 2:
            print(f"⚠️ Skipping TTS: Text is empty or too short.")
            # Create a silent 1s file as placeholder to prevent downstream crashes
            cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "1", output_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return

        # Fast Retry Logic for Stability (Handle 503/Rate Limits without long blocking)
        max_retries = 2
        base_wait = 0.5
        
        try:
            import edge_tts
        except ImportError:
            print("⚠️ edge_tts module not installed — skipping Azure cloud TTS.")
            return

        for attempt in range(max_retries):
            try:
                communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
                await communicate.save(output_path)
                return # Success
            except Exception as e:
                print(f"⚠️ TTS Error (Attempt {attempt+1}/{max_retries}): {e}")
                if "No audio was received" in str(e):
                    print(f"❌ Critical TTS Failure. Text snippet: {text[:50]}...")
                
                if attempt < max_retries - 1:
                    import random
                    wait_time = (base_wait * (attempt + 1)) + random.uniform(0.1, 0.5)
                    print(f"⏳ Waiting {wait_time:.2f}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    print("❌ TTS Failed after max retries — using clean fallback.")
                    raise e

    def _safe_remove(self, path: str):
        """Safe remove with retry for Windows file locking issues."""
        import time
        if not os.path.exists(path):
            return
            
        for i in range(5):
            try:
                os.remove(path)
                return
            except PermissionError:
                if i < 4:
                    time.sleep(0.5)
                else:
                    print(f"⚠️ Could not delete temp file: {path}")
            except Exception as e:
                print(f"⚠️ Error deleting {path}: {e}")
                return

    def _apply_audio_processing(self, input_path: str, output_path: str):
        """
        Applies YouTube-safe loudness normalisation, compression, and silence padding.
        Falls back to a lighter chain (no loudnorm) if the file is silent/too short,
        then falls back to a raw copy if even that fails.
        """
        import shutil

        # Detect silent placeholder: loudnorm fails on pure silence.
        # A 1-second silent MP3 created by anullsrc is typically < 4 KB.
        try:
            _size = os.path.getsize(input_path) if os.path.exists(input_path) else 0
        except OSError:
            _size = 0
        _is_silent_placeholder = _size < 8_000  # < 8 KB → treat as silent

        if _is_silent_placeholder:
            # Just copy — no point running loudnorm on silence
            try:
                shutil.copy2(input_path, output_path)
                return True
            except Exception:
                return False

        # Full chain for real speech
        effects = [
            "apad=pad_dur=0.3",
            "acompressor=threshold=-14dB:ratio=2.5:attack=20:release=200",
            "alimiter=limit=0.95",
            "loudnorm=I=-14:TP=-1.5:LRA=11",
        ]
        cmd = ["ffmpeg", "-y", "-i", input_path,
               "-af", ",".join(effects), "-ac", "2", "-ar", "44100", output_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            # Decode error — skip the FFmpeg version banner (first non-empty content lines)
            raw_err = e.stderr.decode(errors="replace")
            real_lines = [
                l for l in raw_err.splitlines()
                if l.strip() and not l.strip().startswith(("ffmpeg version", "built", "configuration", "lib", "Copyright"))
            ]
            print(f"⚠️ Audio processing failed: {' | '.join(real_lines[:3])}")

        # Fallback 1: lighter chain without loudnorm
        simple_cmd = ["ffmpeg", "-y", "-i", input_path,
                      "-af", "acompressor=threshold=-14dB:ratio=2.5:attack=20:release=200,alimiter=limit=0.95",
                      "-ac", "2", "-ar", "44100", output_path]
        try:
            subprocess.run(simple_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            pass

        # Fallback 2: raw copy (preserves audio even if no processing applied)
        try:
            shutil.copy2(input_path, output_path)
            return True
        except Exception:
            return False

    GEMINI_TTS_VOICES = {
        "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
        "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
        "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
        "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
        "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
    }

    def _resolve_gemini_voice(self, voice: Optional[str]) -> str:
        """Resolves a Gemini prebuilt voice name (maps Edge-TTS voice names cleanly)."""
        env_voice = os.getenv("GEMINI_TTS_VOICE", "Kore").strip() or "Kore"
        if not voice:
            return env_voice
        clean = str(voice).strip()
        for v in self.GEMINI_TTS_VOICES:
            if clean.lower() == v.lower():
                return v
        # Map common male/female Edge-TTS voice names to natural Gemini voices
        lower = clean.lower()
        if any(m in lower for m in ("midhun", "guy", "andrew", "brian", "christopher", "male", "davis", "tony")):
            return "Puck"
        return env_voice

    def _generate_gemini_tts(
        self,
        text: str,
        output_path: str,
        voice_name: Optional[str] = None,
        tone: str = "confident",
        api_key: Optional[str] = None,
    ) -> bool:
        """
        PRIMARY TTS engine: generates voiceover via Google Gemini TTS
        (gemini-2.5-flash-preview-tts) using google-genai, converting raw 24kHz
        PCM output into a valid WAV/MP3 file.
        """
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            return False

        clean_text = self._sanitize_text(text)
        if not clean_text or len(clean_text.strip()) < 2:
            return False

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return False

        model_name = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts").strip()
        gemini_voice = self._resolve_gemini_voice(voice_name)
        prompt_text = (
            f"Read the following script in a {tone}, natural short-form voiceover style:\n\n"
            f"{clean_text}"
        )

        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=gemini_voice,
                            )
                        )
                    ),
                ),
            )

            audio_data = None
            mime_type = ""
            if hasattr(resp, "candidates") and resp.candidates:
                for cand in resp.candidates:
                    parts = getattr(getattr(cand, "content", None), "parts", None) or []
                    for part in parts:
                        inline = getattr(part, "inline_data", None)
                        if inline and getattr(inline, "data", None):
                            audio_data = inline.data
                            mime_type = getattr(inline, "mime_type", "") or ""
                            break
                    if audio_data:
                        break

            if not audio_data:
                return False

            if isinstance(audio_data, str):
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = bytes(audio_data)

            if len(audio_bytes) < 500:
                return False

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            mime_lower = mime_type.lower()
            is_wav_or_mp3 = (
                audio_bytes.startswith(b"RIFF")
                or audio_bytes.startswith(b"ID3")
                or audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
                or "mpeg" in mime_lower
                or "mp3" in mime_lower
            )
            is_pcm = ("pcm" in mime_lower or "l16" in mime_lower or not is_wav_or_mp3)

            if is_pcm:
                rate_match = re.search(r"rate=(\d+)", mime_lower)
                sample_rate = int(rate_match.group(1)) if rate_match else 24000
                tmp_wav = output_path if output_path.lower().endswith(".wav") else f"{output_path}.gemini_pcm.wav"
                with wave.open(tmp_wav, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_bytes)
                if tmp_wav != output_path:
                    conv = subprocess.run(
                        ["ffmpeg", "-y", "-i", tmp_wav, "-codec:a", "libmp3lame", "-q:a", "2", output_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    self._safe_remove(tmp_wav)
                    if conv.returncode != 0:
                        return False
            else:
                with open(output_path, "wb") as af:
                    af.write(audio_bytes)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 500:
                print(f"🎙️ [GEMINI TTS] Primary voiceover generated ({gemini_voice}, {os.path.getsize(output_path)//1024}KB) -> {output_path}")
                return True
            return False
        except Exception as e:
            print(f"⚠️ [GEMINI TTS] Failed ({e}) — falling back to secondary TTS.")
            return False

    def _generate_pocket_tts(self, text: str, output_path: str) -> bool:
        """Generates voiceover using Kyutai pocket-tts if voice sample is provided."""
        try:
            from config import ENABLE_POCKET_TTS, POCKET_TTS_VOICE_SAMPLE
            if not ENABLE_POCKET_TTS:
                return False

            sample_file = POCKET_TTS_VOICE_SAMPLE
            if not os.path.exists(sample_file):
                return False

            print(f"🎙️ [POCKET-TTS CLONING] Generating zero-shot voiceover from '{os.path.basename(sample_file)}'...")
            cmd = [
                "pocket-tts", "generate",
                "--text", text,
                "--output", output_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✨ [POCKET-TTS SUCCESS] Voiceover saved -> {output_path}")
                return True
            return False
        except Exception:
            return False

    def generate_voiceover(
        self,
        text: str,
        voice: str = "Kore",
        filename: str = "voiceover.mp3",
        tone: str = "confident",
        api_key: Optional[str] = None,
    ) -> str:
        """
        Generates VO using Gemini TTS (PRIMARY), Pocket-TTS (optional), or Edge-TTS (FALLBACK)
        and applies loudness normalization.
        """
        os.makedirs(TEMP_DIR, exist_ok=True)
        raw_path = os.path.join(TEMP_DIR, f"raw_{filename}")
        processed_path = os.path.join(TEMP_DIR, filename)

        # 1. PRIMARY: Gemini TTS (via google-genai)
        tts_success = self._generate_gemini_tts(
            text, raw_path, voice_name=voice, tone=tone, api_key=api_key
        )

        # 2. Optional: Kyutai pocket-tts zero-shot voice cloning
        if not tts_success:
            tts_success = self._generate_pocket_tts(text, raw_path)

        # 3. FALLBACK: Edge-TTS (Microsoft Azure Neural)
        if not tts_success:
            try:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError("loop closed")
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(self._amake_voiceover(text, voice, raw_path))
            except Exception as e:
                print(f"⚠️ [TTS] edge-tts notice: {e}")

        # Ensure raw_path exists and is non-empty
        if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
            print(f"🔊 Generating clean fallback audio track for: {filename}")
            cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "5.0", raw_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 4. Apply Normalization/Improvement
        if self._apply_audio_processing(raw_path, processed_path):
            self._safe_remove(raw_path)
            return processed_path

        return raw_path if (os.path.exists(raw_path) and os.path.getsize(raw_path) > 0) else processed_path

    def generate_narration(self, text: str, filename: str) -> str:
        """
        Generates narration using Edge-TTS Malayalam voiceover engine (ml-IN-MidhunNeural).
        """
        output_path = os.path.join(TEMP_DIR, filename)
        os.makedirs(TEMP_DIR, exist_ok=True)
        print(f"🎙️ [Malayalam TTS Engine] Synthesizing voiceover for: {filename}")

        from config import NARRATOR_VOICE
        return self.generate_voiceover(text, NARRATOR_VOICE, filename)
    
    def _apply_narrator_processing(self, input_path: str, output_path: str, volume: float) -> bool:
        """
        Narrator-specific audio processing.
        Optimized for clear, authoritative narration.
        """
        # Narrator-optimized loudness normalization
        loudnorm_filter = "loudnorm=I=-16:TP=-1.5:LRA=11"
        
        # Narrator Effects Chain: Clarity + Presence + Authority
        effects = [
            "acompressor=threshold=-18dB:ratio=3:attack=5:release=150",  # Smooth compression
            "alimiter=limit=0.95",                  # Safety ceiling
            loudnorm_filter,
            f"volume={volume}"                      # Apply configured volume
        ]
        
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", ",".join(effects),
            "-ac", "2", "-ar", "44100",
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Narrator audio processing failed: {e.stderr.decode()[:100]}")
            return False
