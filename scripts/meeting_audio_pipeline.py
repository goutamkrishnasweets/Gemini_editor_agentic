#!/usr/bin/env python3
"""
meeting_audio_pipeline.py — Automated Meeting Recording & LLM Transcription Pipeline

Workflow:
1. Records System Audio (+ optional Microphone) using global FFmpeg with a vocal bandpass filter.
2. Runs Faster-Whisper via Audio_Modules.speech_boundary_detector to extract word/sentence timestamps.
3. Sends audio WAV + timestamped transcript to Gemini via Gemini_Modules.lyric_rhythm_aligner.
4. Saves a structured meeting_transcript_<timestamp>.json file optimized for LLM/AI context windows.

Usage:
    python scripts/meeting_audio_pipeline.py --duration 30
    python scripts/meeting_audio_pipeline.py  (Press Ctrl+C to stop recording)
"""

import argparse
import datetime
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Optional

# Setup Pathing & Logging
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("meeting_audio_pipeline")


class FFmpegAudioRecorder:
    """Handles global FFmpeg system & microphone audio recording on Windows."""

    def __init__(
        self,
        output_wav: str,
        mic_device: Optional[str] = None,
        sample_rate: int = 16000,
        enable_vocal_filter: bool = True
    ):
        self.output_wav = output_wav
        self.mic_device = mic_device
        self.sample_rate = sample_rate
        self.enable_vocal_filter = enable_vocal_filter
        self.process: Optional[subprocess.Popen] = None

    def build_ffmpeg_command(self) -> list:
        cmd = ["ffmpeg", "-y", "-loglevel", "warning"]

        # Input 1: WASAPI Default System Loopback (Captures Meeting Speakers / Other Participants)
        cmd.extend(["-f", "wasapi", "-i", "default"])

        # Input 2: Microphone (if specified)
        if self.mic_device:
            cmd.extend(["-f", "dshow", "-i", f"audio={self.mic_device}"])
            # Mix System Audio + Mic Audio into single channel
            if self.enable_vocal_filter:
                filter_graph = "[0:a][1:a]amix=inputs=2:duration=first[aout];[aout]highpass=f=120,lowpass=f=3800,speechnorm[filtered]"
                cmd.extend(["-filter_complex", filter_graph, "-map", "[filtered]"])
            else:
                filter_graph = "[0:a][1:a]amix=inputs=2:duration=first[aout]"
                cmd.extend(["-filter_complex", filter_graph, "-map", "[aout]"])
        else:
            # Single Source (System Audio Loopback) with Vocal Pass Filter
            if self.enable_vocal_filter:
                cmd.extend(["-af", "highpass=f=120,lowpass=f=3800,speechnorm"])

        # Mono 16kHz WAV output (Optimal for Faster-Whisper ASR)
        cmd.extend([
            "-ac", "1",
            "-ar", str(self.sample_rate),
            self.output_wav
        ])
        return cmd

    def start_recording(self):
        cmd = self.build_ffmpeg_command()
        logger.info("🎙️ Starting FFmpeg Audio Recording...")
        logger.info("  └─ Command: %s", " ".join(cmd))
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def stop_recording(self):
        if not self.process:
            return

        logger.info("🛑 Stopping FFmpeg Audio Recording...")
        try:
            # Send 'q' to FFmpeg stdin for clean WAV container closing
            if self.process.stdin:
                self.process.stdin.write(b"q")
                self.process.stdin.flush()
            self.process.wait(timeout=5)
        except Exception:
            logger.warning("FFmpeg did not close gracefully on 'q', sending terminate...")
            self.process.terminate()
            self.process.wait(timeout=3)
        finally:
            self.process = None
        logger.info("✅ WAV file saved to: %s", self.output_wav)


def format_llm_prompt(whisper_data: Dict[str, Any], gemini_data: Dict[str, Any], duration_sec: float) -> str:
    """Formats the extracted transcript and Gemini analysis into a ready-to-paste LLM prompt string."""
    sentences = whisper_data.get("sentences", [])
    transcript_lines = []

    for s in sentences:
        start_fmt = f"{int(s.get('start', 0) // 60):02d}:{int(s.get('start', 0) % 60):02d}"
        end_fmt = f"{int(s.get('end', 0) // 60):02d}:{int(s.get('end', 0) % 60):02d}"
        text = s.get("text", "").strip()
        transcript_lines.append(f"[{start_fmt} - {end_fmt}]: {text}")

    full_transcript_str = "\n".join(transcript_lines) if transcript_lines else "No spoken text detected."

    prompt = f"""### MEETING AUDIO TRANSCRIPT & INTELLIGENCE REPORT
**Duration:** {duration_sec:.1f} seconds
**Dominant Emotion/Vibe:** {gemini_data.get('dominant_emotion', 'neutral')}
**Language:** {gemini_data.get('language', 'Auto-detected')}

---
### FULL TIMESTAMPED TRANSCRIPT (AUTHORITATIVE ASR):
{full_transcript_str}

---
### GEMINI SEMANTIC SUMMARY:
{json.dumps(gemini_data.get('sections', []), indent=2)}

---
### INSTRUCTION FOR DOWNSTREAM LLM:
Please review the transcript above and provide:
1. Executive Summary & Key Highlights
2. Action Items & Assigned Tasks
3. Main Discussion Points by Topic
"""
    return prompt


def run_meeting_pipeline(
    output_dir: str = "data/recordings",
    duration: Optional[int] = None,
    mic_device: Optional[str] = None,
    model_size: str = "tiny"
) -> str:
    """Executes the full recording -> whisper -> gemini -> JSON export workflow."""

    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = f"meeting_{timestamp_str}.wav"
    json_filename = f"meeting_transcript_{timestamp_str}.json"

    wav_path = os.path.join(output_dir, wav_filename)
    json_path = os.path.join(output_dir, json_filename)

    recorder = FFmpegAudioRecorder(
        output_wav=wav_path,
        mic_device=mic_device,
        enable_vocal_filter=True
    )

    t_start = time.time()
    recorder.start_recording()

    print("\n" + "=" * 60)
    print(" 🎙️ RECORDING MEETING / SYSTEM AUDIO...")
    if duration:
        print(f" ⏳ Recording will stop automatically after {duration} seconds.")
    else:
        print(" 💡 Press Ctrl+C to STOP recording and start AI transcription.")
    print("=" * 60 + "\n")

    try:
        if duration:
            time.sleep(duration)
        else:
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\n⏹️ Stop signal received. Processing recording...")
    finally:
        recorder.stop_recording()

    t_end = time.time()
    recording_duration = round(t_end - t_start, 2)

    if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
        logger.error("❌ Audio recording failed or produced an empty file.")
        return ""

    # ── Step 2: Faster-Whisper Speech Boundary Extraction ────────────────────
    logger.info("🎙️ Step 1/2: Running Faster-Whisper ASR timestamp extraction...")
    from Audio_Modules.speech_boundary_detector import extract_speech_boundaries

    whisper_res = extract_speech_boundaries(
        audio_wav_path=wav_path,
        output_json_path=os.path.join(output_dir, f"whisper_{timestamp_str}.json"),
        model_size=model_size
    )

    # ── Step 3: Gemini Multimodal Audio Perception ────────────────────────────
    logger.info("🧠 Step 2/2: Running Gemini Multimodal Audio Intelligence...")
    from Gemini_Modules.lyric_rhythm_aligner import analyze_music

    gemini_res = analyze_music(
        audio_path=wav_path,
        cache_dir=output_dir
    )

    # ── Step 4: Export Structured JSON for LLMs ──────────────────────────────
    llm_prompt = format_llm_prompt(
        whisper_data=whisper_res,
        gemini_data=gemini_res,
        duration_sec=recording_duration
    )

    payload = {
        "meeting_metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "recording_duration_sec": recording_duration,
            "audio_file": os.path.abspath(wav_path),
            "whisper_model": model_size
        },
        "whisper_transcript": whisper_res,
        "gemini_analysis": gemini_res,
        "llm_ready_prompt": llm_prompt
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("✅ Successful! Meeting JSON transcript saved to:")
    logger.info("   📄 %s", os.path.abspath(json_path))
    return os.path.abspath(json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meeting Audio Recorder & LLM Transcription Pipeline")
    parser.add_argument("--duration", type=int, default=None, help="Recording duration in seconds (optional).")
    parser.add_argument("--output-dir", type=str, default="data/recordings", help="Output directory for WAV & JSON.")
    parser.add_argument("--mic-device", type=str, default=None, help="Name of microphone input device for dual recording.")
    parser.add_argument("--model-size", type=str, default="tiny", help="Faster-Whisper model size (tiny, base, small, medium, large-v3).")

    args = parser.parse_args()
    run_meeting_pipeline(
        output_dir=args.output_dir,
        duration=args.duration,
        mic_device=args.mic_device,
        model_size=args.model_size
    )
