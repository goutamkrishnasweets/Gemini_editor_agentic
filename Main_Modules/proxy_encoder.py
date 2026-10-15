"""
Main_Modules/proxy_encoder.py
=============================
Unified proxy encoder + Gemini-optimized video compressor.

Two different jobs, deliberately separate:

  encode_for_gemini()   Tuned for Gemini video understanding. Used by the
                        forensic analyzer.
  encode_proxy()        General 480p proxy for the editor / auditor /
                        downloader.

Backward compatible API: encode_proxy, ensure_proxy, get_proxy_path,
encode_proxy_480p, encode_gemini_upload.

Why these settings (facts about how Gemini ingests video):
  * Gemini samples video at 1 frame/sec by default, whatever fps the file has.
    Encoding at 2 fps is WASTED unless the request also sets
    VideoMetadata(fps=2). The forensic analyzer does that. Never encode at a
    higher fps than you tell Gemini to sample.
  * Token cost is per sampled frame (~258 tokens/frame at default media
    resolution) + ~32 tokens/sec of audio. It does NOT depend on the pixel
    dimensions you upload. Shrinking resolution saves upload size/latency, not
    tokens. Frame count (fps x duration) is the only lever that saves tokens.
  * Gemini resizes frames internally (roughly to fit ~768px), so uploading
    1080p buys nothing. Short edge 480 keeps burned-in text/watermarks legible
    while staying small. Never upscale a smaller source.
  * Gemini downsamples audio to 16 kHz mono. Sending 48 kHz is dead weight.
  * Timestamps in Gemini's answer are on the PROXY timeline. The proxy must keep
    the exact same duration/timeline as the source or transcript timestamps
    (from Whisper, on the original audio) will not line up. We never trim,
    speed-change, or pad, and we verify duration drift after encoding.
  * At 1-2 fps there are very few frames to encode, so a slower x264 preset is
    nearly free and gives a smaller file.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("proxy_encoder")

# ─── General proxy defaults (encode_proxy) ───────────────────────────────────
PROXY_WIDTH = 854
PROXY_HEIGHT = 480
PROXY_CRF = 28
PROXY_PRESET = "veryfast"
AUDIO_BITRATE = "64k"

# ─── Gemini-tuned defaults (encode_for_gemini) ────────────────────────────────
GEMINI_SHORT_EDGE = 480
GEMINI_CRF = 28
GEMINI_PRESET = "medium"       # cheap at 1-2 fps, better compression than veryfast
GEMINI_AUDIO_BITRATE = "32k"
GEMINI_AUDIO_RATE = 16000      # Gemini resamples to 16 kHz mono anyway
MAX_DURATION_DRIFT_SEC = 0.75  # warn if proxy duration differs from source by more

# Token accounting (approximate, default media resolution)
TOKENS_PER_FRAME = 258
TOKENS_PER_AUDIO_SEC = 32

WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")


# ─── Binary resolution ────────────────────────────────────────────────────────

def _ffmpeg_bin() -> str:
    """Resolve FFmpeg binary path from system PATH or common Windows locations."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\tools\ffmpeg\bin\ffmpeg.exe",
    ]
    for p in common_paths:
        if os.path.isfile(p):
            return p
    raise RuntimeError(
        "[proxy_encoder] FFmpeg binary not found.\n"
        "Install via: choco install ffmpeg  OR  winget install ffmpeg"
    )


def _ffprobe_bin(ffmpeg_bin: str) -> str:
    if ffmpeg_bin in ("ffmpeg", "ffmpeg.exe"):
        return "ffprobe"
    d, name = os.path.split(ffmpeg_bin)
    ext = ".exe" if name.lower().endswith(".exe") else ""
    cand = os.path.join(d, "ffprobe" + ext)
    return cand if os.path.isfile(cand) else "ffprobe"


def _safe_remove(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _sanitize_filename(name: str) -> str:
    """Sanitize string to prevent path traversal and Windows reserved names."""
    clean = "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()
    if clean.upper() in WINDOWS_RESERVED_NAMES:
        clean = f"clip_{clean}"
    return clean or "video_clip"


# ─── Probing ──────────────────────────────────────────────────────────────────

def _parse_rate(rate: str) -> float:
    try:
        if "/" in rate:
            n, d = rate.split("/")
            return float(n) / float(d) if float(d) else 0.0
        return float(rate)
    except Exception:
        return 0.0


def probe_video(path: str, ffmpeg_bin: Optional[str] = None) -> Dict[str, Any]:
    """
    ffprobe a media file. Returns DISPLAY dimensions (rotation applied), so
    coordinates Gemini returns can be mapped back onto what a viewer sees.

    {"width","height","duration_sec","fps","has_audio","rotation","size_bytes"}
    All numeric fields are 0 on failure — callers must tolerate that.
    """
    info = {
        "width": 0, "height": 0, "duration_sec": 0.0, "fps": 0.0,
        "has_audio": False, "rotation": 0,
        "size_bytes": os.path.getsize(path) if os.path.isfile(path) else 0
    }
    try:
        ffprobe = _ffprobe_bin(ffmpeg_bin or _ffmpeg_bin())
        res = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
        )
        if res.returncode != 0 or not res.stdout.strip():
            return info
        data = json.loads(res.stdout)
        streams = data.get("streams", [])
        v = next((s for s in streams if s.get("codec_type") == "video"), None)
        info["has_audio"] = any(s.get("codec_type") == "audio" for s in streams)

        dur = 0.0
        if v:
            w, h = int(v.get("width", 0) or 0), int(v.get("height", 0) or 0)
            rot = 0
            try:
                rot = int(float((v.get("tags") or {}).get("rotate", 0)))
            except Exception:
                rot = 0
            for sd in v.get("side_data_list", []) or []:
                if "rotation" in sd:
                    try:
                        rot = int(float(sd["rotation"]))
                    except Exception:
                        pass
            if abs(rot) % 180 == 90:
                w, h = h, w
            info.update({
                "width": w, "height": h, "rotation": rot,
                "fps": round(_parse_rate(v.get("avg_frame_rate", "0")) or _parse_rate(v.get("r_frame_rate", "0")), 3)
            })
            try:
                dur = float(v.get("duration") or 0.0)
            except Exception:
                dur = 0.0
        try:
            fmt_dur = float((data.get("format") or {}).get("duration") or 0.0)
        except Exception:
            fmt_dur = 0.0
        info["duration_sec"] = round(max(dur, fmt_dur), 3)
    except Exception as e:
        logger.debug(f"[proxy_encoder] probe failed for {path}: {e}")
    return info


def _verify_video_input(ffmpeg_bin: str, input_path: str) -> float:
    """Back-compat helper: duration in seconds (0.0 on failure)."""
    return probe_video(input_path, ffmpeg_bin)["duration_sec"]


def recommend_gemini_fps(duration_sec: float) -> float:
    """
    Frames/sec to sample for a clip of this length.
    """
    if duration_sec <= 60:
        return 2.0
    if duration_sec <= 600:
        return 1.0
    return 0.5


def estimate_gemini_tokens(duration_sec: float, fps: float, has_audio: bool = True) -> int:
    return int(duration_sec * (TOKENS_PER_FRAME * fps + (TOKENS_PER_AUDIO_SEC if has_audio else 0)))


# ─── Gemini-optimized encode ──────────────────────────────────────────────────

def encode_for_gemini(
    input_path: str,
    output_dir: Optional[str] = None,
    *,
    fps: Optional[float] = None,
    short_edge: int = GEMINI_SHORT_EDGE,
    crf: int = GEMINI_CRF,
    preset: str = GEMINI_PRESET,
    audio_bitrate: str = GEMINI_AUDIO_BITRATE,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Compress a video for Gemini File API upload.

    - Aspect ratio preserved, short edge clamped to `short_edge`, NEVER upscaled,
      dimensions forced even (16:9 -> 854x480, 9:16 -> 480x854, 1:1 -> 480x480).
    - Constant frame rate = `fps` (auto by duration if None).
    - Audio: AAC 16 kHz mono (what Gemini actually uses).
    - Timeline is untouched (no trim / speed / pad); drift is measured and reported.
    - Cache key includes source size+mtime and encode params.
    """
    t_start = time.time()
    input_path = str(Path(input_path).resolve())
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"[encode_for_gemini] Video file not found: {input_path}")
    if input_path.lower().endswith(_IMAGE_EXTS):
        raise ValueError("[encode_for_gemini] Images are not videos — send images to Gemini directly.")

    orig_size = os.path.getsize(input_path)
    if orig_size == 0:
        raise ValueError(f"[encode_for_gemini] Input video file is empty: {input_path}")
    orig_size_mb = round(orig_size / (1024 * 1024), 2)

    ffmpeg = _ffmpeg_bin()
    src = probe_video(input_path, ffmpeg)
    if src["duration_sec"] <= 0 or src["width"] <= 0:
        raise RuntimeError(f"[encode_for_gemini] Unreadable video (ffprobe found no video stream/duration): {input_path}")

    if fps is None:
        fps = recommend_gemini_fps(src["duration_sec"])
    fps = float(fps)
    if fps <= 0:
        raise ValueError("fps must be > 0")
    fps_tag = f"{fps:g}"

    if output_dir is None:
        output_dir = str(Path(input_path).parent)
    os.makedirs(output_dir, exist_ok=True)

    st = os.stat(input_path)
    stamp = hashlib.sha1(f"{st.st_size}-{st.st_mtime_ns}-{crf}-{preset}-{audio_bitrate}".encode()).hexdigest()[:8]
    safe_stem = _sanitize_filename(Path(input_path).stem)
    out_path = str(Path(output_dir) / f"{safe_stem}_gemini_{fps_tag}fps_{short_edge}p_{stamp}.mp4")

    def _result(status: str, out_probe: Dict[str, Any]) -> Dict[str, Any]:
        gsize = os.path.getsize(out_path) / (1024 * 1024)
        drift = round(out_probe["duration_sec"] - src["duration_sec"], 3)
        return {
            "gemini_ready_path": out_path,
            "fps": fps,
            "width": out_probe["width"], "height": out_probe["height"],
            "source_width": src["width"], "source_height": src["height"],
            "duration_sec": out_probe["duration_sec"],
            "source_duration_sec": src["duration_sec"],
            "duration_drift_sec": drift,
            "audio_present": out_probe["has_audio"],
            "original_size_mb": orig_size_mb,
            "gemini_size_mb": round(gsize, 2),
            "size_reduction_pct": round((1 - gsize / orig_size_mb) * 100, 1) if orig_size_mb > 0 else 0.0,
            "gemini_token_est": estimate_gemini_tokens(out_probe["duration_sec"], fps, out_probe["has_audio"]),
            "elapsed_sec": round(time.time() - t_start, 2),
            "status": status,
        }

    # Cache hit check
    if os.path.isfile(out_path) and not overwrite:
        cp = probe_video(out_path, ffmpeg)
        if cp["duration_sec"] > 0 and cp["width"] > 0:
            logger.info(f"[encode_for_gemini] Cached: {os.path.basename(out_path)}")
            return _result("cached", cp)
        logger.warning("[encode_for_gemini] Cached file is corrupt — re-encoding.")
        _safe_remove(out_path)

    S = int(short_edge)
    vf = (
        f"fps={fps_tag},"
        f"scale=w='if(gt(iw,ih),-2,trunc(min(iw,{S})/2)*2)':h='if(gt(iw,ih),trunc(min(ih,{S})/2)*2,-2)':flags=lanczos,"
        f"format=yuv420p"
    )
    gop = max(2, int(round(fps * 5)))

    tmp_path = str(Path(output_dir) / f"_tmp_{uuid.uuid4().hex}.mp4")
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", vf,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-g", str(gop), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", audio_bitrate, "-ar", str(GEMINI_AUDIO_RATE), "-ac", "1",
        "-map_metadata", "-1",
        "-movflags", "+faststart",
        tmp_path,
    ]

    timeout = max(300.0, src["duration_sec"] * 2.0)
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        if res.returncode != 0:
            raise RuntimeError(f"[encode_for_gemini] FFmpeg failed (code {res.returncode}):\n{res.stderr[-500:]}")
        op = probe_video(tmp_path, ffmpeg)
        if op["duration_sec"] <= 0:
            raise RuntimeError("[encode_for_gemini] Output has zero duration — encoding failed.")
        shutil.move(tmp_path, out_path)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"[encode_for_gemini] FFmpeg timed out (>{timeout:.0f}s)")
    finally:
        if os.path.exists(tmp_path):
            _safe_remove(tmp_path)

    out = _result("encoded", op)
    if abs(out["duration_drift_sec"]) > MAX_DURATION_DRIFT_SEC:
        logger.warning(
            f"[encode_for_gemini] Duration drift {out['duration_drift_sec']:+.2f}s "
            f"(src {src['duration_sec']:.2f}s -> proxy {out['duration_sec']:.2f}s). "
            f"Transcript timestamps may not line up with the proxy."
        )
    logger.info(
        f"[encode_for_gemini] {os.path.basename(input_path)} -> {os.path.basename(out_path)} | "
        f"{out['width']}x{out['height']} @ {fps:g}fps | {orig_size_mb}MB -> {out['gemini_size_mb']}MB | "
        f"~{out['gemini_token_est']} tokens | {out['elapsed_sec']}s"
    )
    return out


def encode_gemini_upload(
    input_path: str,
    output_dir: Optional[str] = None,
    target_fps: int = 1,
    target_height: int = 480,
    crf: int = 28,
    audio_bitrate: str = GEMINI_AUDIO_BITRATE,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Back-compat wrapper over encode_for_gemini().
    """
    return encode_for_gemini(
        input_path, output_dir,
        fps=float(target_fps), short_edge=target_height, crf=crf,
        audio_bitrate=audio_bitrate, overwrite=overwrite,
    )


# ─── General proxy (behavior unchanged) ───────────────────────────────────────

def encode_proxy(
    input_path: str,
    output_path: Optional[str] = None,
    width: int = PROXY_WIDTH,
    height: int = PROXY_HEIGHT,
    crf: int = PROXY_CRF,
    preset: str = PROXY_PRESET,
    overwrite: bool = False,
    fps: Optional[int] = None,
) -> str:
    """
    Encode video to 480p proxy with dynamic aspect ratio (no letterbox black bars).
    Preserves exact function signature for backward compatibility with master_ai_editor,
    gemini_clip_auditor, and downloader_main.
    """
    input_path = str(Path(input_path).resolve())
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"[proxy_encoder] Input video not found: {input_path}")

    if output_path is None:
        stem = Path(input_path).stem
        parent = Path(input_path).parent
        output_path = str(parent / f"{stem}_proxy480p.mp4")
    output_path = str(Path(output_path).resolve())

    if os.path.isfile(output_path) and not overwrite:
        logger.info(f"[proxy_encoder] Proxy already exists, skipping encode: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ffmpeg = _ffmpeg_bin()

    target_short_edge = min(width, height) if (width > 0 and height > 0) else 480
    fps_clause = f"fps={fps}," if (fps is not None and fps > 0) else ""
    vf = (
        f"{fps_clause}"
        f"scale='if(gt(iw,ih),-2,{target_short_edge})':'if(gt(iw,ih),{target_short_edge},-2)':flags=lanczos,"
        f"format=yuv420p"
    )

    tmp_path = str(Path(output_path).parent / f"_tmp_{uuid.uuid4().hex}.mp4")
    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        "-ar", "48000",
        "-ac", "1",
        "-movflags", "+faststart",
        tmp_path,
    ]

    logger.info(f"[proxy_encoder] Encoding proxy: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        if res.returncode != 0:
            _safe_remove(tmp_path)
            raise RuntimeError(f"[proxy_encoder] FFmpeg failed (code {res.returncode}):\n{res.stderr[-500:]}")
        shutil.move(tmp_path, output_path)
    except subprocess.TimeoutExpired:
        _safe_remove(tmp_path)
        raise RuntimeError(f"[proxy_encoder] FFmpeg timed out (>300s) for {input_path}")

    proxy_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"[proxy_encoder] ✓ Proxy encoded: {os.path.basename(output_path)} ({proxy_size_mb:.2f} MB)")
    return output_path


def get_proxy_path(raw_video_path: str) -> str:
    """Return expected proxy path for a given raw video."""
    stem = Path(raw_video_path).stem
    parent = Path(raw_video_path).parent
    return str(parent / f"{stem}_proxy480p.mp4")


def ensure_proxy(raw_video_path: str, **kwargs) -> str:
    """Return existing proxy if found, else encode and return new proxy."""
    proxy_path = get_proxy_path(raw_video_path)
    if os.path.isfile(proxy_path):
        logger.info(f"[proxy_encoder] Using cached proxy: {os.path.basename(proxy_path)}")
        return proxy_path
    return encode_proxy(raw_video_path, output_path=proxy_path, overwrite=False, **kwargs)


# Backward compatibility alias
encode_proxy_480p = encode_proxy


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Proxy encoder / Gemini-optimized video compressor")
    parser.add_argument("input", type=str, help="Path to raw video file")
    parser.add_argument("--gemini", action="store_true", help="Use the Gemini-tuned encode (encode_for_gemini)")
    parser.add_argument("--output", type=str, default=None, help="Output proxy path (general proxy only)")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS (Gemini mode: auto by duration if omitted)")
    parser.add_argument("--crf", type=int, default=None, help="CRF quality")
    parser.add_argument("--overwrite", action="store_true", help="Re-encode even if output exists")
    args = parser.parse_args()

    if args.gemini:
        info = encode_for_gemini(
            args.input, fps=args.fps,
            crf=args.crf if args.crf is not None else GEMINI_CRF,
            overwrite=args.overwrite,
        )
        print(json.dumps(info, indent=2))
    else:
        out = encode_proxy(
            input_path=args.input, output_path=args.output,
            crf=args.crf if args.crf is not None else PROXY_CRF,
            overwrite=args.overwrite,
            fps=int(args.fps) if args.fps else None,
        )
        print(f"[OK] Proxy ready: {out}")
