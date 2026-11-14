"""FFmpeg / ffprobe helpers shared by extraction and video assembly."""
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg

from config import BASE_DIR


def _find(name: str, env: str) -> str | None:
    """Locate a tool: explicit env var, then PATH, then the copy bundled under Qwen3-TTS/tools."""
    explicit = os.getenv(env)
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which(name)
    if found:
        return found
    bundled = sorted(BASE_DIR.glob(f"Qwen3-TTS/tools/*/bin/{name}.exe")) + sorted(BASE_DIR.glob(f"tools/*/bin/{name}*"))
    return str(bundled[-1]) if bundled else None


FFMPEG = _find("ffmpeg", "FFMPEG_PATH") or imageio_ffmpeg.get_ffmpeg_exe()
FFPROBE = _find("ffprobe", "FFPROBE_PATH")   # optional: probe() falls back to parsing `ffmpeg -i`


def run(args, timeout=1800, log=None):
    """Run ffmpeg. `log(text)` receives the command line and the result."""
    command = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)]
    started = time.monotonic()
    if log:
        log("ffmpeg " + subprocess.list2cmdline(command[1:]))
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    elapsed = time.monotonic() - started
    if result.returncode:
        if log:
            log(f"ffmpeg failed after {elapsed:.1f}s: {result.stderr[-800:]}")
        raise RuntimeError("Media processing failed: " + result.stderr[-1600:])
    if log:
        log(f"ffmpeg finished in {elapsed:.1f}s")
    return elapsed


def _fraction(value) -> float | None:
    try:
        a, _, b = str(value).partition("/")
        return float(a) / float(b or 1) if float(b or 1) else None
    except (ValueError, ZeroDivisionError):
        return None


def probe(path: Path) -> dict:
    """Duration, streams and size of a media file. Uses ffprobe when available."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing media file: {path.name}")
    if FFPROBE:
        result = subprocess.run([FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        if result.returncode:
            raise ValueError(f"Cannot read media file {path.name}: {result.stderr.strip()[-300:]}")
        data = json.loads(result.stdout)
        video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
        audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)

        def seconds(stream):
            value = stream.get("duration") if stream else None
            return float(value) if value not in (None, "N/A") else None
        duration = float(data["format"].get("duration") or 0)
        return {
            "duration": duration, "has_audio": audio is not None, "has_video": video is not None,
            "width": int(video["width"]) if video else None, "height": int(video["height"]) if video else None,
            "video_codec": video.get("codec_name") if video else None, "pix_fmt": video.get("pix_fmt") if video else None,
            "fps": _fraction(video.get("avg_frame_rate")) if video else None,
            "video_duration": seconds(video) or (duration if video else None),
            "audio_codec": audio.get("codec_name") if audio else None,
            "audio_duration": seconds(audio) or (duration if audio else None),
            "frames": int(video["nb_frames"]) if video and str(video.get("nb_frames", "")).isdigit() else None,
        }
    result = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=60)
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise ValueError(f"Cannot read media duration: {path.name}")
    h, m, s = map(float, match.groups())
    duration = h * 3600 + m * 60 + s
    size = re.search(r"Video: (\w+).*?\b(\d{2,5})x(\d{2,5})\b", result.stderr)
    audio = re.search(r"Audio: (\w+)", result.stderr)
    return {"duration": duration, "has_audio": bool(audio), "has_video": bool(size),
            "width": int(size[2]) if size else None, "height": int(size[3]) if size else None,
            "video_codec": size[1] if size else None, "pix_fmt": None, "fps": None,
            "video_duration": duration if size else None, "audio_codec": audio[1] if audio else None,
            "audio_duration": duration if audio else None, "frames": None}


def grab_frame(source: Path, at: float, destination: Path, log=None):
    """Save one frame of a video as an image (for validation and previews)."""
    run(["-ss", f"{max(0, at):.3f}", "-i", source, "-frames:v", "1", "-q:v", "3", destination], timeout=120, log=log)
    return destination


def mean_volume(path: Path) -> float | None:
    """Mean audio level in dB (None if it cannot be measured). Silence is around -91 dB."""
    result = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    match = re.search(r"mean_volume: (-?[\d.]+|-inf) dB", result.stderr)
    if not match:
        return None
    return float("-inf") if match[1] == "-inf" else float(match[1])


def is_faststart(path: Path) -> bool:
    """True when the MP4 index (moov) comes before the media data (mdat), so it can start playing while loading."""
    with open(path, "rb") as f:
        position, size = 0, os.path.getsize(path)
        while position + 8 <= size:
            f.seek(position)
            header = f.read(8)
            length, kind = int.from_bytes(header[:4], "big"), header[4:8]
            if length == 1:
                length = int.from_bytes(f.read(8), "big")
            if kind == b"moov":
                return True
            if kind == b"mdat":
                return False
            if length < 8:
                return False
            position += length
    return False
