"""YouTube link -> transcript + extracted frames.

Run:  python app.py   then open http://127.0.0.1:5000
No API keys needed.
"""
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import imageio_ffmpeg
import yt_dlp
from flask import Flask, jsonify, request, send_from_directory
from youtube_transcript_api import YouTubeTranscriptApi

BASE = Path(__file__).parent
JOBS = BASE / "jobs"
JOBS.mkdir(exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MAX_FRAMES = 80

app = Flask(__name__, static_folder=None)


def video_id_from(url: str):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    return url if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()) else None


def get_transcript(video_id: str):
    """Return (segments, error). Segments: [{start, text}]."""
    api = YouTubeTranscriptApi()
    try:
        try:
            fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        except Exception:
            # fall back to whatever language exists
            first = next(iter(api.list(video_id)))
            fetched = first.fetch()
        segs = [{"start": round(s.start, 1), "text": s.text.replace("\n", " ").strip()}
                for s in fetched]
        return segs, None
    except Exception as e:  # no captions, disabled, blocked, ...
        return [], f"Transcript unavailable: {type(e).__name__}"


def extract_frames(url: str, job: Path, interval: int | None):
    """Download a low-res copy, pull a frame every `interval` seconds."""
    opts = {
        "format": "bv*[height<=480]/b[height<=480]/worst",
        "outtmpl": str(job / "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    duration = info.get("duration") or 0
    if not interval:
        interval = max(5, round(duration / 40)) if duration else 10
    # keep within the frame cap
    if duration and duration / interval > MAX_FRAMES:
        interval = int(duration / MAX_FRAMES) + 1

    video = next(job.glob("video.*"))
    subprocess.run(
        [FFMPEG, "-loglevel", "error", "-i", str(video),
         "-vf", f"fps=1/{interval},scale=640:-2",
         "-frames:v", str(MAX_FRAMES), "-q:v", "4",
         str(job / "frame_%04d.jpg")],
        check=True,
    )
    video.unlink(missing_ok=True)
    frames = sorted(job.glob("frame_*.jpg"))
    return info, interval, [
        {"time": i * interval, "url": f"/frames/{job.name}/{f.name}"}
        for i, f in enumerate(frames)
    ]


@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.get("/frames/<job>/<name>")
def frame(job, name):
    return send_from_directory(JOBS / job, name)


@app.post("/api/extract")
def extract():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    interval = data.get("interval")
    interval = int(interval) if str(interval or "").isdigit() and int(interval) > 0 else None
    vid = video_id_from(url)
    if not vid:
        return jsonify(error="That doesn't look like a YouTube link."), 400

    # keep only the 5 most recent jobs on disk
    old = sorted((p for p in JOBS.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    for p in old[:-4]:
        shutil.rmtree(p, ignore_errors=True)

    job = JOBS / uuid.uuid4().hex[:10]
    job.mkdir()
    transcript, t_err = get_transcript(vid)
    try:
        info, used_interval, frames = extract_frames(f"https://www.youtube.com/watch?v={vid}", job, interval)
    except Exception as e:
        return jsonify(error=f"Could not get video frames: {e}",
                       transcript=transcript, transcript_error=t_err), 502

    return jsonify(
        video_id=vid,
        title=info.get("title"),
        channel=info.get("uploader"),
        duration=info.get("duration"),
        interval=used_interval,
        transcript=transcript,
        transcript_error=t_err,
        frames=frames,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
