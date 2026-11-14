"""YouTube link -> transcript + frames. No API keys needed."""
import re
import uuid
from pathlib import Path

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

from config import JOBS_DIR
import jobs
from media_tools import FFMPEG, probe, run

MAX_FRAMES = 80


def video_id_from(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    url = url.strip()
    return url if re.fullmatch(r"[A-Za-z0-9_-]{11}", url) else None


def get_transcript(video_id: str):
    """Return (segments, error). English preferred, else the first available track."""
    api = YouTubeTranscriptApi()
    try:
        try:
            fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        except Exception:
            fetched = next(iter(api.list(video_id))).fetch()
        segs = [{"start": round(s.start, 3), "duration": round(s.duration, 3),
                 "text": s.text.replace("\n", " ").strip()} for s in fetched]
        return segs, None
    except Exception as e:
        return [], f"Transcript unavailable: {type(e).__name__}"


def _new_job() -> Path:
    job = JOBS_DIR / uuid.uuid4().hex[:10]
    job.mkdir()
    return job


def extract_frames(video_id: str, interval: int | None, transcript=None, transcript_error=None):
    """Keep source video, source audio, JPGs and timed captions together in one job."""
    job = _new_job()
    opts = {
        "format": "bv*[height<=720]+ba/b[height<=720]/best",
        "merge_output_format": "mp4",
        "ffmpeg_location": FFMPEG,
        "outtmpl": str(job / "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)

    video = next(p for p in job.glob("video.*") if p.suffix not in {".part", ".ytdl"})
    metadata = probe(video)
    duration = metadata["duration"]
    if not interval:
        interval = max(5, round(duration / 40)) if duration else 10
    if duration and duration / interval > MAX_FRAMES:
        interval = int(duration / MAX_FRAMES) + 1

    run(
        ["-i", str(video),
         "-vf", f"fps=1/{interval}:start_time=0:round=up,scale=640:-2", "-frames:v", str(MAX_FRAMES), "-q:v", "4",
         str(job / "frame_%04d.jpg")],
    )
    audio = None
    if metadata["has_audio"]:
        audio = "source_audio.wav"
        run(["-i", video, "-vn", "-ac", "1", "-ar", "24000", job / audio])

    frames = [
        {"time": i * interval, "url": f"/media/jobs/{job.name}/{f.name}"}
        for i, f in enumerate(sorted(job.glob("frame_*.jpg")))
    ]
    saved = dict(job_id=job.name, created_at=jobs.now(), video_id=video_id, title=info.get("title"),
                 channel=info.get("uploader"), duration=duration, width=metadata["width"], height=metadata["height"],
                 interval=interval, transcript=transcript or [], transcript_error=transcript_error, frames=frames,
                 source_video=video.name, source_audio=audio)
    jobs.write_json(job / "job.json", saved)
    return saved


def reextract(job_id: str, video_id: str):
    """Download a job's source media again into its existing folder. Its video sessions are kept.

    The download goes to a fresh folder first, so a failed download leaves the old job untouched.
    """
    target = jobs.folder(job_id)
    created_at = jobs.read_json(target / "job.json").get("created_at") if (target / "job.json").exists() else None
    transcript, transcript_error = get_transcript(video_id)
    fresh = extract_frames(video_id, None, transcript, transcript_error)
    source = JOBS_DIR / fresh["job_id"]
    for old in [*target.glob("video.*"), *target.glob("frame_*.jpg"), target / "source_audio.wav"]:
        old.unlink(missing_ok=True)
    for item in source.iterdir():
        if item.name != "job.json":
            item.replace(target / item.name)
    for frame in fresh["frames"]:
        frame["url"] = frame["url"].replace(f"/jobs/{fresh['job_id']}/", f"/jobs/{job_id}/")
    fresh.update(job_id=job_id, created_at=created_at or fresh["created_at"], reextracted_at=jobs.now())
    jobs.write_json(target / "job.json", fresh)
    (source / "job.json").unlink()
    source.rmdir()
    return jobs.load(job_id)
