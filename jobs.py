"""Saved extraction jobs: one folder per video under data/jobs with job.json, the source media and video sessions."""
import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import JOBS_DIR

_io_lock = threading.RLock()


def now():
    return datetime.now(timezone.utc).isoformat()


def folder(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{10,32}", job_id or ""):
        raise ValueError("Invalid job ID.")
    path = JOBS_DIR / job_id
    if not path.is_dir():
        raise FileNotFoundError("Job not found.")
    return path


def write_json(path: Path, data):
    with _io_lock:
        temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        # Windows can briefly deny the rename while antivirus or indexing has the old file open.
        for attempt in range(40):
            try:
                temp.replace(path)
                return
            except PermissionError:
                if attempt == 39:
                    temp.unlink(missing_ok=True)
                    raise
                time.sleep(0.025)


def read_json(path: Path):
    with _io_lock:
        return json.loads(path.read_text(encoding="utf-8"))


def _frame_files(path: Path):
    return sorted(p.name for p in path.glob("frame_*.jpg"))


def load(job_id: str) -> dict:
    """The saved job, plus what is actually on disk (`media`). Older jobs without job.json load as `legacy`."""
    path = folder(job_id)
    if (path / "job.json").exists():
        job = read_json(path / "job.json")
    else:
        files = _frame_files(path)
        job = {"job_id": job_id, "legacy": True, "title": None, "video_id": None, "duration": None, "transcript": [],
               "created_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
               "frames": [{"time": None, "url": f"/media/jobs/{job_id}/{name}"} for name in files],
               "source_video": None, "source_audio": None}
    job["media"] = media_status(job)
    return job


def media_status(job: dict) -> dict:
    """Which source files of a job exist on disk, and a plain message when generation is not possible."""
    path = JOBS_DIR / job["job_id"]
    video = bool(job.get("source_video")) and (path / job["source_video"]).is_file()
    audio = bool(job.get("source_audio")) and (path / job["source_audio"]).is_file()
    frames = [f for f in job.get("frames") or [] if (path / Path(f["url"]).name).is_file()]
    problem = None
    if job.get("legacy"):
        problem = ("This older job kept only its JPG frames. Its video, audio, transcript and even its YouTube link were "
                   "not saved, so it cannot be used for Generate Video. Paste the video's link and extract it again.")
    elif not video or not audio:
        missing = " and ".join(name for name, ok in (("video", video), ("audio", audio)) if not ok)
        problem = f"This job's source {missing} file is missing from disk. Re-extract it to download the media again."
    elif not job.get("transcript"):
        problem = "This job has no transcript, so there is no script to plan visuals from."
    return {"video": video, "audio": audio, "frames": len(frames), "usable": problem is None, "problem": problem,
            "can_reextract": bool(job.get("video_id"))}


def narration(job: dict) -> dict:
    """The audio the final video plays: the job's voiced script when it has one, otherwise the original audio.

    A voiced script is job.json["narration"] = {"file": ..., "segments": [{start, text}, ...]} with its file in the
    job folder. Returns {kind, path, label, transcript}.
    """
    path = JOBS_DIR / job["job_id"]
    voiced = job.get("narration") or {}
    if voiced.get("file") and (path / voiced["file"]).is_file() and voiced.get("segments"):
        return {"kind": "voiced", "path": path / voiced["file"], "transcript": voiced["segments"],
                "label": "voiced script saved in this job"}
    return {"kind": "original", "path": path / (job.get("source_audio") or "source_audio.wav"),
            "transcript": job.get("transcript") or [],
            "label": "original audio of the video (this job has no voiced script)"}


def list_jobs():
    """Every job folder, newest first. Folders without job.json are listed as legacy instead of being hidden."""
    result = []
    for path in JOBS_DIR.iterdir():
        if not path.is_dir() or not re.fullmatch(r"[a-f0-9]{10,32}", path.name):
            continue
        try:
            job = load(path.name)
        except (ValueError, OSError):
            continue
        if job.get("legacy") and not job["frames"]:
            continue    # empty folder: nothing to show
        result.append({"job_id": job["job_id"], "title": job.get("title"), "video_id": job.get("video_id"),
                       "duration": job.get("duration"), "created_at": job.get("created_at"),
                       "legacy": bool(job.get("legacy")), "frames": len(job["frames"]), "usable": job["media"]["usable"]})
    return sorted(result, key=lambda j: j.get("created_at") or "", reverse=True)
