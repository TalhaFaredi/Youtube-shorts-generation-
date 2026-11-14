"""Generate Video sessions: plan -> approval -> visuals -> assembly, with a cost ledger and an activity log.

Each session lives in data/jobs/<job>/video_sessions/<session>/ :
    session.json   settings, plan and state        ledger.json   every billable event
    activity.log   what happened, in plain words   assets/        one file per generated image or clip
    final.mp4      the assembled video
All work runs in background threads and all state is on disk, so a page refresh or an app restart loses nothing.
Every paid call goes through Session.charge_begin(), which enforces the budget cap and writes the ledger.
"""
import io
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import jobs
import video_assembly
import video_plan
import visual_providers as providers
from config import DEFAULT_BUDGET, OPENROUTER_API_KEY, VIDEO_POLL_TIMEOUT
from media_tools import grab_frame, probe
from video_plan import FPS, MODE_NAMES, MODES, PlanError

RUNNING = {"planning", "generating", "assembling"}
STEPS = ["Read the job's inputs", "Plan the visuals", "Approve the plan", "Generate the visuals",
         "Assemble the video", "Check the final video"]
PLAN_ATTEMPTS = 3            # first try + up to two retries with the errors fed back
PLANNER_FRAMES = 8           # source frames shown to the planner
WORKERS = 3                  # visuals generated at the same time
FATAL = {"missing_key", "invalid_key", "no_credits"}    # stop the whole batch, not just one visual
TRANSIENT = {"rate_limit", "network", "provider"}       # worth one automatic retry
RESOLUTION_PIXELS = {"360p": 360, "480p": 480, "720p": 720, "768p": 768, "1080p": 1080, "1k": 1024, "2k": 1536, "4k": 2160}

_sessions: dict = {}
_active: set = set()
_registry = threading.RLock()


class BusyError(ValueError):
    pass


class BudgetPause(Exception):
    """The next paid call would go over the budget cap."""


def _clock():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _money(value):
    return f"${value:.4f}" if value < 1 else f"${value:.2f}"


# ---------- image checks ----------

def inspect_image(source) -> dict:
    """Decode an image and measure it: {width, height, contrast, fingerprint}. Raises ValueError if it cannot be read."""
    import numpy as np
    from PIL import Image
    try:
        with Image.open(source) as image:
            image.load()
            rgb = image.convert("RGB")
    except Exception as e:
        raise ValueError(f"the file is not a readable image ({type(e).__name__})") from None
    gray = np.asarray(rgb.convert("L").resize((9, 8)), dtype=np.int16)
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return {"width": rgb.width, "height": rgb.height,
            "contrast": float(np.asarray(rgb.convert("L").resize((64, 64)), dtype=np.float32).std()),
            "fingerprint": "".join("1" if b else "0" for b in bits), "image": rgb}


def fingerprint_distance(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def nearest_ratio(width, height, choices):
    """Closest 'a:b' from `choices` to width/height. Returns (ratio, exact)."""
    target, best = width / height, None
    for choice in choices:
        match = re.fullmatch(r"(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)", choice)
        if not match:
            continue
        value = float(match[1]) / float(match[2])
        distance = abs(value / target - 1)
        if best is None or distance < best[1]:
            best = (choice, distance)
    if not best:
        return None, False
    return best[0], best[1] < 0.01


# ---------- session ----------

class Session:
    def __init__(self, job_id, session_id):
        self.job_id, self.id = job_id, session_id
        self.dir = jobs.folder(job_id) / "video_sessions" / session_id
        self.assets = self.dir / "assets"
        self.lock = threading.RLock()
        self.state = jobs.read_json(self.dir / "session.json")
        self.ledger = jobs.read_json(self.dir / "ledger.json") if (self.dir / "ledger.json").exists() else []
        self.abort, self.abort_reason = None, ""    # set when a batch must stop: "budget" or "provider"

    # -- persistence --
    def save(self):
        with self.lock:
            self.state["updated_at"] = jobs.now()
            jobs.write_json(self.dir / "session.json", self.state)
            jobs.write_json(self.dir / "ledger.json", self.ledger)

    def log(self, text):
        with self.lock, open(self.dir / "activity.log", "a", encoding="utf-8") as f:
            lines = str(text).splitlines() or [""]
            f.write(f"[{_clock()}] {lines[0]}\n" + "".join(f"    {line}\n" for line in lines[1:]))

    def step(self, number, text):
        with self.lock:
            self.state.update(step=number, step_text=text)
            self.save()

    def visual(self, visual_id):
        for v in self.state["visuals"]:
            if v["id"] == visual_id:
                return v
        raise FileNotFoundError("Visual not found.")

    # -- money --
    def totals(self):
        with self.lock:
            actual = sum(e["cost"] or 0 for e in self.ledger if e["cost_kind"] == "actual" and e["status"] in ("ok", "failed"))
            computed = sum(e["cost"] or 0 for e in self.ledger if e["cost_kind"] == "estimated" and e["status"] == "ok")
            unconfirmed = sum(e["estimate"] or 0 for e in self.ledger if e["status"] in ("pending", "interrupted"))
            remaining_work = sum(v.get("estimate") or 0 for v in self.state["visuals"] if v["status"] in ("pending", "failed"))
            cap = self.state["settings"]["budget"]
            spent = actual + computed
            return {"actual": actual, "computed": computed, "unconfirmed": unconfirmed, "spent": spent,
                    "remaining_work": remaining_work, "estimated_total": spent + unconfirmed + remaining_work,
                    "budget": cap, "remaining": cap - spent - unconfirmed}

    def charge_begin(self, *, step, label, model, estimate, visual_id=None, attempt=1, provider_id=None, units=None):
        """Reserve the estimated cost of a paid call against the budget and open its ledger line.

        Raises BudgetPause, before anything is sent to a provider, when the call would exceed the cap.
        """
        with self.lock:
            totals = self.totals()
            if totals["spent"] + totals["unconfirmed"] + estimate["cost"] > totals["budget"] + 1e-9:
                self.log(f"BUDGET: {label} is estimated at {_money(estimate['cost'])} but only {_money(max(0, totals['remaining']))} "
                         f"of the {_money(totals['budget'])} cap is left. Not sent. Session paused.")
                raise BudgetPause(f"{label} is estimated at {_money(estimate['cost'])}, but only "
                                  f"{_money(max(0, totals['remaining']))} of the {_money(totals['budget'])} budget is left.")
            entry = {"id": len(self.ledger) + 1, "time": _clock(), "step": step, "label": label, "visual_id": visual_id,
                     "provider": "OpenRouter", "model": model, "attempt": attempt, "units": units,
                     "estimate": estimate["cost"], "estimate_basis": estimate["basis"], "cost": None,
                     "cost_kind": None, "status": "pending", "error": None, "latency": None, "provider_id": provider_id}
            self.ledger.append(entry)
            self.log(f"REQUEST {label} -> {model} (estimated {_money(estimate['cost'])}: {estimate['basis']}; "
                     f"{_money(totals['remaining'])} of the budget left before this call)")
            self.save()
            return entry

    def charge_end(self, entry, *, cost, reported, units, latency, note=""):
        """Close a ledger line with what the call really cost."""
        with self.lock:
            if cost is None:   # the provider did not report a cost: fall back to the computed estimate, labelled as such
                cost, reported = entry["estimate"], False
            entry.update(cost=cost, cost_kind="actual" if reported else "estimated", status="ok", units=units,
                         latency=round(latency, 1) if latency is not None else None)
            self.log(f"RESPONSE {entry['label']}: ok, {units}, cost {_money(cost)} "
                     f"({'actual, reported by the provider' if reported else 'estimated: ' + entry['estimate_basis']})"
                     f"{', ' + format(latency, '.1f') + ' s' if latency is not None else ''}{'. ' + note if note else ''}")
            self.save()

    def charge_fail(self, entry, error, cost=0.0):
        with self.lock:
            entry.update(cost=cost, cost_kind="actual", status="failed", error=str(error)[:600])
            self.log(f"ERROR {entry['label']}: {error} (charged {_money(cost)})")
            self.save()

    # -- view for the UI --
    def view(self):
        with self.lock:
            state = deepcopy(self.state)
            ledger = deepcopy(self.ledger)
            totals = self.totals()
        running = (self.job_id, self.id) in _active
        if state["status"] in RUNNING and not running:
            state["status"] = "interrupted"
            state["message"] = ("The app stopped while this session was running. Nothing is lost: finished visuals are kept "
                                "and are not paid for again. Click Resume to continue.")
        base = f"/media/jobs/{self.job_id}/video_sessions/{self.id}"
        for v in state["visuals"]:
            lines = [e for e in ledger if e["visual_id"] == v["id"]]
            v["cost"] = sum(e["cost"] or 0 for e in lines if e["status"] in ("ok", "failed"))
            v["cost_kind"] = "estimated" if any(e["cost_kind"] == "estimated" for e in lines) else "actual"
            v["calls"] = len(lines)
            if v.get("asset") and v["status"] == "done":
                v["asset_url"] = f"{base}/assets/{v['asset']}?v={v.get('version', 1)}"
        done = sum(v["status"] == "done" for v in state["visuals"])
        count = len(state["visuals"])
        progress = {"planning": 8, "awaiting_approval": 15, "completed": 100}.get(self.state["status"])
        if progress is None:
            progress = 15 + (70 * done / count if count else 0) + (10 if self.state["status"] == "assembling" else 0)
        end = state.get("finished_at") or jobs.now()
        elapsed = (datetime.fromisoformat(end) - datetime.fromisoformat(state["created_at"])).total_seconds()
        if state.get("final"):
            state["final"]["url"] = f"{base}/final.mp4?v={state['final'].get('version', 1)}"
            state["final"]["download_url"] = f"{base}/final.mp4?download={state['final']['download_name']}"
        state.update(running=running, cost=totals, ledger=ledger, steps=STEPS, progress=round(progress),
                     elapsed=round(elapsed), done=done, count=count)
        return state

    def summary(self):
        totals = self.totals()
        view_status = self.state["status"]
        if view_status in RUNNING and (self.job_id, self.id) not in _active:
            view_status = "interrupted"
        return {"id": self.id, "created_at": self.state["created_at"], "mode": self.state["mode"],
                "mode_name": MODE_NAMES[self.state["mode"]], "status": view_status,
                "visuals": len(self.state["visuals"]), "done": sum(v["status"] == "done" for v in self.state["visuals"]),
                "cost": totals["spent"], "unconfirmed": totals["unconfirmed"], "has_final": bool(self.state.get("final"))}


def _session_dir(job_id, session_id):
    if not re.fullmatch(r"\d{8}-\d{6}-[a-f0-9]{4}", session_id or ""):
        raise ValueError("Invalid session ID.")
    path = jobs.folder(job_id) / "video_sessions" / session_id
    if not (path / "session.json").exists():
        raise FileNotFoundError("Video session not found.")
    return path


def get(job_id, session_id) -> Session:
    with _registry:
        key = (job_id, session_id)
        if key not in _sessions:
            _session_dir(job_id, session_id)
            _sessions[key] = Session(job_id, session_id)
        return _sessions[key]


def list_sessions(job_id):
    root = jobs.folder(job_id) / "video_sessions"
    result = []
    for path in sorted(root.glob("*/session.json"), reverse=True) if root.is_dir() else []:
        try:
            result.append(get(job_id, path.parent.name).summary())
        except (ValueError, OSError, KeyError):
            continue
    return result


def job_cost(job_id):
    """{sessions, cost} for the Saved jobs list."""
    sessions = list_sessions(job_id)
    return {"sessions": len(sessions), "cost": sum(s["cost"] for s in sessions)}


# ---------- creating a session ----------

def options_for(job) -> dict:
    """What Generate Video would use for this job, and why it cannot run if it cannot. No paid calls."""
    info = {"usable": job["media"]["usable"], "problem": job["media"]["problem"], "can_reextract": job["media"]["can_reextract"]}
    if not job["media"]["usable"]:
        return info
    narration = jobs.narration(job)
    seconds = probe(narration["path"])["duration"]
    info["narration"] = {"kind": narration["kind"], "label": narration["label"], "seconds": seconds}
    info["original_seconds"] = job["duration"]
    try:
        info["video_seconds"] = video_plan.final_frames(seconds, job["duration"]) / FPS
    except ValueError as e:
        info.update(usable=False, problem=str(e), can_reextract=False)
    path = jobs.folder(job["job_id"])
    info["source_files"] = {"video": job["source_video"], "audio": narration["path"].name,
                            "frames": [Path(f["url"]).name for f in job["frames"] if (path / Path(f["url"]).name).is_file()]}
    info["width"], info["height"] = job.get("width"), job.get("height")
    return info


def create(job_id, options) -> dict:
    """Validate the settings, create the session folder and start planning in the background."""
    if not isinstance(options, dict):
        raise ValueError("A JSON object is required.")
    job = jobs.load(job_id)
    if not job["media"]["usable"]:
        raise ValueError(job["media"]["problem"])
    mode = options.get("mode")
    if mode not in MODES:
        raise ValueError("Choose Images only, Videos only, or Videos + Images.")
    if not OPENROUTER_API_KEY:
        raise ValueError("OpenRouter key is missing. Add open_router_api=sk-or-... to .env and restart the app.")
    catalog = providers.catalog()
    if catalog["error"]:
        raise ValueError(catalog["error"])
    defaults, notes = catalog["defaults"], []

    planner = providers.model("planner", options.get("planner_model") or defaults["planner"]["id"])
    settings = {"planner_model": planner["id"], "planner_name": planner["name"], "planner_vision": planner["vision"],
                "planner_json_mode": planner["json_mode"]}
    # Say so when a default is not the exact model that was asked for by name.
    used = {"planner": True, "image": "image" in MODES[mode], "video": "video" in MODES[mode]}
    for kind in ("planner", "image", "video"):
        is_default = (options.get(f"{kind}_model") or defaults[kind]["id"]) == defaults[kind]["id"]
        if used[kind] and is_default and defaults[kind]["note"]:
            notes.append(defaults[kind]["note"])

    width, height = job.get("width"), job.get("height")
    if not width or not height:
        source = probe(jobs.folder(job_id) / job["source_video"])
        width, height = source["width"], source["height"]
    if width % 2 or height % 2:
        notes.append(f"The original is {width}x{height}; H.264 needs even sizes, so the final video is {width - width % 2}x{height - height % 2}.")
        width, height = width - width % 2, height - height % 2

    if "image" in MODES[mode]:
        image = providers.model("image", options.get("image_model") or defaults["image"]["id"])
        ratio, exact = nearest_ratio(width, height, image["aspect_ratios"])
        if not exact:
            notes.append(f"{image['name']} cannot make {width}x{height} exactly"
                         + (f"; requesting {ratio}, the closest shape. Images are cropped to fit, never stretched." if ratio
                            else " and takes no aspect ratio; images are cropped to fit, never stretched."))
        quality = options.get("image_quality") or ("medium" if "medium" in image["qualities"] else None)
        if quality and quality not in image["qualities"]:
            raise ValueError(f"{image['name']} does not offer quality {quality!r}.")
        want_refs = options.get("reference_images", True)
        if want_refs and not image["max_references"]:
            notes.append(f"{image['name']} does not accept reference images, so source frames are not passed to it.")
        settings.update(image_model=image["id"], image_name=image["name"], image_ratio=ratio, image_quality=quality,
                        reference_images=bool(want_refs and image["max_references"]),
                        image_max_references=image["max_references"])
    if "video" in MODES[mode]:
        video = providers.model("video", defaults["video"]["id"] or "")
        ratio, exact = nearest_ratio(width, height, video["aspect_ratios"])
        if not exact:
            notes.append(f"{video['name']} cannot make {width}x{height} exactly; requesting {ratio}. Clips are cropped to fit, never stretched.")
        available = sorted(video["resolutions"], key=lambda r: RESOLUTION_PIXELS.get(r.lower(), 9999))
        resolution = options.get("video_resolution") or next(
            (r for r in available if RESOLUTION_PIXELS.get(r.lower(), 0) >= min(width, height)), available[-1])
        if resolution not in video["resolutions"]:
            raise ValueError(f"{video['name']} does not offer {resolution}.")
        settings.update(video_model=video["id"], video_name=video["name"], video_ratio=ratio, video_resolution=resolution,
                        video_route="OpenRouter", reference_video=bool(options.get("reference_video", False)))
        notes.append("HeyGen Video runs through your OpenRouter account (model heygen/heygen-video-1). "
                     "A HeyGen API key is not needed and is not used.")
        if not settings["reference_video"]:
            notes.append("Source frames are not passed to HeyGen Video: clips are generated from the text prompt only. "
                         "Turning references on switches HeyGen to its reference mode, which costs about 3.5x more per second.")

    try:
        budget = float(options.get("budget") or DEFAULT_BUDGET)
        min_seconds = float(options.get("min_seconds") or 2.5)
    except (TypeError, ValueError):
        raise ValueError("Budget and minimum seconds must be numbers.") from None
    if not 0 < budget <= 500:
        raise ValueError("The budget cap must be between $0.01 and $500.")
    if not 1 <= min_seconds <= 20:
        raise ValueError("Minimum seconds per visual must be between 1 and 20.")
    system_prompt = (options.get("system_prompt") or "").strip() or video_plan.DEFAULT_SYSTEM_PROMPT
    if len(system_prompt) > 20000:
        raise ValueError("The system prompt is too long (maximum 20,000 characters).")

    narration = jobs.narration(job)
    narration_seconds = probe(narration["path"])["duration"]
    frames = video_plan.final_frames(narration_seconds, job["duration"])   # ValueError when the narration is too long
    settings.update(budget=budget, min_seconds=min_seconds, system_prompt=system_prompt,
                    system_prompt_edited=system_prompt != video_plan.DEFAULT_SYSTEM_PROMPT)

    session_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    state = {
        "id": session_id, "job_id": job_id, "job_title": job.get("title") or job_id, "created_at": jobs.now(),
        "updated_at": jobs.now(), "finished_at": None, "mode": mode, "mode_name": MODE_NAMES[mode],
        "status": "planning", "step": 1, "step_text": "reading the job's inputs", "message": "", "error": None,
        "settings": settings, "notes": notes,
        "narration": {"kind": narration["kind"], "label": narration["label"], "file": narration["path"].name,
                      "seconds": narration_seconds},
        "source": {"video": job["source_video"], "width": width, "height": height, "seconds": job["duration"],
                   "frames_shown": [], "frames_how": ""},
        "video": {"frames": frames, "seconds": frames / FPS, "fps": FPS, "width": width, "height": height},
        "plan": None, "visuals": [], "final": None, "balance": {},
    }
    with _registry:
        if any(j == job_id for j, _ in _active):
            raise BusyError("This job already has a video session running. Wait for it to finish or open it.")
        path = jobs.folder(job_id) / "video_sessions" / session_id
        (path / "assets").mkdir(parents=True)
        jobs.write_json(path / "session.json", state)
        jobs.write_json(path / "ledger.json", [])
        session = _sessions[(job_id, session_id)] = Session(job_id, session_id)
    session.log(f"SESSION {session_id} started for job \"{state['job_title']}\" ({job_id}). Mode: {MODE_NAMES[mode]}. "
                f"Budget cap {_money(budget)}.")
    for note in notes:
        session.log("NOTE " + note)
    _start(session, _plan)
    return session.view()


# ---------- background work ----------

def _start(session, work, *args):
    with _registry:
        if (session.job_id, session.id) in _active:
            raise BusyError("This session is already running.")
        if any(j == session.job_id for j, _ in _active):
            raise BusyError("Another video session of this job is running. Wait for it to finish.")
        _active.add((session.job_id, session.id))
    session.abort = None
    threading.Thread(target=_guard, args=(session, work, *args), daemon=True, name=f"video-{session.id}").start()


def _guard(session, work, *args):
    try:
        work(session, *args)
    except BudgetPause as e:
        _pause(session, str(e))
    except Exception as e:
        message = str(e) if isinstance(e, (providers.ProviderError, ValueError, RuntimeError)) else f"{type(e).__name__}: {e}"
        session.log(f"SESSION STOPPED: {message}")
        if not isinstance(e, (providers.ProviderError, ValueError)):
            session.log(traceback.format_exc())
        with session.lock:
            session.state.update(status="failed", error=message,
                                 message="The session stopped. Finished work is kept. Fix the problem, then click Resume.")
    finally:
        try:
            _record_balance(session, "latest")
            session.save()
        finally:
            with _registry:
                _active.discard((session.job_id, session.id))


def _pause(session, reason):
    with session.lock:
        session.state.update(status="paused_budget", error=None,
                             message=f"Paused before going over the budget: {reason} Raise the budget cap, then click Resume.")
    session.log("SESSION PAUSED (budget). Nothing over the cap was sent.")


def _record_balance(session, key):
    usage = providers.key_usage()
    with session.lock:
        if usage is not None:
            session.state["balance"][key] = usage
            session.state["balance"][key + "_at"] = _clock()


def _reference_files(session, job, visual, enabled, limit):
    if not enabled:
        return []
    folder = jobs.folder(session.job_id)
    names = [Path(job["frames"][i]["url"]).name for i in visual["reference_frames"] if i < len(job["frames"])]
    return [name for name in names if (folder / name).is_file()][:limit]


def _estimate(session, visual, position, count):
    settings = session.state["settings"]
    if visual["type"] == "image":
        return providers.estimate_image(providers.model("image", settings["image_model"]), settings["image_quality"],
                                        len(visual["reference_files"]))
    return providers.estimate_video(providers.model("video", settings["video_model"]), visual["clip_seconds"],
                                    settings["video_resolution"], len(visual["reference_files"]))


def _plan(session):
    """Steps 1-2: read the inputs, ask the planner, validate its JSON (retrying with the errors fed back)."""
    state, settings = session.state, session.state["settings"]
    job = jobs.load(session.job_id)
    folder = jobs.folder(session.job_id)
    narration = jobs.narration(job)
    if "start" not in state["balance"]:
        _record_balance(session, "start")
    segments = video_plan.transcript_segments(narration["transcript"], state["video"]["seconds"])
    if not segments:
        raise ValueError("This job's transcript has no spoken text to plan visuals from.")

    available = [(i, f) for i, f in enumerate(job["frames"]) if (folder / Path(f["url"]).name).is_file()]
    step = max(1, -(-len(available) // PLANNER_FRAMES))
    shown = available[::step][:PLANNER_FRAMES]
    frames = [{"index": i, "time": f["time"], "file": Path(f["url"]).name,
               "caption": video_plan.text_between(segments, f["time"], f["time"] + job.get("interval", 5)) or "no speech"}
              for i, f in shown]
    vision = settings["planner_vision"]
    with session.lock:
        state["source"]["frames_shown"] = [{"index": f["index"], "file": f["file"], "time": f["time"]} for f in frames]
        state["source"]["frames_how"] = ("attached to the planner as images" if vision else
                                         "described to the planner in words (this planner model cannot read images)")
        state["source"]["video_note"] = (f"{job['source_video']} is the source of these frames and of the size and length. "
                                         "It is not sent to any model: neither the planner nor the generators accept a video file here.")
    session.log(f"INPUTS read from disk: transcript ({len(segments)} timed lines), narration {narration['path'].name} "
                f"({state['narration']['seconds']:.2f} s, {narration['label']}), source video {job['source_video']} "
                f"({state['source']['width']}x{state['source']['height']}, {state['source']['seconds']:.2f} s), "
                f"{len(frames)} of {len(available)} source frames {state['source']['frames_how']}: "
                + ", ".join(f["file"] for f in frames))

    cheapest = []
    if "image" in MODES[state["mode"]]:
        cheapest.append(providers.estimate_image(providers.model("image", settings["image_model"]), settings["image_quality"], 0)["cost"])
    if "video" in MODES[state["mode"]]:
        cheapest.append(providers.estimate_video(providers.model("video", settings["video_model"]), video_plan.VIDEO_MIN,
                                                 settings["video_resolution"], 0)["cost"])
    max_visuals = max(1, min(video_plan.MAX_VISUALS, int(session.totals()["remaining"] / min(cheapest)),
                             int(state["video"]["seconds"] / settings["min_seconds"])))
    settings["max_visuals"] = max_visuals
    request = video_plan.planner_request(
        mode=state["mode"], title=state["job_title"], duration=state["video"]["seconds"],
        narration_label=state["narration"]["label"], width=state["video"]["width"], height=state["video"]["height"],
        aspect_ratio=settings.get("image_ratio") or settings.get("video_ratio") or "", segments=segments,
        frames=frames, frames_attached=vision, limits={"min_seconds": settings["min_seconds"], "max_visuals": max_visuals})
    content = [{"type": "text", "text": request}]
    if vision:
        for f in frames:
            content.append({"type": "image_url", "image_url": {"url": providers.data_url(folder / f["file"])}})
    planner = providers.model("planner", settings["planner_model"])
    session.log("PLANNER SYSTEM PROMPT" + (" (edited by you)" if settings["system_prompt_edited"] else " (default)")
                + ":\n" + settings["system_prompt"])
    session.log("PLANNER INPUT" + (f" (+ {len(frames)} attached images)" if vision else "") + ":\n" + request)

    feedback, problems = None, []
    for attempt in range(1, PLAN_ATTEMPTS + 1):
        session.step(2, f"planning the visuals with {settings['planner_name']} (attempt {attempt} of {PLAN_ATTEMPTS})")
        estimate = providers.estimate_planner(planner, len(request) + len(settings["system_prompt"]) + len(feedback or ""),
                                              len(frames) if vision else 0)
        entry = session.charge_begin(step="planner", label=f"Planner, attempt {attempt}", model=planner["id"],
                                     estimate=estimate, attempt=attempt)
        try:
            reply = providers.chat(planner["id"], settings["system_prompt"],
                                   content + ([{"type": "text", "text": feedback}] if feedback else []),
                                   settings["planner_json_mode"])
        except providers.ProviderError as e:
            session.charge_fail(entry, e)
            raise
        session.charge_end(entry, cost=reply["cost"], reported=reply["cost_reported"], latency=reply["latency"],
                           units=f"{reply['tokens_in']} tokens in / {reply['tokens_out']} tokens out")
        entry.update(tokens_in=reply["tokens_in"], tokens_out=reply["tokens_out"])
        session.log(f"PLANNER RAW RESPONSE (attempt {attempt}):\n{reply['text']}")
        try:
            style, visuals = video_plan.validate_plan(
                video_plan.parse_json(reply["text"]), mode=state["mode"], duration=state["video"]["seconds"],
                segments=segments, frame_count=len(job["frames"]), min_seconds=settings["min_seconds"],
                max_visuals=max_visuals)
            for i, v in enumerate(visuals):
                v["clip_seconds"] = video_plan.clip_seconds(v, i, len(visuals)) if v["type"] == "video" else None
                v["reference_files"] = _reference_files(
                    session, job, v, settings.get("reference_images") if v["type"] == "image" else settings.get("reference_video"),
                    min(2, settings.get("image_max_references", 2)) if v["type"] == "image" else 2)
                estimate = _estimate(session, v, i, len(visuals))
                v.update(estimate=estimate["cost"], estimate_basis=estimate["basis"])
            total = sum(v["estimate"] for v in visuals)
            left = session.totals()["remaining"]
            if total > left:
                raise PlanError([f"This plan would cost about {_money(total)} to generate but only {_money(left)} of the budget "
                                 f"is left. Use fewer visuals" + (" and shorter or fewer video clips." if state["mode"] != "images" else ".")])
            break
        except PlanError as e:
            problems = e.problems
            session.log(f"PLAN REJECTED (attempt {attempt}): " + " | ".join(problems))
            feedback = ("Your previous reply was rejected by the validator. Problems:\n- " + "\n- ".join(problems)
                        + "\n\nYour previous reply was:\n" + reply["text"][:6000]
                        + "\n\nReturn the complete corrected JSON object. Fix every problem listed.")
    else:
        raise ValueError(f"The planner did not produce a valid plan in {PLAN_ATTEMPTS} attempts. Last problems: "
                         + " ".join(problems) + " Try again, or choose another planning model.")

    with session.lock:
        for v in visuals:
            ratio = settings["image_ratio"] if v["type"] == "image" else settings["video_ratio"]
            v.update(prompt=video_plan.generation_prompt(v.pop("scene"), style, v["type"], ratio or "", v["reference_files"]),
                     status="pending", error=None, asset=None, version=0, provider_job_id=None, note=None)
        state["visuals"] = visuals
        state["plan"] = {"style": style, "attempts": attempt, "approved_at": None}
        total = sum(v["estimate"] for v in visuals)
        state.update(status="awaiting_approval", step=3, error=None,
                     step_text="waiting for you to approve the plan",
                     message=f"The plan has {len(visuals)} visuals and is estimated at {_money(total)} to generate. "
                             "Nothing has been generated yet. Review it, edit any prompt, then click Approve & generate.")
    session.log(f"PLAN ACCEPTED on attempt {attempt}: {len(visuals)} visuals ("
                + ", ".join(f"{v['id']}:{v['type']} {v['start']:.2f}-{v['end']:.2f}s" for v in visuals)
                + f"). Estimated generation cost {_money(total)}. Waiting for approval.")


def _progress_text(session):
    visuals = session.state["visuals"]
    busy = [v for v in visuals if v["status"] == "generating"]
    done = sum(v["status"] == "done" for v in visuals)
    if len(busy) == 1:
        v = busy[0]
        return f"{'image' if v['type'] == 'image' else 'video clip'} {v['id']} of {len(visuals)} ({done} done)"
    if busy:
        return f"visuals {', '.join(str(v['id']) for v in busy)} of {len(visuals)} in progress ({done} done)"
    return f"{done} of {len(visuals)} visuals done"


def _generate(session, only=None):
    """Step 4 (then 5-6): generate every missing visual in parallel, then assemble if all of them exist."""
    state = session.state
    with session.lock:
        todo = [v for v in state["visuals"] if v["status"] != "done" and (only is None or v["id"] in only)]
        for v in todo:
            v.update(status="pending", error=None)
        state.update(status="generating", error=None, message="", final=None if todo else state.get("final"))
    session.step(4, _progress_text(session))
    if todo:
        session.log(f"GENERATION started for {len(todo)} visual(s): " + ", ".join(f"{v['id']} ({v['type']})" for v in todo)
                    + f". Estimated {_money(sum(v['estimate'] for v in todo))}. Up to {WORKERS} at a time.")
        job = jobs.load(session.job_id)
        with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix=f"visual-{session.id}") as pool:
            list(pool.map(lambda v: _one(session, job, v), todo))
    if session.abort == "budget":
        raise BudgetPause(session.abort_reason)
    missing = [v for v in state["visuals"] if v["status"] != "done"]
    if missing:
        with session.lock:
            state.update(status="failed", error="; ".join(f"visual {v['id']}: {v.get('error') or 'not generated'}" for v in missing)[:900],
                         message=f"{len(missing)} of {len(state['visuals'])} visuals are missing. No final video was built: a missing "
                                 "visual is never replaced by a placeholder. Fix the problem and click Resume, or regenerate those visuals.")
        session.log(f"GENERATION incomplete: {len(missing)} visual(s) missing ({', '.join(str(v['id']) for v in missing)}). "
                    "No final video built.")
        return
    _assemble(session)


def _one(session, job, visual):
    if session.abort:
        return      # an earlier visual hit the budget or a fatal provider error: do not send more requests
    for attempt in (1, 2):
        try:
            with session.lock:
                visual.update(status="generating", error=None)
            session.step(4, _progress_text(session))
            (_image if visual["type"] == "image" else _clip)(session, job, visual, attempt)
            with session.lock:
                visual["status"] = "done"
            session.step(4, _progress_text(session))
            return
        except BudgetPause as e:
            with session.lock:
                visual["status"] = "pending"
                session.abort, session.abort_reason = "budget", str(e)
            return
        except providers.ProviderError as e:
            with session.lock:
                visual.update(status="failed", error=str(e))
                if e.kind in FATAL:
                    session.abort = "provider"
            if e.kind in TRANSIENT and attempt == 1 and not session.abort:
                session.log(f"RETRY visual {visual['id']} once in 5 s after a temporary error.")
                time.sleep(5)
                continue
            return
        except Exception as e:
            with session.lock:
                visual.update(status="failed", error=f"{e}" if isinstance(e, (ValueError, RuntimeError)) else f"{type(e).__name__}: {e}")
            session.log(f"ERROR visual {visual['id']}: {visual['error']}")
            return
        finally:
            session.save()


def _other_fingerprints(session, visual):
    return [(v["id"], v["fingerprint"]) for v in session.state["visuals"]
            if v["id"] != visual["id"] and v["status"] == "done" and v.get("fingerprint")]


def _check_picture(session, visual, info, what):
    """Blank / duplicate checks shared by images and clip frames. Raises ValueError with the reason."""
    if info["contrast"] < 4:
        raise ValueError(f"the generated {what} is blank (almost a single colour)")
    for other_id, fingerprint in _other_fingerprints(session, visual):
        if fingerprint_distance(info["fingerprint"], fingerprint) <= 2:
            raise ValueError(f"the generated {what} looks the same as visual {other_id} (duplicate)")


def _image(session, job, visual, attempt):
    settings, folder = session.state["settings"], jobs.folder(session.job_id)
    references = [folder / name for name in visual["reference_files"]]
    estimate = providers.estimate_image(providers.model("image", settings["image_model"]), settings["image_quality"], len(references))
    entry = session.charge_begin(step="image", label=f"Image {visual['id']}", model=settings["image_model"], estimate=estimate,
                                 visual_id=visual["id"], attempt=attempt)
    session.log(f"IMAGE {visual['id']} prompt ({settings['image_ratio'] or 'model default shape'}, quality {settings['image_quality'] or 'model default'}"
                + (f", references: {', '.join(visual['reference_files'])}" if references else ", no references") + f"):\n{visual['prompt']}")
    try:
        result = providers.generate_image(settings["image_model"], visual["prompt"], settings["image_ratio"],
                                          settings["image_quality"], references)
    except providers.ProviderError as e:
        session.charge_fail(entry, e)
        raise
    session.charge_end(entry, cost=result["cost"], reported=result["cost_reported"], latency=result["latency"],
                       units=f"1 image, {result['tokens']} image tokens" if result["tokens"] else "1 image")
    if result["cost_reported"]:
        providers.record_image_cost(settings["image_model"], settings["image_quality"], len(references), result["cost"])
    try:
        info = inspect_image(io.BytesIO(result["bytes"]))
        _check_picture(session, visual, info, "image")
    except ValueError as e:
        raise ValueError(f"Image {visual['id']} was generated and charged, but failed validation: {e}. Regenerate it.") from None
    with session.lock:
        visual["version"] = visual.get("version", 0) + 1
        name = f"visual_{visual['id']:03d}.png"
        info["image"].save(session.assets / name, format="PNG")
        wanted, got = session.state["video"]["width"] / session.state["video"]["height"], info["width"] / info["height"]
        note = None
        if abs(got / wanted - 1) > 0.03:
            note = (f"provider returned {info['width']}x{info['height']} (shape {got:.3f}); the video is {wanted:.3f}, "
                    "so this image is cropped to fit when assembled")
        visual.update(asset=name, width=info["width"], height=info["height"], fingerprint=info["fingerprint"],
                      contrast=round(info["contrast"], 1), note=note, prompt_edited=False)
    session.log(f"SAVED assets/{name} ({info['width']}x{info['height']}, {(session.assets / name).stat().st_size // 1024} KB). "
                f"Checks: decodes, not blank (contrast {info['contrast']:.0f}), not a duplicate." + (f" NOTE {note}." if note else ""))


def _clip(session, job, visual, attempt):
    settings, folder = session.state["settings"], jobs.folder(session.job_id)
    references = [folder / name for name in visual["reference_files"]]
    seconds = visual["clip_seconds"]
    # A provider job this visual already submitted (in flight, or finished and paid for before the app stopped)
    # is reused: never submitted or paid for twice.
    entry = next((e for e in session.ledger if e["visual_id"] == visual["id"] and e["status"] in ("pending", "ok")
                  and e["provider_id"] and e["provider_id"] == visual.get("provider_job_id")), None)
    already_paid = bool(entry and entry["status"] == "ok")
    if entry:
        session.log(f"CLIP {visual['id']}: reusing provider job {entry['provider_id']} "
                    + ("(finished and already paid for; downloading it again, no new charge)." if already_paid
                       else "(already submitted; waiting for it, no new charge)."))
    else:
        estimate = providers.estimate_video(providers.model("video", settings["video_model"]), seconds,
                                            settings["video_resolution"], len(references))
        entry = session.charge_begin(step="video", label=f"Clip {visual['id']}", model=settings["video_model"], estimate=estimate,
                                     visual_id=visual["id"], attempt=attempt, units=f"{seconds} s at {settings['video_resolution']}")
        session.log(f"CLIP {visual['id']} prompt ({seconds} s, {settings['video_resolution']}, {settings['video_ratio']}"
                    + (f", references: {', '.join(visual['reference_files'])}" if references else ", text to video")
                    + f"):\n{visual['prompt']}")
        try:
            submitted = providers.submit_video(settings["video_model"], visual["prompt"], seconds, settings["video_resolution"],
                                               settings["video_ratio"], references)
        except providers.ProviderError as e:
            session.charge_fail(entry, e)
            raise
        with session.lock:
            entry["provider_id"] = visual["provider_job_id"] = submitted["job_id"]
            session.save()      # saved before waiting, so a restart continues this job instead of paying again
        session.log(f"CLIP {visual['id']} accepted by the provider: job {submitted['job_id']}. Waiting for it to render.")

    started, wait, job_id = time.monotonic(), 0.0 if already_paid else 6.0, entry["provider_id"]
    while True:
        if time.monotonic() - started > VIDEO_POLL_TIMEOUT:
            raise RuntimeError(f"Clip {visual['id']} is still rendering at the provider after {VIDEO_POLL_TIMEOUT // 60} minutes "
                               f"(job {job_id}). Click Resume to keep waiting for the same clip; it is not paid for twice.")
        time.sleep(wait)
        wait = min(max(wait, 5) * 1.3, 20)
        try:
            status = providers.video_status(job_id)
        except providers.ProviderError as e:
            if e.kind in FATAL:
                raise
            session.log(f"CLIP {visual['id']}: status check failed ({e}); trying again.")
            continue
        if status["status"] == "completed":
            break
        if status["status"] in ("failed", "cancelled", "expired"):
            session.charge_fail(entry, f"provider job {job_id} {status['status']}: {status.get('error') or 'no reason given'}",
                                cost=status.get("cost") or 0.0)
            with session.lock:
                visual["provider_job_id"] = None
            raise providers.ProviderError(f"Clip {visual['id']}: HeyGen Video job {status['status']}. {status.get('error') or ''}".strip(),
                                          "bad_request")
    name = f"visual_{visual['id']:03d}.mp4"
    temp = session.assets / (name + ".part")
    providers.download_video(job_id, temp)
    if not already_paid:
        session.charge_end(entry, cost=status.get("cost"), reported=status.get("cost") is not None,
                           latency=time.monotonic() - started, units=f"{seconds} s at {settings['video_resolution']}",
                           note=f"provider job {job_id}")
    try:
        info = probe(temp)
        if not info["has_video"]:
            raise ValueError("the file has no video stream")
        if (info["video_duration"] or 0) < seconds - 0.6:
            raise ValueError(f"the clip is {info['video_duration']:.2f} s but {seconds} s was requested")
        still = grab_frame(temp, (info["video_duration"] or seconds) / 2, session.dir / f"check_{visual['id']:03d}.jpg")
        picture = inspect_image(still)
        still.unlink(missing_ok=True)
        _check_picture(session, visual, picture, "clip")
    except (ValueError, RuntimeError) as e:
        temp.unlink(missing_ok=True)
        with session.lock:
            visual["provider_job_id"] = None
        raise ValueError(f"Clip {visual['id']} was generated and charged, but failed validation: {e}. Regenerate it.") from None
    with session.lock:
        visual["version"] = visual.get("version", 0) + 1
        temp.replace(session.assets / name)
        wanted, got = session.state["video"]["width"] / session.state["video"]["height"], info["width"] / info["height"]
        note = None
        if abs(got / wanted - 1) > 0.03:
            note = f"provider returned {info['width']}x{info['height']}; cropped to the video's shape when assembled"
        visual.update(asset=name, width=info["width"], height=info["height"], seconds=round(info["video_duration"], 2),
                      fingerprint=picture["fingerprint"], contrast=round(picture["contrast"], 1), note=note, prompt_edited=False)
    session.log(f"SAVED assets/{name} ({info['width']}x{info['height']}, {info['video_duration']:.2f} s, "
                f"{(session.assets / name).stat().st_size // 1024} KB). Checks: decodes, length ok, middle frame not blank, not a duplicate.")


def _assemble(session):
    """Steps 5-6: build final.mp4 with ffmpeg, then check it. A video that fails a check is not published."""
    state = session.state
    with session.lock:
        state.update(status="assembling", error=None, message="")
    session.step(5, "assembling the video with ffmpeg")
    entry = {"id": len(session.ledger) + 1, "time": _clock(), "step": "assembly", "label": "Assembly (ffmpeg on this computer)",
             "visual_id": None, "provider": "local", "model": "ffmpeg", "attempt": 1, "units": None, "estimate": 0.0,
             "estimate_basis": "local, free", "cost": 0.0, "cost_kind": "actual", "status": "pending", "error": None,
             "latency": None, "provider_id": None}
    with session.lock:
        session.ledger.append(entry)
    started = time.monotonic()
    job = jobs.load(session.job_id)
    narration = jobs.narration(job)
    output = session.dir / "final.building.mp4"
    try:
        result = video_assembly.assemble(
            visuals=state["visuals"], assets_dir=session.assets, work_dir=session.dir / "work", narration=narration["path"],
            output=output, width=state["video"]["width"], height=state["video"]["height"], log=session.log,
            progress=lambda text: session.step(5, text))
    except Exception as e:
        entry.update(status="failed", error=str(e)[:600], latency=round(time.monotonic() - started, 1))
        raise RuntimeError(f"Assembly failed: {e}") from None
    elapsed = time.monotonic() - started
    entry.update(status="ok", units=f"ffmpeg ran for {elapsed:.1f} s", latency=round(elapsed, 1))
    with session.lock:
        for v in state["visuals"]:
            v["assembly_note"] = result["notes"].get(v["id"])

    session.step(6, "checking the final video")
    ok, checks, facts = video_assembly.verify(output, original_seconds=state["source"]["seconds"],
                                              narration_seconds=state["narration"]["seconds"],
                                              width=state["video"]["width"], height=state["video"]["height"])
    for check in checks:
        session.log(f"CHECK {'PASS' if check['ok'] else 'FAIL'}: {check['name']} ({check['detail']})")
    if not ok:
        rejected = session.dir / "final.rejected.mp4"
        output.replace(rejected)
        with session.lock:
            state["final_checks"] = checks
        raise RuntimeError("The assembled video failed its checks and was not published: "
                           + "; ".join(f"{c['name']} ({c['detail']})" for c in checks if not c["ok"]))
    final = session.dir / "final.mp4"
    output.replace(final)
    shutil.rmtree(session.dir / "work", ignore_errors=True)
    title = re.sub(r"[^\w\- ]+", "", state["job_title"]).strip().replace(" ", "_")[:60] or session.job_id
    with session.lock:
        version = (state.get("final_version") or 0) + 1
        state["final_version"] = version
        state["final"] = {"file": "final.mp4", "version": version, "checks": checks, **facts,
                          "original_seconds": state["source"]["seconds"], "size_kb": final.stat().st_size // 1024,
                          "download_name": f"{title}-{state['mode']}-{session.id}.mp4", "built_at": _clock()}
        state.update(status="completed", step=6, step_text="done", finished_at=jobs.now(), error=None,
                     message="The video is ready.")
    session.log(f"FINAL VIDEO saved: final.mp4 ({facts['duration']:.2f} s, {facts['width']}x{facts['height']}, "
                f"{final.stat().st_size // 1024} KB), {facts['ratio'] * 100:.1f}% of the original length. "
                f"Session cost {_money(session.totals()['spent'])}.")


# ---------- actions from the UI ----------

def _idle(session):
    if (session.job_id, session.id) in _active:
        raise BusyError("This session is running. Wait for the current step to finish.")


def approve(job_id, session_id):
    session = get(job_id, session_id)
    _idle(session)
    with session.lock:
        if session.state["status"] != "awaiting_approval":
            raise ValueError("There is no plan waiting for approval in this session.")
        session.state["plan"]["approved_at"] = jobs.now()
    session.log("APPROVED by you. Generation starts.")
    _start(session, _generate)
    return session.view()


def resume(job_id, session_id):
    session = get(job_id, session_id)
    _idle(session)
    with session.lock:
        status = session.state["status"]
        if status == "completed":
            raise ValueError("This session is already complete.")
        if status == "awaiting_approval":
            raise ValueError("The plan is waiting for your approval. Click Approve & generate.")
        # Calls that were in flight when the app stopped: their outcome is unknown, so they stay counted as unconfirmed.
        for entry in session.ledger:
            if entry["status"] == "pending" and not (entry["step"] == "video" and entry["provider_id"]):
                entry.update(status="interrupted", error="The app stopped during this call. The provider may or may not have charged for it.")
                session.log(f"NOTE {entry['label']} was interrupted; its estimated {_money(entry['estimate'])} stays counted as unconfirmed.")
        for v in session.state["visuals"]:
            if v["status"] == "generating":
                v["status"] = "pending"
        session.log(f"RESUME requested (was: {status}).")
        if not session.state["visuals"]:
            session.state.update(status="planning", error=None, message="")
            work = _plan
        elif not session.state["plan"]["approved_at"]:
            session.state.update(status="awaiting_approval", step=3, step_text="waiting for you to approve the plan",
                                 error=None, message="The plan is ready. Review it, then click Approve & generate.")
            session.save()
            return session.view()
        else:
            work = _generate
        session.save()
    _start(session, work)
    return session.view()


def set_budget(job_id, session_id, value):
    session = get(job_id, session_id)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("The budget cap must be a number.") from None
    if not 0 < value <= 500:
        raise ValueError("The budget cap must be between $0.01 and $500.")
    with session.lock:
        old = session.state["settings"]["budget"]
        session.state["settings"]["budget"] = value
        session.save()
    session.log(f"BUDGET cap changed from {_money(old)} to {_money(value)}.")
    return session.view()


def edit_prompt(job_id, session_id, visual_id, prompt):
    """Change a visual's generation prompt. Free: nothing is generated until you approve or regenerate."""
    session = get(job_id, session_id)
    _idle(session)
    prompt = (prompt or "").strip()
    if not 40 <= len(prompt) <= 6000:
        raise ValueError("The prompt must be between 40 and 6,000 characters.")
    with session.lock:
        visual = session.visual(visual_id)
        visual.update(prompt=prompt, prompt_edited=visual["status"] == "done")
        session.save()
    session.log(f"PROMPT of visual {visual_id} edited by you:\n{prompt}")
    return session.view()


def regenerate(job_id, session_id, visual_id, prompt=None):
    """Generate one visual again (a new paid call), keeping the previous file, then rebuild the final video."""
    session = get(job_id, session_id)
    _idle(session)
    if prompt is not None and prompt.strip():
        edit_prompt(job_id, session_id, visual_id, prompt)
    with session.lock:
        if not session.state["plan"] or not session.state["plan"]["approved_at"]:
            raise ValueError("Approve the plan first. Before approval you can edit prompts for free.")
        visual = session.visual(visual_id)
        if visual.get("asset") and (session.assets / visual["asset"]).is_file():
            old = session.assets / visual["asset"]
            kept = old.with_name(f"{old.stem}.v{visual.get('version', 1)}{old.suffix}")
            old.replace(kept)
            session.log(f"REGENERATE visual {visual_id} requested by you. Previous file kept as assets/{kept.name}.")
        visual.update(status="pending", asset=None, provider_job_id=None, error=None, fingerprint=None)
        session.save()
    _start(session, _generate, [visual_id])
    return session.view()


def recheck_balance(job_id, session_id):
    session = get(job_id, session_id)
    _record_balance(session, "latest")
    session.save()
    return session.view()


def read_log(job_id, session_id, after=0):
    """Activity log lines from line number `after` on."""
    session = get(job_id, session_id)
    path = session.dir / "activity.log"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return {"lines": lines[after:], "next": len(lines)}
