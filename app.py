"""YouTube extractor, transcript tools, voice generation and Generate Video.

Run:  uv run app.py   then open http://127.0.0.1:5000
"""
import hashlib
import os
import re
import socket
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.serving import ThreadedWSGIServer

import db
import jobs
import llm
import tts
import video_plan
import video_sessions
import visual_providers
import youtube
from config import BASE_DIR, DATA_DIR, DEFAULT_BUDGET, HEYGEN_API_KEY, PORT, TTS_ENABLED
from media_tools import FFMPEG, FFPROBE

app = Flask(__name__, static_folder=None)


def _build_stamp() -> str:
    """Short hash of the code this process loaded. The page shows it, so a stale server or a stale tab is obvious."""
    digest = hashlib.sha1()
    for path in sorted([*BASE_DIR.glob("*.py"), BASE_DIR / "index.html", *BASE_DIR.glob("static/*")]):
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()[:8]


BUILD = _build_stamp()
STARTED = time.strftime("%Y-%m-%d %H:%M:%S")


class LocalServer(ThreadedWSGIServer):
    """Exclusive bind: a second copy of the app fails to start instead of silently sharing the port."""
    allow_reuse_address = False

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


@app.after_request
def no_stale_pages(response):
    if request.path == "/" or request.path.startswith(("/api/", "/static/")):
        response.headers["Cache-Control"] = "no-store"
    response.headers["X-App-Build"] = BUILD
    return response


@app.get("/api/health")
def health():
    return jsonify(build=BUILD, started_at=STARTED, process_id=os.getpid(), port=PORT,
                   ffmpeg=Path(FFMPEG).name, ffprobe=bool(FFPROBE))


def _err(msg: str, code: int = 400):
    return jsonify(error=msg), code


@app.get("/")
def index():
    page = (BASE_DIR / "index.html").read_text(encoding="utf-8")
    return Response(page.replace("__BUILD__", BUILD).replace("__STARTED__", STARTED), mimetype="text/html")


@app.get("/favicon.ico")
def favicon():
    return Response(status=204)


@app.get("/static/<path:path>")
def static_files(path):
    return send_from_directory(BASE_DIR / "static", path)


@app.get("/media/<path:path>")
def media(path):
    """Files under data/. `?download=<name>` saves the file instead of showing it."""
    name = request.args.get("download")
    if name:
        return send_from_directory(DATA_DIR, path, as_attachment=True,
                                   download_name=re.sub(r"[^\w.\- ]+", "_", name) or Path(path).name)
    return send_from_directory(DATA_DIR, path)


# ---------- YouTube ----------

@app.post("/api/extract")
def extract():
    data = request.get_json(force=True)
    vid = youtube.video_id_from(data.get("url") or "")
    if not vid:
        return _err("That doesn't look like a YouTube link.")
    interval = str(data.get("interval") or "")
    interval = int(interval) if interval.isdigit() and int(interval) > 0 else None

    transcript, t_err = youtube.get_transcript(vid)
    try:
        job = youtube.extract_frames(vid, interval, transcript, t_err)
    except Exception as e:
        return jsonify(error=f"Could not get video frames: {e}", transcript=transcript, transcript_error=t_err), 502

    return jsonify(jobs.load(job["job_id"]))


# ---------- Saved jobs ----------

def _reply(action, code=200):
    """Run an action and turn its expected errors into clear JSON messages."""
    try:
        return jsonify(action()), code
    except FileNotFoundError as e:
        return _err(str(e), 404)
    except video_sessions.BusyError as e:
        return _err(str(e), 409)
    except visual_providers.ProviderError as e:
        return _err(str(e), 502)
    except (ValueError, TypeError) as e:
        return _err(str(e))


@app.get("/api/jobs")
def job_list():
    result = jobs.list_jobs()
    for job in result:
        job["video"] = video_sessions.job_cost(job["job_id"])
    return jsonify(result)


@app.get("/api/jobs/<job_id>")
def job_get(job_id):
    return _reply(lambda: jobs.load(job_id))


@app.post("/api/jobs/<job_id>/reextract")
def job_reextract(job_id):
    """Download a job's source media again. Older jobs that never saved their link need `url`."""
    def action():
        job = jobs.load(job_id)
        vid = job.get("video_id") or youtube.video_id_from((request.get_json(silent=True) or {}).get("url") or "")
        if not vid:
            raise ValueError("This job has no saved link. Paste the video's YouTube link to extract it again.")
        try:
            return youtube.reextract(job_id, vid)
        except Exception as e:
            raise ValueError(f"Could not download the video again: {e}") from None
    return _reply(action)


# ---------- Generate Video ----------

@app.get("/api/video/catalog")
def video_catalog():
    """Live models and prices, the default planner prompt, and which keys are set (never the keys themselves)."""
    data = dict(visual_providers.catalog(force=request.args.get("refresh") == "1"))
    data.update(system_prompt=video_plan.DEFAULT_SYSTEM_PROMPT, default_budget=DEFAULT_BUDGET,
                heygen_key_set=bool(HEYGEN_API_KEY), modes=video_plan.MODE_NAMES)
    return jsonify(data)


@app.get("/api/video/key")
def video_key():
    return jsonify(visual_providers.check_key())


@app.get("/api/jobs/<job_id>/video-options")
def video_options(job_id):
    return _reply(lambda: video_sessions.options_for(jobs.load(job_id)))


@app.get("/api/jobs/<job_id>/video-sessions")
def session_list(job_id):
    return _reply(lambda: video_sessions.list_sessions(job_id))


@app.post("/api/jobs/<job_id>/video-sessions")
def session_create(job_id):
    return _reply(lambda: video_sessions.create(job_id, request.get_json(silent=True)), 201)


@app.get("/api/jobs/<job_id>/video-sessions/<session_id>")
def session_get(job_id, session_id):
    return _reply(lambda: video_sessions.get(job_id, session_id).view())


@app.get("/api/jobs/<job_id>/video-sessions/<session_id>/log")
def session_log(job_id, session_id):
    return _reply(lambda: video_sessions.read_log(job_id, session_id, request.args.get("after", 0, type=int)))


@app.post("/api/jobs/<job_id>/video-sessions/<session_id>/<action>")
def session_action(job_id, session_id, action):
    body = request.get_json(silent=True) or {}
    actions = {
        "approve": lambda: video_sessions.approve(job_id, session_id),
        "resume": lambda: video_sessions.resume(job_id, session_id),
        "budget": lambda: video_sessions.set_budget(job_id, session_id, body.get("budget")),
        "balance": lambda: video_sessions.recheck_balance(job_id, session_id),
    }
    if action not in actions:
        return _err("Unknown action.", 404)
    return _reply(actions[action])


@app.post("/api/jobs/<job_id>/video-sessions/<session_id>/visuals/<int:visual_id>")
def visual_edit(job_id, session_id, visual_id):
    body = request.get_json(silent=True) or {}
    return _reply(lambda: video_sessions.edit_prompt(job_id, session_id, visual_id, body.get("prompt")))


@app.post("/api/jobs/<job_id>/video-sessions/<session_id>/visuals/<int:visual_id>/regenerate")
def visual_regenerate(job_id, session_id, visual_id):
    body = request.get_json(silent=True) or {}
    return _reply(lambda: video_sessions.regenerate(job_id, session_id, visual_id, body.get("prompt")))


# ---------- Transcript text tools ----------

@app.get("/api/text/languages")
def text_languages():
    return jsonify(llm.LANGUAGES)


@app.get("/api/text/models")
def text_models():
    return jsonify(llm.MODELS)


def _text_request():
    """Common fields of the transcript-tool requests, or an error message."""
    d = request.get_json(force=True)
    f = {
        "language": d.get("language"),
        "model": d.get("model") or llm.MODELS[0],
        "text": (d.get("text") or "").strip(),
        "prompt": (d.get("prompt") or "").strip(),
    }
    if f["language"] not in llm.LANGUAGES:
        return d, f, "Unknown language."
    if f["model"] not in llm.MODELS:
        return d, f, "Unknown model."
    if not f["text"]:
        return d, f, "There is no transcript text to work with."
    return d, f, None


@app.post("/api/text/ideas")
def text_ideas():
    """Several new video ideas similar to the transcript."""
    _, f, error = _text_request()
    if error:
        return _err(error)
    try:
        return jsonify(llm.ideas(f["text"], f["language"], f["model"], f["prompt"]))
    except Exception as e:
        return _err(str(e), 502)


@app.post("/api/text/generate")
def text_generate():
    """mode: translate | custom | idea. Result length is matched to the source text."""
    d, f, error = _text_request()
    mode, idea = d.get("mode"), d.get("idea")
    if error:
        return _err(error)
    if mode not in llm.TASKS:
        return _err("Unknown mode.")
    if mode == "custom" and not f["prompt"]:
        return _err("Describe the changes you want.")
    if mode == "idea" and not (isinstance(idea, dict) and idea.get("title")):
        return _err("Pick an idea first.")
    try:
        return jsonify(llm.generate(mode, f["text"], f["language"], f["model"], f["prompt"], idea))
    except Exception as e:
        return _err(str(e), 502)


# ---------- Voice ----------

def _audio_result(name: str):
    return jsonify(audio_url=f"/media/outputs/{name}")


def _uploaded_wav() -> Path | None:
    """Save the uploaded reference audio as a temporary 24 kHz mono WAV."""
    f = request.files.get("audio")
    if not f or not f.filename:
        return None
    raw = tts.temp_path(Path(f.filename).suffix or ".bin")
    f.save(raw)
    wav = tts.temp_path(".wav")
    try:
        tts.to_wav(raw, wav)
    finally:
        raw.unlink(missing_ok=True)
    return wav


@app.get("/api/tts/status")
def tts_status():
    if not TTS_ENABLED:
        return jsonify(ready=False, error="Voice models are switched off for this instance (TTS_ENABLED=0).")
    return jsonify(tts.status())


@app.post("/api/tts/preset")
def tts_preset():
    d = request.get_json(force=True)
    text = (d.get("text") or "").strip()
    if not text or not d.get("speaker"):
        return _err("Text and speaker are required.")
    try:
        return _audio_result(tts.generate_preset(text, d.get("language") or "auto", d["speaker"], d.get("instruct")))
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}", 500)


@app.post("/api/tts/clone")
def tts_clone():
    """Clone from an uploaded clip. If `save_name` is given, also store the voice."""
    text = (request.form.get("text") or "").strip()
    if not text:
        return _err("Text is required.")
    wav = _uploaded_wav()
    if not wav:
        return _err("Reference audio is required.")
    ref_text = request.form.get("ref_text")
    language = request.form.get("language") or "auto"
    save_name = (request.form.get("save_name") or "").strip()
    try:
        out = {"audio_url": f"/media/outputs/{tts.generate_clone(wav, ref_text, text, language)}"}
        if save_name:
            out["voice"] = tts.save_voice(save_name, wav, ref_text)
        return jsonify(out)
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}", 500)
    finally:
        wav.unlink(missing_ok=True)


@app.get("/api/voices")
def voices_list():
    return jsonify(db.list_voices())


@app.post("/api/voices")
def voices_create():
    """Save a voice without generating anything."""
    name = (request.form.get("name") or "").strip()
    if not name:
        return _err("Name is required.")
    wav = _uploaded_wav()
    if not wav:
        return _err("Reference audio is required.")
    try:
        return jsonify(tts.save_voice(name, wav, request.form.get("ref_text")))
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}", 500)
    finally:
        wav.unlink(missing_ok=True)


@app.post("/api/voices/<voice_id>/generate")
def voices_generate(voice_id):
    if not db.get_voice(voice_id):
        return _err("Voice not found.", 404)
    d = request.get_json(force=True)
    text = (d.get("text") or "").strip()
    if not text:
        return _err("Text is required.")
    try:
        return _audio_result(tts.generate_saved(voice_id, text, d.get("language") or "auto"))
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}", 500)


@app.delete("/api/voices/<voice_id>")
def voices_delete(voice_id):
    tts.delete_voice(voice_id)
    return jsonify(ok=True)


if __name__ == "__main__":
    # Bind before anything else. If the port is taken, say so and stop: two copies must never share it.
    try:
        server = LocalServer("127.0.0.1", PORT, app)
    except OSError:
        raise SystemExit(f"Port {PORT} is already in use: another copy of this app is probably running. "
                         "Stop it first, then start again.") from None
    db.init_db()
    if TTS_ENABLED:
        tts.start_loading()
    print(f"YouTube Automation build {BUILD} running at http://127.0.0.1:{PORT} (process {os.getpid()})", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
