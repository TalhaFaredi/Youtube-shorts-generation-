"""YouTube extractor + Qwen3-TTS voice generation.

Run:  uv run app.py   then open http://127.0.0.1:5000
"""
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import db
import llm
import tts
import youtube
from config import BASE_DIR, DATA_DIR

app = Flask(__name__, static_folder=None)


def _err(msg: str, code: int = 400):
    return jsonify(error=msg), code


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/media/<path:path>")
def media(path):
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
        info, used_interval, frames = youtube.extract_frames(vid, interval)
    except Exception as e:
        return jsonify(error=f"Could not get video frames: {e}", transcript=transcript, transcript_error=t_err), 502

    return jsonify(
        video_id=vid, title=info.get("title"), channel=info.get("uploader"), duration=info.get("duration"),
        interval=used_interval, transcript=transcript, transcript_error=t_err, frames=frames,
    )


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
    db.init_db()
    tts.start_loading()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
