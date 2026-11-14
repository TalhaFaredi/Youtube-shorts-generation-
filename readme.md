# Voice Cloner Application

A local Flask web app for creating short-form video assets:

- **Voice generation** with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
  - **Preset voices**: 9 built-in speakers with optional style instructions ("Very happy", "Speak slowly and calmly").
  - **Voice cloning**: clone any voice from a short clip (MP3, WAV, M4A, OGG…).
  - **Saved voices**: store cloned voices in SQLite and reuse them without re-uploading.
- **YouTube extractor**: paste a link and get the transcript plus frames from the video.
- **Generate Video**: AI-generated images, HeyGen Video clips, or both, planned from the job's transcript and played over its narration, with a live cost panel, plan approval and one downloadable MP4.
- **Transcript tools** (Qwen through [OpenRouter](https://openrouter.ai)), each with a choice of 10 output languages:
  - **Translate** the transcript.
  - **Change it** with your own instructions.
  - **New video on a similar idea**: get several ideas first, then write the script for any of them.

  Pick the model from a list (Qwen 3.7 Plus by default). Every result is matched to the length of the original transcript.

Voice generation and the YouTube extractor run on your own machine with no API keys. Transcript tools and Generate Video use one OpenRouter key: the planner, images and HeyGen Video clips all go through OpenRouter. These features send the transcript (and, for Generate Video, a few source frames) to OpenRouter and use its credits.

## Screenshots

| Preset voices | Clone a voice |
|---|---|
| ![Preset voices](docs/screenshots/preset-voices.png) | ![Clone a voice](docs/screenshots/clone-voice.png) |

| Saved voices | YouTube extractor |
|---|---|
| ![Saved voices](docs/screenshots/saved-voices.png) | ![YouTube extractor](docs/screenshots/youtube.png) |

## Requirements

- **NVIDIA GPU** with CUDA. Both voice models together need roughly 10 GB of free VRAM.
- **Python 3.12** (uv installs it for you).
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** package manager.
- About **10 GB of free disk** for the model downloads.

- An **OpenRouter API key**, only for the transcript tools.

ffmpeg is bundled through the `imageio-ffmpeg` package, so you don't need to install it separately.

## Install and run

```bash
git clone https://github.com/TalhaFaredi/Voice-clonner-Application.git
cd Voice-clonner-Application
uv sync          # creates .venv and installs everything, including CUDA PyTorch
uv run app.py
```

For the transcript tools, create a `.env` file in the project folder first:

```
open_router_api=sk-or-...
```

**Generate Video** uses the same key for the planner, images (GPT Image 2) and video clips (HeyGen Video, `heygen/heygen-video-1` on OpenRouter). No HeyGen key is needed. Optional settings are listed in `.env.example`. Restart the app after changing `.env`.

Open **http://127.0.0.1:5000**.

On start, the voice models load in the background. The voice buttons stay disabled until the models are ready. The YouTube extractor works right away.

## How the models are downloaded and cached

The app uses two models from Hugging Face:

| Used for | Model | Setting |
|---|---|---|
| Preset voices | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `PRESET_MODEL` |
| Voice cloning | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `CLONE_MODEL` |

- **First start:** nothing to do by hand. On the first `uv run app.py`, both models download automatically. They are several GB each, so the first start takes a few minutes, depending on your connection.
- **Cache location:** the standard Hugging Face cache.
  - Windows: `C:\Users\<you>\.cache\huggingface\hub\`
  - Linux / macOS: `~/.cache/huggingface/hub/`
- **Later starts:** the models load from the cache with no download. Startup takes about a minute while they load onto the GPU.
- **Moving the cache:** set `HF_HOME` before starting, for example `HF_HOME=D:\hf-cache`.
- **Downloading ahead of time** (optional):
  ```bash
  uv run python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'); d('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"
  ```
- **Running offline:** after the first download, set `HF_HUB_OFFLINE=1` to stop any network checks.
- **Smaller model:** for a GPU with less memory, use the 0.6B cloning model: `CLONE_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base`. It uses less VRAM and gives lower quality.

## Configuration

Set these as environment variables before `uv run app.py`:

| Variable | Default | Purpose |
|---|---|---|
| `PRESET_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Model for preset voices |
| `CLONE_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Model for cloning and saved voices |
| `TTS_DEVICE` | `cuda:0` | GPU to use |
| `HF_HOME` | Hugging Face default | Where models are cached |
| `open_router_api` | – | OpenRouter API key (also read from `OPENROUTER_API_KEY`) |
| `OPENROUTER_MODEL` | `qwen/qwen3.7-plus` | Default model for the transcript tools |

## Using the app

**Generate Video**:

Turns a saved job into a new video: AI-generated images and/or clips that follow the script, played over the job's narration.

1. Open the job from **Saved jobs** (or extract a video), then click **Generate Video**. The card lists what it reads from the job on disk: the narration (the voiced script if the job has one, otherwise the original audio), the timestamped transcript, the original video's size and length, and the source JPGs.
2. Choose **Images only**, **Videos only** or **Videos + Images**, the planning model (default Qwen 3.7 Plus) and the image model (default GPT Image 2: no model is named exactly "GPT Image", so the closest match is used and the page says so). Both lists come live from OpenRouter, with prices. Set the budget cap (default $5) and the minimum seconds per visual. The planner's system prompt can be read and edited.
3. **Plan visuals** (about one cent). The planner sees the transcript, the mode, the limits and up to 8 source frames, and returns JSON that the app validates: order, no gaps or overlaps, full coverage of the narration, clip lengths (5–30 s), minimum time per visual, and a maximum count that fits the budget. An invalid plan is sent back with the errors, up to twice.
4. Review the plan: time range, transcript text, prompt and estimated cost of every visual. Edit any prompt for free, then click **Approve & generate**. Nothing is generated before that.
5. Visuals are generated 3 at a time. Each is checked (decodes, not blank, not a duplicate, clip length) and saved as its own file, viewable and downloadable in the plan table. Before every paid call its estimated cost is checked against what is left of the budget; if it would go over, the session pauses instead. A visual that cannot be generated is shown as missing; no placeholder is ever used and no final video is built without it.
6. ffmpeg places the visuals in order, each on screen while its words are spoken, with 0.42 s crossfades centred on each boundary (the next visual is fully visible 0.21 s after its words start). Everything is scaled and cropped to the original video's size, never stretched. Stills get a slow zoom; clips are trimmed, or slowed by up to 10% and then held. The clips' own audio is dropped, so the narration is the only sound. The final H.264/AAC MP4 is checked (at most 105% of the original length, audio present and not silent, audio and video the same length, same resolution, fast start) and is not published if a check fails. A narration longer than 105% is reported as an error; speech is never cut.

The panel at the top stays visible while you scroll: session, step, progress, cost spent (actual, as reported by OpenRouter), estimated total, budget cap (editable) and remaining. Below it are the plan table with every visual, the cost of every call (retries and failures included), and an activity log with every request, response, saved file and ffmpeg command, including the planner's exact prompt, input and raw reply.

**Resume and regenerate:** sessions run in the background and are saved to disk, so closing the tab or restarting the app loses nothing. **Resume** continues an interrupted or stopped session: finished visuals are never generated or paid for again, and a clip already submitted to HeyGen is collected from the same provider job. A call that was in flight when the app stopped stays counted as "unconfirmed" at its estimate. **Regenerate** makes one visual again (a new paid call; the previous file is kept as `visual_NNN.vN.png`) and rebuilds the final video.

**Older jobs:** jobs from before source media was saved are listed as "Older job … media missing". The page explains why they cannot be used and offers **Re-extract** (paste the link, since those jobs never saved it).

**Transcript tools**: extract a video, then open the **Translate / rewrite** tab.
1. Pick what to do: translate, change it with your own instructions, or a new video on a similar idea.
2. Pick the output language: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish or Italian.
3. Pick the model. Qwen 3.7 Plus is the default; the list is `MODELS` in `llm.py`.
4. Click **Generate**. The result appears in the box next to the original and can be edited and copied.

For a **new video**, the button is **Suggest ideas**: you get 5 ideas, each with a title and a short description. Click **Write script** on any of them to see its script; you can write scripts for several ideas and switch between them with **Show script**.

**How the length is matched:** the app counts the original transcript, asks the model for a draft, counts the draft itself, and sends back only the sentences that need to grow or shrink, each with an exact length. Models often deliver only part of the change they are asked for, so each correction asks for more, based on how much the previous one delivered. It stops within 3% of the original (at least ±2 words for short texts) or after 5 corrections, and shows both lengths. Words are counted for most languages and characters for Chinese and Japanese. Across languages that are counted differently, it converts with the length of a natural translation (about 1.5 Chinese characters, 2.25 Japanese characters or 0.75 Korean words per English word).

**Preset voices**: type text, pick a speaker and language, and optionally add a style instruction.

**Clone a voice**
1. Upload a clean clip of 3 seconds or more, with one speaker and little background noise.
2. Type exactly what the clip says. This is recommended: leaving it empty clones from the voice only, which sounds less similar.
3. Type the new text, then click **Clone & generate**.
4. To keep the voice, give it a name in **Save this voice as**.

**Saved voices**: click **Use** on a voice, type text, and click **Generate**. Each voice has a preview player and a delete button.

**Note on emotion:** style instructions only work with preset voices. The cloning model (`Base`) doesn't support them. To get a cloned voice in a certain mood, record the reference clip in that mood and save one voice per mood.

## Where data is stored

Everything is in `data/`, which is git-ignored:

```
data/
├── app.db             SQLite: saved voice records
├── voices/<id>/
│   ├── ref.wav        reference clip (converted to 24 kHz mono)
│   └── prompt.pt      precomputed voice prompt (fast reuse)
├── outputs/           generated audio (.wav)
├── jobs/<job>/        source video, audio, JPGs and job.json (transcript, size, length)
│   └── video_sessions/<session>/
│       ├── session.json   settings, plan and state
│       ├── ledger.json    every billable call and its cost
│       ├── activity.log   what happened, in plain words
│       ├── assets/        visual_001.png, visual_002.mp4, …
│       └── final.mp4      the finished video
└── tmp/               temporary uploads
```

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Flask routes |
| `tts.py` | Qwen3-TTS loading, preset generation, cloning, saved voices |
| `db.py` | SQLite access |
| `youtube.py` | YouTube transcript and frame extraction |
| `llm.py` | Transcript tools through OpenRouter, with length matching |
| `jobs.py` | Saved jobs: job.json, which source media is on disk, narration choice |
| `video_sessions.py` | Generate Video sessions: plan, approval, generation, budget cap, ledger, log, resume |
| `video_plan.py` | Planner prompt, plan validation, timeline math (pure functions, unit-tested) |
| `video_assembly.py` | ffmpeg assembly of the final MP4 and its checks |
| `visual_providers.py` | OpenRouter: live catalog and prices, planner, images, HeyGen Video clips |
| `media_tools.py` | ffmpeg / ffprobe helpers |
| `config.py` | Paths, model names, device |
| `index.html` | Web page (vanilla HTML/JS) |
| `pyproject.toml` / `uv.lock` | Dependencies |
| `specs/v1.md` | v1 spec (YouTube extractor) |

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/tts/status` | – | `ready`, `languages`, `speakers` |
| POST | `/api/tts/preset` | JSON: `text`, `speaker`, `language`, `instruct?` | `audio_url` |
| POST | `/api/tts/clone` | form: `audio`, `text`, `ref_text?`, `language`, `save_name?` | `audio_url`, `voice?` |
| GET | `/api/voices` | – | list of saved voices |
| POST | `/api/voices` | form: `name`, `audio`, `ref_text?` | saved voice |
| POST | `/api/voices/<id>/generate` | JSON: `text`, `language` | `audio_url` |
| DELETE | `/api/voices/<id>` | – | `ok` |
| POST | `/api/extract` | JSON: `url`, `interval?` | transcript and frames |
| GET | `/api/text/languages` | – | list of output languages |
| GET | `/api/text/models` | – | list of models, default first |
| GET | `/api/jobs` | – | saved extraction jobs |
| GET | `/api/jobs/<id>` | – | source manifest, transcript, frames and duration |
| POST | `/api/jobs/<id>/reextract` | JSON: `url?` | download the job's media again |
| GET | `/api/video/catalog` | – | live models and prices, defaults, default planner prompt |
| GET | `/api/jobs/<id>/video-options` | – | inputs Generate Video would use, or why it cannot run |
| GET / POST | `/api/jobs/<id>/video-sessions` | POST JSON: `mode`, `planner_model?`, `image_model?`, `image_quality?`, `budget?`, `min_seconds?`, `system_prompt?` | sessions / a new session (plans in the background) |
| GET | `/api/jobs/<id>/video-sessions/<s>` | – | full session state, ledger and costs |
| GET | `/api/jobs/<id>/video-sessions/<s>/log?after=n` | – | activity log lines |
| POST | `/api/jobs/<id>/video-sessions/<s>/approve`, `/resume`, `/budget` | `/budget`: JSON `budget` | session state |
| POST | `/api/jobs/<id>/video-sessions/<s>/visuals/<n>` | JSON: `prompt` | edit a prompt (free) |
| POST | `/api/jobs/<id>/video-sessions/<s>/visuals/<n>/regenerate` | JSON: `prompt?` | generate one visual again |
| GET | `/api/health` | – | build stamp, process ID, start time |
| POST | `/api/text/ideas` | JSON: `language`, `text`, `model?`, `prompt?` | list of `title`, `pitch` |
| POST | `/api/text/generate` | JSON: `mode` (`translate`, `custom`, `idea`), `language`, `text`, `model?`, `prompt?`, `idea?` (required for `idea`) | `text`, `count`, `target`, `unit`, `within_tolerance`, `source_count`, `source_unit` |

## Troubleshooting

- **"Port 5000 is in use":** another copy of the app is running. Stop it first; two copies can never share the port. The build stamp under the page title must match `/api/health`; the page warns if it does not.
- **Plan visuals is disabled:** the text under the button says why (missing key, job without source media, …).
- **Session stopped or paused:** the panel shows the reason. Fix it (for example add credits or raise the budget cap), then click **Resume**.
- **"An Application Control policy has blocked this file" (Windows):** Smart App Control is blocking the Python or PyTorch files. Turn it off in *Windows Security → App & browser control → Smart App Control*. On many Windows versions it can't be turned back on without resetting Windows.
- **Voice buttons stay disabled:** the models are still downloading or loading. Watch the console. The page shows an error if loading fails.
- **CUDA out of memory:** close other GPU apps, or switch to the 0.6B cloning model (see above).
- **Slow generation on Windows:** FlashAttention isn't available on Windows, so the app uses PyTorch's default attention. It works, a little slower.

## Tests

`uv run python -m unittest discover -s tests -v` runs the unit tests for plan validation, timeline math, cost estimates, the ledger and budget cap, resume and image checks. They make no provider calls (one test mocks the provider). `tests/browser_check.py` drives the running app in Chrome without spending anything.
