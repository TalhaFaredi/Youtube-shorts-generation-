import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = Path(os.getenv("APP_DATA_DIR") or BASE_DIR / "data")
JOBS_DIR = DATA_DIR / "jobs"        # one folder per extracted video (media, transcript, video sessions)
VOICES_DIR = DATA_DIR / "voices"    # saved voices: <id>/ref.wav + prompt.pt
OUTPUTS_DIR = DATA_DIR / "outputs"  # generated audio
TMP_DIR = DATA_DIR / "tmp"
DB_PATH = DATA_DIR / "app.db"

for d in (JOBS_DIR, VOICES_DIR, OUTPUTS_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

PORT = int(os.getenv("APP_PORT", "5000"))

PRESET_MODEL = os.getenv("PRESET_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
CLONE_MODEL = os.getenv("CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
DEVICE = os.getenv("TTS_DEVICE", "cuda:0")
# TTS_ENABLED=0 starts the app without loading the voice models (used by test instances).
TTS_ENABLED = os.getenv("TTS_ENABLED", "1") != "0"

OPENROUTER_API_KEY = (os.getenv("open_router_api") or os.getenv("OPENROUTER_API_KEY") or "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.7-plus")

# ---- Generate Video ----
# Requested defaults. The app looks each one up in OpenRouter's live catalog and says so in the UI
# when the exact model is missing and a closest match is used instead.
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "qwen/qwen3.7-plus")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "openai/gpt-image-2")
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "heygen/heygen-video-1")   # HeyGen Video, through OpenRouter
# Not used yet: HeyGen Video runs through OpenRouter. Read only so the UI can say whether it is set.
HEYGEN_API_KEY = (os.getenv("HEYGEN_API_KEY") or "").strip()
VIDEO_POLL_TIMEOUT = int(os.getenv("VIDEO_POLL_TIMEOUT", "1800"))
DEFAULT_BUDGET = float(os.getenv("VIDEO_BUDGET", "5"))
