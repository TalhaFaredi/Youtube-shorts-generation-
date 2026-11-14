"""OpenRouter adapters for Generate Video: live model catalog, planner, images, HeyGen Video clips.

Nothing here is called at startup and nothing here decides whether to spend: callers go through
video_sessions, which checks the budget and records every call in the session ledger.
"""
import base64
import difflib
import json
import mimetypes
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from config import DATA_DIR, HEYGEN_API_KEY, IMAGE_MODEL, OPENROUTER_API_KEY, PLANNER_MODEL, VIDEO_MODEL

API = "https://openrouter.ai/api/v1"
CATALOG_TTL = 1800
OBSERVATIONS = DATA_DIR / "image_cost_observations.json"

# What the user asked for by name. The catalog lookup reports whether a model with exactly this name exists.
REQUESTED = {
    "planner": ("Qwen 3.7 Plus", PLANNER_MODEL),
    "image": ("GPT Image", IMAGE_MODEL),
    "video": ("HeyGen Video", VIDEO_MODEL),
}
# Image output tokens assumed before this app has seen a real bill for a model (OpenAI's published
# portrait sizes for GPT Image). Replaced by measured costs after the first real image.
ASSUMED_IMAGE_TOKENS = {"low": 410, "medium": 1600, "high": 6250, "auto": 1600}
ASSUMED_REFERENCE_TOKENS = 350      # input tokens per reference image
UNKNOWN_IMAGE_PRICE = 0.25          # used for the budget check only when a model publishes no price

_lock = threading.Lock()
_catalog = None
_catalog_time = 0.0
_must_reason: set[str] = set()


class ProviderError(RuntimeError):
    """A provider call that failed. `kind` is one of: missing_key, invalid_key, no_credits, rate_limit,
    bad_request, provider, network."""

    def __init__(self, message, kind="provider", status=None):
        super().__init__(message)
        self.kind, self.status = kind, status


# ---------- HTTP ----------

def _headers():
    if not OPENROUTER_API_KEY:
        raise ProviderError("OpenRouter key is missing. Add open_router_api=sk-or-... to .env and restart the app.",
                            "missing_key")
    return {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}


def _request(method, url, what, **kwargs):
    """One OpenRouter call. Returns parsed JSON or raises ProviderError with a plain message."""
    kwargs.setdefault("timeout", 120)
    try:
        response = requests.request(method, url, headers=_headers(), **kwargs)
    except requests.RequestException as e:
        raise ProviderError(f"{what}: could not reach OpenRouter ({type(e).__name__}).", "network") from None
    try:
        data = response.json()
    except ValueError:
        data = {}
    if response.ok and isinstance(data, dict) and not data.get("error"):
        return data
    error = data.get("error") if isinstance(data, dict) else None
    detail = (error.get("message") if isinstance(error, dict) else str(error or "")) or response.text[:300]
    status = response.status_code
    if status == 401:
        raise ProviderError(f"{what}: OpenRouter rejected the API key (HTTP 401). Check open_router_api in .env. [{detail}]",
                            "invalid_key", status)
    if status == 402:
        raise ProviderError(f"{what}: not enough OpenRouter credits (HTTP 402). Add credits or raise the key's limit. [{detail}]",
                            "no_credits", status)
    if status == 429:
        raise ProviderError(f"{what}: OpenRouter rate limit reached (HTTP 429). [{detail}]", "rate_limit", status)
    kind = "bad_request" if 400 <= status < 500 else "provider"
    raise ProviderError(f"{what}: provider error (HTTP {status}). {detail}"[:700], kind, status)


def _public(url):
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json()


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


# ---------- catalog ----------

def _simplify(name: str) -> str:
    """'OpenAI: GPT Image 2' -> 'gptimage2' (vendor prefix, spaces and punctuation removed)."""
    return re.sub(r"[^a-z0-9]", "", name.split(":", 1)[-1].lower())


def _resolve(kind, models):
    """Find the requested default in the live list. Returns {id, exact, note}."""
    wanted_name, wanted_id = REQUESTED[kind]
    by_id = {m["id"]: m for m in models}
    if wanted_id in by_id:
        model = by_id[wanted_id]
        if _simplify(model["name"]) == _simplify(wanted_name):
            return {"requested": wanted_name, "id": wanted_id, "exact": True, "note": ""}
        return {"requested": wanted_name, "id": wanted_id, "exact": False,
                "note": f'No model is named exactly "{wanted_name}". Using the closest match: {model["name"]} ({wanted_id}).'}
    close = difflib.get_close_matches(wanted_id, list(by_id), n=1, cutoff=0.4)
    if close:
        return {"requested": wanted_name, "id": close[0], "exact": False,
                "note": f'"{wanted_name}" ({wanted_id}) is not in OpenRouter\'s catalog. Using the closest match: '
                        f'{by_id[close[0]]["name"]} ({close[0]}).'}
    return {"requested": wanted_name, "id": None, "exact": False,
            "note": f'"{wanted_name}" ({wanted_id}) is not in OpenRouter\'s catalog and nothing similar was found.'}


def _image_pricing(model_id):
    try:
        endpoints = _public(f"{API}/images/models/{model_id}/endpoints").get("endpoints") or []
        return endpoints[0].get("pricing") or [] if endpoints else []
    except Exception:
        return None


def catalog(force=False) -> dict:
    """Live OpenRouter models and prices for the planner, image and video dropdowns (cached for 30 minutes)."""
    global _catalog, _catalog_time
    with _lock:
        if _catalog and not force and time.monotonic() - _catalog_time < CATALOG_TTL:
            return _catalog
        result = {"planner_models": [], "image_models": [], "video_models": [], "defaults": {}, "error": None,
                  "keys": {"openrouter": bool(OPENROUTER_API_KEY), "heygen": bool(HEYGEN_API_KEY)},
                  "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            for m in _public(f"{API}/models")["data"]:
                arch = m.get("architecture") or {}
                price = m.get("pricing") or {}
                if arch.get("output_modalities") != ["text"] or "text" not in arch.get("input_modalities", []):
                    continue
                try:
                    price_in, price_out = float(price.get("prompt", -1)), float(price.get("completion", -1))
                except (TypeError, ValueError):
                    continue
                if price_in < 0 or price_out < 0:
                    continue   # routers with variable pricing cannot be budgeted
                result["planner_models"].append({
                    "id": m["id"], "name": m.get("name") or m["id"], "vision": "image" in arch.get("input_modalities", []),
                    "price_in": price_in * 1e6, "price_out": price_out * 1e6,   # USD per million tokens
                    "json_mode": "response_format" in (m.get("supported_parameters") or []),
                })
            images = _public(f"{API}/images/models")["data"]
            with ThreadPoolExecutor(16) as pool:
                prices = list(pool.map(_image_pricing, [m["id"] for m in images]))
            for m, pricing in zip(images, prices):
                params = m.get("supported_parameters") or {}
                refs = params.get("input_references") or {}
                result["image_models"].append({
                    "id": m["id"], "name": m.get("name") or m["id"],
                    "max_references": int(refs.get("max", 0)) if isinstance(refs, dict) else 0,
                    "min_references": int(refs.get("min", 0)) if isinstance(refs, dict) else 0,
                    "aspect_ratios": (params.get("aspect_ratio") or {}).get("values") or [],
                    "qualities": (params.get("quality") or {}).get("values") or [],
                    "formats": (params.get("output_format") or {}).get("values") or [],
                    "pricing": pricing,
                })
            for m in _public(f"{API}/videos/models")["data"]:
                result["video_models"].append({
                    "id": m["id"], "name": m.get("name") or m["id"], "durations": m.get("supported_durations") or [],
                    "resolutions": m.get("supported_resolutions") or [], "aspect_ratios": m.get("supported_aspect_ratios") or [],
                    "pricing": m.get("pricing_skus") or {}, "frame_images": m.get("supported_frame_images") or [],
                })
            result["planner_models"].sort(key=lambda m: m["name"].lower())
            result["image_models"].sort(key=lambda m: m["name"].lower())
            for kind, key in (("planner", "planner_models"), ("image", "image_models"), ("video", "video_models")):
                result["defaults"][kind] = _resolve(kind, result[key])
            for m in result["image_models"]:
                m["estimate"] = estimate_image(m, "medium", 0)
        except Exception as e:
            result["error"] = f"Could not load OpenRouter's live model catalog ({type(e).__name__}). Check the internet connection and reload."
            if _catalog:
                return _catalog
            return result            # not cached: the next request tries again
        _catalog, _catalog_time = result, time.monotonic()
        return result


def model(kind: str, model_id: str) -> dict:
    """Catalog entry for a planner / image / video model, or ValueError."""
    for m in catalog()[f"{kind}_models"]:
        if m["id"] == model_id:
            return m
    raise ValueError(f"The {kind} model {model_id!r} is not in OpenRouter's current catalog.")


# ---------- cost estimates (before a call) ----------

def _observations():
    try:
        return json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def record_image_cost(model_id, quality, references, cost):
    """Remember what a real image cost, so later estimates use measured prices."""
    if not cost:
        return
    with _lock:
        data = _observations()
        entry = data.setdefault(f"{model_id}|{quality}|{'refs' if references else 'norefs'}", {"n": 0, "total": 0.0})
        entry["n"] += 1
        entry["total"] += cost
        OBSERVATIONS.write_text(json.dumps(data, indent=1), encoding="utf-8")


def estimate_image(image_model: dict, quality: str, references: int) -> dict:
    """Estimated price of one image: {cost, basis}. `basis` shows units x rate or the measured average."""
    seen = _observations().get(f"{image_model['id']}|{quality}|{'refs' if references else 'norefs'}")
    if seen and seen["n"]:
        return {"cost": seen["total"] / seen["n"], "basis": f"average of {seen['n']} earlier image(s) with this model and quality"}
    pricing = image_model.get("pricing")
    if not pricing:
        return {"cost": UNKNOWN_IMAGE_PRICE, "basis": f"model publishes no price; ${UNKNOWN_IMAGE_PRICE:.2f} assumed for the budget check"}
    total, parts = 0.0, []
    # One output price (the first listed tier) plus, with references, one input price.
    output = next((p for p in pricing if p.get("billable") == "output_image"), None)
    reference = next((p for p in pricing if p.get("billable") == "input_image"), None) if references else None
    if output:
        rate, unit = float(output.get("cost_usd") or 0), output.get("unit")
        if unit == "image":
            total += rate
            parts.append(f"1 image x ${rate:g}")
        elif unit == "megapixel":
            total += rate
            parts.append(f"~1 megapixel x ${rate:g}")
        elif unit == "token":
            tokens = ASSUMED_IMAGE_TOKENS.get(quality, 1600)
            total += tokens * rate
            parts.append(f"~{tokens} image tokens (assumed) x ${rate * 1e6:g}/M")
    if reference:
        rate, unit = float(reference.get("cost_usd") or 0), reference.get("unit")
        if unit == "image":
            total += references * rate
            parts.append(f"{references} reference image(s) x ${rate:g}")
        elif unit == "token":
            total += references * ASSUMED_REFERENCE_TOKENS * rate
            parts.append(f"{references} reference(s) x ~{ASSUMED_REFERENCE_TOKENS} tokens x ${rate * 1e6:g}/M")
    if not total:
        return {"cost": UNKNOWN_IMAGE_PRICE, "basis": f"price format not recognised; ${UNKNOWN_IMAGE_PRICE:.2f} assumed for the budget check"}
    return {"cost": total, "basis": " + ".join(parts)}


def estimate_video(video_model: dict, seconds: int, resolution: str, references: int) -> dict:
    """Estimated price of one clip from OpenRouter's published per-second rates."""
    pricing, res = video_model.get("pricing") or {}, resolution.lower()
    prefix = ("long_" if seconds > 15 else "") + ("reference_" if references else "")
    rate = pricing.get(f"{prefix}duration_seconds_{res}") or pricing.get(f"duration_seconds_{res}") or pricing.get("duration_seconds")
    if rate is None:
        return {"cost": seconds * 0.2, "basis": f"no published rate for {resolution}; $0.20/s assumed for the budget check"}
    rate = float(rate)
    cost, basis = seconds * rate, f"{seconds} s x ${rate:g}/s ({resolution}{', over 15 s' if seconds > 15 else ''}{', with references' if references else ''})"
    if references and pricing.get("reference_input_images"):
        per_image = float(pricing["reference_input_images"])
        cost += references * per_image
        basis += f" + {references} reference image(s) x ${per_image:g}"
    return {"cost": cost, "basis": basis}


def estimate_planner(planner_model: dict, characters: int, images: int) -> dict:
    """Rough planner price: ~4 characters per token, ~1,100 tokens per attached frame, up to ~2,500 output tokens."""
    tokens_in, tokens_out = characters // 4 + images * 1100 + 900, 2500
    cost = (tokens_in * planner_model["price_in"] + tokens_out * planner_model["price_out"]) / 1e6
    return {"cost": cost, "basis": f"~{tokens_in} tokens in x ${planner_model['price_in']:g}/M + ~{tokens_out} out x ${planner_model['price_out']:g}/M"}


# ---------- paid calls ----------

def chat(model_id: str, system: str, content: list, json_mode: bool) -> dict:
    """One planner call. Returns {text, tokens_in, tokens_out, cost, cost_reported, latency, generation_id}."""
    body = {"model": model_id, "temperature": 0.4, "max_tokens": 8000,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            # Reasoning off keeps planning fast and cheap. Models that cannot turn it off get the lowest effort.
            "reasoning": {"effort": "low"} if model_id in _must_reason else {"enabled": False}}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    started = time.monotonic()
    try:
        data = _request("POST", f"{API}/chat/completions", "Planner", json=body, timeout=300)
    except ProviderError as e:
        if "reasoning is mandatory" in str(e).lower() and model_id not in _must_reason:
            _must_reason.add(model_id)
            return chat(model_id, system, content, json_mode)
        raise
    usage = data.get("usage") or {}
    try:
        text = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        raise ProviderError("Planner: OpenRouter returned no message.", "provider") from None
    return {"text": text, "tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens"),
            "cost": usage.get("cost"), "cost_reported": usage.get("cost") is not None,
            "latency": time.monotonic() - started, "generation_id": data.get("id")}


def generate_image(model_id: str, prompt: str, aspect_ratio: str | None, quality: str | None, references: list[Path]) -> dict:
    """One image. Returns {bytes, media_type, cost, cost_reported, tokens, latency}."""
    body = {"model": model_id, "prompt": prompt, "n": 1}
    if aspect_ratio:
        body["aspect_ratio"] = aspect_ratio
    if quality:
        body["quality"] = quality
    if references:
        body["input_references"] = [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in references]
    started = time.monotonic()
    data = _request("POST", f"{API}/images", "Image", json=body, timeout=420)
    try:
        item = data["data"][0]
        raw = base64.b64decode(item["b64_json"], validate=True)
    except (KeyError, IndexError, ValueError, TypeError):
        raise ProviderError("Image: the provider returned no image data.", "provider") from None
    usage = data.get("usage") or {}
    return {"bytes": raw, "media_type": item.get("media_type"), "cost": usage.get("cost"),
            "cost_reported": usage.get("cost") is not None,
            "tokens": (usage.get("completion_tokens_details") or {}).get("image_tokens") or usage.get("completion_tokens"),
            "latency": time.monotonic() - started}


def submit_video(model_id: str, prompt: str, seconds: int, resolution: str, aspect_ratio: str, references: list[Path]) -> dict:
    """Start one clip. Returns {job_id, status}. The job keeps running at the provider; poll it with video_status()."""
    # HeyGen Video always renders its own audio track (sending generate_audio=false is rejected with HTTP 400).
    # That track is dropped during assembly: the final video's only audio is the narration.
    body = {"model": model_id, "prompt": prompt, "duration": int(seconds), "resolution": resolution,
            "aspect_ratio": aspect_ratio}
    if references:
        body["input_references"] = [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in references]
    data = _request("POST", f"{API}/videos", "Video clip", json=body, timeout=120)
    if not data.get("id"):
        raise ProviderError("Video clip: OpenRouter returned no job ID.", "provider")
    return {"job_id": data["id"], "status": data.get("status") or "pending"}


def video_status(job_id: str) -> dict:
    """{status, cost, error}. status: pending | in_progress | completed | failed | cancelled | expired."""
    data = _request("GET", f"{API}/videos/{job_id}", "Video clip status", timeout=60)
    usage = data.get("usage") or {}
    return {"status": data.get("status"), "cost": usage.get("cost"), "error": data.get("error")}


def download_video(job_id: str, destination: Path):
    """Save a finished clip. OpenRouter proxies the bytes, so this needs the API key and should run right away."""
    try:
        with requests.get(f"{API}/videos/{job_id}/content", params={"index": 0}, headers=_headers(),
                          stream=True, timeout=(20, 300)) as response:
            if not response.ok:
                raise ProviderError(f"Video clip download failed (HTTP {response.status_code}).", "provider", response.status_code)
            with open(destination, "wb") as out:
                for chunk in response.iter_content(1024 * 1024):
                    out.write(chunk)
    except requests.RequestException as e:
        raise ProviderError(f"Video clip download failed ({type(e).__name__}).", "network") from None


def key_usage() -> float | None:
    """Total USD this OpenRouter key has spent so far (None if it cannot be read)."""
    try:
        return float(_request("GET", f"{API}/key", "Key check", timeout=30)["data"]["usage"])
    except (ProviderError, KeyError, TypeError, ValueError):
        return None


def check_key() -> dict:
    """Is the configured key usable? {ok, message, remaining}"""
    try:
        data = _request("GET", f"{API}/key", "Key check", timeout=30)["data"]
    except ProviderError as e:
        return {"ok": False, "message": str(e), "kind": e.kind}
    return {"ok": True, "message": "", "remaining": data.get("limit_remaining"), "usage": data.get("usage")}
