"""Transcript text tools through OpenRouter (Qwen): translate, custom rewrite, new video idea.

Every result is held to the same length as the source text:
1. the model writes a draft,
2. the length is counted here in code,
3. if it is off, selected sentences are sent back with an exact per-sentence length to hit.
"""
import json
import re

import requests

from config import OPENROUTER_API_KEY, OPENROUTER_MODEL

API_URL = "https://openrouter.ai/api/v1/chat/completions"
LANGUAGES = ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]

# OpenRouter models offered in the page. The first one is the default.
MODELS = list(dict.fromkeys([
    OPENROUTER_MODEL,
    "qwen/qwen3.7-plus",
    "qwen/qwen3.7-max",
    "qwen/qwen3.7-flash",
    "deepseek/deepseek-v4.1-flash",
    "google/gemini-3.8-flash",
    "openai/gpt-5.6-terra",
    "anthropic/claude-sonnet-5.5",
    "moonshotai/kimi-k3.1-pro",
    "z-ai/glm-5.3",
    "x-ai/grok-4.7",
]))
IDEA_COUNT = 5

TOLERANCE = 0.03       # accepted length difference
MIN_SLACK = 2          # short texts: always accept +-2 units (3% of 40 words is 1 word)
MAX_FIT_PASSES = 5     # length corrections after the first draft
MAX_SENTENCE_CHANGE = 0.4  # how much one sentence may grow or shrink in a single pass
MAX_GAIN = 2.5         # most a correction may over-ask to make up for a model that under-delivers

# Chinese and Japanese are counted in characters, every other language in words.
CHAR_LANGUAGES = {"Chinese", "Japanese"}
# Units that carry the same content as one English word (measured on natural translations).
# Used to convert "same length" between languages that are counted differently.
RATES = {"Chinese": 1.5, "Japanese": 2.25, "Korean": 0.75}

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")   # kana + Chinese characters
_KANA = re.compile(r"[぀-ヿ]")
_HANGUL = re.compile(r"[가-힯]")

SYSTEM = (
    "You write and translate spoken video scripts. "
    "Reply with the final script text only: no title, no notes, no headings, no markdown, no quotation marks around it."
)

TASKS = {
    "translate": "Translate the transcript below into {language}. Keep the meaning, tone and order of ideas.",
    "custom": (
        "Rewrite the transcript below, applying these changes requested by the user:\n{prompt}\n\n"
        "Write the result in {language}."
    ),
    "idea": (
        "Write the complete spoken script for a new video, in {language}, based on this idea:\n"
        "Title: {title}\nIdea: {pitch}\n\n"
        "Copy the hook, structure, pacing and style of the transcript below, but not its subject: "
        "do not reuse sentences from it.{extra}"
    ),
}

IDEAS_PROMPT = (
    "Study the transcript below: its topic, niche, hook, structure, pacing and style. "
    "Suggest {count} new video ideas that are similar (same niche, format and audience) "
    "but each with a different specific subject or angle.{extra}\n"
    "Write the titles and descriptions in {language}.\n"
    'Reply with JSON only, no code fence: [{{"title": "...", "pitch": "two or three sentences on what the video covers"}}]\n\n'
    "Transcript:\n{text}"
)


# ---------- length ----------

def _detect(text: str) -> str | None:
    """Chinese, Japanese or Korean if the text is mostly in that script, else None."""
    size = len(re.sub(r"\s", "", text)) or 1
    if len(_HANGUL.findall(text)) > 0.3 * size:
        return "Korean"
    if len(_CJK.findall(text)) > 0.3 * size:
        return "Japanese" if _KANA.search(text) else "Chinese"
    return None


def measure(text: str, language: str | None) -> int:
    """Length in characters for Chinese/Japanese, in words otherwise."""
    return len(_CJK.findall(text)) if language in CHAR_LANGUAGES else len(text.split())


def target_length(source: str, language: str) -> tuple[int, str]:
    """Length the result needs in `language` to match the source text, as (count, unit)."""
    source_language = _detect(source)
    english_words = measure(source, source_language) / RATES.get(source_language, 1.0)
    unit = "characters" if language in CHAR_LANGUAGES else "words"
    return max(1, round(english_words * RATES.get(language, 1.0))), unit


# ---------- model ----------

_MUST_REASON: set[str] = set()   # models that refuse to run with reasoning off


def _chat(user: str, model: str) -> str:
    # Reasoning off: with it on, models spend minutes counting words before answering.
    # Some models cannot turn it off; they get the lowest effort instead.
    reasoning = {"effort": "low"} if model in _MUST_REASON else {"enabled": False}
    r = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "temperature": 0.7,
            "max_tokens": 16000,   # enough for a long script; without it some models reserve their full limit in credits
            "reasoning": reasoning,
        },
        timeout=300,
    )
    data = r.json()
    if r.status_code != 200 or "choices" not in data:
        message = (data.get("error") or {}).get("message") or r.text[:200]
        if "reasoning is mandatory" in message.lower() and model not in _MUST_REASON:
            _MUST_REASON.add(model)
            return _chat(user, model)
        raise RuntimeError(f"OpenRouter error: {message}")
    return (data["choices"][0]["message"].get("content") or "").strip()


# ---------- fitting the length ----------

CHUNK_WORDS = 15   # long runs without punctuation are fitted in pieces of this many words


def _sentences(text: str) -> list[tuple[int, str]]:
    """Split into (paragraph index, sentence) pairs. Very long unpunctuated runs are cut into pieces."""
    out = []
    for p, para in enumerate(x for x in re.split(r"\n+", text) if x.strip()):
        for s in re.split(r"(?<=[.!?…])\s+|(?<=[。！？])", para):
            words = s.split()
            if len(words) > 2 * CHUNK_WORDS:   # only happens in space-separated text
                out += [(p, " ".join(words[i:i + CHUNK_WORDS])) for i in range(0, len(words), CHUNK_WORDS)]
            elif s.strip():
                out.append((p, s.strip()))
    return out


def _budgets(counts: list[int], target: int) -> list[int]:
    """New length per sentence so the total reaches `target`.

    Only as many sentences as needed are changed, longest first, each by a limited amount.
    """
    budgets = list(counts)
    delta = target - sum(counts)
    for i in sorted(range(len(counts)), key=lambda i: counts[i], reverse=True):
        if delta == 0:
            break
        step = min(abs(delta), max(1, int(counts[i] * MAX_SENTENCE_CHANGE)))
        if delta < 0:
            step = min(step, counts[i] - 1)
        budgets[i] += step if delta > 0 else -step
        delta += -step if delta > 0 else step
    return budgets


def _fit(text: str, language: str, target: int, unit: str, model: str) -> str | None:
    """Move `text` toward `target` by rewriting selected sentences to an exact length each.

    Models cannot hit a large total reliably, but they follow a small per-sentence length well.
    """
    sents = _sentences(text)
    counts = [measure(s, language) for _, s in sents]
    budgets = _budgets(counts, target)
    todo = [i for i in range(len(sents)) if budgets[i] != counts[i]]
    if not todo:
        return None

    lines = "\n".join(f"[{i + 1}] ({counts[i]} {unit} now, must be {budgets[i]}) {sents[i][1]}" for i in todo)
    counting = ("Count only Chinese characters and kana, not punctuation." if unit == "characters"
                else "A word is anything between two spaces.")
    reply = _chat(
        f"Below are numbered sentences from a spoken script in {language}. "
        f"Each line says how long the sentence is now and how long it must be, counted in {unit}. {counting} "
        f"Rewrite each sentence to exactly that length. To shorten, cut filler and minor detail. "
        f"To lengthen, add meaningful detail that fits the sentence: never filler words, interjections or repetition. "
        f"Keep the meaning and the natural spoken tone. Never merge or split sentences.\n"
        f"Reply with every sentence on its own line as `[number] sentence` and nothing else.\n\n{lines}",
        model,
    )
    fixed = dict(re.findall(r"^\[(\d+)\]\s*(?:\([^)]*\)\s*)?(.+)$", reply, flags=re.M))
    if any(str(i + 1) not in fixed for i in todo):
        return None

    paragraphs: dict[int, list[str]] = {}
    for i, (p, sentence) in enumerate(sents):
        paragraphs.setdefault(p, []).append(fixed[str(i + 1)].strip() if i in todo else sentence)
    joiner = "" if language in CHAR_LANGUAGES else " "
    return "\n\n".join(joiner.join(v) for v in paragraphs.values())


# ---------- entry point ----------

def _check_key():
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OpenRouter key missing: set open_router_api in .env")


def ideas(text: str, language: str, model: str, prompt: str = "") -> list[dict]:
    """Several new video ideas similar to the transcript, as [{"title", "pitch"}]."""
    _check_key()
    extra = f"\nExtra direction from the user: {prompt}" if prompt else ""
    reply = _chat(IDEAS_PROMPT.format(count=IDEA_COUNT, extra=extra, language=language, text=text), model)
    match = re.search(r"\[.*\]", reply, flags=re.S)
    try:
        items = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        items = []
    items = [{"title": str(i.get("title", "")).strip(), "pitch": str(i.get("pitch", "")).strip()}
             for i in items if isinstance(i, dict) and i.get("title")]
    if not items:
        raise RuntimeError("The model did not return a usable list of ideas. Try again or pick another model.")
    return items


def generate(mode: str, text: str, language: str, model: str, prompt: str = "", idea: dict | None = None) -> dict:
    _check_key()
    target, unit = target_length(text, language)
    slack = max(MIN_SLACK, round(target * TOLERANCE))
    low, high = target - slack, target + slack

    extra = f"\nExtra direction from the user: {prompt}" if prompt else ""
    idea = idea or {}
    task = TASKS[mode].format(language=language, prompt=prompt, extra=extra,
                              title=idea.get("title", ""), pitch=idea.get("pitch", ""))
    best = _chat(
        f"{task}\n\nLength: about {target} {unit}, so it takes the same time to speak as the original.\n\n"
        f"Transcript:\n{text}",
        model,
    )
    best_count = measure(best, language)

    # Models often deliver only part of a requested length change (Korean: about half),
    # so `gain` learns how much to over-ask from what the previous pass actually delivered.
    gain = 1.0
    for _ in range(MAX_FIT_PASSES):
        if low <= best_count <= high:
            break
        wanted = target - best_count
        fitted = _fit(best, language, best_count + round(wanted * gain), unit, model)
        if not fitted:
            continue
        got = measure(fitted, language) - best_count
        if got * wanted > 0:
            gain = min(MAX_GAIN, max(1.0, gain * wanted / got))
        if abs(measure(fitted, language) - target) < abs(best_count - target):
            best, best_count = fitted, measure(fitted, language)

    source_language = _detect(text)
    return {
        "text": best,
        "unit": unit,
        "target": target,
        "count": best_count,
        "within_tolerance": low <= best_count <= high,
        "source_count": measure(text, source_language),
        "source_unit": "characters" if source_language in CHAR_LANGUAGES else "words",
    }
