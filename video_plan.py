"""Visual plan for Generate Video: planner prompt, strict validation of the planner's JSON, timeline math.

Pure functions only (no network, no files), so everything here is unit-tested.
"""
import json
import math
import re

FPS = 24                 # final video frame rate (HeyGen Video clips are 24 fps too)
CROSSFADE_FRAMES = 10    # 0.42 s crossfade centred on each boundary: the next visual is fully on screen 0.21 s after its text starts
SNAP = 0.5               # planner times this close to the narration start/end are snapped onto them
JOIN = 0.1               # neighbouring visuals this close together are joined exactly
MAX_VISUALS = 60
VIDEO_MIN, VIDEO_MAX = 5, 30     # HeyGen Video clip length in whole seconds
MODES = {"images": {"image"}, "videos": {"video"}, "mixed": {"image", "video"}}
MODE_NAMES = {"images": "Images only", "videos": "Videos only", "mixed": "Videos + Images"}

DEFAULT_SYSTEM_PROMPT = """You are the visual director for a narrated explainer video. You receive the timestamped transcript of the narration and plan the visuals that play over it. The narration audio already exists; you only decide what is on screen and when.

YOUR JOB
- Read the transcript in order and split it into consecutive visuals.
- Start a new visual whenever the subject, example, step or idea changes, so that each visual shows what is being said at that moment.
- Each visual must illustrate its own part of the transcript literally and specifically: the objects, actions and relationships the words describe. No generic stock imagery, no decorative filler, no repeating the same picture.
- Keep one consistent look across the whole video and describe it once in "style" (medium, lighting, colour palette, camera feel).

RULES FOR EVERY GENERATION PROMPT
- English, and fully self-contained: the image or video model sees only that one prompt, never the transcript or the other prompts. Name the subject, setting, composition, lighting and colours explicitly.
- Realistic and concrete: a real-looking scene, or a clean physical or 3D setup that makes the idea visible.
- No text of any kind in the picture: no words, letters, numbers, captions, labels, equations, logos or watermarks. When the narration mentions a formula or a number, show the idea itself instead.
- No real or identifiable people and no brands.
- For a video clip: one continuous shot with one simple motion and a mostly steady camera. No dialogue, no speech, no music.

TIMING
- Use the transcript's own timestamps. A visual starts when the first words it covers are spoken and ends where the next visual starts.
- Visuals are contiguous: the first starts at 0, each one starts exactly where the previous one ends, and the last one ends exactly at the end of the narration. No gaps and no overlaps.
- Respect the limits in the request: allowed visual types, minimum time on screen, clip lengths, maximum number of visuals.

SOURCE FRAMES
- You may be given frames from the original video. Use them to understand what the narration refers to. Never copy their text, captions or layout.
- If a frame would help the generator with a subject or composition, list its index in "reference_frames" (at most 2 per visual). Otherwise leave that list empty.

OUTPUT
Return one JSON object and nothing else:
{"style": "the shared visual style, 1-3 sentences",
 "visuals": [
   {"id": 1, "type": "image", "start": 0.0, "end": 4.8,
    "text": "the exact transcript words this visual covers",
    "reason": "why this picture fits these words",
    "prompt": "the self-contained generation prompt",
    "reference_frames": []}
 ]}
"type" is "image" or "video". "start" and "end" are seconds."""


class PlanError(ValueError):
    """The planner's output broke one or more rules. `problems` is fed back to the planner on a retry."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__(" ".join(self.problems))


# ---------- transcript ----------

def transcript_segments(transcript, duration):
    """Timed caption lines -> [{start, end, text}] in order. A line ends where the next begins.

    Bracketed sound tags such as [music] are removed; lines left empty are dropped.
    """
    lines = []
    for s in sorted(transcript or [], key=lambda s: s["start"]):
        text = " ".join(re.sub(r"\[[^\]]*\]|\([^)]*(?:music|applause|laughter)[^)]*\)", " ", s.get("text") or "",
                               flags=re.I).split())
        if text and 0 <= s["start"] < duration:
            lines.append({"start": float(s["start"]), "text": text})
    for i, line in enumerate(lines):
        line["end"] = lines[i + 1]["start"] if i + 1 < len(lines) else float(duration)
    return [line for line in lines if line["end"] > line["start"]]


def text_between(segments, start, end):
    """Transcript words spoken while a visual is on screen: lines that begin inside [start, end)."""
    inside = [s["text"] for s in segments if start - 0.05 <= s["start"] < end - 0.05]
    if not inside:   # the visual sits inside one long line
        inside = [s["text"] for s in segments if s["start"] <= start < s["end"]]
    return " ".join(inside)


# ---------- planner request ----------

def planner_request(*, mode, title, duration, narration_label, width, height, aspect_ratio, segments, frames,
                    frames_attached, limits):
    """The text part of the planner's input (frames are attached separately when the model can read images)."""
    allowed = {"images": 'every visual has type "image"', "videos": 'every visual has type "video"',
               "mixed": 'use both types: "video" where motion helps the words, "image" elsewhere (at least one of each)'}[mode]
    lines = [
        f"MODE: {MODE_NAMES[mode]} - {allowed}.",
        f"VIDEO TITLE: {title}",
        f"NARRATION: {duration:.2f} seconds ({narration_label}). The last visual must end at {duration:.2f}.",
        f"OUTPUT FRAME: {aspect_ratio} ({width}x{height} pixels).",
        "LIMITS:",
    ]
    if "image" in MODES[mode]:
        lines.append(f"- image: on screen for at least {limits['min_seconds']:g} seconds.")
    if "video" in MODES[mode]:
        lines.append(f"- video: each clip lasts {max(VIDEO_MIN, limits['min_seconds']):g} to {VIDEO_MAX} seconds (HeyGen Video). "
                     f"Clips longer than 15 seconds cost more than twice as much per second, so prefer 5-15 seconds.")
    lines += [f"- at most {limits['max_visuals']} visuals in total (budget).", "", "TRANSCRIPT (seconds):"]
    lines += [f"[{s['start']:.2f} - {s['end']:.2f}] {s['text']}" for s in segments]
    lines.append("")
    if not frames:
        lines.append("SOURCE FRAMES: none available for this job.")
    elif frames_attached:
        lines.append("SOURCE FRAMES from the original video (attached as images, in this order):")
        lines += [f"- frame {f['index']}: at {f['time']:g} s" for f in frames]
    else:
        lines.append("SOURCE FRAMES from the original video (this model cannot read images, so each frame is described "
                     "by the words spoken at that moment):")
        lines += [f"- frame {f['index']}: at {f['time']:g} s - \"{f['caption']}\"" for f in frames]
    return "\n".join(lines)


def parse_json(text):
    """The planner's reply as a dict. Accepts a ```json fence around it."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        data = json.loads(cleaned)
    except ValueError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        try:
            data = json.loads(match.group(0)) if match else None
        except ValueError:
            data = None
    if not isinstance(data, dict):
        raise PlanError(["The reply was not a JSON object. Return exactly one JSON object with \"style\" and \"visuals\"."])
    return data


# ---------- validation ----------

def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def snap(seconds):
    """Round a time to the frame grid."""
    return round(seconds * FPS) / FPS


def validate_plan(raw, *, mode, duration, segments, frame_count, min_seconds, max_visuals):
    """Check the planner's JSON against every rule. Returns (style, visuals) or raises PlanError listing all problems.

    Returned visuals: {id, type, start, end, duration, text, planner_text, reason, scene, reference_frames}.
    Times are snapped to the frame grid; `text` is the transcript text actually spoken in that time range.
    """
    problems = []
    style = raw.get("style")
    if not isinstance(style, str) or len(style.strip()) < 15:
        problems.append('"style" must describe the shared visual style in at least one sentence.')
    items = raw.get("visuals")
    if not isinstance(items, list) or not items:
        raise PlanError(problems + ['"visuals" must be a non-empty list.'])
    if len(items) > max_visuals:
        problems.append(f"The plan has {len(items)} visuals but at most {max_visuals} are allowed. Merge neighbouring visuals.")

    cleaned = []
    for n, item in enumerate(items, 1):
        if not isinstance(item, dict):
            problems.append(f"Visual {n} is not an object.")
            continue
        kind, start, end = item.get("type"), item.get("start"), item.get("end")
        if kind not in MODES[mode]:
            problems.append(f'Visual {n}: type must be {" or ".join(sorted(MODES[mode]))} in this mode, not {kind!r}.')
        if not _number(start) or not _number(end):
            problems.append(f'Visual {n}: "start" and "end" must be numbers in seconds.')
            continue
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not 40 <= len(prompt.strip()) <= 4000:
            problems.append(f'Visual {n}: "prompt" must be a self-contained description of 40 to 4000 characters.')
        for field in ("text", "reason"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                problems.append(f'Visual {n}: "{field}" is missing.')
        refs = item.get("reference_frames") or []
        if not isinstance(refs, list) or any(type(r) is not int or not 0 <= r < frame_count for r in refs):
            problems.append(f'Visual {n}: "reference_frames" must list source frame indices between 0 and {frame_count - 1}.'
                            if frame_count else f'Visual {n}: there are no source frames, so "reference_frames" must be empty.')
            refs = []
        cleaned.append({"n": n, "type": kind, "start": float(start), "end": float(end), "prompt": prompt,
                        "planner_text": item.get("text"), "reason": item.get("reason"), "refs": list(dict.fromkeys(refs))[:2]})
    if not cleaned:
        raise PlanError(problems)

    # Order, coverage, gaps and overlaps.
    if abs(cleaned[0]["start"]) > SNAP:
        problems.append(f"The first visual starts at {cleaned[0]['start']:.2f} but must start at 0.")
    cleaned[0]["start"] = 0.0
    if abs(cleaned[-1]["end"] - duration) > SNAP:
        problems.append(f"The last visual ends at {cleaned[-1]['end']:.2f} but must end at {duration:.2f}, the end of the narration.")
    cleaned[-1]["end"] = float(duration)
    for a, b in zip(cleaned, cleaned[1:]):
        gap = b["start"] - a["end"]
        if gap > JOIN:
            problems.append(f"Gap of {gap:.2f} s between visual {a['n']} (ends {a['end']:.2f}) and visual {b['n']} (starts {b['start']:.2f}).")
        elif gap < -JOIN:
            problems.append(f"Visual {b['n']} starts at {b['start']:.2f}, before visual {a['n']} ends at {a['end']:.2f} (overlap or wrong order).")
        b["start"] = a["end"]

    visuals = []
    for item in cleaned:
        start, end = snap(item["start"]), snap(item["end"])
        length = end - start
        if length <= 0:
            problems.append(f"Visual {item['n']} has no duration.")
            continue
        if item["type"] == "image" and length < min_seconds - 0.05:
            problems.append(f"Visual {item['n']} (image) lasts {length:.2f} s; the minimum is {min_seconds:g} s. Merge it with a neighbour.")
        if item["type"] == "video":
            low = max(VIDEO_MIN, min_seconds)
            if length < low - 0.25:
                problems.append(f"Visual {item['n']} (video) lasts {length:.2f} s; a clip must be at least {low:g} s.")
            if length > VIDEO_MAX:
                problems.append(f"Visual {item['n']} (video) lasts {length:.2f} s; a clip can be at most {VIDEO_MAX} s. Split it.")
        visuals.append({"id": len(visuals) + 1, "type": item["type"], "start": start, "end": end, "duration": length,
                        "text": text_between(segments, start, end), "planner_text": (item["planner_text"] or "").strip(),
                        "reason": (item["reason"] or "").strip(), "scene": (item["prompt"] or "").strip(),
                        "reference_frames": item["refs"]})
    if mode == "mixed" and visuals and {v["type"] for v in visuals} != {"image", "video"}:
        problems.append('Videos + Images mode needs at least one "video" visual and at least one "image" visual.')
    if problems:
        raise PlanError(problems)
    return style.strip(), visuals


def generation_prompt(scene, style, kind, aspect_ratio, references):
    """The full prompt sent to the image or video model: the planner's scene plus the shared style and fixed rules."""
    orientation = {"9:16": "vertical", "16:9": "horizontal widescreen", "1:1": "square"}.get(aspect_ratio, "")
    parts = [scene.strip(), f"Style (shared by every visual in this video): {style.strip()}",
             f"Format: {orientation + ' ' if orientation else ''}{aspect_ratio} frame, main subject kept inside the central area."]
    if kind == "video":
        parts.append("One continuous shot with a mostly steady camera. No dialogue, no speech, no music.")
    if references:
        parts.append("The attached reference image(s) are only a loose guide for subject and composition. "
                     "Do not copy any text, captions or titles from them.")
    parts.append("Strict rules: no text, letters, numbers, captions, labels, logos or watermarks anywhere in the frame. "
                 "No real or identifiable people.")
    return "\n\n".join(parts)


# ---------- timeline ----------

def timeline(visuals, fade_frames=CROSSFADE_FRAMES):
    """Frame-exact assembly plan for contiguous visuals joined by crossfades.

    A crossfade is centred on the boundary between two visuals, so each neighbour is extended by half of it.
    Because every crossfade overlaps two segments, the segments add up to more than the timeline:
        sum(segment frames) - sum(crossfade frames) == total frames
    Returns {total_frames, segments: [{id, start_frame, end_frame, lead, tail, frames}],
             fades: [{frames, offset_frames}]}  where fades[i] joins segment i and i+1 and offset_frames is
    measured from the start of the video (the `offset` of ffmpeg's xfade filter).
    """
    edges = [round(visuals[0]["start"] * FPS)] + [round(v["end"] * FPS) for v in visuals]
    if edges[0] != 0 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("Visuals must start at 0, be in order and each last at least one frame.")
    lengths = [b - a for a, b in zip(edges, edges[1:])]
    halves = []
    for left, right in zip(lengths, lengths[1:]):
        # Never fade for longer than a third of the shorter neighbour.
        halves.append(min(fade_frames // 2, min(left, right) // 3))
    segments, fades = [], []
    for i, visual in enumerate(visuals):
        lead = halves[i - 1] if i else 0
        tail = halves[i] if i < len(halves) else 0
        segments.append({"id": visual["id"], "start_frame": edges[i], "end_frame": edges[i + 1],
                         "lead": lead, "tail": tail, "frames": lengths[i] + lead + tail})
    for i, half in enumerate(halves):
        fades.append({"frames": 2 * half, "offset_frames": edges[i + 1] - half})
    return {"total_frames": edges[-1], "segments": segments, "fades": fades}


def clip_seconds(visual, position, count):
    """Whole seconds to request for a video visual: its time on screen plus the crossfade overlap, within HeyGen's limits."""
    overlap = (CROSSFADE_FRAMES // 2) * ((position > 0) + (position < count - 1)) / FPS
    return min(VIDEO_MAX, max(VIDEO_MIN, math.ceil(visual["duration"] + overlap - 1e-6)))


def final_frames(narration_seconds, original_seconds):
    """Length of the final video in frames, or ValueError when the narration cannot fit the 105% limit.

    The video covers the whole narration (rounded up to a full frame); speech is never cut.
    """
    limit = original_seconds * 1.05
    if narration_seconds > limit + 1e-6:
        raise ValueError(f"The narration is {narration_seconds:.2f} s, longer than 105% of the original video "
                         f"({original_seconds:.2f} s x 1.05 = {limit:.2f} s). Shorten the script or the narration; "
                         f"speech is never cut automatically.")
    frames = math.ceil(narration_seconds * FPS - 1e-6)
    return min(frames, math.floor(limit * FPS + 1e-6)) if frames / FPS > limit else frames
