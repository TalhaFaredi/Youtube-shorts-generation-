"""Assemble generated visuals and the narration into the final MP4 with ffmpeg, then verify it."""
import math
import shutil
from pathlib import Path

import video_plan
from media_tools import is_faststart, mean_volume, probe, run
from video_plan import FPS

MAX_STRETCH = 1.10     # a clip that is too short is slowed by at most 10%; any remainder holds the last frame
ZOOM = 0.06            # stills zoom by 6% over their time on screen


def _even(n):
    return int(n) + int(n) % 2


def _cover(width, height):
    return f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"


def _render_image(asset, destination, frames, width, height, zoom_in, log):
    # zoompan moves in whole pixels, so the still is enlarged first; the zoom then looks smooth after scaling down.
    factor = max(2, math.ceil(1440 / width))
    big_w, big_h = _even(width * factor), _even(height * factor)
    zoom = f"1+{ZOOM}*on/{frames}" if zoom_in else f"1+{ZOOM}-{ZOOM}*on/{frames}"
    filters = (f"{_cover(big_w, big_h)},zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:"
               f"s={width}x{height}:fps={FPS},setsar=1,format=yuv420p")
    run(["-loop", "1", "-framerate", FPS, "-i", asset, "-vf", filters, "-frames:v", frames, "-an",
         "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-r", FPS, destination], log=log)
    return None


def _render_clip(asset, destination, frames, width, height, log):
    """Fit a generated clip to its slot. Returns a note when the clip had to be trimmed, slowed or held."""
    have, need = probe(asset)["video_duration"] or probe(asset)["duration"], frames / FPS
    filters, note = [_cover(width, height)], None
    if have >= need:
        if have - need > 0.05:
            note = f"clip is {have:.2f} s; trimmed to the {need:.2f} s it is on screen"
    else:
        stretch = min(MAX_STRETCH, need / have)
        filters.append(f"setpts={stretch:.6f}*PTS")
        hold = need - have * stretch
        note = f"clip is {have:.2f} s but {need:.2f} s is needed: slowed by {(stretch - 1) * 100:.0f}%"
        if hold > 0.02:
            note += f" and the last frame held for {hold:.2f} s"
    filters += [f"fps={FPS}", "tpad=stop_mode=clone:stop_duration=3", "setsar=1", "format=yuv420p"]
    run(["-i", asset, "-vf", ",".join(filters), "-frames:v", frames, "-an",
         "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-r", FPS, destination], log=log)
    return note


def assemble(*, visuals, assets_dir, work_dir, narration, output, width, height, log, progress=lambda text: None):
    """Build `output` from the visuals (in order, crossfaded) with the narration as its only audio.

    Returns {frames, duration, notes: {visual id: text}, seconds}. Every ffmpeg command goes to `log`.
    """
    plan = video_plan.timeline(visuals)
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    notes, parts, elapsed = {}, [], 0.0
    for i, (visual, segment) in enumerate(zip(visuals, plan["segments"])):
        progress(f"fitting visual {i + 1} of {len(visuals)} to its slot")
        part = work_dir / f"part_{visual['id']:03d}.mp4"
        asset = assets_dir / visual["asset"]
        if visual["type"] == "image":
            _render_image(asset, part, segment["frames"], width, height, i % 2 == 0, log)
        else:
            note = _render_clip(asset, part, segment["frames"], width, height, log)
            if note:
                notes[visual["id"]] = note
                log(f"Visual {visual['id']}: {note}.")
        got = probe(part)
        if got["frames"] is not None and got["frames"] != segment["frames"]:
            raise RuntimeError(f"Visual {visual['id']} rendered {got['frames']} frames instead of {segment['frames']}.")
        parts.append(part)

    progress("joining visuals with crossfades")
    silent = work_dir / "timeline.mp4"
    encode = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", FPS,
              "-frames:v", plan["total_frames"], "-an"]
    if len(parts) == 1:
        run(["-i", parts[0], *encode, silent], log=log)
    else:
        inputs, graph, last = [], [], "0:v"
        for part in parts:
            inputs += ["-i", part]
        for i, fade in enumerate(plan["fades"]):
            label = f"x{i + 1}"
            if fade["frames"]:
                graph.append(f"[{last}][{i + 1}:v]xfade=transition=fade:duration={fade['frames'] / FPS:.6f}:"
                             f"offset={fade['offset_frames'] / FPS:.6f}[{label}]")
            else:   # neighbours too short to fade: plain cut
                graph.append(f"[{last}][{i + 1}:v]concat=n=2:v=1:a=0[{label}]")
            last = label
        run([*inputs, "-filter_complex", ";".join(graph), "-map", f"[{last}]", *encode, silent], log=log)

    progress("adding the narration")
    duration = plan["total_frames"] / FPS
    # apad + -t: the audio is padded with silence up to the last video frame; speech is never cut.
    run(["-i", silent, "-i", narration, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-af", "apad", "-t", f"{duration:.6f}",
         "-movflags", "+faststart", output], log=log)
    return {"frames": plan["total_frames"], "duration": duration, "notes": notes, "timeline": plan}


def verify(output: Path, *, original_seconds, narration_seconds, width, height):
    """Checks on the final MP4. Returns (all_ok, [{name, ok, detail}], facts)."""
    info = probe(output)
    volume = mean_volume(output)
    ratio = info["duration"] / original_seconds if original_seconds else float("inf")
    video_seconds, audio_seconds = info["video_duration"] or 0, info["audio_duration"] or 0
    checks = [
        ("Length is at most 105% of the original", ratio <= 1.05 + 1e-6,
         f"final {info['duration']:.2f} s / original {original_seconds:.2f} s = {ratio * 100:.1f}%"),
        ("Covers the whole narration", info["duration"] >= narration_seconds - 0.05,
         f"narration {narration_seconds:.2f} s, final {info['duration']:.2f} s"),
        ("Has an audio track", info["has_audio"], info["audio_codec"] or "no audio stream"),
        ("Audio is not silent", volume is not None and volume > -60,
         "level could not be measured" if volume is None else f"mean level {volume:.1f} dB"),
        ("Audio and video are the same length", abs(video_seconds - audio_seconds) <= 0.1,
         f"video {video_seconds:.3f} s, audio {audio_seconds:.3f} s"),
        ("Resolution matches the original", (info["width"], info["height"]) == (width, height),
         f"final {info['width']}x{info['height']}, original {width}x{height}"),
        ("H.264 video, AAC audio, yuv420p", (info["video_codec"], info["audio_codec"], info["pix_fmt"]) == ("h264", "aac", "yuv420p"),
         f"{info['video_codec']} / {info['audio_codec']} / {info['pix_fmt']}"),
        ("Fast start (index at the front)", is_faststart(output), "moov atom before mdat" if is_faststart(output) else "moov atom after mdat"),
    ]
    result = [{"name": name, "ok": bool(ok), "detail": detail} for name, ok, detail in checks]
    facts = {"duration": info["duration"], "ratio": ratio, "width": info["width"], "height": info["height"],
             "video_seconds": video_seconds, "audio_seconds": audio_seconds, "mean_volume": volume, "fps": info["fps"]}
    return all(c["ok"] for c in result), result, facts
