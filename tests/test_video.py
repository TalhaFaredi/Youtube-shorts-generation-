"""Unit tests for Generate Video: plan validation, timeline math, cost ledger and budget cap.

No network and no paid calls: providers are never called here.
Run:  .venv/Scripts/python.exe -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Use a throw-away data folder before any app module is imported.
_DATA = tempfile.mkdtemp(prefix="video-tests-")
os.environ["APP_DATA_DIR"] = _DATA
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jobs  # noqa: E402
import video_plan as vp  # noqa: E402
import video_sessions as vs  # noqa: E402
import visual_providers as providers  # noqa: E402
from video_plan import FPS, PlanError  # noqa: E402

SEGMENTS = vp.transcript_segments([
    {"start": 0.32, "text": "Want to predict a house price from its"},
    {"start": 1.96, "text": "[music] size? That's linear regression."},
    {"start": 5.04, "text": "Each dot is one house."},
    {"start": 8.00, "text": "price goes up. Draw a line."},
    {"start": 11.14, "text": "[music]"},
    {"start": 12.96, "text": "are the errors."},
], 20.0)


def visual(kind, start, end, prompt="A clean flat illustration of a scatter plot with cyan dots on a navy background."):
    return {"type": kind, "start": start, "end": end, "text": "words", "reason": "fits", "prompt": prompt, "reference_frames": []}


def validate(items, mode="images", duration=20.0, min_seconds=2.5, max_visuals=10, frames=4, style="Flat navy and cyan diagrams."):
    return vp.validate_plan({"style": style, "visuals": items}, mode=mode, duration=duration, segments=SEGMENTS,
                            frame_count=frames, min_seconds=min_seconds, max_visuals=max_visuals)


class TranscriptTests(unittest.TestCase):
    def test_sound_tags_removed_and_empty_lines_dropped(self):
        self.assertEqual([s["text"] for s in SEGMENTS][1], "size? That's linear regression.")
        self.assertNotIn("[music]", " ".join(s["text"] for s in SEGMENTS))
        self.assertEqual(len(SEGMENTS), 5)

    def test_lines_are_contiguous_and_end_at_duration(self):
        for a, b in zip(SEGMENTS, SEGMENTS[1:]):
            self.assertEqual(a["end"], b["start"])
        self.assertEqual(SEGMENTS[-1]["end"], 20.0)

    def test_text_between(self):
        self.assertEqual(vp.text_between(SEGMENTS, 0, 5.04), "Want to predict a house price from its size? That's linear regression.")
        self.assertEqual(vp.text_between(SEGMENTS, 9, 10), "price goes up. Draw a line.")   # inside one long line


class PlanValidationTests(unittest.TestCase):
    def test_valid_plan(self):
        style, visuals = validate([visual("image", 0, 5.04), visual("image", 5.04, 12.96), visual("image", 12.96, 20)])
        self.assertEqual(style, "Flat navy and cyan diagrams.")
        self.assertEqual([v["id"] for v in visuals], [1, 2, 3])
        self.assertEqual(visuals[0]["start"], 0)
        self.assertEqual(visuals[-1]["end"], 20)
        self.assertIn("Each dot is one house.", visuals[1]["text"])

    def test_times_are_snapped_to_frames_and_small_gaps_joined(self):
        _, visuals = validate([visual("image", 0.2, 5.03), visual("image", 5.08, 20.3)])
        self.assertEqual(visuals[0]["start"], 0)
        self.assertEqual(visuals[0]["end"], visuals[1]["start"])
        self.assertEqual(visuals[1]["end"], 20)
        for v in visuals:
            self.assertAlmostEqual(v["start"] * FPS, round(v["start"] * FPS))

    def assertProblem(self, items, text, **kw):
        with self.assertRaises(PlanError) as ctx:
            validate(items, **kw)
        self.assertTrue(any(text in p for p in ctx.exception.problems), ctx.exception.problems)

    def test_gap_rejected(self):
        self.assertProblem([visual("image", 0, 5), visual("image", 7, 20)], "Gap of 2.00 s")

    def test_overlap_rejected(self):
        self.assertProblem([visual("image", 0, 8), visual("image", 6, 20)], "before visual 1 ends")

    def test_wrong_order_rejected(self):
        self.assertProblem([visual("image", 10, 20), visual("image", 0, 10)], "before visual 1 ends")

    def test_must_start_at_zero_and_cover_the_end(self):
        self.assertProblem([visual("image", 2, 20)], "must start at 0")
        self.assertProblem([visual("image", 0, 15)], "must end at 20.00")

    def test_type_must_match_mode(self):
        self.assertProblem([visual("video", 0, 20)], "type must be image", mode="images")
        self.assertProblem([visual("image", 0, 20)], "type must be video", mode="videos")

    def test_mixed_needs_both_types(self):
        self.assertProblem([visual("image", 0, 10), visual("image", 10, 20)], "at least one \"video\"", mode="mixed")
        _, visuals = validate([visual("video", 0, 8), visual("image", 8, 20)], mode="mixed")
        self.assertEqual({v["type"] for v in visuals}, {"image", "video"})

    def test_minimum_seconds(self):
        self.assertProblem([visual("image", 0, 1), visual("image", 1, 20)], "minimum is 2.5 s")

    def test_video_clip_limits(self):
        self.assertProblem([visual("video", 0, 3), visual("video", 3, 20)], "at least 5 s", mode="videos")
        self.assertProblem([visual("video", 0, 40)], "at most 30 s", mode="videos", duration=40)

    def test_max_visuals_from_budget(self):
        items = [visual("image", i * 4, (i + 1) * 4) for i in range(5)]
        self.assertProblem(items, "at most 3 are allowed", max_visuals=3)

    def test_prompt_style_and_fields_required(self):
        self.assertProblem([visual("image", 0, 20, prompt="short")], "40 to 4000 characters")
        self.assertProblem([visual("image", 0, 20)], '"style"', style="")
        item = visual("image", 0, 20)
        del item["reason"]
        self.assertProblem([item], '"reason" is missing')

    def test_reference_frames_checked(self):
        item = visual("image", 0, 20)
        item["reference_frames"] = [9]
        self.assertProblem([item], "reference_frames", frames=4)

    def test_parse_json_accepts_fence_and_rejects_prose(self):
        self.assertEqual(vp.parse_json('```json\n{"style": "x", "visuals": []}\n```')["style"], "x")
        with self.assertRaises(PlanError):
            vp.parse_json("Sure! Here is your plan.")

    def test_all_problems_reported_together(self):
        with self.assertRaises(PlanError) as ctx:
            validate([visual("video", 1, 5), visual("image", 9, 15)])
        self.assertGreaterEqual(len(ctx.exception.problems), 3)

    def test_generation_prompt_carries_style_and_rules(self):
        text = vp.generation_prompt("A scatter plot.", "Flat navy diagrams.", "video", "9:16", ["frame_0001.jpg"])
        for needed in ("A scatter plot.", "Flat navy diagrams.", "9:16", "No dialogue", "no text", "reference image"):
            self.assertIn(needed.lower(), text.lower())


class TimelineTests(unittest.TestCase):
    def plan(self, edges):
        return [{"id": i + 1, "start": a, "end": b} for i, (a, b) in enumerate(zip(edges, edges[1:]))]

    def test_segments_minus_fades_equal_total(self):
        for edges in ([0, 10.5, 21, 31.0417], [0, 3, 3.5, 9, 20], [0, 12]):
            t = vp.timeline(self.plan(edges))
            self.assertEqual(sum(s["frames"] for s in t["segments"]) - sum(f["frames"] for f in t["fades"]), t["total_frames"])
            self.assertEqual(t["total_frames"], round(edges[-1] * FPS))

    def test_crossfade_is_centred_on_each_boundary(self):
        t = vp.timeline(self.plan([0, 10.5, 21, 31.0417]))
        self.assertEqual([f["frames"] for f in t["fades"]], [10, 10])
        # boundary 10.5 s = frame 252: the fade runs 247..257, so the next visual is fully on screen 5 frames (0.21 s) after its words start
        self.assertEqual(t["fades"][0]["offset_frames"], 252 - 5)
        self.assertLessEqual(5 / FPS, 0.25)

    def test_xfade_offsets_match_the_ffmpeg_formula(self):
        # ffmpeg chains: offset_i = sum(lengths of inputs 0..i) - sum(fade durations 0..i)
        t = vp.timeline(self.plan([0, 4, 9.5, 15, 20]))
        running = 0
        for i, fade in enumerate(t["fades"]):
            running += t["segments"][i]["frames"]
            self.assertEqual(fade["offset_frames"], running - fade["frames"])
            running -= fade["frames"]

    def test_short_neighbours_get_shorter_fades(self):
        t = vp.timeline(self.plan([0, 0.5, 10]))   # 12-frame first visual
        self.assertEqual(t["fades"][0]["frames"], 8)      # at most a third of the shorter side, per half
        t = vp.timeline(self.plan([0, 0.1, 10]))
        self.assertEqual(t["fades"][0]["frames"], 0)      # too short: plain cut

    def test_invalid_timelines_rejected(self):
        with self.assertRaises(ValueError):
            vp.timeline([{"id": 1, "start": 1, "end": 5}])
        with self.assertRaises(ValueError):
            vp.timeline([{"id": 1, "start": 0, "end": 5}, {"id": 2, "start": 5, "end": 5}])

    def test_clip_seconds(self):
        self.assertEqual(vp.clip_seconds({"duration": 6.2}, 0, 3), 7)     # +0.21 s fade overlap, rounded up
        self.assertEqual(vp.clip_seconds({"duration": 6.2}, 1, 3), 7)
        self.assertEqual(vp.clip_seconds({"duration": 2.0}, 0, 1), 5)     # HeyGen minimum
        self.assertEqual(vp.clip_seconds({"duration": 29.9}, 1, 3), 30)   # HeyGen maximum

    def test_final_length(self):
        self.assertEqual(vp.final_frames(31.033, 31.03), 745)            # whole narration, rounded up to a frame
        self.assertLessEqual(745 / FPS, 31.03 * 1.05)
        with self.assertRaises(ValueError) as ctx:
            vp.final_frames(40, 31.03)
        self.assertIn("never cut", str(ctx.exception))


class EstimateTests(unittest.TestCase):
    def test_video_estimate_uses_published_rates(self):
        model = {"pricing": {"duration_seconds_480p": "0.02", "long_duration_seconds_480p": "0.045",
                             "reference_duration_seconds_480p": "0.07", "reference_input_images": "0.04"}}
        self.assertAlmostEqual(providers.estimate_video(model, 6, "480p", 0)["cost"], 0.12)
        self.assertAlmostEqual(providers.estimate_video(model, 20, "480p", 0)["cost"], 0.90)
        self.assertAlmostEqual(providers.estimate_video(model, 6, "480p", 1)["cost"], 6 * 0.07 + 0.04)
        self.assertIn("6 s x $0.02/s", providers.estimate_video(model, 6, "480p", 0)["basis"])

    def test_image_estimate_per_image_and_tokens(self):
        flat = {"id": "x/flat", "pricing": [{"billable": "output_image", "unit": "image", "cost_usd": 0.03}]}
        self.assertAlmostEqual(providers.estimate_image(flat, "medium", 0)["cost"], 0.03)
        tokens = {"id": "x/tokens", "pricing": [{"billable": "output_image", "unit": "token", "cost_usd": 3e-05}]}
        self.assertAlmostEqual(providers.estimate_image(tokens, "medium", 0)["cost"], 1600 * 3e-05)
        self.assertIn("assumed", providers.estimate_image(tokens, "medium", 0)["basis"])
        self.assertEqual(providers.estimate_image({"id": "x/none", "pricing": None}, "medium", 0)["cost"], providers.UNKNOWN_IMAGE_PRICE)


class LedgerTests(unittest.TestCase):
    """The budget cap and the ledger, on a real session folder (no provider calls)."""

    def setUp(self):
        self.job_id = "abcdef0123"
        root = Path(_DATA) / "jobs" / self.job_id / "video_sessions" / "20260101-000000-abcd"
        (root / "assets").mkdir(parents=True, exist_ok=True)
        state = {"id": root.name, "job_id": self.job_id, "status": "generating", "settings": {"budget": 0.20},
                 "created_at": jobs.now(), "finished_at": None, "final": None, "step": 4, "step_text": "",
                 "visuals": [{"id": 1, "status": "pending", "estimate": 0.05}, {"id": 2, "status": "done", "estimate": 0.05}]}
        jobs.write_json(root / "session.json", state)
        jobs.write_json(root / "ledger.json", [])
        self.session = vs.Session(self.job_id, root.name)

    def charge(self, cost, **kw):
        entry = self.session.charge_begin(step="image", label="Image", model="m", estimate={"cost": cost, "basis": "test"}, **kw)
        return entry

    def test_actual_and_computed_costs(self):
        a = self.charge(0.05)
        self.session.charge_end(a, cost=0.04, reported=True, units="1 image", latency=1.0)
        b = self.charge(0.03)
        self.session.charge_end(b, cost=None, reported=False, units="1 image", latency=1.0)   # provider gave no cost
        t = self.session.totals()
        self.assertAlmostEqual(t["actual"], 0.04)
        self.assertAlmostEqual(t["computed"], 0.03)
        self.assertAlmostEqual(t["spent"], 0.07)
        self.assertEqual(b["cost_kind"], "estimated")

    def test_failed_calls_count_and_pending_reserved(self):
        a = self.charge(0.05)
        self.session.charge_fail(a, "HTTP 500", cost=0.01)
        self.charge(0.06)                                   # in flight
        t = self.session.totals()
        self.assertAlmostEqual(t["spent"], 0.01)
        self.assertAlmostEqual(t["unconfirmed"], 0.06)
        self.assertAlmostEqual(t["remaining"], 0.20 - 0.07)
        self.assertAlmostEqual(t["estimated_total"], 0.01 + 0.06 + 0.05)   # + the one pending visual

    def test_budget_cap_blocks_before_sending(self):
        a = self.charge(0.15)
        self.session.charge_end(a, cost=0.15, reported=True, units="", latency=0)
        with self.assertRaises(vs.BudgetPause):
            self.charge(0.06)                                # 0.15 + 0.06 > 0.20
        self.assertEqual(len(self.session.ledger), 1)        # nothing recorded for the blocked call
        self.charge(0.05)                                    # exactly at the cap is allowed

    def test_ledger_and_log_persist(self):
        a = self.charge(0.02, visual_id=1)
        self.session.charge_end(a, cost=0.02, reported=True, units="1 image", latency=2.0)
        again = vs.Session(self.job_id, self.session.id)
        self.assertEqual(len(again.ledger), 1)
        self.assertAlmostEqual(again.totals()["spent"], 0.02)
        log = (self.session.dir / "activity.log").read_text(encoding="utf-8")
        self.assertIn("REQUEST Image", log)
        self.assertIn("RESPONSE Image: ok", log)

    def test_header_total_equals_ledger_and_visual_lines(self):
        for visual_id, cost in ((1, 0.03), (2, 0.04), (2, 0.01)):
            e = self.charge(cost, visual_id=visual_id)
            self.session.charge_end(e, cost=cost, reported=True, units="", latency=0)
        view = self.session.view()
        ledger_sum = sum(e["cost"] for e in view["ledger"])
        visual_sum = sum(v["cost"] for v in view["visuals"])
        self.assertAlmostEqual(view["cost"]["spent"], ledger_sum)
        self.assertAlmostEqual(ledger_sum, visual_sum)
        self.assertEqual(view["visuals"][1]["calls"], 2)       # regeneration is counted, not hidden

    def test_running_state_without_a_worker_shows_interrupted(self):
        self.assertEqual(self.session.view()["status"], "interrupted")


class ResumeTests(unittest.TestCase):
    """A clip the provider already finished (and billed) before the app stopped is downloaded again, never re-submitted."""

    def test_paid_clip_is_not_submitted_again(self):
        from unittest import mock
        job_id = "abcdef0456"
        root = Path(_DATA) / "jobs" / job_id / "video_sessions" / "20260101-000000-beef"
        (root / "assets").mkdir(parents=True, exist_ok=True)
        clip = {"id": 3, "type": "video", "status": "generating", "clip_seconds": 6, "reference_files": [], "prompt": "p" * 50,
                "provider_job_id": "gen-vid-1-abc", "estimate": 0.12}
        state = {"id": root.name, "job_id": job_id, "status": "generating", "created_at": jobs.now(), "finished_at": None,
                 "final": None, "step": 4, "step_text": "", "visuals": [clip],
                 "settings": {"budget": 1.0, "video_model": "heygen/heygen-video-1", "video_resolution": "480p", "video_ratio": "9:16"},
                 "video": {"width": 360, "height": 640}}
        ledger = [{"id": 1, "visual_id": 3, "status": "ok", "cost": 0.06, "cost_kind": "actual", "estimate": 0.12,
                   "provider_id": "gen-vid-1-abc", "label": "Clip 3", "estimate_basis": "", "step": "video"}]
        jobs.write_json(root / "session.json", state)
        jobs.write_json(root / "ledger.json", ledger)
        session = vs.Session(job_id, root.name)

        def fake_download(_job, dest):
            Path(dest).write_bytes(b"x")
        with mock.patch.object(providers, "submit_video") as submit, \
             mock.patch.object(providers, "video_status", return_value={"status": "completed", "cost": 0.06, "error": None}), \
             mock.patch.object(providers, "download_video", side_effect=fake_download), \
             mock.patch.object(vs, "probe", return_value={"has_video": True, "video_duration": 6.5, "width": 480, "height": 832}), \
             mock.patch.object(vs, "grab_frame", side_effect=lambda src, t, dest: dest), \
             mock.patch.object(vs, "inspect_image", return_value={"contrast": 40, "fingerprint": "1" * 64}), \
             mock.patch.object(vs.time, "sleep"):
            vs._clip(session, {}, session.state["visuals"][0], 1)
        submit.assert_not_called()
        self.assertEqual(len(session.ledger), 1)                  # no new ledger line, no new charge
        self.assertAlmostEqual(session.totals()["spent"], 0.06)
        self.assertEqual(session.state["visuals"][0]["asset"], "visual_003.mp4")


class ImageCheckTests(unittest.TestCase):
    def test_blank_and_duplicate_detection(self):
        from PIL import Image, ImageDraw
        blank = Image.new("RGB", (90, 160), (12, 20, 40))
        busy = Image.new("RGB", (90, 160), (12, 20, 40))
        draw = ImageDraw.Draw(busy)
        for i in range(0, 90, 10):
            draw.ellipse((i, 160 - i * 1.6, i + 8, 168 - i * 1.6), fill=(60, 200, 240))
        draw.line((0, 150, 90, 20), fill=(255, 200, 0), width=4)
        a, b = Path(_DATA) / "blank.png", Path(_DATA) / "busy.png"
        blank.save(a)
        busy.save(b)
        self.assertLess(vs.inspect_image(a)["contrast"], 4)
        self.assertGreater(vs.inspect_image(b)["contrast"], 4)
        self.assertEqual(vs.fingerprint_distance(vs.inspect_image(b)["fingerprint"], vs.inspect_image(b)["fingerprint"]), 0)
        with self.assertRaises(ValueError):
            a.write_bytes(b"not an image")
            vs.inspect_image(a)

    def test_nearest_ratio(self):
        self.assertEqual(vs.nearest_ratio(360, 640, ["1:1", "9:16", "16:9"]), ("9:16", True))
        self.assertEqual(vs.nearest_ratio(360, 640, ["1:1", "2:3", "3:2"]), ("2:3", False))


if __name__ == "__main__":
    unittest.main()
