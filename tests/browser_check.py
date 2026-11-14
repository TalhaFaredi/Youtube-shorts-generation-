"""Real-browser check of the running app (no paid calls): Saved jobs, Generate Video controls, a session view.

Run against the live app:  .venv/Scripts/python.exe tests/browser_check.py --job <job_id> [--session <id>] --out <dir>
"""
import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def chrome():
    for path in (Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
                 Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")):
        if path.exists():
            return str(path)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    ap.add_argument("--job", required=True)
    ap.add_argument("--session")
    ap.add_argument("--legacy-job")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"errors": [], "failed_requests": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome(), headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.on("pageerror", lambda e: report["errors"].append(str(e)))
        page.on("console", lambda m: report["errors"].append("console: " + m.text) if m.type == "error" else None)
        page.on("response", lambda r: report["failed_requests"].append(f"{r.status} {r.url}") if r.status >= 400 else None)
        page.add_init_script("localStorage.clear()")
        page.goto(args.base, wait_until="networkidle")
        health = page.request.get(args.base + "/api/health").json()
        report["server_build"] = health["build"]
        report["page_build"] = page.locator("#build").get_attribute("data-build")
        report["build_banner"] = page.locator("#build").inner_text()
        report["health_banner_visible"] = page.locator("#app-health").is_visible()

        # Saved jobs
        page.wait_for_function("document.querySelectorAll('#job-select option').length > 1")
        report["saved_jobs"] = page.locator("#job-select option").all_inner_texts()
        report["yt_status_after_jobs"] = page.locator("#yt-status").inner_text()

        if args.legacy_job:
            page.select_option("#job-select", args.legacy_job)
            page.click("#job-open")
            page.wait_for_selector("#job-problem:not([hidden])")
            report["legacy_message"] = page.locator("#job-problem-text").inner_text()
            report["legacy_reextract_visible"] = page.locator("#reextract-go").is_visible()
            report["legacy_url_box_visible"] = page.locator("#reextract-url").is_visible()
            page.screenshot(path=str(out / "legacy-job.png"))

        page.select_option("#job-select", args.job)
        page.click("#job-open")
        page.wait_for_selector("#yt-out:not([hidden])")
        page.click("#open-video")
        page.wait_for_selector("#yt-video:not([hidden])")
        if page.locator("#vs-select option").count() > 1:
            page.select_option("#vs-select", "")      # the setup form of a new session
        page.wait_for_function("document.querySelectorAll('#vs-planner option').length > 10")
        page.wait_for_function("document.querySelector('#vs-inputs').innerText.includes('Narration')")
        report["modes"] = page.locator("input[name=vs-mode]").evaluate_all("els => els.map(e => e.value + (e.checked ? ' (checked)' : ''))")
        report["mode_labels"] = page.locator(".modes label").all_inner_texts()
        sel = lambda s: page.locator(f"{s} option:checked").inner_text()
        report["planner_default"] = sel("#vs-planner")
        report["planner_note"] = page.locator("#vs-planner-note").inner_text()
        report["image_default"] = sel("#vs-image")
        report["image_note"] = page.locator("#vs-image-note").inner_text()
        report["planner_options"] = page.locator("#vs-planner option").count()
        report["image_options"] = page.locator("#vs-image option").count()
        report["plan_button_enabled"] = page.locator("#vs-plan").is_enabled()
        report["plan_hint"] = page.locator("#vs-plan-why").inner_text()
        report["inputs"] = page.locator("#vs-inputs").inner_text()
        page.screenshot(path=str(out / "setup-images.png"), full_page=True)
        page.check("input[name=vs-mode][value=videos]")
        report["video_model"] = sel("#vs-video")
        report["video_model_disabled_reason"] = page.locator("#vs-video-note").inner_text()
        report["image_select_hidden_in_videos_mode"] = not page.locator("#vs-image").is_visible()
        page.screenshot(path=str(out / "setup-videos.png"), full_page=True)
        page.check("input[name=vs-mode][value=images]")

        if args.session:
            page.select_option("#vs-select", args.session)
            page.wait_for_selector("#vp:not([hidden])")
            page.wait_for_function("document.querySelectorAll('#vs-plan-table tr').length > 1")
            page.wait_for_function("document.querySelector('#vs-log').innerText.length > 100")
            report["panel"] = page.locator("#vp").inner_text()
            report["panel_sticky"] = page.locator("#vp").evaluate("e => getComputedStyle(e).position")
            report["plan_rows"] = page.locator("#vs-plan-table tr").count() - 1
            report["plan_images_loaded"] = page.locator("#vs-plan-table img").evaluate_all("els => els.map(e => e.naturalWidth + 'x' + e.naturalHeight)")
            report["plan_clips"] = page.locator("#vs-plan-table video").count()
            report["ledger_last_row"] = page.locator("#vs-ledger tr").last.inner_text()
            report["final_visible"] = page.locator("#vs-final").is_visible()
            if report["final_visible"]:
                page.wait_for_function("document.querySelector('#vs-final-video').readyState >= 1", timeout=20000)
                report["final_video"] = page.locator("#vs-final-video").evaluate(
                    "v => ({duration: v.duration, width: v.videoWidth, height: v.videoHeight, error: v.error && v.error.code})")
                page.locator("#vs-final-video").evaluate("v => { v.muted = true; return v.play(); }")
                page.wait_for_timeout(1500)
                report["final_played_seconds"] = page.locator("#vs-final-video").evaluate("v => v.currentTime")
            page.screenshot(path=str(out / "session-top.png"))
            page.screenshot(path=str(out / "session-full.png"), full_page=True)
            page.mouse.wheel(0, 2500)
            page.wait_for_timeout(300)
            report["panel_visible_after_scroll"] = page.locator("#vp").evaluate(
                "e => { const r = e.getBoundingClientRect(); return r.top >= -1 && r.top < 50; }")
            page.screenshot(path=str(out / "session-scrolled.png"))
        browser.close()
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
