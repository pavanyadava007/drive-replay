"""Records the AR-HUD demo video from the page itself (site/hud.html?demo=1), frame by frame.

    python scripts/record_hud_demo.py [--out site/media] [--fps 30] [--ffmpeg ~/.local/bin/ffmpeg]

The page has a deterministic demo script (site/hud.js, SCRIPT): window.HUD.demoAt(T) sets scene, playhead,
latency, noise level, methods and captions for video time T and draws synchronously. This script serves site/ on a
local port, steps T at the frame rate in headless Chromium (Playwright), takes one screenshot per frame and pipes the
frames into ffmpeg (H.264, yuv420p, +faststart). No screen capture, so no dropped frames; every frame shows the
exported benchmark data exactly as the interactive page does.

Writes to --out: drive-replay-hud-demo.mp4 (1920x1080), drive-replay-hud-demo-4x5.mp4 (1080x1350, feed format),
drive-replay-hud-loop.mp4 and .gif (12 s excerpt for the README) and drive-replay-hud-poster.png (1280x720).
Needs site/data/hud (scripts/export_mpred_demo.py) and site/data/mpred.json (scripts/reproduce_mpred.py).
"""
from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
# (name, CSS viewport, device scale factor, page mode): output size = viewport x scale
FORMATS = {"landscape": ((1280, 720), 1.5, "1"), "portrait": ((720, 900), 1.5, "portrait")}
LOOP = (21.5, 33.5)  # seconds of the landscape video for the README loop: CTRV and EKF + MLP in the turn
POSTER_T = 36.0


def serve(directory: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

    handler = functools.partial(Quiet, directory=str(directory))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def record(page, url: str, out: Path, fps: int, ffmpeg: str, crf: int) -> float:
    page.goto(url)
    page.wait_for_function("window.HUD && window.HUD.state && window.HUD.state.doc")
    page.wait_for_timeout(300)
    duration = page.evaluate("window.HUD.duration")
    n = int(round(duration * fps))
    enc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(fps),
                            "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", "-r", str(fps), str(out)], stdin=subprocess.PIPE)
    t0 = time.monotonic()
    for i in range(n):
        page.evaluate(f"window.HUD.demoAt({i / fps})")
        enc.stdin.write(page.screenshot(type="jpeg", quality=94))
        if i % (fps * 10) == 0:
            print(f"  {out.name}: {i / fps:5.1f} / {duration:.0f} s ({time.monotonic() - t0:.0f} s)", flush=True)
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return duration


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=SITE / "media")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or str(Path.home() / ".local/bin/ffmpeg"))
    ap.add_argument("--only", choices=list(FORMATS))
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    args.out.mkdir(parents=True, exist_ok=True)
    srv, port = serve(SITE)
    names = {"landscape": "drive-replay-hud-demo.mp4", "portrait": "drive-replay-hud-demo-4x5.mp4"}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for fmt, ((w, h), scale, mode) in FORMATS.items():
                if args.only and fmt != args.only:
                    continue
                page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=scale)
                errors = []
                page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
                url = f"http://127.0.0.1:{port}/hud.html?demo={mode}&frame=1"
                record(page, url, args.out / names[fmt], args.fps, args.ffmpeg, args.crf)
                if errors:
                    raise RuntimeError(f"page errors: {errors}")
                if fmt == "landscape":
                    page.evaluate(f"window.HUD.demoAt({POSTER_T})")
                    poster(page.screenshot(type="png"), args.out / "drive-replay-hud-poster.png", scale)
                page.close()
            browser.close()
    finally:
        srv.shutdown()
    src = args.out / names["landscape"]
    if src.exists():
        a, b = LOOP
        ff = [args.ffmpeg, "-y", "-loglevel", "error", "-ss", str(a), "-t", str(b - a), "-i", str(src)]
        h264 = ["-an", "-c:v", "libx264", "-crf", "26", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        subprocess.run(ff + ["-vf", "scale=960:-2:flags=lanczos", *h264, str(args.out / "drive-replay-hud-loop.mp4")], check=True)
        gif = ("fps=10,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];"
               "[b][p]paletteuse=dither=bayer:bayer_scale=4")
        subprocess.run(ff + ["-vf", gif, str(args.out / "drive-replay-hud-loop.gif")], check=True)
    for f in sorted(args.out.iterdir()):
        print(f"{f.name:34s} {f.stat().st_size / 1e6:6.2f} MB")


def poster(png: bytes, out: Path, scale: float) -> None:
    """The frame at POSTER_T, scaled to 1280x720, with a play button (it links to the video)."""
    import io

    from PIL import Image, ImageDraw

    im = Image.open(io.BytesIO(png)).convert("RGB").resize((1280, 720), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    cx, cy, r = 640, 400, 62
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(11, 11, 11, 200), outline=(255, 255, 255, 230), width=4)
    d.polygon([(cx - 20, cy - 32), (cx - 20, cy + 32), (cx + 34, cy)], fill=(255, 255, 255, 240))
    im.save(out, optimize=True)


if __name__ == "__main__":
    main()
