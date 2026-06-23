"""BLOCKER 1: find game-1 terminal (victory line) + new-game reset ('Happy settling').

Scans forward from t=1100 in coarse steps, extracts a frame via portable ffmpeg
from the live stream URL, OCRs the log panel (crop 0.645,0,1.0,0.3), and looks
for terminal markers: 'won', 'victory', 'Happy settling'. Saves frames + OCR.
"""

import json
import os
import subprocess
import sys

import cv2
import easyocr
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
OUT = "/tmp/phantom/blockers"
URL_FILE = f"{OUT}/stream_url.txt"

# Log-panel crop (same as M0 spike). Also a WIDER full-log crop for terminal banners,
# which can appear lower / center-screen on Colonist.
LX0, LY0, LX1, LY1 = 0.645, 0.0, 1.0, 0.42

_reader = None


def reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def get_url():
    if os.path.exists(URL_FILE):
        u = open(URL_FILE).read().strip()
        if u:
            return u
    u = (
        subprocess.check_output(
            [
                "yt-dlp",
                "--js-runtimes",
                "node",
                "-f",
                "bestvideo[height<=1080]/best",
                "-g",
                "https://www.youtube.com/watch?v=9Sm86ml04aI",
            ],
            stderr=subprocess.DEVNULL,
        )
        .decode()
        .strip()
        .splitlines()[0]
    )
    open(URL_FILE, "w").write(u)
    return u


def grab(url, t, dst):
    subprocess.run(
        [FFMPEG, "-nostdin", "-ss", str(t), "-i", url, "-frames:v", "1", "-q:v", "2", "-y", dst],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def ocr_log(img):
    h, w = img.shape[:2]
    crop = img[int(LY0 * h) : int(LY1 * h), int(LX0 * w) : int(LX1 * w)]
    lines = reader().readtext(crop, detail=0, paragraph=False)
    return crop, lines


def ocr_full(img):
    """OCR a center band too: victory banner can be a modal in screen center."""
    h, w = img.shape[:2]
    crop = img[int(0.25 * h) : int(0.75 * h), int(0.20 * w) : int(0.80 * w)]
    lines = reader().readtext(crop, detail=0, paragraph=False)
    return crop, lines


KEYS = (
    "won",
    "victory",
    "happy settling",
    "wins",
    "winner",
    "game over",
    "settling",
    "new game",
    "rematch",
)


def main():
    url = get_url()
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1100
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 2400
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    results = {}
    for t in range(start, end + 1, step):
        f = f"{OUT}/frames/scan_{t}.png"
        try:
            grab(url, t, f)
        except Exception as e:
            print(f"[t={t}] grab FAILED: {e}")
            continue
        img = cv2.imread(f)
        _, log_lines = ocr_log(img)
        joined = " | ".join(log_lines).lower()
        hits = [k for k in KEYS if k in joined]
        full_hits = []
        if not hits:
            # cheap: only do the expensive center OCR if log looks terminal-ish
            pass
        results[t] = {"log": log_lines, "hits": hits}
        flag = "  <<< TERMINAL?" if hits else ""
        print(f"[t={t}] {len(log_lines)} log lines hits={hits}{flag}")
        for ln in log_lines:
            print(f"     | {ln}")
    json.dump(results, open(f"{OUT}/ocr/scan_{start}_{end}.json", "w"), indent=2)
    print(f"\nsaved scan_{start}_{end}.json")


if __name__ == "__main__":
    main()
