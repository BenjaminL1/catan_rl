"""Fine sub-second scan + full-frame OCR to catch the brief victory banner."""

import subprocess
import sys

import cv2
import easyocr
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
OUT = "/tmp/phantom/blockers"
URL = open(f"{OUT}/stream_url.txt").read().strip()
R = easyocr.Reader(["en"], gpu=False, verbose=False)


def grab(t, dst):
    subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-ss",
            f"{t:.2f}",
            "-i",
            URL,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            dst,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


KEYS = ("won", "victory points", "won the game", "wins", "winner", "happy settling", "victory")


def main():
    a = float(sys.argv[1])
    b = float(sys.argv[2])
    step = float(sys.argv[3])
    t = a
    while t <= b + 1e-6:
        f = f"{OUT}/frames/fine_{t:.2f}.png"
        grab(t, f)
        img = cv2.imread(f)
        H, W = img.shape[:2]
        # log panel
        logc = img[0 : int(0.42 * H), int(0.645 * W) : W]
        log = R.readtext(logc, detail=0, paragraph=False)
        # full-screen center band (victory modal)
        cen = img[int(0.20 * H) : int(0.80 * H), int(0.18 * W) : int(0.82 * W)]
        full = R.readtext(cen, detail=0, paragraph=False)
        joined = (" | ".join(log + full)).lower()
        hits = [k for k in KEYS if k in joined]
        print(f"[t={t:.2f}] log={len(log)} full={len(full)} hits={hits}")
        for ln in log:
            print(f"   LOG | {ln}")
        for ln in full:
            low = ln.lower()
            if any(
                k in low for k in ("won", "victor", "win", "phantom", "rayman", "point", "settl")
            ):
                print(f"   FULL| {ln}")
        t += step


if __name__ == "__main__":
    main()
