"""Whiteboard sketch animation (vendored ai-img2sketch logic).

Credit: the pencil-sketch + progressive-reveal approach is adapted from
ai-img2sketch by DasLearning (https://github.com/daslearning-org/ai-img2sketch)
and the storyboard-ai project by Yogendra Yatnalkar. Licensed under their
respective open-source licenses (see README). No ML model is used here — this is
pure NumPy + OpenCV.

animate(png_path, target_duration, fps) renders a whiteboard-style reveal that
draws the image tile-by-tile in reading order with a moving marker/hand, lasting
target_duration seconds (the drawing completes just before the end and the final
frame is held) so the clip matches the narration length.
"""
from __future__ import annotations

import logging
import subprocess

import cv2
import numpy as np

from pipeline.config import (DEFAULT_FPS, VIDEO_HEIGHT, VIDEO_WIDTH,
                             ffmpeg_bin, new_tmp)

log = logging.getLogger("sketchnote.sketch")

# Fraction of the clip spent actively "drawing"; the rest holds the final frame.
DRAW_FRACTION = 0.9
TILE = 36  # px granularity of the tile-by-tile reveal


def _fit_on_canvas(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize a BGR image preserving aspect ratio, centered on a white canvas."""
    ih, iw = img.shape[:2]
    scale = min((w - 60) / iw, (h - 60) / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _prepare(png_path: str, w: int, h: int) -> np.ndarray:
    """Load the final art as crisp BGR (black ink on white), fit to canvas."""
    img = cv2.imread(png_path, cv2.IMREAD_COLOR)
    if img is None:
        return np.full((h, w, 3), 255, dtype=np.uint8)
    return _fit_on_canvas(img, w, h)


def _inked_tiles(art: np.ndarray):
    """Return ink-containing tile boxes (x0, y0, x1, y1) in reading order."""
    h, w = art.shape[:2]
    gray = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY)
    ink = gray < 245  # dark pixels = strokes to draw
    boxes = []
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
            if ink[y0:y1, x0:x1].any():
                boxes.append((x0, y0, x1, y1))
    return boxes


def _draw_hand(frame: np.ndarray, x: int, y: int) -> None:
    """Draw a simple marker-in-hand whose nib points at (x, y)."""
    back = (x + 64, y - 120)
    grip = (x + 40, y - 78)
    cv2.line(frame, (x, y), back, (60, 60, 60), 16)       # marker body
    cv2.line(frame, (x, y), back, (130, 130, 130), 6)     # body highlight
    cv2.circle(frame, (x, y), 7, (40, 40, 40), -1)        # nib
    cv2.ellipse(frame, grip, (40, 30), -55, 0, 360, (165, 195, 230), -1)  # palm
    cv2.ellipse(frame, (grip[0] + 18, grip[1] + 14), (16, 11), -55, 0, 360,
                (150, 180, 220), -1)                       # fingers


def animate(
    png_path: str,
    target_duration: float,
    fps: int = DEFAULT_FPS,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
) -> str:
    """Render a whiteboard reveal of png_path (tile-by-tile + hand) as an MP4."""
    target_duration = max(0.5, float(target_duration))
    total_frames = max(1, int(round(target_duration * fps)))
    draw_frames = max(1, int(total_frames * DRAW_FRACTION))

    art = _prepare(png_path, width, height)  # crisp BGR art, white bg
    white = np.full((height, width, 3), 255, dtype=np.uint8)
    tiles = _inked_tiles(art)
    if not tiles:  # blank art: nothing to draw, just hold white
        tiles = [(width // 2, height // 2,
                  width // 2 + TILE, height // 2 + TILE)]
    n = len(tiles)

    out_path = new_tmp(suffix=".mp4", prefix="sketch_")
    cmd = [
        ffmpeg_bin(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    shown = white.copy()  # persistent canvas accumulating drawn tiles
    drawn = 0
    try:
        for f in range(total_frames):
            p = min(1.0, f / float(draw_frames))
            target = int(round(p * n))
            while drawn < target:  # reveal newly completed tiles
                x0, y0, x1, y1 = tiles[drawn]
                shown[y0:y1, x0:x1] = art[y0:y1, x0:x1]
                drawn += 1
            done = drawn >= n and p >= 1.0
            frame = shown if done else shown.copy()
            if not done:  # show the hand at the tile being drawn
                x0, y0, x1, y1 = tiles[min(drawn, n - 1)]
                _draw_hand(frame, (x0 + x1) // 2, (y0 + y1) // 2)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            err = proc.stderr.read().decode("utf-8", "ignore")
            raise RuntimeError(f"ffmpeg sketch encode failed: {err[:400]}")
    finally:
        if proc.stderr:
            proc.stderr.close()
    log.info("sketch: %.2fs @ %dfps, %d tiles -> %s",
             target_duration, fps, n, out_path)
    return out_path


if __name__ == "__main__":  # quick manual check
    import sys

    logging.basicConfig(level=logging.INFO)
    src = sys.argv[1] if len(sys.argv) > 1 else "assets/sample_card.png"
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    print(animate(src, dur))
