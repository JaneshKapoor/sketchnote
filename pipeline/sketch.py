"""Whiteboard sketch animation (vendored ai-img2sketch logic).

Credit: the pencil-sketch + progressive-reveal approach is adapted from
ai-img2sketch by DasLearning (https://github.com/daslearning-org/ai-img2sketch)
and the storyboard-ai project by Yogendra Yatnalkar. Licensed under their
respective open-source licenses (see README). No ML model is used here — this is
pure NumPy + OpenCV.

animate(png_path, target_duration, fps) first converts the card to clean line
art (grayscale -> invert -> blur -> color-dodge), then reveals it along detected
stroke contours in reading order with a moving marker/hand. The drawing
completes at ~75% of target_duration and the finished frame is held for the rest
so the clip matches the narration length and feels natural at 1.0x playback.
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
DRAW_FRACTION = 0.75
STROKE_STRIDE = 6   # sample every Nth contour pixel as a reveal point
REVEAL_R = 12       # px radius of art painted around each revealed point


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


def _line_art(art: np.ndarray) -> np.ndarray:
    """Convert BGR art to clean line art via the classic color-dodge sketch."""
    gray = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (0, 0), sigmaX=3)
    dodge = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(dodge, cv2.COLOR_GRAY2BGR)


def _stroke_points(line: np.ndarray):
    """Ordered (x, y) points tracing the ink strokes in reading order."""
    gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
    ink = (gray < 200).astype(np.uint8) * 255
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if len(c) >= 4]
    # Reading order: top-to-bottom in coarse bands, then left-to-right.
    contours.sort(key=lambda c: (int(c[:, 0, 1].min()) // 48,
                                 int(c[:, 0, 0].min())))
    points: list[tuple[int, int]] = []
    for c in contours:
        cp = c.reshape(-1, 2)
        points.extend((int(x), int(y)) for x, y in cp[::STROKE_STRIDE])
    return points


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
    """Render a stroke-order whiteboard reveal of png_path (+ hand) as an MP4."""
    target_duration = max(0.5, float(target_duration))
    total_frames = max(1, int(round(target_duration * fps)))
    draw_frames = max(1, int(total_frames * DRAW_FRACTION))

    art = _prepare(png_path, width, height)  # crisp BGR art, white bg
    line = _line_art(art)                     # clean black-on-white strokes
    white = np.full((height, width, 3), 255, dtype=np.uint8)
    points = _stroke_points(line)
    if not points:  # blank art: nothing to draw, just hold white
        points = [(width // 2, height // 2)]
    n = len(points)

    out_path = new_tmp(suffix=".mp4", prefix="sketch_")
    cmd = [
        ffmpeg_bin(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    shown = white.copy()  # persistent canvas accumulating drawn strokes
    drawn = 0
    try:
        for f in range(total_frames):
            p = min(1.0, f / float(draw_frames))
            target = int(round(p * n))
            while drawn < target:  # paint art around newly revealed points
                x, y = points[drawn]
                x0, y0 = max(0, x - REVEAL_R), max(0, y - REVEAL_R)
                x1, y1 = min(width, x + REVEAL_R), min(height, y + REVEAL_R)
                shown[y0:y1, x0:x1] = line[y0:y1, x0:x1]
                drawn += 1
            done = drawn >= n and p >= 1.0
            if done:  # snap to the full clean line art for a crisp hold
                frame = line
            else:
                frame = shown.copy()
                hx, hy = points[min(drawn, n - 1)]
                _draw_hand(frame, hx, hy)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            err = proc.stderr.read().decode("utf-8", "ignore")
            raise RuntimeError(f"ffmpeg sketch encode failed: {err[:400]}")
    finally:
        if proc.stderr:
            proc.stderr.close()
    log.info("sketch: %.2fs @ %dfps, %d stroke-points -> %s",
             target_duration, fps, n, out_path)
    return out_path


def animate_beat(
    full_png: str,
    base_png: str | None,
    target_duration: float,
    fps: int = DEFAULT_FPS,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
) -> str:
    """Render the INCREMENTAL reveal of a single storyboard beat as an MP4.

    ``full_png``  — cumulative diagram image including the new node.
    ``base_png``  — cumulative diagram from the *previous* beat (or None for
                    the very first beat, in which case animation starts from
                    a blank white canvas).

    The prior nodes shown in ``base_png`` are rendered as a static background;
    only the pixels that are *new* in ``full_png`` (i.e. the new box + arrow)
    are revealed progressively in reading/stroke order.  Drawing completes at
    ~75 % of ``target_duration``; the finished frame is held for the remainder.
    """
    target_duration = max(0.5, float(target_duration))
    total_frames = max(1, int(round(target_duration * fps)))
    draw_frames = max(1, int(total_frames * DRAW_FRACTION))

    # ── Prepare images ──────────────────────────────────────────────────────
    full_art = _prepare(full_png, width, height)
    full_line = _line_art(full_art)

    if base_png is not None:
        base_art = _prepare(base_png, width, height)
        base_line = _line_art(base_art)
    else:
        base_line = np.full((height, width, 3), 255, dtype=np.uint8)

    # ── New-ink mask: pixels present in full but absent in base ─────────────
    full_gray = cv2.cvtColor(full_line, cv2.COLOR_BGR2GRAY)
    base_gray = cv2.cvtColor(base_line, cv2.COLOR_BGR2GRAY)
    new_ink = ((full_gray < 200) & (base_gray >= 200)).astype(np.uint8) * 255

    # Find stroke points only within the new region.
    contours, _ = cv2.findContours(new_ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if len(c) >= 4]
    contours.sort(key=lambda c: (int(c[:, 0, 1].min()) // 48,
                                 int(c[:, 0, 0].min())))
    points: list[tuple[int, int]] = []
    for c in contours:
        cp = c.reshape(-1, 2)
        points.extend((int(x), int(y)) for x, y in cp[::STROKE_STRIDE])
    if not points:
        # Nothing new to draw — just hold the full frame for the whole clip.
        points = []

    n = len(points)
    out_path = new_tmp(suffix=".mp4", prefix="beat_")
    cmd = [
        ffmpeg_bin(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    shown = base_line.copy()   # start from the already-drawn prior state
    drawn = 0
    try:
        for f in range(total_frames):
            if n == 0:
                # No new ink — show complete frame immediately.
                proc.stdin.write(full_line.tobytes())
                continue
            p = min(1.0, f / float(draw_frames))
            target = int(round(p * n))
            while drawn < target:
                x, y = points[drawn]
                x0 = max(0, x - REVEAL_R); y0 = max(0, y - REVEAL_R)
                x1 = min(width, x + REVEAL_R); y1 = min(height, y + REVEAL_R)
                shown[y0:y1, x0:x1] = full_line[y0:y1, x0:x1]
                drawn += 1
            done = drawn >= n and p >= 1.0
            if done:
                frame = full_line
            else:
                frame = shown.copy()
                hx, hy = points[min(drawn, n - 1)]
                _draw_hand(frame, hx, hy)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            err = proc.stderr.read().decode("utf-8", "ignore")
            raise RuntimeError(f"ffmpeg beat encode failed: {err[:400]}")
    finally:
        if proc.stderr:
            proc.stderr.close()
    log.info("beat: %.2fs @ %dfps, %d new stroke-points -> %s",
             target_duration, fps, n, out_path)
    return out_path


if __name__ == "__main__":  # quick manual check
    import sys

    logging.basicConfig(level=logging.INFO)
    src = sys.argv[1] if len(sys.argv) > 1 else "assets/sample_card.png"
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    print(animate(src, dur))
