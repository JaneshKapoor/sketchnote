"""ffmpeg helpers: mux narration onto a sketch clip, and concat chapter clips."""
from __future__ import annotations

import logging
import os
import subprocess

from pipeline.config import DEFAULT_FPS, ffmpeg_bin, new_tmp

log = logging.getLogger("sketchnote.video")


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'ignore')[:500]}"
        )


def mux(video_path: str, wav_path: str, fps: int = DEFAULT_FPS) -> str:
    """Mux audio onto a (silent) video. Clip length == audio length (-shortest).

    Re-encodes both streams to uniform params so the concat demuxer can later
    copy-join the per-chapter clips without parameter mismatches.
    """
    out_path = new_tmp(suffix=".mp4", prefix="chapter_")
    cmd = [
        ffmpeg_bin(), "-y", "-loglevel", "error",
        "-i", video_path, "-i", wav_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-shortest", out_path,
    ]
    _run(cmd)
    log.info("muxed -> %s", out_path)
    return out_path


def concat(clip_paths: list[str]) -> str:
    """Concatenate chapter clips into one MP4 (concat demuxer, stream copy).

    Falls back to a re-encode filter if stream copy fails (param mismatch).
    """
    clips = [c for c in clip_paths if c and os.path.exists(c)]
    if not clips:
        raise ValueError("concat: no input clips")
    if len(clips) == 1:
        return clips[0]

    list_path = new_tmp(suffix=".txt", prefix="concat_")
    with open(list_path, "w", encoding="utf-8") as fh:
        for c in clips:
            safe = os.path.abspath(c).replace("\\", "/").replace("'", "'\\''")
            fh.write(f"file '{safe}'\n")

    out_path = new_tmp(suffix=".mp4", prefix="final_")
    try:
        _run([
            ffmpeg_bin(), "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-c", "copy", out_path,
        ])
    except RuntimeError as exc:
        log.warning("concat copy failed (%s); re-encoding", type(exc).__name__)
        out_path = _concat_reencode(clips)
    log.info("concat %d clips -> %s", len(clips), out_path)
    return out_path


def _concat_reencode(clips: list[str]) -> str:
    """Robust concat via the concat filter, re-encoding to uniform output."""
    out_path = new_tmp(suffix=".mp4", prefix="final_")
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error"]
    for c in clips:
        cmd += ["-i", c]
    n = len(clips)
    streams = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n))
    filt = f"{streams}concat=n={n}:v=1:a=1[v][a]"
    cmd += [
        "-filter_complex", filt, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        out_path,
    ]
    _run(cmd)
    return out_path
