"""Shared configuration and small utilities for Sketchnote.

Loads credentials from .env (via python-dotenv) and exposes cache/output
directories plus an ffmpeg-binary resolver. Tokens are read from the
environment only and are NEVER printed or logged.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()  # loads .env if present; no-op otherwise
except Exception:  # pragma: no cover - dotenv optional at import time
    pass

# --- Directories -----------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("SKETCHNOTE_CACHE", ROOT / ".sketchnote_cache"))
OUTPUT_DIR = Path(os.environ.get("SKETCHNOTE_OUTPUT", ROOT / "outputs"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model / app identifiers ----------------------------------------------
MODAL_APP_NAME = os.environ.get("MODAL_APP_NAME", "sketchnote")
NEMOTRON_PARSE_MODEL = "nvidia/NVIDIA-Nemotron-Parse-v1.1"
SUMMARIZER_MODEL = "openbmb/MiniCPM4.1-8B"
SUMMARIZER_FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TTS_MODEL = "hexgrad/Kokoro-82M"
SDXL_TURBO_MODEL = "stabilityai/sdxl-turbo"

# --- TTS defaults ----------------------------------------------------------
TTS_SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"

# --- Video defaults --------------------------------------------------------
DEFAULT_FPS = 25
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720


def get_token(name: str) -> str | None:
    """Return an env token by name without ever logging its value."""
    return os.environ.get(name)


def ffmpeg_bin() -> str:
    """Resolve an ffmpeg executable.

    Prefers a system ffmpeg (provided by packages.txt on the Space); falls back
    to the binary bundled with imageio-ffmpeg for local/dev environments.
    """
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "ffmpeg not found. Install system ffmpeg or `pip install imageio-ffmpeg`."
        ) from exc


def ffprobe_bin() -> str | None:
    """Resolve ffprobe if available (system only; imageio bundles only ffmpeg)."""
    return shutil.which("ffprobe")


def new_tmp(suffix: str = "", prefix: str = "sketchnote_") -> str:
    """Create a unique temp file path inside CACHE_DIR and return it."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=str(CACHE_DIR))
    os.close(fd)
    return path
