"""Sketchnote pipeline package.

Modules:
    pdf_parser  -- PyMuPDF fast path + Nemotron Parse hard path -> chapters
    llm         -- thin Modal client (MiniCPM summarization) + prompt templates
    tts         -- Kokoro narration -> (wav_path, duration)
    visuals     -- whiteboard concept card / SVG (+ optional SDXL-Turbo)
    sketch      -- vendored ai-img2sketch animation
    video       -- ffmpeg mux + concat
"""

__all__ = [
    "pdf_parser",
    "llm",
    "tts",
    "visuals",
    "sketch",
    "video",
]
