"""Thin client the Gradio Space uses to call the Modal GPU functions
(MiniCPM summarization, Nemotron Parse, optional SDXL-Turbo). Also holds the
prompt templates and a NON-AI extractive fallback used only when Modal is not
reachable, so the demo never hard-fails.

No proprietary hosted model API is ever called here — only OUR OWN models that
we self-host on Modal (looked up via the Modal SDK).
"""
from __future__ import annotations

import logging
import re

from pipeline.config import MODAL_APP_NAME

log = logging.getLogger("sketchnote.llm")

# --- Prompt templates (source of truth; mirrored in modal_app.py) ----------
SUMMARIZE_PROMPT = (
    "You are scripting an educational whiteboard sketch-note video.\n"
    "Read the chapter and return STRICT JSON ONLY (no markdown, no code "
    "fences, no commentary) with exactly these keys:\n"
    '  "narration_script": a spoken-style summary of 80-150 words, clear and '
    "engaging, plain text only (no bullets, no markdown).\n"
    '  "visual_concepts": an array of 3 to 5 short noun phrases (2-4 words '
    "each) naming the key ideas to draw.\n\n"
    "Chapter title: {title}\n"
    "Chapter text:\n{text}\n\n"
    "Return only the JSON object."
)


def _modal_function(name: str):
    """Look up a deployed Modal function by app + name, or return None."""
    try:
        import modal

        try:  # modal >= 0.64
            return modal.Function.from_name(MODAL_APP_NAME, name)
        except AttributeError:  # older SDK
            return modal.Function.lookup(MODAL_APP_NAME, name)
    except Exception as exc:  # noqa: BLE001
        log.warning("Modal lookup for %r unavailable: %s", name, type(exc).__name__)
        return None


def summarize_chapter(title: str, text: str) -> dict:
    """Return {"narration_script": str, "visual_concepts": [str, ...]}.

    Primary: MiniCPM4.1-8B on Modal. Fallback: extractive (no model).
    """
    fn = _modal_function("summarize_chapter")
    if fn is not None:
        try:
            result = fn.remote(title, text)
            return _coerce_summary(result, title, text)
        except Exception as exc:  # noqa: BLE001
            log.warning("Modal summarize failed (%s); using extractive fallback",
                        type(exc).__name__)
    return _extractive_summary(title, text)


def parse_pages(images: list[bytes]) -> list[dict]:
    """Call the Nemotron Parse Modal function. Raises if unavailable."""
    fn = _modal_function("parse_pages")
    if fn is None:
        raise RuntimeError("parse_pages Modal function not available")
    return fn.remote(images)


def generate_image(prompt: str) -> bytes:
    """Call the optional SDXL-Turbo Modal function. Raises if unavailable."""
    fn = _modal_function("generate_image")
    if fn is None:
        raise RuntimeError("generate_image Modal function not available")
    return fn.remote(prompt)


# --- Coercion / fallback ---------------------------------------------------
def _coerce_summary(result, title: str, text: str) -> dict:
    """Validate/normalize a Modal summary result into the expected shape."""
    if not isinstance(result, dict):
        return _extractive_summary(title, text)
    script = str(result.get("narration_script", "")).strip()
    concepts = result.get("visual_concepts", [])
    if isinstance(concepts, str):
        concepts = [concepts]
    concepts = [str(c).strip() for c in concepts if str(c).strip()][:5]
    if not script:
        return _extractive_summary(title, text)
    if not concepts:
        concepts = _keywords(text, title)
    return {"narration_script": script, "visual_concepts": concepts}


def _extractive_summary(title: str, text: str) -> dict:
    """Non-AI fallback: first few sentences + keyword extraction."""
    text = re.sub(r"\s+", " ", text or "").strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    script, words = [], 0
    for s in sentences:
        script.append(s)
        words += len(s.split())
        if words >= 120:
            break
    narration = " ".join(script) or f"This section covers {title}."
    return {"narration_script": narration[:1200],
            "visual_concepts": _keywords(text, title)}


_STOP = set("the a an and or of to in for on with is are was were be by as at from "
            "this that these those it its their his her our your you we they them "
            "into over under than then so such can could may might will would also "
            "which who whom whose what when where why how each other more most some "
            "all any not no but if while during because through about called using "
            "used use describes called like such many one two three first second "
            "called also called known very much often usually within without".split())


def _keywords(text: str, title: str, k: int = 4) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", (text or "").lower())
    freq: dict[str, int] = {}
    for w in words:
        if w in _STOP:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq, key=lambda w: (-freq[w], w))[:k]
    out = [w.capitalize() for w in ranked]
    return out or [title or "Key idea"]
