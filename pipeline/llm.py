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
# Built with <<TITLE>>/<<TEXT>> + str.replace (NOT str.format) because the
# example JSON below contains literal { } braces.
STORYBOARD_PROMPT = (
    "You are scripting an educational whiteboard sketch-note video.\n"
    "Read the chapter and return STRICT JSON ONLY (no markdown, no code "
    "fences, no commentary).\n"
    'Return an object with a single key "beats": an array of 3 to 6 beats.\n'
    "Each beat is an object with exactly these keys:\n"
    '  "say": one complete spoken sentence of 15-25 words, in your own words '
    "(no bullets, no page numbers, do not repeat the chapter title).\n"
    '  "node": a 2-4 word label naming the single idea drawn for this beat.\n'
    '  "connects_to": the exact node label of an EARLIER beat this builds on, '
    "or null for the first beat.\n"
    "Every node label must be unique. connects_to must match an earlier node "
    "label exactly, or be null.\n\n"
    "Chapter title: <<TITLE>>\n"
    "Chapter text:\n<<TEXT>>\n\n"
    "Return only the JSON object, for example: "
    '{"beats":[{"say":"...","node":"First Idea","connects_to":null},'
    '{"say":"...","node":"Second Idea","connects_to":"First Idea"}]}'
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
    """Return {"beats": [{"say","node","connects_to"}, ...], "warning": str|None}.

    Primary: MiniCPM4.1-8B storyboard on Modal. Fallback: extractive beats. The
    raw Modal response is logged, and any fallback is reported via "warning" so
    the UI can show it loudly instead of silently degrading.
    """
    fn = _modal_function("summarize_chapter")
    if fn is None:
        warn = "AI model (Modal) not reachable — using a non-AI extractive fallback."
        log.warning(warn)
        return {"beats": _extractive_beats(title, text), "warning": warn}

    try:
        raw = fn.remote(title, text)
        log.info("Modal summarize raw response for %r: %r", title, raw)
        beats = _coerce_storyboard(raw)
        if beats:
            return {"beats": beats, "warning": None}
        warn = ("MiniCPM returned no usable storyboard JSON — using a non-AI "
                "extractive fallback. (See logs for the raw response.)")
        log.warning("%s raw=%r", warn, raw)
    except Exception as exc:  # noqa: BLE001
        warn = (f"MiniCPM call failed ({type(exc).__name__}) — using a non-AI "
                "extractive fallback.")
        log.warning(warn)
    return {"beats": _extractive_beats(title, text), "warning": warn}


def parse_pages(images: list[bytes]) -> list[dict]:
    """Call the Nemotron Parse Modal function. Raises if unavailable."""
    fn = _modal_function("parse_pages")
    if fn is None:
        raise RuntimeError("parse_pages Modal function not available")
    return fn.remote(images)


def generate_image(prompt: str, negative_prompt: str = "") -> bytes:
    """Call the optional SDXL-Turbo Modal function. Raises if unavailable."""
    fn = _modal_function("generate_image")
    if fn is None:
        raise RuntimeError("generate_image Modal function not available")
    return fn.remote(prompt, negative_prompt)


# --- Coercion / fallback ---------------------------------------------------
def _coerce_storyboard(raw) -> list[dict] | None:
    """Validate a Modal storyboard into [{say, node, connects_to}, ...].

    Enforces unique node labels and connects_to referencing an EARLIER node (or
    None). Returns None when fewer than two usable beats are present so callers
    can fall back to extractive beats.
    """
    if not isinstance(raw, dict):
        return None
    beats_in = raw.get("beats")
    if not isinstance(beats_in, list):
        return None
    beats: list[dict] = []
    labels: list[str] = []
    for b in beats_in:
        if not isinstance(b, dict):
            continue
        say = str(b.get("say", "")).strip()
        node = " ".join(str(b.get("node", "")).strip().split()[:4])
        if not say or not node or node in labels:
            continue
        ct = b.get("connects_to")
        ct = str(ct).strip() if ct not in (None, "", "null", "None") else None
        if ct is not None and ct not in labels:
            ct = None  # must reference an already-seen (earlier) node
        beats.append({"say": say[:300], "node": node[:40], "connects_to": ct})
        labels.append(node)
        if len(beats) >= 6:
            break
    if len(beats) < 2:
        return None
    beats[0]["connects_to"] = None  # first beat is always the root
    return beats


def _extractive_beats(title: str, text: str) -> list[dict]:
    """Non-AI fallback storyboard: sentences -> beats with keyword node labels.

    Chains beats linearly (each connects to the previous) so the diagram still
    renders a sensible left-to-right flow.
    """
    clean = re.sub(r"\s+", " ", text or "").strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean)
                 if len(s.split()) >= 6][:6]
    if not sentences:
        sentences = [f"This section introduces {title}."]
    kws = _keywords(clean, title, k=max(3, len(sentences)))
    beats: list[dict] = []
    labels: list[str] = []
    prev = None
    for i, s in enumerate(sentences):
        node = kws[i] if i < len(kws) else f"Idea {i + 1}"
        node = " ".join(str(node).split()[:4])
        while node in labels:  # keep labels unique
            node = f"{node} {len(labels) + 1}"
        beats.append({"say": s[:300], "node": node[:40], "connects_to": prev})
        labels.append(node)
        prev = node
    return beats


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
