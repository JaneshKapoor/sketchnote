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
    "each) naming the key ideas to draw.\n"
    '  "diagram": an object with "nodes" and "edges" that diagrams how the key '
    "ideas connect. "
    '"nodes" is an array of 3 to 6 objects, each {"id": a short unique id, '
    '"label": a real concept name of 1-4 words from the chapter}. '
    '"edges" is an array of [from_id, to_id] pairs showing the flow/'
    "relationship between nodes (use the real ids). Keep labels concrete and "
    "specific to the chapter, never placeholders.\n\n"
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


def generate_image(prompt: str, negative_prompt: str = "") -> bytes:
    """Call the optional SDXL-Turbo Modal function. Raises if unavailable."""
    fn = _modal_function("generate_image")
    if fn is None:
        raise RuntimeError("generate_image Modal function not available")
    return fn.remote(prompt, negative_prompt)


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
    diagram = _coerce_diagram(result.get("diagram")) or _fallback_diagram(concepts)
    return {"narration_script": script, "visual_concepts": concepts,
            "diagram": diagram}


def _coerce_diagram(raw) -> dict | None:
    """Validate a model-authored diagram into {nodes:[{id,label}], edges:[[a,b]]}.

    Returns None when fewer than two usable nodes are present so callers can
    fall back to a concept-derived diagram.
    """
    if not isinstance(raw, dict):
        return None
    nodes, ids = [], set()
    for nd in raw.get("nodes") or []:
        if isinstance(nd, dict):
            nid = str(nd.get("id", "")).strip()
            label = str(nd.get("label", "")).strip()
        else:
            label = str(nd).strip()
            nid = label
        if not label:
            continue
        nid = nid or label
        if nid in ids:
            continue
        ids.add(nid)
        nodes.append({"id": nid, "label": label[:40]})
        if len(nodes) >= 6:
            break
    if len(nodes) < 2:
        return None
    edges = []
    for e in raw.get("edges") or []:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            a, b = str(e[0]).strip(), str(e[1]).strip()
        elif isinstance(e, dict):
            a, b = str(e.get("from", "")).strip(), str(e.get("to", "")).strip()
        else:
            continue
        if a in ids and b in ids and a != b:
            edges.append([a, b])
    return {"nodes": nodes, "edges": edges}


def _fallback_diagram(concepts) -> dict | None:
    """Build a simple linear flow diagram from the visual concepts."""
    items = [str(c).strip() for c in (concepts or []) if str(c).strip()][:6]
    if len(items) < 2:
        return None
    nodes = [{"id": f"n{i}", "label": c[:40]} for i, c in enumerate(items)]
    edges = [[f"n{i}", f"n{i + 1}"] for i in range(len(items) - 1)]
    return {"nodes": nodes, "edges": edges}


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
    concepts = _keywords(text, title)
    return {"narration_script": narration[:1200],
            "visual_concepts": concepts,
            "diagram": _fallback_diagram(concepts)}


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
