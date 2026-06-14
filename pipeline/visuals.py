"""Visual builder: a clean black-on-white "whiteboard card" from the chapter
title + visual concepts. White background and black strokes make the sketch
reveal in sketch.py look natural. SDXL-Turbo is an OPTIONAL upgrade that, on any
failure, falls back to the card so the pipeline never hard-fails.
"""
from __future__ import annotations

import logging
import textwrap

import matplotlib

matplotlib.use("Agg")  # headless / thread-safe rendering
import matplotlib.pyplot as plt  # noqa: E402

from pipeline.config import VIDEO_HEIGHT, VIDEO_WIDTH, new_tmp  # noqa: E402

log = logging.getLogger("sketchnote.visuals")


def build_visual(concepts, title: str, use_sdxl: bool = False) -> str:
    """Return a PNG path for the chapter's visual. Tries SDXL only if asked."""
    if use_sdxl:
        try:
            return _sdxl_card(concepts, title)
        except Exception as exc:  # noqa: BLE001
            log.warning("SDXL image failed (%s); using concept card",
                        type(exc).__name__)
    return _concept_card(concepts, title)


def _norm_concepts(concepts) -> list[str]:
    if isinstance(concepts, str):
        concepts = [concepts]
    out = []
    for c in concepts or []:
        s = str(c).strip(" -•\t")
        if s:
            out.append(s)
    return out[:5] or ["Key idea"]


def _concept_card(concepts, title: str) -> str:
    items = _norm_concepts(concepts)
    out_path = new_tmp(suffix=".png", prefix="card_")

    fig = plt.figure(figsize=(VIDEO_WIDTH / 100, VIDEO_HEIGHT / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    wrapped = "\n".join(textwrap.wrap(title or "Sketchnote", width=34)[:3])
    ax.text(0.5, 0.90, wrapped, ha="center", va="top", color="black",
            fontsize=30, fontweight="bold", family="DejaVu Sans")
    ax.plot([0.08, 0.92], [0.79, 0.79], color="black", linewidth=2.5)

    # Bulleted keywords down the left/center.
    y = 0.68
    for item in items:
        line = "\n".join(textwrap.wrap(f"• {item}", width=40)[:2])
        ax.text(0.12, y, line, ha="left", va="top", color="black",
                fontsize=20, family="DejaVu Sans")
        y -= 0.12

    # Simple left-to-right flow diagram for the first few concepts.
    _flow_diagram(ax, items[:3])

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    log.info("concept card -> %s (%d items)", out_path, len(items))
    return out_path


def _flow_diagram(ax, items: list[str]) -> None:
    if len(items) < 2:
        return
    n = len(items)
    box_w, box_h, y = 0.22, 0.10, 0.12
    gap = (1.0 - n * box_w) / (n + 1)
    xs = [gap + i * (box_w + gap) for i in range(n)]
    for i, (x, item) in enumerate(zip(xs, items)):
        ax.add_patch(plt.Rectangle((x, y), box_w, box_h, fill=False,
                                   edgecolor="black", linewidth=2))
        label = "\n".join(textwrap.wrap(item, width=16)[:2])
        ax.text(x + box_w / 2, y + box_h / 2, label, ha="center", va="center",
                fontsize=12, color="black")
        if i < n - 1:
            ax.annotate("", xy=(xs[i + 1], y + box_h / 2),
                        xytext=(x + box_w, y + box_h / 2),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2))


def _sdxl_prompt(title: str, items: list[str]) -> str:
    """Build a prompt biased toward clean, monochrome whiteboard line-art."""
    subject = ", ".join(items)
    return (
        f"black and white line drawing, whiteboard marker sketch of {title}: "
        f"{subject}. thick clean black outlines, coloring book style, "
        f"monochrome, no color, no shading, no text, flat white background, "
        f"simple minimal doodle, hand-drawn")


def _sdxl_card(concepts, title: str) -> str:
    """Render via Modal SDXL-Turbo, then compose the title above the art.

    Raises on any failure so build_visual can fall back to the concept card.
    """
    from pipeline import llm

    items = _norm_concepts(concepts)
    png_bytes = llm.generate_image(_sdxl_prompt(title, items))
    raw_path = new_tmp(suffix=".png", prefix="sdxlraw_")
    with open(raw_path, "wb") as fh:
        fh.write(png_bytes)
    return _compose_titled(raw_path, title)


def _compose_titled(img_path: str, title: str) -> str:
    """Place the AI illustration on a white card under a bold chapter title."""
    import matplotlib.image as mpimg

    out_path = new_tmp(suffix=".png", prefix="sdxl_")
    art = mpimg.imread(img_path)

    fig = plt.figure(figsize=(VIDEO_WIDTH / 100, VIDEO_HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor("white")

    wrapped = "\n".join(textwrap.wrap(title or "Sketchnote", width=34)[:2])
    fig.text(0.5, 0.96, wrapped, ha="center", va="top", color="black",
             fontsize=30, fontweight="bold", family="DejaVu Sans")

    ax = fig.add_axes([0.06, 0.04, 0.88, 0.80])
    ax.imshow(art)
    ax.axis("off")

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    log.info("sdxl titled card -> %s", out_path)
    return out_path


if __name__ == "__main__":  # quick manual check
    logging.basicConfig(level=logging.INFO)
    p = build_visual(["evaporation", "condensation", "precipitation"],
                     "The Water Cycle")
    print(p)
