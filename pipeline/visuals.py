"""Visual builder: a clean black-on-white "whiteboard card" from the chapter
title + an LLM-authored concept diagram. White background and black strokes make
the sketch reveal in sketch.py look natural.

Default visual: a hand-drawn (matplotlib ``plt.xkcd``) node/edge diagram built
from the model's ``diagram`` field. SDXL-Turbo is an OPTIONAL upgrade (off by
default) restricted to a single text-free icon. Both paths fall back to a plain
concept card so the pipeline never hard-fails.
"""
from __future__ import annotations

import logging
import math
import textwrap

import matplotlib

matplotlib.use("Agg")  # headless / thread-safe rendering
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

from pipeline.config import VIDEO_HEIGHT, VIDEO_WIDTH, new_tmp  # noqa: E402

log = logging.getLogger("sketchnote.visuals")

# Suppress text/letters/etc. so the optional SDXL icon stays a clean glyph.
SDXL_NEGATIVE = ("text, letters, words, labels, watermark, signature, caption, "
                 "numbers, multiple objects, clutter, color, shading, gradient")


def build_storyboard_frame(beats: list[dict], title: str,
                           upto: int | None = None) -> str:
    """Render a cumulative storyboard diagram, filling a FIXED canvas.

    ``beats`` is always the chapter's FULL beat list so the layout is computed
    once; ``upto`` selects how many nodes to draw (1..k).  Nodes 1..upto are
    rendered at their FINAL positions — the diagram fills in on a stable canvas
    instead of being re-laid-out (which used to move every node each beat).

    Each beat dict has ``{say, node, connects_to}``.  Uses ``plt.xkcd()`` for a
    hand-drawn whiteboard feel.  Returns the path to the saved PNG.
    """
    out_path = new_tmp(suffix=".png", prefix="story_")
    n_full = len(beats)
    k = n_full if upto is None else max(0, min(upto, n_full))
    if n_full == 0 or k == 0:
        # Empty diagram: just the title + rule on a white card.
        fig = plt.figure(figsize=(VIDEO_WIDTH / 100, VIDEO_HEIGHT / 100), dpi=100)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        wrapped = "\n".join(textwrap.wrap(title or "Sketchnote", width=36)[:2])
        ax.text(0.5, 0.95, wrapped, ha="center", va="top", color="black",
                fontsize=26, fontweight="bold")
        ax.plot([0.08, 0.92], [0.82, 0.82], color="black", linewidth=2.5)
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
        return out_path

    # Layout + box size computed ONCE from the FULL beat set so positions are
    # identical across every beat frame (the key fix for the moving-node bug).
    label_to_idx = {}
    for i, b in enumerate(beats):
        label_to_idx.setdefault(b["node"], i)
    pos, bw, bh = _graph_positions(beats)
    # Scale arrow gap, font and wrap width to the (fixed) box size.
    shrink = max(12.0, min(bw, bh) * VIDEO_HEIGHT * 0.36)
    fs = max(9, int(round(13 * bw / 0.24)))
    wrap_w = max(8, int(round(14 * bw / 0.24)))

    with plt.xkcd():
        fig = plt.figure(figsize=(VIDEO_WIDTH / 100, VIDEO_HEIGHT / 100), dpi=100)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        # Title + horizontal rule.
        wrapped = "\n".join(textwrap.wrap(title or "Sketchnote", width=36)[:2])
        ax.text(0.5, 0.95, wrapped, ha="center", va="top", color="black",
                fontsize=26, fontweight="bold")
        ax.plot([0.08, 0.92], [0.82, 0.82], color="black", linewidth=2.5)

        # Edges first (under boxes) — only between nodes that are visible (< k).
        for b in beats[:k]:
            ct = b.get("connects_to")
            dst_i = label_to_idx.get(b["node"])
            src = label_to_idx.get(ct) if ct else None
            if src is not None and src != dst_i and src < k:
                ax.annotate("", xy=pos[dst_i], xytext=pos[src],
                            arrowprops=dict(arrowstyle="-|>", color="black",
                                            lw=2, shrinkA=shrink, shrinkB=shrink))

        # Boxes + labels for the first k nodes at their FINAL positions.
        for i in range(k):
            cx, cy = pos[i]
            ax.add_patch(plt.Rectangle((cx - bw / 2, cy - bh / 2), bw, bh,
                                       fill=False, edgecolor="black", linewidth=2.5))
            text = "\n".join(textwrap.wrap(beats[i]["node"], width=wrap_w)[:3])
            ax.text(cx, cy, text, ha="center", va="center",
                    fontsize=fs, color="black")

        fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    log.info("storyboard frame (%d/%d beats) -> %s", k, n_full, out_path)
    return out_path


def _illustration_prompt(title: str, nodes: list[str]) -> str:
    """Prompt a SINGLE clean, text-free illustration for the whole chapter."""
    subjects = ", ".join(nodes[:4]) if nodes else title
    return (
        f"a clean minimalist black ink line illustration about {title}, "
        f"depicting {subjects}, thick clean black outlines, coloring book "
        f"line art, monochrome, no color, no shading, flat white background, "
        f"hand-drawn, simple, uncluttered, centered")


def chapter_illustration(title: str, beats: list[dict]) -> str | None:
    """Generate ONE text-free SDXL illustration for the whole chapter.

    The node labels inform the prompt so the picture reflects the chapter's
    concepts, but no text is rendered (we overlay readable labels ourselves in
    ``build_image_label_frame``).  Returns the saved PNG path, or ``None`` on
    failure so the caller can fall back to the diagram.
    """
    from pipeline import llm

    nodes = [b["node"] for b in beats if b.get("node")]
    try:
        png_bytes = llm.generate_image(_illustration_prompt(title, nodes),
                                       negative_prompt=SDXL_NEGATIVE)
    except Exception as exc:  # noqa: BLE001
        log.warning("SDXL illustration failed (%s); using diagram instead",
                    type(exc).__name__)
        return None
    out_path = new_tmp(suffix=".png", prefix="illus_")
    with open(out_path, "wb") as fh:
        fh.write(png_bytes)
    log.info("chapter illustration -> %s (%d bytes)", out_path, len(png_bytes))
    return out_path


def build_image_label_frame(beats: list[dict], title: str,
                            bg_path: str, upto: int | None = None) -> str:
    """Render the SDXL chapter image with our node labels overlaid on top.

    ``beats`` is the chapter's FULL beat list (so label positions are computed
    once and stay fixed); ``upto`` selects how many labels to draw (1..k).  The
    labels are placed at positions from the ``connects_to`` graph (no boxes, no
    arrows — per the image-only design) with a white halo so the technical terms
    stay readable over the illustration.  Returns the path to the saved PNG.
    """
    out_path = new_tmp(suffix=".png", prefix="imglbl_")
    n_full = len(beats)
    k = n_full if upto is None else max(0, min(upto, n_full))
    pos, bw, _bh = _graph_positions(beats) if n_full else ([], 0.24, 0.13)
    fs = max(11, int(round(15 * bw / 0.24)))
    wrap_w = max(8, int(round(16 * bw / 0.24)))

    art = mpimg.imread(bg_path)
    fig = plt.figure(figsize=(VIDEO_WIDTH / 100, VIDEO_HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Illustration fills the area below the title.
    ax.imshow(art, extent=(0.04, 0.96, 0.04, 0.80), aspect="auto", zorder=0)

    # Title + rule on the white margin above the image.
    wrapped = "\n".join(textwrap.wrap(title or "Sketchnote", width=36)[:2])
    ax.text(0.5, 0.97, wrapped, ha="center", va="top", color="black",
            fontsize=26, fontweight="bold", zorder=6)
    ax.plot([0.08, 0.92], [0.84, 0.84], color="black", linewidth=2.5, zorder=6)

    # First k node labels overlaid at their FINAL positions — white halo keeps
    # them readable. Positions never move because layout used the full set.
    for i in range(k):
        cx, cy = pos[i]
        text = "\n".join(textwrap.wrap(beats[i]["node"], width=wrap_w)[:3])
        ax.text(cx, cy, text, ha="center", va="center", color="black",
                fontsize=fs, fontweight="bold", zorder=5,
                path_effects=[pe.withStroke(linewidth=5, foreground="white")])

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    log.info("image+label frame (%d/%d labels) -> %s", k, n_full, out_path)
    return out_path


def build_visual(concepts, title: str, diagram=None,
                 use_sdxl: bool = False) -> str:
    """Return a PNG path for the chapter's visual.

    Order: optional SDXL icon (if asked) -> LLM diagram card -> concept card.
    """
    if use_sdxl:
        try:
            return _sdxl_card(concepts, title)
        except Exception as exc:  # noqa: BLE001
            log.warning("SDXL image failed (%s); using diagram card",
                        type(exc).__name__)
    try:
        nodes, edges = _norm_diagram(diagram)
        if len(nodes) >= 2:
            return _diagram_card(nodes, edges, title)
    except Exception as exc:  # noqa: BLE001
        log.warning("diagram card failed (%s); using concept card",
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


def _norm_diagram(diagram):
    """Turn a {nodes:[{id,label}], edges:[[a,b]]} dict into drawable parts.

    Returns (nodes, edges) where nodes is a list of label strings and edges is
    a list of (src_index, dst_index) tuples referencing those labels.
    """
    nodes, index = [], {}
    for nd in (diagram or {}).get("nodes") or []:
        if isinstance(nd, dict):
            nid = str(nd.get("id", "")).strip()
            label = str(nd.get("label", nd.get("id", ""))).strip()
        else:
            label = str(nd).strip()
            nid = label
        nid = nid or label
        if not label or nid in index:
            continue
        index[nid] = len(nodes)
        nodes.append(label)
        if len(nodes) >= 6:
            break
    edges = []
    for e in (diagram or {}).get("edges") or []:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            a, b = str(e[0]).strip(), str(e[1]).strip()
        elif isinstance(e, dict):
            a, b = str(e.get("from", "")).strip(), str(e.get("to", "")).strip()
        else:
            continue
        if a in index and b in index and index[a] != index[b]:
            edges.append((index[a], index[b]))
    return nodes, edges


def _graph_positions(beats: list[dict]):
    """Lay out beat nodes from the ``connects_to`` graph.

    Returns ``(positions, bw, bh)`` where ``positions[i]`` is the (x, y) center
    of ``beats[i]``.  Branching graphs become a left->right layered tree; pure
    chains become a serpentine flow — so the shape reflects the chapter's actual
    structure instead of a fixed grid.  Box size adapts so everything fits.
    """
    n = len(beats)
    if n == 0:
        return [], 0.24, 0.13
    idx: dict[str, int] = {}
    for i, b in enumerate(beats):
        idx.setdefault(b["node"], i)
    parent: list[int | None] = [None] * n
    for i, b in enumerate(beats):
        ct = b.get("connects_to")
        if ct and ct in idx and idx[ct] != i:
            parent[i] = idx[ct]

    # Longest-path depth from roots (bounded passes -> cycle-safe).
    depth = [0] * n
    for _ in range(n):
        changed = False
        for i in range(n):
            pi = parent[i]
            if pi is not None and depth[i] <= depth[pi]:
                depth[i] = depth[pi] + 1
                changed = True
        if not changed:
            break

    layers: dict[int, list[int]] = {}
    for i in range(n):
        layers.setdefault(depth[i], []).append(i)
    n_layers = max(depth) + 1
    max_w = max(len(v) for v in layers.values())

    x_lo, x_hi, y_lo, y_hi = 0.10, 0.90, 0.12, 0.72
    pos: list[tuple[float, float]] = [(0.5, 0.5)] * n

    # Pure chain of >3 nodes -> serpentine grid (uses vertical space).
    if max_w == 1 and n_layers == n and n > 3:
        cols = 3
        rows = math.ceil(n / cols)
        bw = min(0.26, (x_hi - x_lo) / cols * 0.82)
        bh = min(0.13, (y_hi - y_lo) / rows * 0.60)
        order = sorted(range(n), key=lambda i: depth[i])
        for slot, i in enumerate(order):
            r, c = divmod(slot, cols)
            if r % 2 == 1:                      # snake: reverse alternate rows
                c = cols - 1 - c
            cx = x_lo + (x_hi - x_lo) * (c + 0.5) / cols
            cy = y_hi - (y_hi - y_lo) * (r + 0.5) / rows
            pos[i] = (cx, cy)
        return pos, bw, bh

    # General / branching -> left-to-right layered tree.
    bw = min(0.24, (x_hi - x_lo) / max(1, n_layers) * 0.82)
    bh = min(0.13, (y_hi - y_lo) / max(1, max_w) * 0.70)
    for d, members in layers.items():
        members = sorted(members)               # stable order within a layer
        cx = 0.5 if n_layers == 1 else \
            x_lo + (x_hi - x_lo) * (d + 0.5) / n_layers
        k = len(members)
        for j, i in enumerate(members):
            cy = (y_lo + y_hi) / 2 if k == 1 else \
                y_hi - (y_hi - y_lo) * (j + 0.5) / k
            pos[i] = (cx, cy)
    return pos, bw, bh


def _node_positions(n: int) -> list[tuple[float, float]]:
    """Lay out n boxes on a centered grid in the area below the title rule."""
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    x_lo, x_hi, y_lo, y_hi = 0.12, 0.88, 0.14, 0.70
    positions: list[tuple[float, float]] = []
    for i in range(n):
        r, c = divmod(i, cols)
        in_row = cols if r < rows - 1 or n % cols == 0 else n % cols
        cx = (x_lo + x_hi) / 2 if in_row == 1 else \
            x_lo + (x_hi - x_lo) * c / (in_row - 1)
        cy = (y_lo + y_hi) / 2 if rows == 1 else \
            y_hi - (y_hi - y_lo) * r / (rows - 1)
        positions.append((cx, cy))
    return positions


def _diagram_card(nodes: list[str], edges, title: str) -> str:
    """Render an LLM-authored node/edge diagram as a hand-drawn xkcd-style card."""
    out_path = new_tmp(suffix=".png", prefix="diag_")
    pos = _node_positions(len(nodes))
    bw, bh = 0.24, 0.13

    with plt.xkcd():
        fig = plt.figure(figsize=(VIDEO_WIDTH / 100, VIDEO_HEIGHT / 100), dpi=100)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        wrapped = "\n".join(textwrap.wrap(title or "Sketchnote", width=36)[:2])
        ax.text(0.5, 0.95, wrapped, ha="center", va="top", color="black",
                fontsize=26, fontweight="bold")
        ax.plot([0.08, 0.92], [0.82, 0.82], color="black", linewidth=2.5)

        for a, b in edges:  # arrows first so boxes sit on top
            ax.annotate("", xy=pos[b], xytext=pos[a],
                        arrowprops=dict(arrowstyle="-|>", color="black",
                                        lw=2, shrinkA=24, shrinkB=24))
        for label, (cx, cy) in zip(nodes, pos):
            ax.add_patch(plt.Rectangle((cx - bw / 2, cy - bh / 2), bw, bh,
                                       fill=False, edgecolor="black", linewidth=2.5))
            text = "\n".join(textwrap.wrap(label, width=14)[:3])
            ax.text(cx, cy, text, ha="center", va="center", fontsize=13,
                    color="black")

        fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    log.info("diagram card -> %s (%d nodes, %d edges)",
             out_path, len(nodes), len(edges))
    return out_path


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
    """Build a prompt for a SINGLE clean, text-free monochrome icon."""
    subject = items[0] if items else title
    return (
        f"a single minimalist black ink icon representing {subject}, "
        f"thick clean black outlines, centered, coloring book line art, "
        f"monochrome, no color, no shading, flat white background, "
        f"simple minimal doodle, hand-drawn")


def _sdxl_card(concepts, title: str) -> str:
    """Render via Modal SDXL-Turbo, then compose the title above the art.

    Raises on any failure so build_visual can fall back to the concept card.
    """
    from pipeline import llm

    items = _norm_concepts(concepts)
    png_bytes = llm.generate_image(_sdxl_prompt(title, items),
                                   negative_prompt=SDXL_NEGATIVE)
    raw_path = new_tmp(suffix=".png", prefix="sdxlraw_")
    with open(raw_path, "wb") as fh:
        fh.write(png_bytes)
    return _compose_titled(raw_path, title)


def _compose_titled(img_path: str, title: str) -> str:
    """Place the AI illustration on a white card under a bold chapter title."""
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
    demo_diagram = {
        "nodes": [{"id": "a", "label": "Evaporation"},
                  {"id": "b", "label": "Condensation"},
                  {"id": "c", "label": "Precipitation"},
                  {"id": "d", "label": "Collection"}],
        "edges": [["a", "b"], ["b", "c"], ["c", "d"], ["d", "a"]],
    }
    p = build_visual(["evaporation", "condensation", "precipitation"],
                     "The Water Cycle", diagram=demo_diagram)
    print(p)
