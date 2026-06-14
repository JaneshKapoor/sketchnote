"""PDF ingestion -> list of chapters.

Fast path (default): PyMuPDF. Prefer the embedded TOC (PDF bookmarks). If no
TOC, fall back to a font-size/regex heading heuristic, then to N equal page
buckets. Hard path (scanned/complex PDFs): render pages to images and route to
the Nemotron Parse Modal function (see modal_app.parse_pages); wired in
pdf_parser.hard_path_chapters and gated by _needs_hard_path.

Chapter = {"title": str, "text": str, "page_start": int, "page_end": int}
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Optional, TypedDict

import fitz  # PyMuPDF

from pipeline.config import CACHE_DIR

log = logging.getLogger("sketchnote.pdf")

# Truncate each chapter body to keep summarization prompts in budget.
MAX_CHARS_PER_CHAPTER = 6000
# Minimum extractable characters/page to consider the text layer "usable".
MIN_TEXT_CHARS_PER_PAGE = 40
CHAPTER_RE = re.compile(r"^\s*(chapter|section|part|unit|lesson)\s+[\dIVXLC]+", re.I)


class Chapter(TypedDict):
    title: str
    text: str
    page_start: int
    page_end: int


def extract_chapters(
    pdf_path: str,
    max_chapters: Optional[int] = None,
    page_range: Optional[tuple[int, int]] = None,
) -> list[Chapter]:
    """Extract chapters from a PDF. See module docstring for routing logic."""
    doc = fitz.open(pdf_path)
    try:
        n_pages = doc.page_count
        lo, hi = _resolve_range(page_range, n_pages)

        if _needs_hard_path(doc, lo, hi):
            log.info("PDF lacks usable text layer/TOC -> trying Nemotron hard path")
            chapters = hard_path_chapters(doc, lo, hi)
            if chapters:
                return _finalize(chapters, max_chapters)
            log.warning("Hard path unavailable/empty -> falling back to bucket split")

        chapters = _from_toc(doc, lo, hi)
        if not chapters:
            chapters = _from_headings(doc, lo, hi)
        if not chapters:
            chapters = _equal_buckets(doc, lo, hi, max_chapters or 3)
        return _finalize(chapters, max_chapters)
    finally:
        doc.close()


def _resolve_range(page_range, n_pages) -> tuple[int, int]:
    if not page_range:
        return 0, n_pages - 1
    lo, hi = page_range
    lo = max(0, lo)
    hi = min(n_pages - 1, hi)
    if hi < lo:
        lo, hi = 0, n_pages - 1
    return lo, hi


def _page_text(doc, i: int) -> str:
    return doc[i].get_text("text")


def _needs_hard_path(doc, lo: int, hi: int) -> bool:
    """True when the text layer is too sparse to trust (likely scanned/image)."""
    sample = range(lo, min(hi, lo + 4) + 1)
    chars = sum(len(_page_text(doc, i)) for i in sample)
    pages = max(1, len(list(sample)))
    return (chars / pages) < MIN_TEXT_CHARS_PER_PAGE


def _from_toc(doc, lo: int, hi: int) -> list[Chapter]:
    """Build chapters from PDF bookmarks (level-1 entries within range)."""
    toc = doc.get_toc(simple=True)  # [[level, title, page1based], ...]
    if not toc:
        return []
    tops = [(t[1].strip(), t[2] - 1) for t in toc if t[0] == 1 and lo <= t[2] - 1 <= hi]
    if len(tops) < 2:
        tops = [(t[1].strip(), t[2] - 1) for t in toc if lo <= t[2] - 1 <= hi]
    if not tops:
        return []
    chapters: list[Chapter] = []
    for idx, (title, start) in enumerate(tops):
        end = (tops[idx + 1][1] - 1) if idx + 1 < len(tops) else hi
        end = max(start, min(end, hi))
        text = "".join(_page_text(doc, p) for p in range(start, end + 1))
        chapters.append(Chapter(title=title or f"Chapter {idx + 1}",
                                text=text, page_start=start, page_end=end))
    return chapters


def _from_headings(doc, lo: int, hi: int) -> list[Chapter]:
    """Detect headings by large font size or 'Chapter N' style regex."""
    starts: list[tuple[int, str]] = []
    for p in range(lo, hi + 1):
        title = _detect_heading(doc[p])
        if title:
            starts.append((p, title))
    if len(starts) < 2:
        return []
    chapters: list[Chapter] = []
    for idx, (start, title) in enumerate(starts):
        end = (starts[idx + 1][0] - 1) if idx + 1 < len(starts) else hi
        end = max(start, min(end, hi))
        text = "".join(_page_text(doc, p) for p in range(start, end + 1))
        chapters.append(Chapter(title=title, text=text,
                                page_start=start, page_end=end))
    return chapters


def _detect_heading(page) -> Optional[str]:
    """Return a heading string for a page, or None."""
    data = page.get_text("dict")
    sizes: list[float] = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes.append(span.get("size", 0))
    if not sizes:
        return None
    big = max(sizes)
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            txt = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
            if not txt:
                continue
            line_size = max((s.get("size", 0) for s in line.get("spans", [])), default=0)
            if CHAPTER_RE.match(txt) or (line_size >= big - 0.1 and big >= 16 and len(txt) < 90):
                return txt
    return None


def _equal_buckets(doc, lo: int, hi: int, n: int) -> list[Chapter]:
    """Final fallback: split the page range into N roughly equal buckets."""
    n = max(1, min(n, hi - lo + 1))
    total = hi - lo + 1
    size = max(1, total // n)
    chapters: list[Chapter] = []
    start = lo
    idx = 0
    while start <= hi:
        end = min(hi, start + size - 1)
        if idx == n - 1:  # last bucket absorbs remainder
            end = hi
        text = "".join(_page_text(doc, p) for p in range(start, end + 1))
        chapters.append(Chapter(title=f"Section {idx + 1}", text=text,
                                page_start=start, page_end=end))
        start = end + 1
        idx += 1
        if idx >= n:
            break
    return chapters


def hard_path_chapters(doc, lo: int, hi: int) -> list[Chapter]:
    """Render pages to PNG bytes and reconstruct chapters via Nemotron Parse.

    Returns [] if the Modal parse function is unavailable so callers can fall
    back to the heuristic paths. Page rendering uses PyMuPDF pixmaps.
    """
    try:
        from pipeline import llm
    except Exception:
        return []

    cached = _parse_cache_get(doc, lo, hi)
    if cached is not None:
        log.info("Using cached Nemotron parse for pages %d-%d", lo, hi)
        return _chapters_from_parsed(cached, lo)

    images: list[bytes] = []
    for p in range(lo, hi + 1):
        pix = doc[p].get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x for legibility
        images.append(pix.tobytes("png"))
    try:
        parsed = llm.parse_pages(images)
    except Exception as exc:  # Modal not deployed / network / etc.
        log.warning("Nemotron parse_pages failed: %s", type(exc).__name__)
        return []
    _parse_cache_put(doc, lo, hi, parsed)
    return _chapters_from_parsed(parsed, lo)


def _parse_cache_key(doc, lo: int, hi: int) -> str:
    """Stable key from file bytes + page range (so re-runs reuse parse output)."""
    h = hashlib.sha256()
    try:
        with open(doc.name, "rb") as fh:
            h.update(fh.read())
    except Exception:
        h.update(str(getattr(doc, "name", "")).encode())
    h.update(f"|{lo}|{hi}".encode())
    return h.hexdigest()[:24]


def _parse_cache_get(doc, lo: int, hi: int):
    path = CACHE_DIR / f"parse_{_parse_cache_key(doc, lo, hi)}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _parse_cache_put(doc, lo: int, hi: int, parsed) -> None:
    path = CACHE_DIR / f"parse_{_parse_cache_key(doc, lo, hi)}.json"
    try:
        path.write_text(json.dumps(parsed), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not write parse cache: %s", type(exc).__name__)


def _chapters_from_parsed(parsed: list[dict], lo: int) -> list[Chapter]:
    """Turn Nemotron Parse page objects into chapters using title/section classes."""
    chapters: list[Chapter] = []
    cur: Optional[Chapter] = None
    for offset, page in enumerate(parsed):
        page_no = lo + offset
        objs = page.get("objects", []) if isinstance(page, dict) else []
        page_text = page.get("text", "") if isinstance(page, dict) else ""
        title = None
        for obj in objs:
            if str(obj.get("class", "")).lower() in {"title", "section", "section-header"}:
                title = (obj.get("text") or "").strip()
                if title:
                    break
        if title:
            if cur:
                cur["page_end"] = page_no - 1 if page_no > cur["page_start"] else page_no
                chapters.append(cur)
            cur = Chapter(title=title, text=page_text,
                          page_start=page_no, page_end=page_no)
        elif cur:
            cur["text"] += "\n" + page_text
            cur["page_end"] = page_no
        else:
            cur = Chapter(title="Section 1", text=page_text,
                          page_start=page_no, page_end=page_no)
    if cur:
        chapters.append(cur)
    return chapters


def _finalize(chapters: list[Chapter], max_chapters: Optional[int]) -> list[Chapter]:
    """Apply max_chapters cap and truncate each chapter body to budget."""
    if max_chapters:
        chapters = chapters[:max_chapters]
    for ch in chapters:
        ch["text"] = _clean(ch["text"])[:MAX_CHARS_PER_CHAPTER]
        ch["title"] = (ch["title"] or "Untitled").strip()[:120]
    return chapters


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


if __name__ == "__main__":  # quick manual check: python -m pipeline.pdf_parser file.pdf
    import sys

    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else "assets/sample.pdf"
    chs = extract_chapters(path, max_chapters=int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    for i, c in enumerate(chs):
        print(f"[{i+1}] {c['title']!r} pages {c['page_start']}-{c['page_end']} "
              f"({len(c['text'])} chars)")
