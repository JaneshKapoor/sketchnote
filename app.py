"""Sketchnote — Gradio app.

Upload a PDF, pick a chapter range, and get a whiteboard sketch-animation video
with synced Kokoro narration, chapter by chapter. Heavy models (MiniCPM,
Nemotron Parse, optional SDXL-Turbo) run on Modal; light work runs here.

Built for the Hugging Face "Build Small" hackathon — every model is < 32B and
fully open-weight / self-hosted (no proprietary hosted model APIs).
"""
from __future__ import annotations

import logging
import traceback

import gradio as gr

from pipeline import llm, pdf_parser, sketch, tts, video
from pipeline import visuals as visuals_mod

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sketchnote.app")

VOICES = ["af_heart", "af_bella", "af_sarah", "am_michael", "am_adam"]


def _page_range(start: int, end: int):
    start, end = int(start or 0), int(end or 0)
    if start > 0 and end >= start:
        return (start - 1, end - 1)  # UI is 1-based; parser is 0-based
    return None


def run_pipeline(pdf_file, max_chapters, page_start, page_end, voice, use_sdxl,
                 progress=gr.Progress()):
    """Ingest -> per chapter {summarize, visual, tts, sketch, mux} -> concat."""
    if not pdf_file:
        return None, "Please upload a PDF first."
    pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name

    progress(0.05, desc="Reading PDF and splitting chapters…")
    chapters = pdf_parser.extract_chapters(
        pdf_path, max_chapters=int(max_chapters), page_range=_page_range(page_start, page_end))
    if not chapters:
        return None, "Could not extract any chapters from this PDF."

    clips, transcript, warnings = [], [], []
    n = len(chapters)
    for i, ch in enumerate(chapters):
        frac = 0.1 + 0.8 * (i / max(1, n))
        progress(frac, desc=f"Chapter {i + 1}/{n}: {ch['title'][:40]}…")
        try:
            clip = _build_chapter(ch, voice, use_sdxl)
            clips.append(clip["path"])
            transcript.append(clip["md"])
            if clip.get("warning"):
                warnings.append(clip["warning"])
        except Exception:  # noqa: BLE001 — never let one chapter kill the run
            log.error("Chapter %d failed:\n%s", i + 1, traceback.format_exc())
            transcript.append(f"### {i + 1}. {ch['title']}\n\n_(skipped — error)_\n")

    if not clips:
        return None, "All chapters failed to render. See logs.\n\n" + "\n".join(transcript)

    progress(0.92, desc="Stitching chapters into the final video…")
    final = video.concat(clips)
    progress(1.0, desc="Done!")

    warn_banner = ""
    if warnings:
        unique = list(dict.fromkeys(warnings))  # dedupe, preserve order
        warn_banner = "\n".join(f"> ⚠️ **{w}**" for w in unique) + "\n\n"
    return final, warn_banner + "\n".join(transcript)


def _build_chapter(ch: dict, voice: str, use_sdxl: bool) -> dict:
    """Render a single chapter to a muxed clip using per-beat synchronization.

    For each beat:
      1. Synthesize just that beat's sentence with Kokoro → get duration d.
      2. Render the cumulative visual (nodes 0..k). With ``use_sdxl`` this is the
         chapter's SDXL illustration with our labels overlaid; otherwise the
         hand-drawn storyboard diagram.
      3. animate_beat reveals only the new label k over d seconds (prior labels
         stay drawn; drawing finishes at ~75 % then holds).
      4. Mux beat audio onto the beat clip.
    All beat clips are concatenated into the chapter clip.

    Returns {path, md, warning}.
    """
    summary = llm.summarize_chapter(ch["title"], ch["text"])
    beats = summary.get("beats") or []
    warning = summary.get("warning")  # surfaced in the UI transcript

    # Graceful fallback: if we somehow have no beats at all, one sentence.
    if not beats:
        beats = [{"say": ch["text"][:300] or f"This section covers {ch['title']}.",
                  "node": ch["title"][:40], "connects_to": None}]

    # Image-only mode: generate ONE text-free illustration for the chapter and
    # overlay our labels on it. Falls back to the diagram if SDXL is unavailable.
    bg_path = visuals_mod.chapter_illustration(ch["title"], beats) if use_sdxl else None

    beat_clips: list[str] = []
    beat_mds: list[str] = []
    prev_png: str | None = None

    for k, beat in enumerate(beats):
        log.info("Chapter %r beat %d/%d: node=%r", ch["title"], k + 1, len(beats),
                 beat["node"])
        # Cumulative visual up to (and including) beat k.
        if bg_path:
            full_png = visuals_mod.build_image_label_frame(
                beats[: k + 1], ch["title"], bg_path)
        else:
            full_png = visuals_mod.build_storyboard_frame(beats[: k + 1], ch["title"])

        wav_path, duration = tts.synthesize(beat["say"], voice=voice)
        silent = sketch.animate_beat(full_png, prev_png, target_duration=duration)
        clip = video.mux(silent, wav_path)
        beat_clips.append(clip)
        beat_mds.append(f"- **{beat['node']}**: {beat['say']}")
        prev_png = full_png

    chapter_clip = video.concat(beat_clips) if len(beat_clips) > 1 else beat_clips[0]

    warn_md = f"\n\n> ⚠️ **{warning}**" if warning else ""
    md = (f"### {ch['title']}{warn_md}\n\n"
          + "\n".join(beat_mds) + "\n")
    return {"path": chapter_clip, "md": md, "warning": warning}


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Sketchnote") as demo:
        gr.Markdown(
            "# ✏️ Sketchnote\n"
            "Turn a PDF (e.g. a textbook) into a **whiteboard sketch-animation "
            "video with synced narration**, chapter by chapter. "
            "All models are open-weight and under 32B parameters.")
        with gr.Row():
            with gr.Column(scale=1):
                pdf_in = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
                max_ch = gr.Slider(1, 8, value=3, step=1, label="Max chapters")
                with gr.Row():
                    p_start = gr.Number(value=0, precision=0, label="First page (0 = auto)")
                    p_end = gr.Number(value=0, precision=0, label="Last page (0 = auto)")
                voice = gr.Dropdown(VOICES, value="af_heart", label="Narration voice")
                use_sdxl = gr.Checkbox(value=True,
                                       label="AI illustration + labels (SDXL-Turbo "
                                             "on Modal) — off uses a hand-drawn "
                                             "concept diagram")
                go = gr.Button("Generate video", variant="primary")
                gr.Examples(examples=[["assets/sample.pdf"]], inputs=[pdf_in],
                            label="Try the sample PDF")
            with gr.Column(scale=1):
                video_out = gr.Video(label="Sketchnote video")
                transcript_out = gr.Markdown(label="Per-chapter transcript")
        go.click(run_pipeline,
                 inputs=[pdf_in, max_ch, p_start, p_end, voice, use_sdxl],
                 outputs=[video_out, transcript_out])
        gr.Markdown(
            "_Tip: keep the chapter range small for a fast demo. If Modal isn't "
            "deployed, Sketchnote falls back to a non-AI extractive summary so it "
            "still produces a video._")
    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.queue().launch()
