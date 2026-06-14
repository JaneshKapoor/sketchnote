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

    clips, transcript = [], []
    n = len(chapters)
    for i, ch in enumerate(chapters):
        frac = 0.1 + 0.8 * (i / max(1, n))
        progress(frac, desc=f"Chapter {i + 1}/{n}: {ch['title'][:40]}…")
        try:
            clip = _build_chapter(ch, voice, use_sdxl)
            clips.append(clip["path"])
            transcript.append(clip["md"])
        except Exception:  # noqa: BLE001 — never let one chapter kill the run
            log.error("Chapter %d failed:\n%s", i + 1, traceback.format_exc())
            transcript.append(f"### {i + 1}. {ch['title']}\n\n_(skipped — error)_\n")

    if not clips:
        return None, "All chapters failed to render. See logs.\n\n" + "\n".join(transcript)

    progress(0.92, desc="Stitching chapters into the final video…")
    final = video.concat(clips)
    progress(1.0, desc="Done!")
    return final, "\n".join(transcript)


def _build_chapter(ch: dict, voice: str, use_sdxl: bool) -> dict:
    """Render a single chapter to a muxed clip; returns {path, md}."""
    summary = llm.summarize_chapter(ch["title"], ch["text"])
    narration = summary.get("narration_script") or ch["text"][:600]
    concepts = summary.get("visual_concepts") or [ch["title"]]
    diagram = summary.get("diagram")

    wav_path, duration = tts.synthesize(narration, voice=voice)
    png_path = visuals_mod.build_visual(concepts, ch["title"], diagram=diagram,
                                        use_sdxl=use_sdxl)
    silent = sketch.animate(png_path, target_duration=duration)
    clip_path = video.mux(silent, wav_path)

    bullets = "\n".join(f"- {c}" for c in concepts)
    md = (f"### {ch['title']}\n\n"
          f"**Visual concepts:**\n{bullets}\n\n"
          f"**Narration:**\n{narration}\n")
    return {"path": clip_path, "md": md}


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
                use_sdxl = gr.Checkbox(value=False,
                                       label="AI icon (SDXL-Turbo on Modal) — "
                                             "off uses a hand-drawn concept diagram")
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
