# Build Prompt — "Sketchnote" (Build Small Hackathon submission)

You are an autonomous coding agent. Build the project described below **from scratch**, working incrementally and verifying each phase before moving on. Do not skip the testing steps. Ask me only if something is genuinely blocking; otherwise make reasonable decisions and keep moving — the deadline is tight.

---

## 1. What we are building

**Sketchnote** turns an uploaded PDF (e.g. a textbook) into a **whiteboard sketch-animation video with synced voice narration**, chapter by chapter. The user uploads a PDF, picks a chapter range, and gets back a video where each chapter's summary is read aloud while a hand-drawn-style sketch animation plays in sync — like an automated visual-notes / explainer-video generator. Think "Golpo.ai, but fully local and open-source."

It is a submission to the **Hugging Face "Build Small" hackathon** and MUST follow the hackathon rules in Section 2 exactly.

---

## 2. HARD CONSTRAINTS (non-negotiable — a violation disqualifies the entry)

1. **Every AI model used must have under 32B total parameters** (total, not active). We will combine several small models; each one individually must be under the cap. Document each model's parameter count in the README.
2. **No proprietary hosted model APIs.** Do NOT call OpenAI, Google/Gemini, Anthropic, ElevenLabs, Cohere's hosted endpoints, or **NVIDIA's hosted NIM / build.nvidia.com endpoints** for the AI work (text, speech, image, or document parsing). All inference must use **open-weight models we host ourselves**. (NVIDIA's models have open weights — we self-host those weights on Modal; we never call NVIDIA's hosted API.)
3. **The app must be a Gradio app**, deployable as a Hugging Face Space (Gradio SDK, or Docker exposing a Gradio interface). The final deliverable is a public Space.
4. **No fine-tuning.** Use the models as released. We do not have time and it is not needed.
5. **Credentials live only in a `.env` file** (see Section 6). The only credentials are *infrastructure* tokens — Hugging Face and Modal — used to **download and host our own open models**. These are NOT "external model APIs" and do not violate rule 2. Never hardcode any token. Never commit `.env`.

If you ever feel tempted to reach for a hosted LLM/TTS/image/parse API "just to make it work," STOP — that breaks rule 2. Use the local/open self-hosted model path instead.

---

## 3. Tech stack & exact models

| Role | Model / Tool | HF ID / source | Params | Where it runs |
|---|---|---|---|---|
| Fast PDF text + TOC (clean digital PDFs) | PyMuPDF (`fitz`) | pip `pymupdf` | n/a (no model) | In the Space |
| **Document parsing / structure (scanned or complex PDFs)** | **NVIDIA Nemotron Parse** | `nvidia/NVIDIA-Nemotron-Parse-v1.1` (or latest v1.2 weights) | compact VLM — **confirm count on model card; expected well under cap** | Modal GPU |
| Summarization + visual-concept generation | **MiniCPM4.1-8B** (primary) | `openbmb/MiniCPM4.1-8B` | 8B | Modal GPU |
| (fallback LLM if MiniCPM is troublesome) | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 7B | Modal GPU |
| Narration (text-to-speech) | **Kokoro-82M** | `hexgrad/Kokoro-82M` (pip `kokoro`) | 0.082B | In the Space (CPU ok) |
| Whiteboard sketch animation | **ai-img2sketch** (NumPy + OpenCV, NO model) | github.com/daslearning-org/ai-img2sketch | n/a | In the Space |
| Image for the sketch (OPTIONAL, time-permitting) | SDXL-Turbo | `stabilityai/sdxl-turbo` | ~3.5B | Modal GPU |
| Video assembly | ffmpeg | system pkg | n/a | In the Space |
| UI | Gradio | pip `gradio` | n/a | In the Space |

**Sponsor alignment (why these models):**
- **MiniCPM4.1-8B** — the sponsor OpenBMB offers a special prize for projects using their models. This both fits the budget and unlocks that prize lane.
- **Nemotron Parse** — the sponsor NVIDIA offers a special prize for projects using Nemotron models. Parse genuinely upgrades our weakest step (PDF ingestion of scanned/complex/multi-column documents, and section/title classification that drives chapter splitting), so it is a real improvement, not a bolt-on.
- Keep MiniCPM as the primary summarizer; only swap to Qwen if MiniCPM blocks you.

**Do NOT** add Nemotron ASR (speech-to-text — wrong direction; we need text-to-speech), Nemotron Embed/ColEmbed VL (retrieval models — only needed for a RAG layer we are not building), or Nemotron Nano Omni (too heavy to wire in the time we have). One parse model + one summarizer + one TTS is the whole AI stack.

**Primary visual path is the cheap/reliable one:** generate a clean concept "card" (keywords + a simple diagram via matplotlib or an LLM-authored SVG rasterized to PNG), then run it through ai-img2sketch. Treat **SDXL-Turbo image generation as an optional upgrade** that we only wire in if the core pipeline is already working end-to-end. The pipeline must never hard-fail because image generation failed — fall back to the concept card.

---

## 4. Architecture

```
User → Gradio Space
        │  upload PDF + choose chapter range
        ▼
  Ingestion:
    • PyMuPDF first — extract text + TOC for clean digital PDFs (fast, in-Space)
    • If no TOC, or pages are scanned/image-based/complex → render pages to images,
      send to Nemotron Parse (Modal GPU) → structured text + object classes
      (title / section / table / caption …) used to recover chapter structure
        │  for each chapter:
        ▼
  Modal GPU endpoint ── MiniCPM4.1-8B ──► { narration_script, visual_concepts[] }
        ▼
  Visual builder:  concept card / SVG  (optional: SDXL-Turbo) → PNG
        ▼
  ai-img2sketch (OpenCV) → silent sketch-animation clip
        ▼
  Kokoro-82M → narration WAV  (generate audio FIRST, measure its duration)
        ▼
  ffmpeg: render the sketch animation to fill the audio duration, mux audio onto video
        ▼
  ffmpeg: concatenate all chapter clips → final MP4
        ▼
  Gradio shows final video + per-chapter transcript
```

**Compute split:** heavy models (Nemotron Parse, MiniCPM, optional SDXL-Turbo) run on **Modal** serverless GPU functions that the Space calls. Light work (Kokoro, OpenCV, ffmpeg, UI) runs **inside the Space**. This keeps the Space lightweight and uses the Modal credits.

> Simpler fallback architecture if Modal integration eats too much time: run the LLM inside a single GPU-enabled HF Space (MiniCPM in 4-bit) and use PyMuPDF-only ingestion, dropping Nemotron Parse. Build the full Modal path first; keep this as the escape hatch.

---

## 5. Repository structure

```
sketchnote/
├── app.py                  # Gradio UI + orchestration of the pipeline
├── modal_app.py            # Modal app: Nemotron Parse, MiniCPM, (optional SDXL-Turbo) GPU functions
├── pipeline/
│   ├── pdf_parser.py       # PyMuPDF text+TOC; route hard PDFs to Nemotron Parse; chapter splitting
│   ├── llm.py              # client that calls the Modal MiniCPM function; prompt templates
│   ├── tts.py              # Kokoro narration → WAV, returns (path, duration)
│   ├── visuals.py          # concept-card / SVG builder (+ optional SDXL-Turbo call)
│   ├── sketch.py           # vendored ai-img2sketch logic, parameterized by target_duration
│   └── video.py            # ffmpeg: mux audio+video per chapter, concat chapters
├── assets/
│   └── sample.pdf          # a short sample PDF for the demo
├── requirements.txt
├── packages.txt            # system deps for the Space (ffmpeg, espeak-ng)
├── .env.example
├── .gitignore              # MUST include .env
└── README.md               # HF Space frontmatter + hackathon write-up
```

Vendor (copy in) the core sketch function from ai-img2sketch rather than depending on its CLI. **Preserve its license and credit DasLearning + Yogendra Yatnalkar (storyboard-ai)** in the README and as a comment in `sketch.py`.

---

## 6. Credentials — `.env` handling

Create a `.env.example` with placeholders (committed) and instruct me to copy it to `.env` (gitignored) and fill in real values. Load it with `python-dotenv`. The ONLY entries:

```
# Hugging Face — used to download open model weights / for the Space
HF_TOKEN=

# Modal — used to host OUR OWN open models (Nemotron Parse, MiniCPM, SDXL-Turbo) on serverless GPU
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
```

Do not add any other API key. If the code ever needs another credential, that's a signal it's about to call a forbidden hosted model — reconsider the approach instead. In particular, **do not add an `NVIDIA_API_KEY` or any build.nvidia.com / NIM key** — we self-host the Nemotron weights on Modal, we do not call NVIDIA's hosted service. Add `.env` to `.gitignore`. Read tokens via `os.environ`; never print or log them.

---

## 7. Module specs

### `pdf_parser.py`
- `extract_chapters(pdf_path, max_chapters=None, page_range=None) -> list[Chapter]` where `Chapter = {title, text, page_start, page_end}`.
- **Fast path (default):** PyMuPDF. Prefer `doc.get_toc()` (PDF bookmarks). If a TOC exists and the document has an extractable text layer, build chapters from it directly. This is cheap and runs in the Space.
- **Hard path (Nemotron Parse):** trigger when (a) there is no usable text layer (scanned/image PDF), or (b) there is no TOC and headings can't be detected reliably. Render the relevant pages to images with PyMuPDF (`page.get_pixmap`), send them to the Nemotron Parse Modal function, and use its returned objects — specifically the `title`/`section` classes — to reconstruct chapter boundaries, plus its extracted text as the chapter body. Cache results so we don't re-parse pages.
- Heading-detection fallback (font-size heuristic / regex like `^Chapter\s+\d+`) sits between the two; final fallback is N equal page buckets.
- Respect `max_chapters` and `page_range` so demos stay short. Truncate each chapter's text to a sane token budget before summarization.

### `modal_app.py`
- One Modal app, one container image with `transformers`, `torch`, `accelerate` (+ `vllm` if preferred), plus whatever Nemotron Parse requires.
- **`parse_pages(images: list[bytes]) -> list[ParsedPage]`** — loads Nemotron Parse weights (download from HF with `HF_TOKEN`; do NOT hit build.nvidia.com), runs document parsing, returns text + object classes + bounding boxes per page. Give it its own GPU/profile; Parse can be VRAM-sensitive, so isolate it from the LLM container if needed.
- **`summarize_chapter(title, text) -> {narration_script, visual_concepts}`** — MiniCPM4.1-8B. Use an A10G (24GB); if cost/availability is an issue, load in 4-bit on a T4. Prompt for **strict JSON only** (no markdown fences); parse defensively, strip fences if present, retry once on parse failure.
  - `narration_script`: ~80–150 words, spoken-style, clear, no markdown.
  - `visual_concepts`: 3–5 short noun phrases / a one-line scene description to drive the visual.
- **(optional) `generate_image(prompt) -> png_bytes`** — SDXL-Turbo, few steps.
- Verify and record each loaded model's parameter count for the README's ≤32B proof.

### `llm.py`
- Thin client the Space uses to invoke the Modal `parse_pages` and `summarize_chapter` functions (Modal SDK lookup or deployed web endpoints). Holds the prompt templates.

### `tts.py`
- Kokoro via `pip install kokoro soundfile`; system dep `espeak-ng` (in `packages.txt`).
- `synthesize(text, voice="af_heart") -> (wav_path, duration_seconds)`. Sample rate 24000. Generate audio FIRST — its duration drives the video length.

### `visuals.py`
- `build_visual(concepts, title) -> png_path`.
- Primary: a clean "whiteboard card" — title + bulleted keywords, optionally a simple matplotlib/SVG diagram, white background, black strokes (so the sketch effect looks natural). Rasterize SVG→PNG with `cairosvg` if you generate SVG.
- Optional upgrade: call the Modal SDXL-Turbo function. Wrap in try/except → fall back to the card on any failure.

### `sketch.py`
- Vendored ai-img2sketch OpenCV logic. Expose `animate(png_path, target_duration, fps=25) -> mp4_path` so the reveal is distributed across `target_duration * fps` frames; hold the final frame if the drawing finishes early so the clip length matches the narration.

### `video.py`
- `mux(video_path, wav_path) -> mp4_path` using ffmpeg (`-c:v libx264 -c:a aac -shortest`), ensuring clip length == audio length.
- `concat(clip_paths) -> final_mp4` using the ffmpeg concat demuxer.

### `app.py`
- Gradio UI: PDF upload, a "max chapters" / page-range control (default small, e.g. 3, to keep demos fast), a Generate button, a progress indicator, the output video player, and a per-chapter transcript panel.
- Orchestrate: ingest → for each chapter { summarize → visual → TTS → sketch(target=audio duration) → mux } → concat → display.
- **Robustness:** wrap each chapter in try/except; on failure, log and skip that chapter but continue; always return *some* video. Never let one bad chapter kill the run.

---

## 8. requirements.txt / packages.txt

`requirements.txt`: `gradio`, `pymupdf`, `kokoro`, `soundfile`, `numpy`, `opencv-python-headless`, `python-dotenv`, `modal`, `cairosvg`, `matplotlib`, `Pillow` (+ `transformers torch accelerate bitsandbytes` only if running the LLM in-Space fallback).

`packages.txt` (Space system deps): `ffmpeg`, `espeak-ng`.

(Nemotron Parse / MiniCPM / SDXL-Turbo heavy deps live in the **Modal** image, not the Space.)

---

## 9. README.md (deliverable)

Include valid Hugging Face Space frontmatter and the hackathon write-up:

```yaml
---
title: Sketchnote
emoji: ✏️
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
tags:
  - build-small-hackathon
  # add the correct track + badge tags per the field guide
---
```

Below the frontmatter: one-paragraph idea pitch; the model table from Section 3 **with confirmed parameter counts proving every model is < 32B** (explicitly list Nemotron Parse, MiniCPM4.1-8B, Kokoro-82M); a "how it works" diagram; setup instructions (`.env` from `.env.example`, Modal deploy command, run command); a note on the OpenBMB (MiniCPM) and NVIDIA (Nemotron Parse) sponsor models used; and **credits to DasLearning / Yogendra Yatnalkar** for the sketch algorithm plus the relevant license notes (ai-img2sketch license + NVIDIA Community Model License for Nemotron Parse).

---

## 10. Build order (do it in this sequence and verify each step)

1. Scaffold repo, `.env.example`, `.gitignore`, `requirements.txt`, `packages.txt`.
2. `pdf_parser.py` — PyMuPDF fast path first. Verify chapter extraction on `assets/sample.pdf`, print chapter titles.
3. `tts.py` — verify Kokoro produces a WAV and a correct duration locally.
4. `sketch.py` — verify a static PNG becomes an animation clip of a requested duration.
5. `video.py` — verify mux + concat produce a playable MP4 (use a placeholder narration + card first, BEFORE wiring any model).
6. **At this point you should have an end-to-end video with dummy summaries. Confirm it plays.**
7. `modal_app.py` + `llm.py` — deploy MiniCPM on Modal, verify JSON summaries from real chapter text.
8. `visuals.py` — concept cards from real `visual_concepts`.
9. `app.py` — wire the full Gradio pipeline (PyMuPDF ingestion); test with the sample PDF, 2–3 chapters.
10. **Add Nemotron Parse:** deploy the `parse_pages` Modal function; wire the hard-path trigger in `pdf_parser.py`; verify on a scanned/complex sample PDF that structure + text come back and chapters are reconstructed. Keep PyMuPDF as the fast path for clean PDFs.
11. Robustness pass (try/except per chapter, fallbacks, caching).
12. (Optional, only if time remains) SDXL-Turbo visual upgrade.
13. Write README, deploy the Space publicly in the hackathon org, confirm it runs there.

Get a working ugly version by step 6; everything after is quality. Nemotron Parse (step 10) is a real upgrade but must not block shipping — if it isn't working in time, ship with PyMuPDF ingestion and note Parse as a path you built toward.

---

## 11. Do NOT

- Do not use Remotion or any JS video renderer (OpenCV+ffmpeg already produces the video).
- Do not call any proprietary hosted model API for text, speech, images, or parsing — including **NVIDIA NIM / build.nvidia.com**. Self-host the open weights on Modal.
- Do not add Nemotron ASR, Embed/ColEmbed VL, or Nano Omni — they don't fit this pipeline or the time budget.
- Do not fine-tune anything.
- Do not hardcode or log tokens; do not commit `.env`.
- Do not try to process huge PDFs live in the demo — keep the chapter range small.

When done, give me: the run command, the Modal deploy command(s), and a short list of anything I need to do manually (fill `.env`, create the Space, record the demo).
