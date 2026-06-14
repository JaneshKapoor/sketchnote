---
title: Sketchnote
emoji: ✏️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
python_version: "3.11"
pinned: false
tags:
  - build-small-hackathon
  - gradio
  - openbmb
  - nvidia
  - video
  # add the correct track + badge tags per the hackathon field guide
---

# ✏️ Sketchnote

**Turn a PDF (e.g. a textbook) into a whiteboard sketch-animation video with
synced voice narration — chapter by chapter.** Upload a PDF, pick a chapter
range, and Sketchnote reads each chapter's summary aloud while a hand-drawn-style
sketch animation plays in sync. Think *"Golpo.ai, but fully local and
open-source."*

This is a submission to the Hugging Face **"Build Small"** hackathon. Every AI
model is **open-weight, self-hosted, and under 32B parameters** — no proprietary
hosted model APIs are used for any AI work (text, speech, image, or parsing).

## Why it fits the rules

| Rule | How Sketchnote complies |
|---|---|
| Every model < 32B params | Largest is MiniCPM4.1-**8B** (see table) |
| No proprietary hosted model APIs | All inference uses open weights we self-host on Modal / in the Space. We **never** call OpenAI/Gemini/Anthropic/ElevenLabs or **NVIDIA NIM / build.nvidia.com** |
| Gradio app, deployable as a Space | `app.py` is a Gradio app (`sdk: gradio`) |
| No fine-tuning | Models used as released |
| Credentials only in `.env` | Only infra tokens (HF + Modal) to download/host **our own** open weights; read via `os.environ`, never logged, `.env` is git-ignored |

## Models (every one < 32B — proof)

| Role | Model | HF ID | Parameters | Where it runs |
|---|---|---|---|---|
| Fast PDF text + TOC | PyMuPDF (`fitz`) | pip `pymupdf` | n/a (no model) | In the Space |
| Document parsing (scanned/complex) | **NVIDIA Nemotron Parse v1.1** | `nvidia/NVIDIA-Nemotron-Parse-v1.1` | **885M (0.885B)** — encoder/decoder VLM (657M ViT-H vision + 256M mBART decoder) | Modal GPU |
| Summarization + visual concepts | **MiniCPM4.1-8B** (primary) | `openbmb/MiniCPM4.1-8B` | **8B** | Modal GPU |
| Fallback LLM | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 7B | Modal GPU |
| Narration (TTS) | **Kokoro-82M** | `hexgrad/Kokoro-82M` | **0.082B** | In the Space (CPU) |
| Whiteboard sketch animation | **ai-img2sketch** (NumPy + OpenCV) | vendored, no model | n/a | In the Space |
| Image (optional upgrade) | SDXL-Turbo | `stabilityai/sdxl-turbo` | ~3.5B | Modal GPU |
| Video assembly | ffmpeg | system pkg | n/a | In the Space |
| UI | Gradio | pip `gradio` | n/a | In the Space |

All AI models are individually well under the 32B cap (largest = 8B).

## How it works

```
User → Gradio Space
        │  upload PDF + choose chapter range
        ▼
  Ingestion:
    • PyMuPDF first — text + TOC for clean digital PDFs (fast, in-Space)
    • If no TOC / scanned / complex → render pages to images,
      send to Nemotron Parse (Modal GPU) → structured text + title/section
      classes used to recover chapter structure  (results cached)
        │  for each chapter:
        ▼
  Modal GPU ── MiniCPM4.1-8B ──► { narration_script, visual_concepts[] }
        ▼
  Visual builder: whiteboard concept card  (optional: SDXL-Turbo) → PNG
        ▼
  Kokoro-82M → narration WAV  (audio FIRST → measures duration)
        ▼
  ai-img2sketch (OpenCV) → sketch reveal sized to the audio duration
        ▼
  ffmpeg: mux audio onto the sketch clip → per-chapter MP4
        ▼
  ffmpeg: concatenate chapter clips → final MP4
        ▼
  Gradio shows the final video + per-chapter transcript
```

**Compute split:** heavy models (Nemotron Parse, MiniCPM, optional SDXL-Turbo)
run on **Modal** serverless GPUs; light work (Kokoro, OpenCV, ffmpeg, UI) runs
**inside the Space**. If Modal isn't deployed, Sketchnote falls back to a non-AI
extractive summary so it still produces a video.

## Repository layout

```
app.py            Gradio UI + pipeline orchestration
modal_app.py      Modal GPU functions: summarize_chapter, parse_pages, generate_image
pipeline/
  config.py       env + paths + ffmpeg resolver (tokens via os.environ, never logged)
  pdf_parser.py   PyMuPDF fast path + Nemotron hard path → chapters (+ parse cache)
  llm.py          Modal client + prompt templates + extractive fallback
  tts.py          Kokoro narration → (wav, duration)
  visuals.py      whiteboard concept card (+ optional SDXL-Turbo)
  sketch.py       vendored ai-img2sketch reveal, sized to narration
  video.py        ffmpeg mux + concat
assets/sample.pdf a short sample for the demo
requirements.txt  Space (in-Space) deps    packages.txt  ffmpeg, espeak-ng
.env.example      infra-token placeholders  .gitignore   includes .env
```

## Setup & run

```bash
# 1. credentials (infra only — HF + Modal)
cp .env.example .env        # then fill HF_TOKEN, MODAL_TOKEN_ID, MODAL_TOKEN_SECRET

# 2. environment (Python 3.11 recommended)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# system deps (Linux): the Space installs these via packages.txt
#   sudo apt-get install ffmpeg espeak-ng

# 3. deploy the GPU models to Modal (self-hosting our own open weights)
modal deploy modal_app.py
modal run modal_app.py      # optional smoke test of summarize_chapter

# 4. run the Gradio app
python app.py               # open the printed local URL
```

If Modal is **not** deployed, the app still runs end-to-end using PyMuPDF
ingestion + the non-AI extractive summarizer (degraded, but never hard-fails).

## Sponsor models

- **OpenBMB — MiniCPM4.1-8B** is the primary summarizer that writes each
  chapter's narration script and visual concepts.
- **NVIDIA — Nemotron Parse v1.1** upgrades our weakest step: ingesting scanned /
  multi-column / complex PDFs and classifying titles & sections to drive chapter
  splitting. We self-host its **open weights** on Modal — we do **not** call
  NVIDIA's hosted NIM / build.nvidia.com service.

## Credits & licenses

- **Sketch animation** adapted from **ai-img2sketch** by **DasLearning**
  (https://github.com/daslearning-org/ai-img2sketch) and the **storyboard-ai**
  project by **Yogendra Yatnalkar**. Credit preserved here and in
  `pipeline/sketch.py`; see those projects for their original licenses.
- **NVIDIA Nemotron Parse v1.1** — NVIDIA Open Model / Community Model License
  (tokenizer under CC-BY-4.0). See the model card.
- **MiniCPM4.1-8B** — OpenBMB model license (see model card).
- **Kokoro-82M** — Apache-2.0 (hexgrad/Kokoro-82M).

## Manual checklist (for the maintainer)

1. `cp .env.example .env` and fill `HF_TOKEN`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`.
2. `modal deploy modal_app.py` (first call downloads weights to a Modal Volume).
3. Create the public HF Space in the hackathon org (Gradio SDK) and push this repo.
4. Add the correct hackathon **track + badge** tags to the frontmatter.
5. Record the demo video.
