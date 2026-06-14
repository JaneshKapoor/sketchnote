"""Modal app: self-hosted open-weight GPU functions for Sketchnote.

We DOWNLOAD open weights from Hugging Face (with HF_TOKEN) and run them on
Modal serverless GPUs. We NEVER call NVIDIA NIM / build.nvidia.com or any other
hosted model API — Nemotron Parse runs from its open weights here.

Functions (looked up by pipeline/llm.py via modal.Function.from_name):
  summarize_chapter(title, text) -> {narration_script, visual_concepts}   MiniCPM4.1-8B (8B)
  parse_pages(images: list[bytes]) -> list[ParsedPage]                    Nemotron Parse v1.1
  generate_image(prompt) -> png_bytes                                     SDXL-Turbo (optional)

Deploy:  modal deploy modal_app.py
"""
from __future__ import annotations

import io
import json
import re

import modal

APP_NAME = "sketchnote"
CACHE_DIR = "/cache"
HF_HOME = f"{CACHE_DIR}/huggingface"

SUMMARIZER_MODEL = "openbmb/MiniCPM4.1-8B"
NEMOTRON_PARSE_MODEL = "nvidia/NVIDIA-Nemotron-Parse-v1.1"
SDXL_TURBO_MODEL = "stabilityai/sdxl-turbo"

app = modal.App(APP_NAME)

# Open .env locally at deploy time -> HF_TOKEN available inside containers.
try:
    hf_secret = modal.Secret.from_dotenv()
except Exception:  # pragma: no cover - allow deploy without local .env
    hf_secret = modal.Secret.from_dict({})

# Persistent HF cache so weights download only once.
cache_vol = modal.Volume.from_name("sketchnote-hf-cache", create_if_missing=True)

_BASE_PKGS = ["huggingface_hub>=0.24", "Pillow>=10.2"]
# MiniCPM4.1-8B needs transformers>=4.56, but its remote modeling code imports
# is_torch_fx_available, which was REMOVED in transformers 5.x -> pin to 4.56.1
# (the exact version the model card config is saved with).
llm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "transformers==4.56.1", "accelerate>=0.33",
                 "sentencepiece", "einops", *_BASE_PKGS)
    .env({"HF_HOME": HF_HOME})
)
# Nemotron Parse v1.1 reference env pins transformers==4.51.3 (+ timm, albumentations).
parse_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "transformers==4.51.3", "accelerate>=0.33",
                 "einops", "timm==1.0.22", "albumentations==2.0.8",
                 "opencv-python-headless", "sentencepiece", *_BASE_PKGS)
    .env({"HF_HOME": HF_HOME})
)
# diffusers 0.38+ eagerly references transformers symbols (e.g.
# Qwen3VLForConditionalGeneration) that don't exist in transformers 4.56.1, so
# pin diffusers to 0.31.0 — well-tested with SDXL-Turbo and this transformers.
sdxl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "diffusers==0.31.0", "transformers==4.56.1",
                 "accelerate>=0.33", *_BASE_PKGS)
    .env({"HF_HOME": HF_HOME})
)

# Prompt mirrored from pipeline/llm.py (containers don't import the Space code).
SUMMARIZE_PROMPT = (
    "You are scripting an educational whiteboard sketch-note video.\n"
    "Read the chapter and return STRICT JSON ONLY (no markdown, no code "
    "fences, no commentary) with exactly these keys:\n"
    '  "narration_script": a spoken-style summary of 80-150 words, clear and '
    "engaging, plain text only (no bullets, no markdown).\n"
    '  "visual_concepts": an array of 3 to 5 short noun phrases (2-4 words '
    "each) naming the key ideas to draw.\n\n"
    "Chapter title: {title}\nChapter text:\n{text}\n\nReturn only the JSON object."
)

_LLM: dict = {}  # warm-container cache: {"model":..., "tokenizer":...}


def _strip_json(raw: str) -> dict | None:
    """Extract a JSON object from model output, tolerating fences/prose."""
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    start, depth = raw.find("{"), 0
    if start < 0:
        return None
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


@app.function(image=llm_image, gpu="A10G", secrets=[hf_secret],
              volumes={CACHE_DIR: cache_vol}, timeout=900,
              scaledown_window=300)
def summarize_chapter(title: str, text: str) -> dict:
    """MiniCPM4.1-8B (8B params) -> strict-JSON chapter summary + visuals."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not _LLM:
        tok = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            SUMMARIZER_MODEL, torch_dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True).eval()
        _LLM.update(model=model, tokenizer=tok)

    tok, model = _LLM["tokenizer"], _LLM["model"]
    prompt = SUMMARIZE_PROMPT.format(title=title, text=text[:6000])

    def _run(user_prompt: str) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        text_in = tok.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
        inputs = tok([text_in], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=512,
                             do_sample=True, temperature=0.6, top_p=0.9)
        return tok.decode(out[0][inputs["input_ids"].shape[-1]:],
                          skip_special_tokens=True)

    parsed = _strip_json(_run(prompt))
    if parsed is None:  # one retry with a stricter nudge
        parsed = _strip_json(_run(prompt + "\nOutput ONLY valid JSON, nothing else."))
    if parsed is None:
        parsed = {"narration_script": "", "visual_concepts": []}
    return parsed


_PARSE: dict = {}  # warm-container cache for Nemotron Parse

# Nemotron Parse task prompt (predict classes + emit markdown).
NEMOTRON_TASK = "</s><s><predict_bbox><predict_classes><output_markdown>"


@app.function(image=parse_image, gpu="A10G", secrets=[hf_secret],
              volumes={CACHE_DIR: cache_vol}, timeout=1200,
              scaledown_window=300)
def parse_pages(images: list[bytes]) -> list[dict]:
    """Nemotron Parse v1.1 -> [{page, text, objects:[{class,text}]}, ...].

    Open weights downloaded from HF; runs locally on the Modal GPU. We do NOT
    call build.nvidia.com / NIM.
    """
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor, GenerationConfig

    if not _PARSE:
        model = AutoModel.from_pretrained(
            NEMOTRON_PARSE_MODEL, trust_remote_code=True,
            torch_dtype=torch.bfloat16).to("cuda").eval()
        processor = AutoProcessor.from_pretrained(
            NEMOTRON_PARSE_MODEL, trust_remote_code=True)
        gen_cfg = GenerationConfig.from_pretrained(
            NEMOTRON_PARSE_MODEL, trust_remote_code=True)
        _PARSE.update(model=model, processor=processor, gen_cfg=gen_cfg)

    model, processor, gen_cfg = _PARSE["model"], _PARSE["processor"], _PARSE["gen_cfg"]
    results: list[dict] = []
    for i, raw in enumerate(images):
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        inputs = processor(images=[image], text=NEMOTRON_TASK,
                           return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, generation_config=gen_cfg,
                                 max_new_tokens=2048)
        md = processor.batch_decode(out, skip_special_tokens=True)[0]
        results.append({"page": i, "text": md, "objects": _md_objects(md)})
    return results


def _md_objects(markdown: str) -> list[dict]:
    """Derive title/section objects from markdown headings for chapter splitting."""
    objs: list[dict] = []
    for line in (markdown or "").splitlines():
        m = re.match(r"^(#{1,3})\s+(.*\S)", line)
        if m:
            cls = "title" if len(m.group(1)) == 1 else "section"
            objs.append({"class": cls, "text": m.group(2).strip()})
    return objs


@app.function(image=sdxl_image, gpu="A10G", secrets=[hf_secret],
              volumes={CACHE_DIR: cache_vol}, timeout=600,
              scaledown_window=300)
def generate_image(prompt: str) -> bytes:
    """Optional: SDXL-Turbo (~3.5B) few-step image. Returns PNG bytes."""
    import torch
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        SDXL_TURBO_MODEL, torch_dtype=torch.float16, variant="fp16").to("cuda")
    image = pipe(prompt=prompt, num_inference_steps=4,
                 guidance_scale=0.0).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@app.local_entrypoint()
def main():
    """Smoke test the deployed functions: `modal run modal_app.py`."""
    summary = summarize_chapter.remote(
        "The Water Cycle",
        "Water evaporates from oceans, condenses into clouds, and falls as rain.")
    print("summary:", summary)

