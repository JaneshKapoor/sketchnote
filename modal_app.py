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
# NOTE: built with <<TITLE>>/<<TEXT>> placeholders + str.replace (NOT str.format)
# because the example JSON below contains literal { } braces.
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
              scaledown_window=300, min_containers=0)
def summarize_chapter(title: str, text: str) -> dict:
    """MiniCPM4.1-8B (8B params) -> strict-JSON storyboard {"beats":[...]}."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not _LLM:
        print(f"[summarize_chapter] cold start: loading {SUMMARIZER_MODEL}")
        tok = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            SUMMARIZER_MODEL, torch_dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True).eval()
        _LLM.update(model=model, tokenizer=tok)
        print("[summarize_chapter] model loaded OK")

    tok, model = _LLM["tokenizer"], _LLM["model"]
    print(f"[summarize_chapter] title={title!r} text_len={len(text or '')}")

    # .replace (NOT .format): the example JSON in the prompt has literal { } braces.
    prompt = (STORYBOARD_PROMPT
              .replace("<<TITLE>>", title or "")
              .replace("<<TEXT>>", (text or "")[:6000]))

    def _run(user_prompt: str) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        # enable_thinking=False: skip MiniCPM's internal chain-of-thought.
        # Wrap in try/except in case older tokenizer templates don't support the kwarg.
        try:
            text_in = tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False)
        except TypeError:
            text_in = tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        inputs = tok([text_in], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=900,
                                 do_sample=True, temperature=0.6, top_p=0.9)
        return tok.decode(out[0][inputs["input_ids"].shape[-1]:],
                          skip_special_tokens=True)

    raw = _run(prompt)
    # ── Log raw output before any parsing (visible in `modal app logs sketchnote`) ──
    print(f"[summarize_chapter] raw output (first 800 chars): {raw[:800]!r}")

    parsed = _strip_json(raw)
    if parsed is None or not parsed.get("beats"):
        print("[summarize_chapter] no beats on first attempt — retrying with nudge")
        raw2 = _run(prompt + "\nOutput ONLY the JSON object, nothing else.")
        print(f"[summarize_chapter] retry raw (first 400 chars): {raw2[:400]!r}")
        parsed = _strip_json(raw2)

    if parsed and parsed.get("beats"):
        print(f"[summarize_chapter] success: {len(parsed['beats'])} beats")
        return parsed

    # Both attempts failed: return the raw text so the client can log it.
    print(f"[summarize_chapter] JSON parse failed after retry. Returning raw.")
    return {"beats": [], "raw": raw[:2000]}


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
def generate_image(prompt: str, negative_prompt: str = "") -> bytes:
    """Optional: SDXL-Turbo (~3.5B) few-step image. Returns PNG bytes.

    A negative_prompt (e.g. to suppress text/letters/watermarks) needs
    classifier-free guidance, so it is only honored with guidance_scale > 1.
    """
    import torch
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        SDXL_TURBO_MODEL, torch_dtype=torch.float16, variant="fp16").to("cuda")
    kwargs = dict(prompt=prompt, num_inference_steps=4, guidance_scale=0.0)
    if negative_prompt:
        kwargs.update(negative_prompt=negative_prompt,
                      num_inference_steps=6, guidance_scale=2.0)
    image = pipe(**kwargs).images[0]
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

