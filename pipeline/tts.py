"""Text-to-speech narration with Kokoro-82M (open weights, runs on CPU).

synthesize(text, voice) -> (wav_path, duration_seconds). Audio is generated
FIRST so its duration can drive the length of the sketch animation. Sample
rate is 24000 Hz. Kokoro is an 0.082B-parameter model — well under the 32B cap.
"""
from __future__ import annotations

import logging

import numpy as np
import soundfile as sf

from pipeline.config import DEFAULT_VOICE, TTS_SAMPLE_RATE, new_tmp

log = logging.getLogger("sketchnote.tts")

# Slightly faster than 1.0 for a more natural, energetic narration pace.
TTS_SPEED = 1.2
_PIPELINE = None  # lazy KPipeline singleton (loading weights is expensive)


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from kokoro import KPipeline

        # 'a' = American English. Kokoro downloads its open weights on first use.
        _PIPELINE = KPipeline(lang_code="a")
    return _PIPELINE


def _to_numpy(audio) -> np.ndarray:
    """Coerce a Kokoro audio chunk (torch tensor or array) to 1-D float32."""
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return audio


def synthesize(text: str, voice: str = DEFAULT_VOICE) -> tuple[str, float]:
    """Synthesize narration to a WAV file.

    Returns (wav_path, duration_seconds). Falls back to a short silent clip if
    the text is empty so the pipeline never hard-fails on a bad chapter.
    """
    text = (text or "").strip()
    wav_path = new_tmp(suffix=".wav", prefix="narration_")
    if not text:
        silent = np.zeros(int(TTS_SAMPLE_RATE * 0.5), dtype=np.float32)
        sf.write(wav_path, silent, TTS_SAMPLE_RATE)
        return wav_path, 0.5

    pipeline = _get_pipeline()
    chunks: list[np.ndarray] = []
    for _graphemes, _phonemes, audio in pipeline(text, voice=voice,
                                                 speed=TTS_SPEED):
        chunks.append(_to_numpy(audio))

    if not chunks:
        raise RuntimeError("Kokoro produced no audio for the given text")

    samples = np.concatenate(chunks)
    sf.write(wav_path, samples, TTS_SAMPLE_RATE)
    duration = len(samples) / float(TTS_SAMPLE_RATE)
    log.info("TTS: %d chars -> %.2fs (%s)", len(text), duration, voice)
    return wav_path, duration


if __name__ == "__main__":  # quick manual check
    logging.basicConfig(level=logging.INFO)
    p, d = synthesize(
        "Hello! This is Sketchnote testing the Kokoro narration voice.",
    )
    print(f"Wrote {p} ({d:.2f}s)")
