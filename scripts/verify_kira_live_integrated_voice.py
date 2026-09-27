#!/usr/bin/env python3
"""Short Metal + frozen-Listener gate for KIRA's integrated emotional voice."""

from __future__ import annotations

import json
import argparse
import math
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx
import numpy as np
import soundfile as sf
from transformers import AutoProcessor

from kira_live.emotion import EmotionState
from kira_live.mlx_listener import KiraMLXListener
from kira_live.mlx_voice import IntegratedSpeechPlan, KiraIntegratedQwenVoice, emotion_style_instruction
from kira_live.weights import resolve_composite_weights
from kira_live.audio_pipeline import overlap_audio


def wer(reference: str, hypothesis: str) -> float:
    left = re.findall(r"[a-z0-9']+", reference.lower())
    right = re.findall(r"[a-z0-9']+", hypothesis.lower())
    row = list(range(len(right) + 1))
    for i, source in enumerate(left, 1):
        nxt = [i]
        for j, target in enumerate(right, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (source != target)))
        row = nxt
    return row[-1] / max(len(left), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phrase", default="Hello, this is Kira. I can hear you clearly, and my voice is steady and natural.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training_runs/kira_live_customvoice_v2")
    parser.add_argument("--simulate-playback", action="store_true")
    args = parser.parse_args()
    phrase = args.phrase
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "kira_integrated_voice.wav"
    report_path = output_dir / "verification.json"
    weights = resolve_composite_weights()
    voice = KiraIntegratedQwenVoice(
        model_id=str(weights.talker.snapshot),
    )
    emotion = EmotionState(
        {"neutral": 0.65, "joy": 0.25, "sadness": 0.02, "anger": 0.02, "fear": 0.02, "surprise": 0.04},
        valence=0.35,
        arousal=0.42,
        confidence=0.82,
        observed_at=time.monotonic(),
    )
    plan = IntegratedSpeechPlan(
        token_ids=(),
        public_text=phrase,
        speaker="Aiden",
        language="English",
        style_instruction=emotion_style_instruction(emotion),
    )
    started = time.perf_counter()
    chunks = []
    first_audio_seconds = None
    sample_rate = 24_000
    def consume(samples, rate):
        nonlocal first_audio_seconds, sample_rate
        sample_rate = rate
        if first_audio_seconds is None:
            first_audio_seconds = time.perf_counter() - started
        chunks.append(samples)
        if args.simulate_playback:
            time.sleep(len(samples) / rate)
    cancelled = threading.Event()
    if args.simulate_playback:
        overlap_audio(voice.iter_audio(plan, cancelled), consume, cancelled)
    else:
        for samples, rate in voice.iter_audio(plan, cancelled):
            consume(samples, rate)
    elapsed = time.perf_counter() - started
    if not chunks:
        raise RuntimeError("The integrated Talker produced no audio.")
    audio = np.concatenate(chunks)
    sf.write(output, audio, sample_rate)

    processor = AutoProcessor.from_pretrained(
        str(weights.listener.snapshot), local_files_only=True
    )
    listener = KiraMLXListener.from_composite_checkpoint(
        weights.listener.model_file, weights.listener.config, processor.tokenizer
    )
    positions = np.arange(round(len(audio) * 16_000 / sample_rate)) * sample_rate / 16_000
    audio_16k = np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)
    encoded = processor.feature_extractor(
        audio_16k,
        sampling_rate=16_000,
        return_tensors="np",
        return_attention_mask=True,
    )
    length = int(encoded["attention_mask"].sum())
    padded = max(100, int(math.ceil(length / 100) * 100))
    decoded = listener.semantic_decode(
        mx.array(encoded["input_features"][:, :, :padded], dtype=mx.bfloat16),
        length,
    )
    score = wer(phrase, decoded.text)
    duration = len(audio) / sample_rate
    report = {
        "model": weights.talker.spec.model_id,
        "revision": weights.talker.spec.revision,
        "speaker": plan.speaker,
        "style_instruction": plan.style_instruction,
        "phrase": phrase,
        "listener_loopback": decoded.text,
        "wer": score,
        "chunks": len(chunks),
        "first_audio_seconds": first_audio_seconds,
        "generation_seconds": elapsed,
        "simulated_concurrent_playback": args.simulate_playback,
        "audio_seconds": duration,
        "realtime_factor": elapsed / duration,
        "output": str(output),
        "accepted": score <= 0.2,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
