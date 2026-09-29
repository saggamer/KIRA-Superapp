"""Small real-weight voice test; no microphone capture or training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
import numpy as np
import sounddevice as sd
import soundfile as sf
from kira_live.audio_pipeline import overlap_audio
from kira_live.emotion import EmotionState
from kira_live.mlx_voice import IntegratedSpeechPlan
from kira_live.native_session import KiraNativeModelBundle
from kira_live.token_bus import remap_token_ids
from scripts.verify_kira_live_integrated_voice import wer


def run(args):
    started = time.perf_counter()
    os.environ['KIRA_LIVE_TALKER_BITS'] = str(args.quant_bits)
    bundle = KiraNativeModelBundle()
    loading = time.perf_counter() - started
    phrase = args.phrase
    rows = []
    with mx.stream(bundle.mlx_cpu_stream), mx.stream(bundle.mlx_stream):
        for turn in range(2):
            mx.random.seed(42)
            plan = IntegratedSpeechPlan((), phrase, 'Aiden', 'English', args.style)
            chunks, gaps = [], []
            first = None
            previous_end = None
            first_generated = None
            generation_end = None
            rate = 24000
            clock = time.perf_counter()
            def source():
                nonlocal first_generated, generation_end
                for samples, sample_rate in bundle.voice.iter_audio(plan, threading.Event()):
                    if first_generated is None:
                        first_generated = time.perf_counter() - clock
                    yield samples, sample_rate
                generation_end = time.perf_counter() - clock
            def consume(samples, sample_rate):
                nonlocal first, previous_end, rate
                now = time.perf_counter()
                if first is None:
                    first = now - clock
                if previous_end is not None:
                    gaps.append(max(0.0, now - previous_end))
                rate = sample_rate
                chunks.append(samples)
                time.sleep(len(samples) / sample_rate)
                previous_end = time.perf_counter()
            overlap_audio(source(), consume, threading.Event(), prebuffer=args.prebuffer)
            audio = np.concatenate(chunks)
            path = args.output / f'voice_{turn}.wav'
            sf.write(path, audio, rate)
            duration = len(audio) / rate
            samples16 = np.interp(np.arange(round(len(audio)*16000/rate))*rate/16000,
                                  np.arange(len(audio)), audio).astype(np.float32)
            tick = time.perf_counter()
            features, length = bundle.features(samples16)
            features_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            decoded = bundle.listener.semantic_decode(features, length, language='English')
            input_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            mapped = mx.array([remap_token_ids(decoded.token_ids, bundle.listener.tokenizer, bundle.thinker.tokenizer)])
            prefix = bundle.thinker.build_live_prefix(bundle.thinker.embed(mapped), listener_unit_ids=mapped)
            reply = bundle.thinker.generate_from_embeddings(prefix.embeddings, ngram_ids=prefix.ngram_ids,
                       max_tokens=24, compute_response_hidden=False)
            thinker_seconds = time.perf_counter() - tick
            row = {'turn': turn, 'first_generated_seconds': first_generated, 'first_playback_seconds': first,
                   'generation_seconds': generation_end, 'audio_seconds': duration,
                   'synthesis_realtime_factor': generation_end/duration,
                   'max_playback_gap_seconds': max(gaps, default=0), 'total_playback_gap_seconds': sum(gaps),
                   'chunks': len(chunks), 'frontend_seconds': features_seconds, 'listener_seconds': input_seconds,
                   'listener_text': decoded.text, 'word_error_rate': wer(phrase, decoded.text),
                   'thinker_seconds': thinker_seconds, 'reply': reply.text,
                   'audio_finite_nonzero': bool(np.isfinite(audio).all() and np.max(np.abs(audio)) > 1e-5),
                   'audio_file': str(path)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    return {'quant_bits': args.quant_bits, 'quantized_linear_layers': bundle.voice.optimization['quantized_linear_layers'],
            'prebuffer_chunks': args.prebuffer, 'loading_seconds': loading, 'rows': rows,
            'coreml_emotion_loaded': bundle.coreml_emotion is not None,
            'limits': 'Two fixed phrases, synthetic playback timing and ASR loopback, capped 24-token Thinker probe. Not live acoustic echo/microphone calibration.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--quant-bits', type=int, choices=(0, 8), default=0)
    p.add_argument('--prebuffer', type=int, choices=(1, 2), default=2)
    p.add_argument('--phrase', default='Kira is ready. We can begin our story now.')
    p.add_argument('--style', default='Speak clearly and naturally.')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    devices = sd.query_devices()
    sd.check_input_settings(device=0, channels=1, samplerate=16000, dtype='float32')
    sd.check_output_settings(device=1, channels=1, samplerate=24000, dtype='float32')
    with ThreadPoolExecutor(max_workers=1) as executor:
        report = executor.submit(run, args).result()
    report['audio_devices'] = [dict(d) for d in devices]
    report['hardware_formats_supported'] = True
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
