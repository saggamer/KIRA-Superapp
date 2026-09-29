"""Exercise the real native session, with a timed simulated speaker and isolated memory."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
import numpy as np
import sounddevice as sd
import soundfile as sf
from kira_live.native_session import KiraNativeModelBundle, KiraNativeLiveSession
from kira_live.workflow_memory import PersistentWorkflowMemory
from scripts.verify_kira_live_integrated_voice import wer


def main():
    output = ROOT/'training_runs/kira_native_voice_latency_2026_09_29'
    output.mkdir(exist_ok=False)
    audio, rate = sf.read(ROOT/'training_runs/kira_voice_latency_2026_09_29_int8/voice_0.wav', dtype='float32')
    samples = np.interp(np.arange(round(len(audio)*16000/rate))*rate/16000,
                        np.arange(len(audio)), audio).astype(np.float32)
    played, events = [], []
    class Speaker:
        latency = .01
        def __init__(self, **kwargs): self.rate = kwargs['samplerate']
        def start(self): pass
        def write(self, values):
            played.append(values[:, 0].copy())
            time.sleep(len(values)/self.rate)
            return False
        def stop(self): pass
        def abort(self): pass
        def close(self): pass
    with tempfile.TemporaryDirectory(dir='/private/tmp') as scratch, ThreadPoolExecutor(max_workers=1) as worker:
        started = time.perf_counter()
        bundle = worker.submit(KiraNativeModelBundle).result()
        load_seconds = time.perf_counter()-started
        with patch('kira_live.native_session.PersistentWorkflowMemory',
                   lambda _: PersistentWorkflowMemory(Path(scratch)/'memory.sqlite3')), patch.object(sd, 'OutputStream', Speaker):
            session = KiraNativeLiveSession(chat_id='latency-fixture', bundle=bundle,
                        on_event=lambda kind, payload: events.append({'kind':kind, **payload}))
            session._running = True
            session.control.start()
            ticket = session.duplex.begin_generation()
            tick = time.perf_counter()
            session._inference.submit(session._run_utterance, samples, {}, ticket, False, tick).result(timeout=120)
            session.stop()
        texts = [event['text'] for event in events if event['kind']=='RESPONSE_FINISHED']
        failures = [event for event in events if event['kind']=='LIVE_ERROR']
        if failures or not played or len(texts)!=1:
            raise RuntimeError(f'Native voice session failed: {failures}')
        voice_audio = np.concatenate(played)
        sf.write(output/'native_reply.wav', voice_audio, 24000)
        def decode_reply():
            with mx.stream(bundle.mlx_cpu_stream), mx.stream(bundle.mlx_stream):
                resampled = np.interp(np.arange(round(len(voice_audio)*16000/24000))*1.5,
                                     np.arange(len(voice_audio)), voice_audio).astype(np.float32)
                features, length = bundle.features(resampled)
                return bundle.listener.semantic_decode(features, length, language='English').text
        recovered = worker.submit(decode_reply).result()
        score = wer(texts[0], recovered)
        report = {'load_seconds':load_seconds, 'events':events, 'reply':texts[0],
                  'listener_loopback':recovered, 'word_error_rate':score,
                  'voice_optimization':bundle.voice.optimization,
                  'completed_without_error':not failures,
                  'accepted':score<=.2 and bool(np.isfinite(voice_audio).all()),
                  'limits':'One real native turn and CoreML emotion inference, isolated memory, timed simulated speaker. No microphone recording or acoustic echo test.'}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        if not report['accepted']:
            raise SystemExit(2)


if __name__=='__main__': main()
