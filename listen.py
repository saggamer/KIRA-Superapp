"""Persistent, fully local microphone capture and speech recognition for Kira."""

import importlib.util
import glob
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from collections import deque

import numpy as np


SAMPLE_RATE = 16000
FRAME_SAMPLES = 256
QWEN_STT_MODEL = "Qwen/Qwen3-ASR-0.6B"
MLX_STT_MODEL = "mlx-community/whisper-small.en-mlx"
FASTER_STT_MODEL = "Systran/faster-distil-whisper-small.en"
DEFAULT_STT_MODEL = os.environ.get("KIRA_STT_MODEL", QWEN_STT_MODEL).strip() or QWEN_STT_MODEL
DEFAULT_ENDPOINT_SECONDS = 0.65
DEFAULT_MAX_TURN_SECONDS = 25.0
DEFAULT_PRE_ROLL_SECONDS = 0.4


class LocalSTTUnavailable(RuntimeError):
    """Raised when no supported local speech-to-text runtime is installed."""


def _module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _cached_huggingface_model(repo_id):
    """Return a complete local HF snapshot without triggering a download."""
    cache_root = os.environ.get(
        "HF_HUB_CACHE",
        os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"),
    )
    model_root = os.path.join(cache_root, "models--" + repo_id.replace("/", "--"), "snapshots")
    for snapshot in sorted(glob.glob(os.path.join(model_root, "*")), reverse=True):
        if os.path.isfile(os.path.join(snapshot, "config.json")):
            return snapshot
    return ""


class LocalSTT:
    """Lazy local STT adapter in deterministic preference order."""

    def __init__(self, model=DEFAULT_STT_MODEL, allow_cloud=False):
        self.model = str(model or DEFAULT_STT_MODEL)
        self.allow_cloud = bool(allow_cloud)
        self._lock = threading.Lock()
        self._engine = None
        self._model_loaded = False
        self._load_error = ""
        self.runtime_model = self.model
        self._whisper_cli = shutil.which("whisper-cli")
        self._whisper_cli_model = os.environ.get("KIRA_WHISPER_CLI_MODEL", "").strip()
        requested = os.environ.get("KIRA_STT_PROVIDER", "auto").strip().lower() or "auto"
        qwen_cached = _cached_huggingface_model(self.model if "qwen3-asr" in self.model.lower() else QWEN_STT_MODEL)

        if requested in {"qwen", "qwen3", "qwen3_asr"}:
            self.provider = "qwen3_asr"
        elif requested in {"mlx", "mlx_whisper"}:
            self.provider = "mlx_whisper"
        elif requested in {"faster", "faster_whisper"}:
            self.provider = "faster_whisper"
        elif requested in {"whisper_cli", "cli"}:
            self.provider = "whisper_cli"
        elif _module_available("qwen_asr") and qwen_cached:
            self.provider = "qwen3_asr"
        elif _module_available("mlx_whisper"):
            self.provider = "mlx_whisper"
        elif _module_available("faster_whisper"):
            self.provider = "faster_whisper"
        elif self._whisper_cli:
            self.provider = "whisper_cli"
        else:
            self.provider = ""
        if self.provider == "qwen3_asr":
            self.runtime_model = qwen_cached or self.model or QWEN_STT_MODEL
        elif self.provider == "mlx_whisper" and self.model == DEFAULT_STT_MODEL:
            self.runtime_model = MLX_STT_MODEL
        elif self.provider == "faster_whisper" and self.model == DEFAULT_STT_MODEL:
            self.runtime_model = FASTER_STT_MODEL

    @property
    def ready(self):
        if self.provider == "qwen3_asr":
            return bool(_module_available("qwen_asr") and _cached_huggingface_model(QWEN_STT_MODEL))
        if self.provider == "mlx_whisper":
            return _module_available("mlx_whisper")
        if self.provider == "faster_whisper":
            return _module_available("faster_whisper")
        if self.provider == "whisper_cli":
            return bool(self._whisper_cli_model and os.path.isfile(self._whisper_cli_model))
        return False

    @property
    def error(self):
        if self._load_error:
            return self._load_error
        if self.provider == "qwen3_asr" and not self.ready:
            return (
                "Qwen3-ASR 0.6B is selected but not installed locally. Run "
                "scripts/setup_live_voice.py once while online, or set "
                "KIRA_STT_PROVIDER=mlx_whisper to use the lighter Apple fallback."
            )
        if self.provider == "whisper_cli" and not self.ready:
            return (
                "whisper-cli is installed, but KIRA_WHISPER_CLI_MODEL must point to a "
                "local whisper.cpp model file."
            )
        if not self.provider:
            return (
                "No local STT runtime is installed. Run scripts/setup_live_voice.py, "
                "install mlx-whisper/faster-whisper, or place whisper-cli on PATH."
            )
        return ""

    def status(self):
        return {
            "ready": self.ready,
            "provider": self.provider or "unavailable",
            "model": self.model,
            "runtime_model": self.runtime_model,
            "model_loaded": self._engine is not None or self._model_loaded,
            "allow_cloud": self.allow_cloud,
            "cloud_enabled": False,
            "error": self.error,
            "adapter_order": ["qwen3_asr", "mlx_whisper", "faster_whisper", "whisper_cli"],
            "license": (
                "apache-2.0"
                if self.provider == "qwen3_asr"
                else ("MIT/permissive fallback" if self.provider else "unavailable")
            ),
        }

    def transcribe(self, samples, sample_rate=SAMPLE_RATE):
        if not self.ready:
            raise LocalSTTUnavailable(self.error)

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Local STT requires {SAMPLE_RATE} Hz audio, got {sample_rate} Hz.")
        if not audio.size:
            return "", {
                "provider": self.provider,
                "model": self.model,
                "runtime_model": self.runtime_model,
            }

        started = time.monotonic()
        if self.provider == "qwen3_asr":
            text = self._transcribe_qwen(audio)
        elif self.provider == "mlx_whisper":
            text = self._transcribe_mlx(audio)
        elif self.provider == "faster_whisper":
            text = self._transcribe_faster(audio)
        else:
            text = self._transcribe_cli(audio)

        return text.strip(), {
            "provider": self.provider,
            "model": self.model,
            "runtime_model": self.runtime_model,
            "processing_seconds": round(time.monotonic() - started, 3),
        }

    def _transcribe_mlx(self, audio):
        import mlx_whisper

        result = mlx_whisper.transcribe(audio, path_or_hf_repo=self.runtime_model)
        self._model_loaded = True
        if isinstance(result, dict):
            return str(result.get("text", ""))
        return str(result or "")

    def _transcribe_qwen(self, audio):
        with self._lock:
            if self._engine is None:
                try:
                    import torch
                    from qwen_asr import Qwen3ASRModel

                    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
                    self._engine = Qwen3ASRModel.from_pretrained(
                        self.runtime_model,
                        dtype=torch.float16 if has_mps else torch.float32,
                        device_map="mps" if has_mps else "cpu",
                        max_inference_batch_size=1,
                        max_new_tokens=256,
                        local_files_only=True,
                    )
                except Exception as exc:
                    self._load_error = f"Unable to load local Qwen3-ASR 0.6B: {exc}"
                    raise LocalSTTUnavailable(self._load_error) from exc

        with tempfile.TemporaryDirectory(prefix="kira_qwen_stt_") as temp_dir:
            wav_path = os.path.join(temp_dir, "turn.wav")
            self._write_wav(wav_path, audio)
            results = self._engine.transcribe(audio=wav_path, language=None)
        self._model_loaded = True
        if isinstance(results, (list, tuple)) and results:
            result = results[0]
        else:
            result = results
        if isinstance(result, dict):
            return str(result.get("text", ""))
        return str(getattr(result, "text", result) or "")

    def _transcribe_faster(self, audio):
        with self._lock:
            if self._engine is None:
                try:
                    from faster_whisper import WhisperModel

                    device = os.environ.get("KIRA_FASTER_WHISPER_DEVICE", "auto").strip() or "auto"
                    compute_type = os.environ.get("KIRA_FASTER_WHISPER_COMPUTE", "int8").strip() or "int8"
                    self._engine = WhisperModel(
                        self.runtime_model,
                        device=device,
                        compute_type=compute_type,
                    )
                except Exception as exc:
                    self._load_error = f"Unable to load faster-whisper model {self.model}: {exc}"
                    raise LocalSTTUnavailable(self._load_error) from exc

        segments, _info = self._engine.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip())

    def _transcribe_cli(self, audio):
        with tempfile.TemporaryDirectory(prefix="kira_stt_") as temp_dir:
            wav_path = os.path.join(temp_dir, "turn.wav")
            output_base = os.path.join(temp_dir, "transcript")
            self._write_wav(wav_path, audio)

            command = [
                self._whisper_cli,
                "-m",
                self._whisper_cli_model,
                "-f",
                wav_path,
                "-otxt",
                "-of",
                output_base,
                "-nt",
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "unknown whisper-cli error").strip()
                raise RuntimeError(f"whisper-cli failed: {detail}")

            output_path = output_base + ".txt"
            if os.path.isfile(output_path):
                with open(output_path, "r", encoding="utf-8") as output_file:
                    return output_file.read()
            return result.stdout or ""

    @staticmethod
    def _write_wav(path, audio):
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm.tobytes())


class TenVadAdapter:
    """Optional TEN VAD wrapper with the frame shape required by its native API."""

    name = "ten_vad"

    def __init__(self):
        from ten_vad import TenVad

        self._vad = TenVad(FRAME_SAMPLES, 0.5)

    def process(self, frame):
        pcm = (np.clip(frame, -1.0, 1.0) * 32767.0).astype(np.int16, copy=False)
        probability, flag = self._vad.process(pcm)
        return bool(flag), float(probability)

    def close(self):
        destroy = getattr(self._vad, "destroy", None)
        if callable(destroy):
            destroy()

    def status(self):
        return {"provider": self.name, "ready": True, "error": ""}


class AdaptiveRMSVad:
    """Noise-floor-aware VAD used when TEN VAD is unavailable."""

    name = "adaptive_rms"

    def __init__(self, initial_noise_floor=0.004, minimum_threshold=0.009, ratio=2.8):
        self.noise_floor = max(float(initial_noise_floor), 0.0001)
        self.minimum_threshold = float(minimum_threshold)
        self.ratio = float(ratio)
        self.threshold = max(self.minimum_threshold, self.noise_floor * self.ratio)

    def process(self, frame):
        audio = np.asarray(frame, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) if audio.size else 0.0
        self.threshold = min(0.15, max(self.minimum_threshold, self.noise_floor * self.ratio))
        speech = rms >= self.threshold
        if not speech:
            adaptation = 0.025 if rms < self.noise_floor * 2.0 else 0.005
            self.noise_floor = (1.0 - adaptation) * self.noise_floor + adaptation * rms
        probability = min(1.0, rms / max(self.threshold, 1e-6))
        return speech, probability

    def close(self):
        return None

    def status(self):
        return {
            "provider": self.name,
            "ready": True,
            "noise_floor": round(self.noise_floor, 6),
            "threshold": round(self.threshold, 6),
            "error": "",
        }


def create_vad(prefer_ten=True):
    if prefer_ten:
        try:
            return TenVadAdapter()
        except Exception:
            pass
    return AdaptiveRMSVad()


class LocalVoiceSession:
    """Persistent 16 kHz mono capture with local VAD, endpointing, and STT."""

    def __init__(
        self,
        stt=None,
        vad=None,
        on_transcript=None,
        on_event=None,
        on_audio_level=None,
        on_error=None,
        on_speech_start=None,
        stream_factory=None,
        endpoint_seconds=DEFAULT_ENDPOINT_SECONDS,
        max_turn_seconds=DEFAULT_MAX_TURN_SECONDS,
        pre_roll_seconds=DEFAULT_PRE_ROLL_SECONDS,
        on_partial=None,
        partial_interval_seconds=1.0,
    ):
        self.stt = stt or LocalSTT()
        self.vad = vad or create_vad(prefer_ten=True)
        self.on_transcript = on_transcript
        self.on_partial = on_partial
        self.partial_interval_samples = max(SAMPLE_RATE, int(partial_interval_seconds * SAMPLE_RATE))
        self._partial_queue = queue.Queue(maxsize=1)
        self._partial_samples = 0
        self._turn_id = 0
        self.on_event = on_event
        self.on_audio_level = on_audio_level
        self.on_error = on_error
        self.on_speech_start = on_speech_start
        self.stream_factory = stream_factory
        self.endpoint_frames = max(1, int(math.ceil(endpoint_seconds * SAMPLE_RATE / FRAME_SAMPLES)))
        self.max_turn_samples = max(FRAME_SAMPLES, int(max_turn_seconds * SAMPLE_RATE))
        self.pre_roll_frames = max(1, int(math.ceil(pre_roll_seconds * SAMPLE_RATE / FRAME_SAMPLES)))

        self._audio_queue = queue.Queue(maxsize=256)
        self._turn_queue = queue.Queue(maxsize=3)
        self._pre_roll = deque(maxlen=self.pre_roll_frames)
        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()
        self._stream = None
        self._capture_thread = None
        self._stt_thread = None
        self._running = False
        self._paused = False
        self._turn_chunks = []
        self._turn_samples = 0
        self._voiced_frames = 0
        self._silence_frames = 0
        self._speech_streak = 0
        self._in_turn = False
        self._dropped_frames = 0
        self._last_level_emit = 0.0
        self._last_error = ""

    @property
    def running(self):
        with self._state_lock:
            return self._running

    def status(self):
        with self._state_lock:
            session = {
                "running": self._running,
                "paused": self._paused,
                "state": "hearing" if self._in_turn else ("paused" if self._paused else "listening"),
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "frame_samples": FRAME_SAMPLES,
                "endpoint_ms": round(self.endpoint_frames * FRAME_SAMPLES * 1000 / SAMPLE_RATE),
                "max_turn_seconds": self.max_turn_samples / SAMPLE_RATE,
                "pre_roll_ms": round(self.pre_roll_frames * FRAME_SAMPLES * 1000 / SAMPLE_RATE),
                "dropped_frames": self._dropped_frames,
                "last_error": self._last_error,
            }
        session["stt"] = self.stt.status()
        session["vad"] = self.vad.status()
        return session

    def start(self):
        with self._state_lock:
            if self._running:
                return self.status()
            self._stop_event.clear()
            self._paused = False
            self._reset_turn(clear_pre_roll=True)
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name="kira-voice-capture",
                daemon=True,
            )
            self._stt_thread = threading.Thread(
                target=self._stt_loop,
                name="kira-local-stt",
                daemon=True,
            )
            self._capture_thread.start()
            self._stt_thread.start()

            try:
                factory = self.stream_factory or self._default_stream_factory
                self._stream = factory(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=FRAME_SAMPLES,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._running = True
            except Exception as exc:
                self._last_error = f"Unable to start microphone capture: {exc}"
                self._stop_event.set()
                self._queue_sentinel()
                self._emit_error(self._last_error, {"stage": "capture_start"})
                return self.status()

        self._emit_event("VOICE_SESSION_STARTED", self.status())
        if not self.stt.ready:
            self._emit_error(self.stt.error, {"stage": "stt_dependency", "stt": self.stt.status()})
        return self.status()

    def stop(self):
        with self._state_lock:
            was_running = self._running
            self._running = False
            self._stop_event.set()
            stream = self._stream
            self._stream = None

        if stream is not None:
            for method_name in ("stop", "close"):
                try:
                    method = getattr(stream, method_name, None)
                    if callable(method):
                        method()
                except Exception:
                    pass

        self._queue_sentinel()
        current = threading.current_thread()
        for worker in (self._capture_thread, self._stt_thread):
            if worker is not None and worker is not current and worker.is_alive():
                worker.join(timeout=1.5)

        with self._state_lock:
            self._reset_turn(clear_pre_roll=True)
            self._paused = False
        try:
            self.vad.close()
        except Exception:
            pass
        if was_running:
            self._emit_event("VOICE_SESSION_STOPPED", self.status())
        return self.status()

    def set_paused(self, paused, flush=True):
        with self._state_lock:
            self._paused = bool(paused)
            if flush:
                self._reset_turn(clear_pre_roll=True)
        self._emit_event("CAPTURE_PAUSED" if paused else "CAPTURE_RESUMED", {"paused": bool(paused)})

    def feed_audio(self, samples):
        """Inject audio for tests or alternate capture transports."""
        audio = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
        if not audio.size:
            return False
        try:
            self._audio_queue.put_nowait(audio)
            return True
        except queue.Full:
            with self._state_lock:
                self._dropped_frames += int(math.ceil(audio.size / FRAME_SAMPLES))
            return False

    def _default_stream_factory(self, **kwargs):
        import sounddevice as sd

        return sd.InputStream(**kwargs)

    def _audio_callback(self, indata, frames, _time_info, status):
        if status:
            self._emit_event("CAPTURE_STATUS", {"status": str(status)})
        if frames <= 0 or self._stop_event.is_set():
            return
        self.feed_audio(indata[:, 0])

    def _capture_loop(self):
        pending = np.empty(0, dtype=np.float32)
        while not self._stop_event.is_set():
            try:
                item = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            pending = np.concatenate((pending, np.asarray(item, dtype=np.float32).reshape(-1)))
            while pending.size >= FRAME_SAMPLES:
                frame = pending[:FRAME_SAMPLES]
                pending = pending[FRAME_SAMPLES:]
                self._process_frame(frame)

    def _process_frame(self, frame):
        with self._state_lock:
            if self._paused or self._stop_event.is_set():
                return

        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        try:
            is_speech, probability = self.vad.process(frame)
        except Exception as exc:
            self._last_error = f"VAD failed: {exc}"
            self._emit_error(self._last_error, {"stage": "vad"})
            self.vad = AdaptiveRMSVad()
            is_speech, probability = self.vad.process(frame)

        now = time.monotonic()
        if now - self._last_level_emit >= 0.05:
            self._last_level_emit = now
            level = min(1.0, rms / 0.12)
            payload = {
                "level": round(level, 4),
                "rms": round(rms, 6),
                "speech": bool(is_speech),
                "vad_probability": round(float(probability), 4),
            }
            self._emit_callback(self.on_audio_level, payload)

        self._pre_roll.append(frame.copy())
        if not self._in_turn:
            self._speech_streak = self._speech_streak + 1 if is_speech else 0
            if self._speech_streak < 2:
                return
            self._in_turn = True
            self._turn_id += 1
            self._turn_chunks = list(self._pre_roll)
            self._turn_samples = sum(chunk.size for chunk in self._turn_chunks)
            self._voiced_frames = self._speech_streak
            self._silence_frames = 0
            payload = {"level": min(1.0, rms / 0.12), "vad_probability": float(probability)}
            self._emit_event("SPEECH_START", payload)
            self._emit_callback(self.on_speech_start, payload)
            return

        self._turn_chunks.append(frame.copy())
        self._turn_samples += frame.size
        if is_speech:
            self._voiced_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        if self._turn_samples >= self.max_turn_samples:
            self._finish_turn("max_duration")
        elif self._silence_frames >= self.endpoint_frames:
            self._finish_turn("silence")
        elif self.on_partial and self._turn_samples - self._partial_samples >= self.partial_interval_samples:
            self._partial_samples = self._turn_samples
            # Keep only the newest preview; final turns use the higher-priority queue.
            try:
                self._partial_queue.get_nowait()
            except queue.Empty:
                pass
            self._partial_queue.put_nowait((np.concatenate(self._turn_chunks), {"turn_id": self._turn_id}))

    def _finish_turn(self, reason):
        chunks = self._turn_chunks
        voiced_frames = self._voiced_frames
        sample_count = self._turn_samples
        self._reset_turn(clear_pre_roll=True)
        if not chunks or voiced_frames * FRAME_SAMPLES < int(0.16 * SAMPLE_RATE):
            self._emit_event("STT_EMPTY", {"reason": "speech_too_short"})
            return

        audio = np.concatenate(chunks).astype(np.float32, copy=False)
        payload = {
            "reason": reason,
            "duration_seconds": round(sample_count / SAMPLE_RATE, 3),
            "voiced_seconds": round(voiced_frames * FRAME_SAMPLES / SAMPLE_RATE, 3),
        }
        self._emit_event("SPEECH_END", payload)
        try:
            self._turn_queue.put_nowait((audio, payload))
        except queue.Full:
            self._emit_error("Local STT queue is full; the completed turn was dropped.", {"stage": "stt_queue"})

    def _stt_loop(self):
        while not self._stop_event.is_set():
            partial = False
            try:
                item = self._turn_queue.get(timeout=0.1)
            except queue.Empty:
                try:
                    item = self._partial_queue.get_nowait()
                    partial = True
                except queue.Empty:
                    continue
            if item is None:
                break
            audio, metadata = item
            if partial and (not self._in_turn or metadata["turn_id"] != self._turn_id):
                continue
            self._emit_event("STT_PROCESSING", metadata)
            try:
                text, stt_metadata = self.stt.transcribe(audio, SAMPLE_RATE)
                metadata = dict(metadata)
                metadata.update(stt_metadata)
                if self._stop_event.is_set():
                    continue
                if text and partial:
                    if self._in_turn and metadata["turn_id"] == self._turn_id:
                        self._emit_callback(self.on_partial, text, metadata)
                elif text:
                    self._emit_callback(self.on_transcript, text, metadata)
                else:
                    self._emit_event("STT_EMPTY", metadata)
            except Exception as exc:
                if partial:
                    continue
                self._last_error = str(exc)
                self._emit_error(self._last_error, {"stage": "stt", "stt": self.stt.status()})

    def _reset_turn(self, clear_pre_roll=False):
        self._partial_samples = 0
        self._turn_chunks = []
        self._turn_samples = 0
        self._voiced_frames = 0
        self._silence_frames = 0
        self._speech_streak = 0
        self._in_turn = False
        if clear_pre_roll:
            self._pre_roll.clear()

    def _queue_sentinel(self):
        for target_queue in (self._audio_queue, self._turn_queue):
            try:
                target_queue.put_nowait(None)
            except queue.Full:
                pass

    def _emit_event(self, kind, payload=None):
        self._emit_callback(self.on_event, str(kind), payload or {})

    def _emit_error(self, message, payload=None):
        details = dict(payload or {})
        details["error"] = str(message)
        self._emit_event("STT_ERROR", details)
        self._emit_callback(self.on_error, str(message), details)

    @staticmethod
    def _emit_callback(callback, *args):
        if not callable(callback):
            return
        try:
            callback(*args)
        except Exception:
            pass


def get_local_stt_status():
    allow_cloud = os.environ.get("KIRA_ALLOW_CLOUD_STT", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    return LocalSTT(allow_cloud=allow_cloud).status()


def kira_listen(timeout=30.0):
    """Compatibility one-turn listener backed only by local STT."""
    result = {"text": None}
    completed = threading.Event()
    stt = LocalSTT()
    if not stt.ready:
        print(f"[Kira local STT unavailable] {stt.error}")
        return None

    def on_transcript(text, _metadata):
        result["text"] = text
        completed.set()

    def on_error(message, _metadata):
        print(f"[Kira local STT error] {message}")
        completed.set()

    session = LocalVoiceSession(stt=stt, on_transcript=on_transcript, on_error=on_error)
    status = session.start()
    if not status.get("running"):
        session.stop()
        return None
    try:
        completed.wait(timeout=max(0.0, float(timeout)))
        return result["text"]
    finally:
        session.stop()


if __name__ == "__main__":
    transcript = kira_listen()
    if transcript:
        print(f"Heard: {transcript!r}")
