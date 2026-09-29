import importlib
import importlib.util
import json
import os
import re
import shutil
import threading
import time
from collections import deque
from kira_live.public_guard import tools_disallowed


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_TEMP_DIR = os.path.join(BASE_DIR, "kira_voice_tmp")
try:
    os.makedirs(VOICE_TEMP_DIR, exist_ok=True)
    os.environ["TMPDIR"] = VOICE_TEMP_DIR
    os.environ["TEMP"] = VOICE_TEMP_DIR
    os.environ["TMP"] = VOICE_TEMP_DIR
except Exception:
    pass

_voice_module = None
_listen_module = None
VOICE_READY = False
LISTEN_READY = False
STTS_TERMINAL_LOG_PATH = os.path.join(VOICE_TEMP_DIR, "stts_terminal.log")
STTS_EVENT_LOG_PATH = os.path.join(VOICE_TEMP_DIR, "stts_events.jsonl")
STTS_SILENT_TEST = os.environ.get("KIRA_STTS_SILENT_TEST", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
_stts_terminal_started = False
_stts_log_lock = threading.Lock()
_voice_load_lock = threading.Lock()
_voice_load_error = ""
_voice_asset_paths = {"model": "", "voices": ""}
_voice_session_lock = threading.Lock()

DEFAULT_STT_PROVIDER = os.environ.get("KIRA_STT_PROVIDER", "local_auto").strip().lower() or "local_auto"
DEFAULT_TURN_DETECTOR = os.environ.get("KIRA_TURN_DETECTOR", "local_vad").strip().lower() or "local_vad"
DEFAULT_TTS_PROVIDER = os.environ.get("KIRA_TTS_PROVIDER", "kokoro_onnx").strip().lower() or "kokoro_onnx"
TEN_RUNTIME_URL = os.environ.get("KIRA_TEN_RUNTIME_URL", "").strip()
QWEN_STT_MODEL = os.environ.get("KIRA_QWEN_STT_MODEL", "").strip()
QWEN_TURN_MODEL = os.environ.get("KIRA_QWEN_TURN_MODEL", QWEN_STT_MODEL).strip()

_voice_session = {
    "active": False,
    "live_mode": False,
    "state": "idle",
    "stt_provider": DEFAULT_STT_PROVIDER,
    "turn_detector": DEFAULT_TURN_DETECTOR,
    "tts_provider": DEFAULT_TTS_PROVIDER,
    "last_partial": "",
    "last_transcript": "",
    "last_source": "",
    "last_confidence": 0.0,
    "speaking_amplitude": 0.0,
    "speaking_envelope": 0.0,
    "last_event_at": 0.0,
}


def _module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _qwen_status():
    configured = bool(QWEN_STT_MODEL or QWEN_TURN_MODEL)
    return {
        "configured": configured,
        "ready": configured and (_module_available("transformers") or _module_available("mlx_lm")),
        "stt_model": QWEN_STT_MODEL,
        "turn_model": QWEN_TURN_MODEL,
        "note": (
            "Qwen audio/turn detection is configured as an optional local adapter."
            if configured
            else "Set KIRA_QWEN_STT_MODEL or KIRA_QWEN_TURN_MODEL to enable a local Qwen audio adapter."
        ),
    }


def _ten_status():
    return {
        "configured": bool(TEN_RUNTIME_URL),
        "local_vad_installed": _module_available("ten_vad"),
        "runtime_url": TEN_RUNTIME_URL,
        "note": (
            "TEN runtime URL is configured; KIRA can route realtime session events through that bridge."
            if TEN_RUNTIME_URL
            else "The local TEN VAD adapter is used automatically when ten-vad is installed."
        ),
    }


def _update_voice_session(**changes):
    with _voice_session_lock:
        _voice_session.update(changes)
        _voice_session["last_event_at"] = time.time()
        return dict(_voice_session)


def _voice_stack_status():
    with _voice_session_lock:
        session = dict(_voice_session)

    qwen = _qwen_status()
    ten = _ten_status()
    endpointing = session.get("turn_detector") or DEFAULT_TURN_DETECTOR
    provider = session.get("stt_provider") or DEFAULT_STT_PROVIDER

    if provider == "qwen" and not qwen.get("ready"):
        provider_state = "qwen_config_needed"
    elif provider == "ten" and not (ten.get("configured") or ten.get("local_vad_installed")):
        provider_state = "ten_runtime_needed"
    elif provider in {"browser", "browser_webspeech"}:
        provider_state = "browser_ready"
    elif provider in {"listen", "python", "local", "local_auto", "native", "qwen3_asr"}:
        provider_state = "local_lazy"
    else:
        provider_state = "configured"

    return {
        "session": session,
        "stt_provider": provider,
        "turn_detector": endpointing,
        "tts_provider": session.get("tts_provider") or DEFAULT_TTS_PROVIDER,
        "provider_state": provider_state,
        "qwen": qwen,
        "ten": ten,
        "recommended_local_tts": "kokoro_onnx",
        "recommended_turn_detector": "ten_vad with adaptive_rms fallback",
    }


def _stts_label():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _stts_text(value, limit=6000):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        return text[:limit] + "... [truncated]"
    return text


def _write_stts_event(kind, text="", extra=None):
    event = {
        "time": time.time(),
        "label": _stts_label(),
        "kind": str(kind or "EVENT").upper(),
        "text": _stts_text(text),
        "extra": extra or {},
    }

    try:
        os.makedirs(VOICE_TEMP_DIR, exist_ok=True)
        with _stts_log_lock:
            with open(STTS_EVENT_LOG_PATH, "a", encoding="utf-8") as event_file:
                event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            with open(STTS_TERMINAL_LOG_PATH, "a", encoding="utf-8") as terminal_file:
                terminal_file.write(f"[{event['label']}] {event['kind']}: {event['text']}\n")

        if os.environ.get("KIRA_STTS_PRINT", "").strip().lower() in {"1", "true", "yes", "on"}:
            print(f"[KIRA STTS] {event['kind']}: {event['text']}")
    except Exception as exc:
        print(f"[!] STTS terminal log failed: {exc}")
    return event


def ensure_stts_terminal():
    global _stts_terminal_started
    if _stts_terminal_started:
        return True
    try:
        os.makedirs(VOICE_TEMP_DIR, exist_ok=True)
        _stts_terminal_started = True
        _write_stts_event(
            "SESSION",
            "Local STT/TTS event route online.",
            {"route": "local_voice", "log": STTS_TERMINAL_LOG_PATH},
        )
        return True
    except Exception as exc:
        print(f"[!] STTS terminal unavailable: {exc}")
        return False


def set_stts_silent_test(enabled=True):
    global STTS_SILENT_TEST
    STTS_SILENT_TEST = bool(enabled)
    ensure_stts_terminal()
    _write_stts_event(
        "CONFIG",
        "Silent STT/TTS test mode " + ("enabled." if STTS_SILENT_TEST else "disabled."),
        {"silent_test": STTS_SILENT_TEST},
    )
    return STTS_SILENT_TEST


def _kokoro_search_roots():
    roots = [BASE_DIR, os.path.join(os.path.expanduser("~"), "Desktop", "Kira_OS")]
    result = []
    for root in roots:
        normalized = os.path.abspath(os.path.expanduser(root))
        if normalized not in result and os.path.isdir(normalized):
            result.append(normalized)
    return result


def _find_kokoro_assets():
    model_name = "kokoro-v1.0.onnx"
    voices_name = "voices-v1.0.bin"
    roots = _kokoro_search_roots()

    candidates = []
    for root in roots:
        for suffix in ("", "models", "voice", "assets"):
            candidate = os.path.join(root, suffix) if suffix else root
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        model_path = os.path.join(candidate, model_name)
        voices_path = os.path.join(candidate, voices_name)
        if os.path.isfile(model_path) and os.path.isfile(voices_path):
            return model_path, voices_path

    found_models = []
    found_voices = []
    ignored = {".git", "__pycache__", "node_modules", ".venv", "venv", "kira_voice_tmp"}
    for root in roots:
        for current, directories, files in os.walk(root):
            directories[:] = [name for name in directories if name not in ignored and not name.startswith(".")]
            if model_name in files:
                found_models.append(os.path.join(current, model_name))
            if voices_name in files:
                found_voices.append(os.path.join(current, voices_name))
            if model_name in files and voices_name in files:
                return os.path.join(current, model_name), os.path.join(current, voices_name)

    return (
        found_models[0] if found_models else "",
        found_voices[0] if found_voices else "",
    )


def get_voice_engine_status():
    model_path, voices_path = _find_kokoro_assets()
    dependencies = {
        "sounddevice": _module_available("sounddevice"),
        "kokoro_onnx": _module_available("kokoro_onnx"),
        "onnxruntime": _module_available("onnxruntime"),
    }
    assets_ready = bool(model_path and voices_path)
    dependencies_ready = all(dependencies.values())
    error = _voice_load_error
    if not assets_ready:
        missing = []
        if not model_path:
            missing.append("kokoro-v1.0.onnx")
        if not voices_path:
            missing.append("voices-v1.0.bin")
        error = "Missing Kokoro assets: " + ", ".join(missing)
    elif not dependencies_ready:
        error = "Missing Kokoro dependencies: " + ", ".join(
            name for name, available in dependencies.items() if not available
        )
    return {
        "ready": bool(VOICE_READY and _voice_module is not None),
        "loaded": _voice_module is not None,
        "available_to_load": assets_ready and dependencies_ready,
        "assets_ready": assets_ready,
        "model_path": model_path,
        "voices_path": voices_path,
        "search_roots": _kokoro_search_roots(),
        "dependencies": dependencies,
        "error": "" if VOICE_READY else error,
    }


def _stable_espeak_config():
    """Stage eSpeak assets at a path its native loader can parse reliably."""
    import espeakng_loader
    from kokoro_onnx.config import EspeakConfig

    source_library = espeakng_loader.get_library_path()
    source_data = espeakng_loader.get_data_path()
    runtime_dir = os.path.join("/tmp", "kira_espeak_runtime")
    library_path = os.path.join(runtime_dir, os.path.basename(source_library))
    data_path = os.path.join(runtime_dir, "espeak-ng-data")

    os.makedirs(runtime_dir, exist_ok=True)
    if not os.path.isfile(library_path):
        shutil.copy2(source_library, library_path)
    if not os.path.isfile(os.path.join(data_path, "phontab")):
        shutil.copytree(source_data, data_path, dirs_exist_ok=True)
    return EspeakConfig(lib_path=library_path, data_path=data_path)


def ensure_voice_engine():
    global _voice_module, VOICE_READY, _voice_load_error, _voice_asset_paths
    if _voice_module is not None and VOICE_READY:
        return True

    with _voice_load_lock:
        if _voice_module is not None and VOICE_READY:
            return True
        model_path, voices_path = _find_kokoro_assets()
        if not model_path or not voices_path:
            missing = []
            if not model_path:
                missing.append("kokoro-v1.0.onnx")
            if not voices_path:
                missing.append("voices-v1.0.bin")
            _voice_load_error = "Missing Kokoro assets: " + ", ".join(missing)
            _voice_module = None
            VOICE_READY = False
            print(f"[!] Voice engine unavailable: {_voice_load_error}")
            return False

        try:
            import onnxruntime as ort
            import sounddevice as sd
            from kokoro_onnx import Kokoro

            available = ort.get_available_providers()
            preferred = [
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = [provider for provider in preferred if provider in available]
            if not providers:
                raise RuntimeError("ONNX Runtime did not report an execution provider.")
            session = ort.InferenceSession(model_path, providers=providers)
            espeak_config = _stable_espeak_config()
            module = type("LazyKiraVoice", (), {})()
            module.sd = sd
            module.kira_voice = Kokoro.from_session(
                session,
                voices_path,
                espeak_config=espeak_config,
            )
            module.model_path = model_path
            module.voices_path = voices_path
            module.providers = providers
            _voice_module = module
            _voice_asset_paths = {"model": model_path, "voices": voices_path}
            _voice_load_error = ""
            VOICE_READY = True
            print(f"[Voice] Kira voice engine loaded from {os.path.dirname(model_path)}.")
        except Exception as exc:
            _voice_module = None
            VOICE_READY = False
            _voice_load_error = str(exc)
            print(f"[!] Voice engine unavailable: {exc}")
    return VOICE_READY


def ensure_listen_engine():
    global _listen_module, LISTEN_READY
    if _listen_module is not None:
        return LISTEN_READY
    try:
        _listen_module = importlib.import_module("listen")
        LISTEN_READY = bool(
            callable(getattr(_listen_module, "kira_listen", None))
            and getattr(_listen_module, "LocalVoiceSession", None)
        )
    except Exception as exc:
        _listen_module = None
        LISTEN_READY = False
        print(f"[!] listen.py unavailable: {exc}")
    return LISTEN_READY


class _TTSRequest:
    def __init__(self, request_id, generation, text, silent, callback):
        self.request_id = request_id
        self.generation = generation
        self.text = text
        self.silent = silent
        self.callback = callback
        self.cancelled = threading.Event()
        self.completed = threading.Event()
        self.event_lock = threading.RLock()
        self.terminal = False
        self.result = {"ok": True, "queued": True, "generation": generation}


class _SerializedTTS:
    def __init__(self):
        self._condition = threading.Condition()
        self._queue = deque()
        self._worker = None
        self._active = None
        self._active_stream = None
        self._generation = 0
        self._request_id = 0

    def submit(self, text, silent, callback=None, wait=True):
        with self._condition:
            self._request_id += 1
            request = _TTSRequest(
                self._request_id,
                self._generation,
                str(text),
                bool(silent),
                callback,
            )
            self._queue.append(request)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._run,
                    name="kira-serialized-tts",
                    daemon=True,
                )
                self._worker.start()
            self._condition.notify_all()

        if wait:
            request.completed.wait()
            return dict(request.result)
        return dict(request.result)

    def stop(self, reason="barge_in"):
        with self._condition:
            self._generation += 1
            requests = list(self._queue)
            self._queue.clear()
            if self._active is not None:
                requests.insert(0, self._active)
            for request in requests:
                request.cancelled.set()
            stream = self._active_stream
            generation = self._generation
            self._condition.notify_all()

        if stream is not None:
            for method_name in ("abort", "stop"):
                try:
                    method = getattr(stream, method_name, None)
                    if callable(method):
                        method()
                        break
                except Exception:
                    pass
        module = _voice_module
        if module is not None:
            try:
                sd = getattr(module, "sd", None)
                if sd is not None and hasattr(sd, "stop"):
                    sd.stop()
            except Exception:
                pass

        interrupted = 0
        for request in requests:
            if self._finish(
                request,
                "TTS_INTERRUPTED",
                "Voice output interrupted.",
                {"reason": reason, "generation": request.generation},
                ok=False,
            ):
                interrupted += 1
        return {"ok": True, "interrupted": interrupted, "generation": generation}

    def status(self):
        with self._condition:
            active = self._active
            return {
                "speaking": bool(active and not active.terminal),
                "generation": self._generation,
                "active_request": active.request_id if active and not active.terminal else None,
                "queued": len(self._queue),
            }

    def _run(self):
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                request = self._queue.popleft()
                self._active = request

            if self._is_cancelled(request):
                self._finish(
                    request,
                    "TTS_INTERRUPTED",
                    "Voice output interrupted before playback.",
                    {"reason": "stale_generation"},
                    ok=False,
                )
                self._clear_active(request)
                continue

            if not self._emit_nonterminal(
                request,
                "TTS_START",
                request.text,
                {"generation": request.generation, "route": "silent_terminal" if request.silent else "audio"},
            ):
                self._clear_active(request)
                continue
            try:
                if request.silent:
                    self._emit_nonterminal(
                        request,
                        "TTS_OUT",
                        request.text,
                        {"generation": request.generation, "route": "silent_terminal"},
                    )
                else:
                    self._synthesize_and_play(request)

                if self._is_cancelled(request):
                    self._finish(
                        request,
                        "TTS_INTERRUPTED",
                        "Voice output interrupted.",
                        {"reason": "barge_in", "generation": request.generation},
                        ok=False,
                    )
                else:
                    self._finish(
                        request,
                        "TTS_DONE",
                        "Voice output completed.",
                        {"generation": request.generation},
                        ok=True,
                    )
            except Exception as exc:
                if self._is_cancelled(request):
                    self._finish(
                        request,
                        "TTS_INTERRUPTED",
                        "Voice output interrupted.",
                        {"reason": "barge_in", "generation": request.generation},
                        ok=False,
                    )
                else:
                    self._finish(
                        request,
                        "TTS_ERROR",
                        str(exc),
                        {"generation": request.generation},
                        ok=False,
                        error=str(exc),
                    )
            finally:
                self._clear_active(request)

    def _synthesize_and_play(self, request):
        if not ensure_voice_engine():
            status = get_voice_engine_status()
            raise RuntimeError(status.get("error") or "Kokoro voice engine is unavailable.")
        if self._is_cancelled(request):
            return

        samples, sample_rate = _voice_module.kira_voice.create(
            request.text,
            voice="af_sarah",
            speed=1.1,
            lang="en-us",
        )
        if self._is_cancelled(request):
            return

        import numpy as np

        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        elif audio.ndim != 2:
            audio = audio.reshape(-1, 1)
        if not audio.size:
            raise RuntimeError("Kokoro returned no audio samples.")

        channels = int(audio.shape[1])
        chunk_frames = max(256, int(sample_rate * 0.04))
        envelope = 0.0
        stream = _voice_module.sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
        )
        with self._condition:
            self._active_stream = stream
        try:
            stream.start()
            total_frames = audio.shape[0]
            for offset in range(0, total_frames, chunk_frames):
                if self._is_cancelled(request):
                    break
                chunk = audio[offset:offset + chunk_frames]
                amplitude = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
                envelope = envelope * 0.72 + min(1.0, amplitude * 4.0) * 0.28
                stream.write(chunk)
                self._emit_nonterminal(
                    request,
                    "TTS_LEVEL",
                    "",
                    {
                        "amplitude": round(min(1.0, amplitude), 5),
                        "envelope": round(min(1.0, envelope), 5),
                        "progress": round(min(1.0, (offset + len(chunk)) / total_frames), 4),
                        "generation": request.generation,
                    },
                )
        finally:
            with self._condition:
                if self._active_stream is stream:
                    self._active_stream = None
            for method_name in ("stop", "close"):
                try:
                    method = getattr(stream, method_name, None)
                    if callable(method):
                        method()
                except Exception:
                    pass

    def _is_cancelled(self, request):
        with self._condition:
            return request.cancelled.is_set() or request.generation != self._generation or request.terminal

    def _clear_active(self, request):
        with self._condition:
            if self._active is request:
                self._active = None
            self._condition.notify_all()

    def _finish(self, request, kind, text, extra, ok, error=""):
        with request.event_lock:
            with self._condition:
                if request.terminal:
                    return False
                request.terminal = True
                request.result = {
                    "ok": bool(ok),
                    "event": kind,
                    "generation": request.generation,
                    "error": error,
                }
            self._emit(request, kind, text, extra)
            request.completed.set()
            return True

    def _emit_nonterminal(self, request, kind, text, extra):
        with request.event_lock:
            with self._condition:
                if (
                    request.terminal
                    or request.cancelled.is_set()
                    or request.generation != self._generation
                ):
                    return False
            self._emit(request, kind, text, extra)
            return True

    @staticmethod
    def _emit(request, kind, text, extra):
        payload = {"request_id": request.request_id, **dict(extra or {})}
        callback = request.callback
        if callable(callback):
            try:
                callback(kind, text, payload)
                return
            except Exception:
                pass
        _write_stts_event(kind, text, payload)


_tts_controller = _SerializedTTS()


def kira_speak(text, silent=None, callback=None, wait=True):
    if not text:
        return {"ok": False, "error": "No text to speak."}
    if silent is None:
        silent = STTS_SILENT_TEST
    return _tts_controller.submit(text, silent=silent, callback=callback, wait=wait)


def stop_speaking(reason="barge_in"):
    return _tts_controller.stop(reason=reason)


def interrupt_speaking(reason="barge_in"):
    return stop_speaking(reason=reason)


class KiraVoiceMixin:
    def _release_text_worker_for_live(self):
        """Avoid keeping a second full Thinker/Orchestrator resident with Live."""
        if getattr(self, "model_worker_process", None) is None:
            return
        lock = getattr(self, "model_worker_io_lock", None)
        if lock is not None and not lock.acquire(blocking=False):
            raise RuntimeError("A text answer is still running. Try Live after it finishes.")
        try:
            self._stop_model_worker()
            self.active_model = None
            self.active_tokenizer = None
            self.current_brain = None
        finally:
            if lock is not None:
                lock.release()

    def get_kira_live_status(self):
        try:
            from kira_live.readiness import kira_live_readiness

            return kira_live_readiness()
        except Exception as exc:
            return {
                "ok": False,
                "ui_ready": True,
                "conversation_ready": False,
                "state": "error",
                "name": "KIRA Live 1",
                "architecture": "one_model_three_weight_islands",
                "prompt_handoffs": False,
                "private_reasoning_visible": False,
                "checks": [],
                "blockers": [str(exc)],
            }

    def configure_voice_stack(self, stt_provider=None, turn_detector=None, tts_provider=None, live_mode=None):
        changes = {}
        if stt_provider:
            changes["stt_provider"] = str(stt_provider).strip().lower()
        if turn_detector:
            changes["turn_detector"] = str(turn_detector).strip().lower()
        if tts_provider:
            changes["tts_provider"] = str(tts_provider).strip().lower()
        # Retain the field for older UI/API clients, but Live Mode is disabled.
        changes["live_mode"] = False

        session = _update_voice_session(**changes) if changes else _update_voice_session()
        self._record_stts_event(
            "VOICE_STACK",
            "Realtime voice stack configured.",
            {
                "stt_provider": session.get("stt_provider"),
                "turn_detector": session.get("turn_detector"),
                "tts_provider": session.get("tts_provider"),
                "live_mode": session.get("live_mode"),
            },
        )
        return self.get_voice_status()

    def toggle_voice_mode(self, state):
        self.stop_live_voice()
        return "Voice conversations are disabled. Use Dictate to enter a text prompt."

    def toggle_voice(self, state):
        return self.toggle_voice_mode(state)

    def get_voice_status(self):
        ensure_listen_engine()
        stack = _voice_stack_status()
        kokoro = get_voice_engine_status()
        live_session = getattr(self, "_live_voice_session", None)
        if live_session is not None:
            local_session = live_session.status()
        elif LISTEN_READY and callable(getattr(_listen_module, "get_local_stt_status", None)):
            local_session = {
                "running": False,
                "paused": False,
                "state": "idle",
                "stt": _listen_module.get_local_stt_status(),
            }
        else:
            local_session = {
                "running": False,
                "state": "unavailable",
                "stt": {"ready": False, "error": "listen.py is unavailable."},
            }
        return {
            "voice_mode_active": bool(getattr(self, "voice_mode_active", False)),
            "voice_ready": kokoro["ready"],
            "listen_ready": LISTEN_READY,
            "voice_loaded": kokoro["loaded"],
            "listen_loaded": _listen_module is not None,
            "stts_ready": True,
            "stts_silent_test": STTS_SILENT_TEST,
            "stts_route": "silent_terminal" if STTS_SILENT_TEST else "audio",
            "stts_terminal_log": STTS_TERMINAL_LOG_PATH,
            "stts_event_log": STTS_EVENT_LOG_PATH,
            "speech_output": "kokoro_onnx" if kokoro["ready"] else "lazy/offline",
            "python_listen_fallback": "listen.py" if LISTEN_READY else "offline",
            "temp_dir": VOICE_TEMP_DIR,
            "realtime_stack": stack,
            "stt_provider": stack.get("stt_provider"),
            "turn_detector": stack.get("turn_detector"),
            "tts_provider": stack.get("tts_provider"),
            "qwen_ready": stack.get("qwen", {}).get("ready", False),
            "ten_ready": stack.get("ten", {}).get("local_vad_installed", False),
            "kokoro": kokoro,
            "local_voice_session": local_session,
            "local_stt": local_session.get("stt", {}),
            "native_voice_ready": bool(
                LISTEN_READY and local_session.get("stt", {}).get("ready", False)
            ),
            "native_voice_active": bool(local_session.get("running", False)),
            # Compatibility aliases for older bundled UIs.
            "native_live_ready": bool(
                LISTEN_READY and local_session.get("stt", {}).get("ready", False)
            ),
            "native_live_active": bool(local_session.get("running", False)),
            "tts": _tts_controller.status(),
        }

    def set_voice_test_mode(self, enabled=True):
        enabled = set_stts_silent_test(enabled)
        stop_speaking(reason="voice_test_mode_changed")
        return self.get_voice_status() | {
            "status": "silent_test_enabled" if enabled else "audio_enabled"
        }

    def _live_tree_requested(self, text):
        if tools_disallowed(text):
            return False
        if re.search(r'\b(?:search|look up|weather|forecast|latest|right now|current prices)\b|\bfind\b.*\b(?:best|spot|place|camping)\b', str(text or ''), re.I):
            return True
        scoped = bool(re.search(
            r"\b(?:tree|desktop|downloads|folder|files|directory|web|internet|apps|applications|processes|ram|computer|delete|move|schedule|send|book|pdf|slides|presentation|document|spreadsheet|code|project|terminal|shell|python|clipboard|archive|zip|calendar|reminder|blender|mcp)\b|(?:^|\s)(?:~/|/Users/|/tmp/)|^(?:open|close|launch|quit)\b",
            str(text or ""), re.I,
        ))
        return scoped and self._should_request_agentic_pathway(text)

    def _execute_live_tree_output(self, model_output, user_request, chat_id):
        """Run Tree branches without starting another LLM worker for Live."""
        if tools_disallowed(user_request):
            return "BLOCKED SAFELY: The user requested no tools. Nothing was executed."
        blocks = self._extract_agentic_blocks(model_output)
        if not blocks:
            blocks = self._build_deterministic_agentic_blocks(user_request, "", chat_id, kira_live=True)
        if not blocks and re.search(r'\b(?:search|look up|weather|forecast|latest|right now)\b|\bfind\b.*\bbest\b', user_request, re.I):
            query = re.sub(r'[\[\]\r\n]', ' ', user_request)[:2000]
            blocks = f'[WEB_SEARCH]\nQUERY: {query}\nVISIBLE: true\n[/WEB_SEARCH]'
        # The shared extractor recognizes the normal Agent-mode tool catalog.
        # Dispatch remains permissioned/sandboxed; Live must not bypass it.
        if not blocks:
            return "BLOCKED SAFELY: No supported live Tree branch was selected. Use Agent mode for the advanced task; nothing was executed."
        self._emit_agent_progress(chat_id, "executing", "KIRA Live 1 is running a Tree branch", 35, self._agentic_branch_labels_from_text(blocks))
        result = self._run_agentic_capabilities_with_watchdog(blocks, user_request, chat_id, "native_live_tree")
        from kira_live.action_contracts import result_links, search_wants_open
        links = result_links(result)
        if links and search_wants_open(user_request, []):
            target = links[0]
            opened = self._run_agentic_capabilities_with_watchdog(
                f'[WEB_OPEN]\nURL: {target}\n[/WEB_OPEN]', user_request, chat_id, 'native_live_tree',
            )
            result += '\n\n' + opened
        self._record_text_execution_evidence(result, chat_id=chat_id)
        self._emit_agent_progress(chat_id, "done", "Live Tree branch returned evidence", 100, [])
        return result

    def start_live_voice(self, chat_id=None, language="English"):
        status = self.get_kira_live_status()
        if not status.get("conversation_ready"):
            return {
                "ok": False,
                "status": "not_ready",
                "error": "KIRA Live 1 is not end-to-end ready. No fallback voice stack was started.",
                "readiness": status,
                "chat_id": chat_id,
            }
        existing = getattr(self, "_live_voice_session", None)
        if existing is not None and getattr(existing, "running", False):
            if chat_id and existing.chat_id != str(chat_id):
                self.stop_live_voice()
            else:
                return {"ok": True, "already_running": True, "session": existing.status(), "chat_id": existing.chat_id}
        chat_id = str(chat_id or self.new_chat()["id"])
        try:
            self._release_text_worker_for_live()
            from kira_live.native_session import KiraNativeLiveSession

            def on_event(kind, payload):
                if kind == 'AUDIO_LEVEL':
                    # Meter updates belong in the UI, not the disk event log.
                    self._emit_native_voice_event(kind, '', {'source': 'kira_live_native', **dict(payload or {})})
                    return
                spoken = str(payload.get("text", "") or "").strip()
                if spoken and kind in {"TRANSCRIPT", "RESPONSE_FINISHED"}:
                    try:
                        self._append_chat_message(
                            chat_id,
                            "user" if kind == "TRANSCRIPT" else "assistant",
                            spoken,
                            "User" if kind == "TRANSCRIPT" else "KIRA Live 1",
                            memory_synced=True,
                        )
                    except Exception as exc:
                        self._record_stts_event(
                            "MEMORY_BRIDGE_WARNING", str(exc),
                            {"source": "kira_live_native"},
                        )
                self._record_stts_event(
                    kind,
                    spoken,
                    {"source": "kira_live_native", **dict(payload or {})},
                )

            session = KiraNativeLiveSession(
                chat_id=str(chat_id or "kira-live"),
                on_event=on_event,
                history_loader=lambda: self._load_chat_messages(chat_id),
                tree_request_detector=self._live_tree_requested,
                tree_executor=lambda output, request: self._execute_live_tree_output(output, request, chat_id),
                language=language,
            )
            self._live_voice_session = session
            session_status = session.start()
            self.voice_mode_active = bool(session_status.get("running"))
            return {
                "ok": self.voice_mode_active,
                "status": "listening" if self.voice_mode_active else "error",
                "session": session_status,
                "readiness": status,
                "chat_id": chat_id,
            }
        except Exception as exc:
            failed_session = getattr(self, "_live_voice_session", None)
            if failed_session is not None:
                try:
                    failed_session.stop()
                except Exception:
                    pass
            self._record_stts_event("LIVE_START_ERROR", str(exc), {"source": "kira_live_native"})
            self._live_voice_session = None
            self.voice_mode_active = False
            return {
                "ok": False,
                "status": "runtime_error",
                "error": str(exc),
                "readiness": status,
                "chat_id": chat_id,
            }

    def _legacy_live_voice(self, chat_id=None):
        """Compatibility entry point for the ordinary local voice session."""
        ensure_stts_terminal()
        if not ensure_listen_engine():
            message = "Local voice is unavailable because listen.py could not be loaded."
            self._record_stts_event("STT_ERROR", message, {"source": "native"})
            return {"ok": False, "error": message, "status": self.get_voice_status()}

        stt_status = _listen_module.get_local_stt_status()
        if not stt_status.get("ready", False):
            message = stt_status.get("error", "Local speech recognition is not ready.")
            self._record_stts_event("STT_ERROR", message, {"source": "native", "stt": stt_status})
            return {"ok": False, "error": message, "status": self.get_voice_status()}

        existing = getattr(self, "_live_voice_session", None)
        if existing is not None and existing.running:
            self._live_voice_chat_id = chat_id
            self.voice_mode_active = True
            return {"ok": True, "already_running": True, "session": existing.status()}

        self._live_voice_chat_id = chat_id
        self.voice_mode_active = True

        def on_event(kind, payload):
            if kind == "STT_ERROR":
                return
            if kind == "SPEECH_START":
                _update_voice_session(state="hearing", last_source="native")
            elif kind == "SPEECH_END":
                _update_voice_session(state="transcribing", last_source="native")
            elif kind == "STT_PROCESSING":
                _update_voice_session(state="transcribing", last_source="native")
            self._record_stts_event(kind, "", {"source": "native", **dict(payload or {})})

        def on_audio_level(payload):
            self._emit_native_voice_event("AUDIO_LEVEL", "", {"source": "native", **dict(payload or {})})

        def on_error(message, payload):
            _update_voice_session(state="error", last_source="native")
            self._record_stts_event("STT_ERROR", message, {"source": "native", **dict(payload or {})})

        def on_speech_start(_payload):
            if _tts_controller.status().get("speaking"):
                stop_speaking(reason="barge_in")

        def on_transcript(text, metadata):
            clean = _stts_text(text, limit=3000)
            if not clean:
                return
            self._record_stts_event("STT_FINAL", clean, {"source": "native", **dict(metadata or {})})
            self.complete_voice_turn(clean, source="native", confidence=1.0, live=False)
            try:
                result = self.send_prompt(
                    clean,
                    "voice",
                    "orchestrator",
                    getattr(self, "_live_voice_chat_id", None),
                )
                if isinstance(result, dict) and result.get("chat_id"):
                    self._live_voice_chat_id = result["chat_id"]
                self._record_stts_event(
                    "VOICE_PROMPT_QUEUED",
                    clean,
                    {"source": "native", "chat_id": getattr(self, "_live_voice_chat_id", None)},
                )
            except Exception as exc:
                _update_voice_session(state="error")
                self._record_stts_event("STT_ERROR", str(exc), {"source": "native", "stage": "send_prompt"})

        session = _listen_module.LocalVoiceSession(
            on_transcript=on_transcript,
            on_event=on_event,
            on_audio_level=on_audio_level,
            on_error=on_error,
            on_speech_start=on_speech_start,
        )
        self._live_voice_session = session
        status = session.start()
        running = bool(status.get("running"))
        if running and not status.get("stt", {}).get("ready", False):
            # Do not capture turns that cannot be transcribed. Returning a clear
            # false status lets the UI use Web Speech until the local model is installed.
            session.stop()
            running = False
            status = session.status()
            status["last_error"] = status.get("stt", {}).get("error", "Local STT is unavailable.")
        if not running:
            self.voice_mode_active = False
            self._live_voice_session = None
        _update_voice_session(
            active=running,
            live_mode=False,
            state="listening" if running else "error",
            stt_provider="local",
            turn_detector=status.get("vad", {}).get("provider", "local_vad"),
            last_source="native",
        )
        self._record_stts_event(
            "VOICE_CAPTURE_START" if running else "STT_ERROR",
            "Local voice capture started." if running else status.get("last_error", "Microphone capture did not start."),
            {"source": "native", "session": status},
        )
        return {"ok": running, "session": status, "error": "" if running else status.get("last_error", "")}

    def stop_live_voice(self):
        session = getattr(self, "_live_voice_session", None)
        self._live_voice_session = None
        session_status = {"running": False, "state": "idle"}
        if session is not None:
            try:
                session_status = session.stop()
            except Exception as exc:
                session_status = {"running": False, "state": "error", "last_error": str(exc)}
        tts_status = stop_speaking(reason="voice_capture_stopped")
        self.voice_mode_active = False
        _update_voice_session(
            active=False,
            live_mode=False,
            state="idle",
            speaking_amplitude=0.0,
            speaking_envelope=0.0,
        )
        self._record_stts_event(
            "VOICE_CAPTURE_STOP",
            "Local voice capture stopped.",
            {"source": "native", "session": session_status, "tts": tts_status},
        )
        return {"ok": True, "session": session_status, "tts": tts_status}

    def set_live_microphone_muted(self, muted=False):
        session = getattr(self, "_live_voice_session", None)
        if session is None or not session.running:
            return {"ok": False, "error": "No active KIRA Live session."}
        return session.set_microphone_muted(bool(muted))

    def interrupt_live_response(self):
        session = getattr(self, "_live_voice_session", None)
        if session is None or not session.running:
            return {"ok": False, "error": "No active KIRA Live session."}
        return session.interrupt_response()

    def record_voice_input(self, text, source="browser"):
        clean = _stts_text(text, limit=3000)
        _update_voice_session(
            active=True,
            state="turn_complete",
            last_transcript=clean,
            last_partial="",
            last_source=source or "voice",
        )
        return self._record_stts_event(
            "STT_IN", clean, {"source": source or "voice", "stack": _voice_stack_status()}
        )

    def start_voice_turn(self, source="browser", live=False):
        session = _update_voice_session(
            active=True,
            live_mode=False,
            state="listening",
            last_partial="",
            last_source=source or "voice",
        )
        return self._record_stts_event(
            "TURN_START",
            "Voice turn started.",
            {"source": source or "voice", "live": False, "stack": _voice_stack_status(), "session": session},
        )

    def record_voice_partial(self, text, source="browser", confidence=0.0):
        clean = _stts_text(text, limit=1200)
        if not clean:
            return {"ok": True, "ignored": True}
        session = _update_voice_session(
            active=True,
            state="hearing",
            last_partial=clean,
            last_source=source or "voice",
            last_confidence=float(confidence or 0.0),
        )
        return self._record_stts_event(
            "STT_PARTIAL",
            clean,
            {"source": source or "voice", "confidence": float(confidence or 0.0), "session": session},
        )

    def complete_voice_turn(self, text, source="browser", confidence=0.0, live=False):
        clean = _stts_text(text, limit=3000)
        if not clean:
            return {"ok": False, "text": "", "error": "No speech text to submit."}
        _update_voice_session(
            active=True,
            live_mode=False,
            state="turn_complete",
            last_transcript=clean,
            last_partial="",
            last_source=source or "voice",
            last_confidence=float(confidence or 0.0),
        )
        event = self._record_stts_event(
            "TURN_END",
            clean,
            {
                "source": source or "voice",
                "confidence": float(confidence or 0.0),
                "live": False,
                "stack": _voice_stack_status(),
            },
        )
        return {"ok": True, "text": clean, "event": event}

    def _record_stts_event(self, kind, text="", extra=None):
        event = _write_stts_event(kind, text, extra)
        self._queue_native_voice_event(event)
        return event

    def _emit_native_voice_event(self, kind, text="", extra=None):
        event = {
            "time": time.time(),
            "label": _stts_label(),
            "kind": str(kind or "EVENT").upper(),
            "text": _stts_text(text),
            "extra": extra or {},
        }
        self._queue_native_voice_event(event)
        return event

    def _queue_native_voice_event(self, event):
        try:
            if hasattr(self, "response_queue"):
                self.response_queue.put({
                    "type": "voice_event",
                    "event": event.get("kind"),
                    "content": event.get("text", ""),
                    "extra": event.get("extra", {}),
                    "label": event.get("label", ""),
                })
        except Exception:
            pass

    def dictate_once(self, request_id=None):
        """Capture one local transcript for the composer, never queue an agent turn."""
        if not self._dictation_lock.acquire(blocking=False):
            return {"ok": False, "error": "Dictation is already running."}
        session = None
        done = self._dictation_cancel
        done.clear()
        result = {"ok": False, "text": "", "error": "No speech recognized. Try again."}
        try:
            self.voice_mode_active = False
            if not ensure_listen_engine():
                return {"ok": False, "error": "Local speech recognition is unavailable."}
            stt = _listen_module.LocalSTT()
            if not stt.ready:
                return {"ok": False, "error": stt.error or "Install a local speech recognition model first."}

            def transcript(text, _metadata):
                if done.is_set():
                    return
                clean = str(text or "").strip()
                result.update(ok=bool(clean), text=clean, error="" if clean else "No speech recognized.")
                done.set()

            def failed(message, _metadata):
                if not done.is_set():
                    result.update(ok=False, error=str(message))
                    done.set()

            def partial(text, metadata):
                if not done.is_set() and hasattr(self, "response_queue"):
                    self.response_queue.put({
                        "type": "dictation_partial", "request_id": request_id,
                        "content": str(text or "").strip(),
                    })

            session = _listen_module.LocalVoiceSession(
                stt=stt, on_transcript=transcript, on_error=failed,
                on_partial=partial, partial_interval_seconds=1.0,
            )
            if done.is_set():
                return {"ok": False, "cancelled": True, "text": ""}
            status = session.start()
            if not status.get("running"):
                return {"ok": False, "error": status.get("last_error") or "Microphone unavailable. Check microphone permissions."}
            done.wait(timeout=30)
            return result
        finally:
            try:
                if session is not None:
                    session.stop()
            finally:
                self._dictation_lock.release()

    def cancel_dictation(self):
        self._dictation_cancel.set()
        return {"ok": True}

    def listen_once(self):
        ensure_stts_terminal()
        _update_voice_session(active=True, state="listening", last_source="listen.py")
        self._record_stts_event("STT_LISTEN_START", "Listening once through the local microphone backend.")
        if not ensure_listen_engine():
            message = "The local Python listening backend is unavailable."
            self._record_stts_event("STT_ERROR", message, {"source": "listen.py"})
            return {"ok": False, "text": "", "error": message}

        try:
            text = _listen_module.kira_listen()
            if text:
                _update_voice_session(
                    active=True,
                    state="turn_complete",
                    last_transcript=_stts_text(text, limit=3000),
                    last_partial="",
                    last_source="listen.py",
                    last_confidence=1.0,
                )
                self._record_stts_event("STT_IN", text, {"source": "listen.py"})
            else:
                _update_voice_session(active=True, state="listening", last_partial="", last_source="listen.py")
                self._record_stts_event("STT_EMPTY", "No speech was recognized.", {"source": "listen.py"})
            return {"ok": bool(text), "text": text or "", "error": "" if text else "I did not catch that."}
        except Exception as exc:
            self._record_stts_event("STT_ERROR", str(exc), {"source": "listen.py"})
            return {"ok": False, "text": "", "error": str(exc)}

    def stop_voice(self):
        return stop_speaking(reason="stop_voice")

    def _speak_if_voice_active(self, text):
        if getattr(self, "voice_mode_active", False):
            _update_voice_session(state="speaking")
            self._trigger_voice(text)

    def _voice_clean(self, text):
        clean = self._strip_private_reasoning(text)
        clean = self._strip_agentic_blocks(clean)
        clean = re.sub(r"```.*?```", "I completed the command and have the result.", clean, flags=re.DOTALL)
        clean = re.sub(r"`([^`]+)`", r"\1", clean)
        return clean.strip()

    def _on_tts_event(self, kind, text, extra):
        live_session = getattr(self, "_live_voice_session", None)
        if kind == "TTS_START":
            _update_voice_session(state="speaking", speaking_amplitude=0.0, speaking_envelope=0.0)
            if live_session is not None and live_session.running:
                live_session.set_paused(True, flush=True)
        elif kind == "TTS_LEVEL":
            _update_voice_session(
                state="speaking",
                speaking_amplitude=float(extra.get("amplitude", 0.0)),
                speaking_envelope=float(extra.get("envelope", 0.0)),
            )
        elif kind in {"TTS_DONE", "TTS_INTERRUPTED", "TTS_ERROR"}:
            if live_session is not None and live_session.running:
                live_session.set_paused(False, flush=True)
            _update_voice_session(
                state="listening" if getattr(self, "voice_mode_active", False) else "idle",
                speaking_amplitude=0.0,
                speaking_envelope=0.0,
            )

        if kind == "TTS_LEVEL":
            self._emit_native_voice_event(kind, text, extra)
        else:
            self._record_stts_event(kind, text, extra)

    def _trigger_voice(self, text):
        clean_text = self._voice_clean(text)
        if not clean_text:
            return {"ok": False, "error": "No speakable text."}
        self._record_stts_event("TTS_QUEUE", clean_text, {"silent_test": STTS_SILENT_TEST})
        return kira_speak(
            clean_text,
            silent=STTS_SILENT_TEST,
            callback=self._on_tts_event,
            wait=False,
        )


if __name__ == "__main__":
    kira_speak("Hardware diagnostics complete. Universal voice module is online.")
