from __future__ import annotations

"""Native full-duplex KIRA Live session.

This module is deliberately lazy-loaded by the desktop app.  It owns one
Listener → Thinker → Talker graph, keeps text transcription diagnostic-only,
and shares one cancellation event across token generation, codec generation,
and waveform playback.
"""

import math
from dataclasses import replace
import os
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
import threading
import time
from typing import Callable

import mlx.core as mx
import numpy as np
from transformers import AutoProcessor

from .audio_devices import pick_audio_device
from .duplex import KiraLiveDuplexController
from .coreml import CoreMLEmotionHeadRunner
from .mlx_emotion import NgramListenerEmotionHead, pool_listener_ngram_states
from .mlx_listener import KiraMLXListener
from .mlx_thinker import KiraMLXThinker
from .mlx_voice import IntegratedSpeechPlan, KiraIntegratedQwenVoice
from .memory_grounding import grounded_recall
from .privacy import sanitize_public_output
from .public_guard import guard_public_reply, guard_live_action_reply
from .session import KiraLiveSession
from .turn_guard import LiveTurnGuard
from .playback_echo import DuplexEchoGuard
from .noise_gate import LiveNoiseGate
from .speech_detector import LiveSpeechDetector
from .audio_pipeline import overlap_audio
from .token_bus import remap_token_ids
from .checkpoint_bundle import resolve_live_checkpoint
from .workflow_memory import PersistentWorkflowMemory


ROOT = Path(__file__).resolve().parent.parent


def _configure_mlx_runtime() -> dict[str, int]:
    """Keep MLX's reusable buffers bounded while leaving model memory untouched."""
    try:
        total_memory = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        cache_limit = min(2 * 1024**3, max(256 * 1024**2, total_memory // 16))
        previous = int(mx.set_cache_limit(cache_limit))
        return {"cache_limit": cache_limit, "previous_cache_limit": previous}
    except Exception:
        # MLX can still run when a cache limit is unavailable (for example in
        # headless test processes without a Metal device).
        return {}


class KiraNativeModelBundle:
    """Load all accepted local checkpoints exactly once for a live session."""

    def __init__(self):
        self.mlx_runtime = _configure_mlx_runtime()
        # The live graph is constructed on the UI bridge thread and evaluated
        # on a serialized audio worker. MLX's ordinary streams are bound to the
        # thread that created them, so use its explicit cross-thread stream.
        # `_lock` in KiraNativeLiveSession guarantees sequential evaluation.
        self.mlx_stream = mx.new_thread_unsafe_stream(mx.gpu)
        self.mlx_cpu_stream = mx.new_thread_unsafe_stream(mx.cpu)
        mx.set_default_stream(self.mlx_stream)
        mx.set_default_stream(self.mlx_cpu_stream)
        checkpoint = resolve_live_checkpoint()
        weights = checkpoint.weights
        self.processor = AutoProcessor.from_pretrained(
            str(weights.listener.snapshot), local_files_only=True
        )
        self.listener = KiraMLXListener.from_composite_checkpoint(
            weights.listener.model_file,
            weights.listener.config,
            self.processor.tokenizer,
        )
        self.thinker = KiraMLXThinker.from_pretrained(weights.thinker.snapshot)
        self.thinker.load_weights(
            str(checkpoint.thinker_addons), strict=False
        )
        self.thinker.eval()
        self.voice = KiraIntegratedQwenVoice(
            # Load the validated local island, including standalone Hub bundles.
            # Resolving its original Hub ID again would require a donor cache.
            model_id=str(weights.talker.snapshot),
        )
        self.emotion = NgramListenerEmotionHead()
        self.emotion.load_weights(str(checkpoint.emotion_head), strict=True)
        normalization = np.load(checkpoint.emotion_normalization)
        self.emotion_mean = mx.array(normalization["mean"], dtype=mx.float32)
        self.emotion_std = mx.array(normalization["std"], dtype=mx.float32)
        self.coreml_emotion = (
            CoreMLEmotionHeadRunner(checkpoint.coreml_emotion)
            if checkpoint.coreml_emotion is not None else None
        )
        # Inference mode prevents any training-only behavior from degrading a
        # live turn and lets MLX reuse the graph more consistently.
        for module in (
            self.listener,
            self.thinker,
            self.voice.model,
            self.emotion,
        ):
            module.eval()

    def features(self, samples: np.ndarray) -> tuple[mx.array, int]:
        encoded = self.processor.feature_extractor(
            samples,
            sampling_rate=16_000,
            return_tensors="np",
            return_attention_mask=True,
        )
        length = int(encoded["attention_mask"].sum())
        padded = max(100, int(math.ceil(length / 100) * 100))
        return (
            mx.array(encoded["input_features"][:, :, :padded], dtype=mx.bfloat16),
            length,
        )


class KiraNativeLiveSession:
    """Always-on microphone session with hard barge-in cancellation."""

    def __init__(
        self,
        *,
        chat_id: str,
        on_event: Callable[[str, dict], None] | None = None,
        history_loader: Callable[[], list[dict]] | None = None,
        tree_request_detector: Callable[[str], bool] | None = None,
        tree_executor: Callable[[str, str], str] | None = None,
        bundle: KiraNativeModelBundle | None = None,
    ):
        self.chat_id = str(chat_id or "kira-live")
        self.on_event = on_event
        self.history_loader = history_loader
        self.tree_request_detector = tree_request_detector
        self.tree_executor = tree_executor
        # MLX streams are thread-affine. Construct every weight island and run
        # every turn on this one persistent inference thread.
        self._inference = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="kira-live-mlx",
        )
        self.bundle = bundle or self._inference.submit(KiraNativeModelBundle).result()
        memory = PersistentWorkflowMemory(ROOT / "kira_live" / "kira_live_memory.sqlite3")
        self.control = KiraLiveSession(memory=memory, chat_id=self.chat_id)
        self.duplex = KiraLiveDuplexController(
            self.control,
            audio_config=replace(self.control.config.audio, start_speech_ms=100, vad_start_threshold=.65),
            stop_playback=self.stop_playback,
            on_utterance=self._on_utterance,
        )
        self._stream = None
        self._output_stream = None
        self._worker: Future | None = None
        self._running = False
        self._lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._mic_muted = False
        self._echo_filter = DuplexEchoGuard()
        self._utterance_playback_overlap = False
        self._noise_gate = LiveNoiseGate(float(os.environ.get("KIRA_LIVE_MIN_DBFS", "-48")))
        self._last_error = ""
        self._last_level_emit = 0.0
        self._turn_guard = LiveTurnGuard()
        self._input_device: int | None = None
        self._output_device: int | None = None

    @property
    def running(self) -> bool:
        return self._running

    def _emit(self, kind: str, **payload) -> None:
        payload.setdefault("chat_id", self.chat_id)
        if self.on_event is not None:
            self.on_event(kind, payload)

    def start(self) -> dict:
        if self._running:
            return self.status()
        import sounddevice as sd

        devices = tuple(dict(item) for item in sd.query_devices())
        defaults = tuple(int(item) for item in sd.default.device)
        self._input_device = pick_audio_device(
            devices, defaults[0], "max_input_channels"
        )
        self._output_device = pick_audio_device(
            devices, defaults[1], "max_output_channels"
        )

        # Fail closed if the speech detector is unavailable. Never silently
        # revert to amplitude-only interruption. Reblock 320-sample capture
        # into the detector's exact 256-sample native hops.
        self._vad = LiveSpeechDetector()

        def callback(indata, frames, _time, status):
            if status:
                self._emit("CAPTURE_STATUS", status=str(status))
            if not self._running or self._mic_muted or frames <= 0:
                return
            frame = np.asarray(indata[:, 0], dtype=np.float32)
            frame = self._echo_filter.clean(frame)
            _speech, probability = self._vad.process(frame)
            probability = self._noise_gate.probability(
                frame, probability, speaking=self.duplex.segmenter.speaking,
                playback=self._echo_filter.recent_playback(),
            )
            now = time.monotonic()
            if now - self._last_level_emit >= 0.08:
                rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
                self._emit("AUDIO_LEVEL", level=min(1.0, rms * 12.0))
                self._last_level_emit = now
            with self._capture_lock:
                if self._mic_muted:
                    return
                if self._echo_filter.recent_playback():
                    self._utterance_playback_overlap = True
                events = self.duplex.ingest_frame(frame, probability)
                if not self.duplex.segmenter.speaking and not events:
                    self._utterance_playback_overlap = False
            for event in events:
                self._emit(event.kind.upper(), probability=event.probability)

        self._running = True
        self.control.start()
        self._stream = sd.InputStream(
            device=self._input_device,
            samplerate=16_000,
            channels=1,
            dtype="float32",
            blocksize=self.control.config.audio.samples_per_frame,
            callback=callback,
        )
        self._stream.start()
        self._emit(
            "LISTENING_STARTED",
            chat_id=self.chat_id,
            vad=getattr(self._vad, "name", "unknown"),
            mlx_runtime=self.bundle.mlx_runtime,
        )
        return self.status()

    def _on_utterance(self, samples: tuple[float, ...], ngrams: dict) -> None:
        ticket = self.duplex.begin_generation()
        overlap = self._utterance_playback_overlap
        self._utterance_playback_overlap = False
        self._worker = self._inference.submit(
            self._run_utterance,
            np.asarray(samples, dtype=np.float32),
            ngrams,
            ticket,
            overlap,
        )

    def _emotion_features(self, acoustic_hidden: mx.array) -> mx.array:
        pooled = pool_listener_ngram_states(acoustic_hidden)
        if self.bundle.coreml_emotion is not None:
            result = self.bundle.coreml_emotion.predict(
                np.asarray(pooled.astype(mx.float32))
            )
            logits = mx.array(result["emotion_logits"])[None]
            prosody = mx.array(result["prosody"])[None]
        else:
            normalized = (pooled - self.bundle.emotion_mean) / self.bundle.emotion_std
            logits, prosody = self.bundle.emotion(normalized[None])
        probabilities = mx.softmax(logits, axis=-1)
        mx.eval(probabilities, prosody)
        values = probabilities[0].tolist()
        mapped = {
            "neutral": values[0] + values[1] * 0.35,
            "joy": values[2],
            "sadness": values[3],
            "anger": values[4] + values[6] * 0.4,
            "fear": values[5],
            "surprise": values[7],
        }
        confidence = max(mapped.values())
        state = self.control.emotions.update(
            mapped,
            valence=float(prosody[0, 0].item()) * 2.0 - 1.0,
            arousal=float(prosody[0, 1].item()),
            confidence=confidence,
        )
        self.control.update_emotion(state)
        return probabilities.astype(mx.bfloat16)

    def _run_utterance(self, samples: np.ndarray, ngrams: dict, ticket, during_playback: bool = False) -> None:
        try:
            with self._lock:
                with mx.stream(self.bundle.mlx_cpu_stream), mx.stream(
                    self.bundle.mlx_stream
                ):
                    features, feature_length = self.bundle.features(samples)
                    decoded = self.bundle.listener.semantic_decode(
                        features, feature_length, cancelled=ticket.cancelled
                    )
                    if ticket.cancelled.is_set() or not decoded.token_ids:
                        return
                    accepted, reason = self._turn_guard.accept(decoded.text, during_playback=during_playback)
                    if not accepted:
                        self._emit(
                            "TRANSCRIPT_IGNORED",
                            text=decoded.text,
                            reason=reason,
                        )
                        return
                    if self.history_loader is not None and self.control.memory is not None:
                        try:
                            self.control.memory.sync_chat_messages(
                                self.chat_id, self.history_loader()
                            )
                        except Exception as exc:
                            # Chat-history import is supplementary. The native
                            # append-only voice log remains available if it fails.
                            self._emit("MEMORY_BRIDGE_WARNING", error=str(exc))
                    prior_memory = ""
                    if self.control.memory is not None:
                        prior_memory = self.control.memory.build_context(
                            self.chat_id,
                            decoded.text,
                            recent=self.control.config.memory.recent_exact_turns,
                            retrieved=self.control.config.memory.retrieval_events,
                            max_chars=self.control.config.memory.prompt_budget_chars,
                        )
                    self.control.transcript_ready(decoded.text)
                    self._emit("TRANSCRIPT", text=decoded.text)
                    emotion = self._emotion_features(decoded.acoustic_hidden)
                    mapped_token_ids = remap_token_ids(
                        decoded.token_ids,
                        self.bundle.listener.tokenizer,
                        self.bundle.thinker.tokenizer,
                    )
                    mapped_tokens = mx.array([mapped_token_ids], dtype=mx.int32)
                    # Exact mapped semantics passed the spoken capability
                    # check. Listener acoustics still enter via emotion; the
                    # older hidden residual changed the meaning of words.
                    semantic = self.bundle.thinker.embed(mapped_tokens)
                    tool_evidence = ""
                    tree_requested = bool(
                        self.tree_executor is not None and self.tree_request_detector is not None
                        and self.tree_request_detector(decoded.text)
                    )
                    if tree_requested:
                        planning_prefix = self.bundle.thinker.build_live_prefix(
                            semantic, memory_context=prior_memory,
                            listener_unit_ids=mapped_tokens, tree_planning=True,
                        )
                        plan = self.bundle.thinker.generate_from_embeddings(
                            planning_prefix.embeddings, emotion_features=emotion,
                            ngram_ids=planning_prefix.ngram_ids,
                            cancelled=ticket.cancelled, max_tokens=160, compute_response_hidden=False,
                        )
                        if ticket.cancelled.is_set():
                            return
                        self._emit("TREE_ACTION_STARTED")
                        tool_evidence = str(self.tree_executor(plan.text, decoded.text) or "")
                        if self.control.memory is not None and tool_evidence:
                            self.control.memory.append(self.chat_id, "tool", tool_evidence, kind="tree_evidence")
                        self._emit("TREE_ACTION_FINISHED", evidence_chars=len(tool_evidence))
                        if ticket.cancelled.is_set():
                            return
                    response_context = prior_memory
                    if tool_evidence:
                        response_context += "\n\nKIRA OS backend evidence (pending/failed actions are not complete):\n" + tool_evidence[-4000:]
                    prefix = self.bundle.thinker.build_live_prefix(
                        semantic,
                        memory_context=response_context,
                        listener_unit_ids=mapped_tokens,
                    )
                    grounded = grounded_recall(decoded.text, prior_memory)
                    if grounded is not None:
                        public_text = grounded
                        response_token_ids = tuple(
                            self.bundle.thinker.tokenizer.encode(
                                grounded, add_special_tokens=False
                            )
                        )
                    else:
                        response = self.bundle.thinker.generate_from_embeddings(
                            prefix.embeddings,
                            emotion_features=emotion,
                            ngram_ids=prefix.ngram_ids,
                            cancelled=ticket.cancelled,
                            # Exact history remains on disk; cap live turns to
                            # keep spoken latency bounded.
                            max_tokens=64,
                            compute_response_hidden=False,
                        )
                        public_text = sanitize_public_output(response.text, live=True)
                        response_token_ids = response.token_ids
                    guarded = guard_public_reply(
                        decoded.text, public_text,
                        observed_local_files=any(marker in tool_evidence for marker in ("READ_FILE:", "LIST_DIR:", "SEARCH_FILES:")),
                    )
                    guarded = guard_live_action_reply(decoded.text, guarded, tool_evidence)
                    if guarded != public_text:
                        public_text = guarded
                        response_token_ids = tuple(self.bundle.thinker.tokenizer.encode(guarded, add_special_tokens=False))
                    if ticket.cancelled.is_set() or not public_text:
                        return
                    speech_plan = self.bundle.voice.build_plan(
                        response_token_ids,
                        public_text,
                        self.bundle.thinker.tokenizer,
                        self.control.emotions.state(),
                    )
                    self.control.response_started()
                    self._emit("RESPONSE_TEXT", text=speech_plan.public_text)
                    chunks = self._play_voice_stream(speech_plan, ticket.cancelled)
                    if ticket.cancelled.is_set() or chunks == 0:
                        return
                    if not ticket.cancelled.is_set():
                        self.control.response_finished(speech_plan.public_text)
                        self._emit("RESPONSE_FINISHED", text=speech_plan.public_text)
        except Exception as exc:
            if ticket.cancelled.is_set() or not self._running:
                # A barge-in intentionally aborts an in-flight PortAudio write
                # and the shared MLX generation ticket. That is a successful
                # interruption, not a live-session failure.
                self._emit("RESPONSE_INTERRUPTED")
                return
            self._last_error = str(exc)
            self._emit("LIVE_ERROR", error=self._last_error)

    def _play_voice_stream(
        self, plan: IntegratedSpeechPlan, cancelled: threading.Event
    ) -> int:
        """Play complete conversational audio continuously with hard barge-in."""
        import sounddevice as sd

        started = time.perf_counter()
        def consume(samples, sample_rate):
            if cancelled.is_set() or not self._running:
                return
            if self._output_stream is None:
                self._output_stream = sd.OutputStream(
                    device=self._output_device, samplerate=sample_rate,
                    channels=1, dtype="float32", blocksize=960,
                )
                self._output_stream.start()
                self._echo_filter.set_output_delay(self._output_stream.latency)
                self._turn_guard.note_playback(plan.public_text)
                self._emit("FIRST_AUDIO", latency_seconds=time.perf_counter() - started)
            for offset in range(0, samples.size, 960):
                if cancelled.is_set() or not self._running:
                    return
                playback = samples[offset:offset + 960]
                self._echo_filter.add_playback(playback, sample_rate)
                stream = self._output_stream
                if stream is None:
                    return
                stream.write(playback[:, None])
        try:
            return overlap_audio(self.bundle.voice.iter_audio(plan, cancelled), consume, cancelled)
        finally:
            self.stop_playback(abort=cancelled.is_set() or not self._running)

    def stop_playback(self, *, abort: bool = True) -> None:
        stream, self._output_stream = self._output_stream, None
        if stream is not None:
            try:
                if abort:
                    stream.abort()
                else:
                    stream.stop()
                stream.close()
            except Exception:
                pass
            self._echo_filter.finish_playback(aborted=abort)

    def set_microphone_muted(self, muted: bool) -> dict:
        with self._capture_lock:
            self._mic_muted = bool(muted)
            self.duplex.segmenter.reset()
        self._emit("MIC_MUTED" if muted else "MIC_RESUMED")
        return {"ok": True, "muted": self._mic_muted}

    def interrupt_response(self) -> dict:
        self.duplex.generation.interrupt()
        self.stop_playback()
        self.control.response_finished()
        self._emit("RESPONSE_INTERRUPTED")
        return {"ok": True, "muted": self._mic_muted}

    def stop(self) -> dict:
        self._running = False
        self.duplex.generation.interrupt()
        self.stop_playback()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        vad = getattr(self, "_vad", None)
        if vad is not None:
            try:
                vad.close()
            except Exception:
                pass
        self.control.stop()
        status = self.status()
        self._inference.shutdown(wait=False, cancel_futures=True)
        return status

    def status(self) -> dict:
        return {
            "running": self._running,
            "microphone_muted": self._mic_muted,
            "echo_reference_frames_cleaned": self._echo_filter.frames_cleaned,
            "echo_canceller": self._echo_filter.name,
            "state": self.control.state.value,
            "chat_id": self.chat_id,
            "last_error": self._last_error,
            "talker": self.bundle.voice.model_id,
            "voice": self.bundle.voice.speaker,
            "input_device": self._input_device,
            "output_device": self._output_device,
            "acoustic_ngrams": self.duplex.ngrams.snapshot(),
        }
