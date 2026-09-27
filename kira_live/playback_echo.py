"""Bounded playback-reference subtraction before microphone VAD.

This is a lightweight echo guard, not a claim of production acoustic echo
cancellation. Uncorrelated user speech passes through, including during TTS.
"""
import threading
import time
from collections import deque
import numpy as np


class PlaybackEchoFilter:
    def __init__(self, sample_rate=16000, history_seconds=0.65):
        self.sample_rate = sample_rate
        self.limit = int(sample_rate * history_seconds)
        self._reference = np.zeros(0, dtype=np.float32)
        self._last_written = -1e9
        self._lock = threading.Lock()
        self.frames_cleaned = 0

    def add_playback(self, samples, sample_rate, *, now=None):
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate != self.sample_rate and values.size:
            positions = np.arange(round(values.size * self.sample_rate / sample_rate)) * sample_rate / self.sample_rate
            values = np.interp(positions, np.arange(values.size), values).astype(np.float32)
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            if timestamp - self._last_written > 0.8:
                self._reference = np.zeros(0, dtype=np.float32)
            self._reference = np.concatenate((self._reference, values))[-self.limit:]
            self._last_written = timestamp

    def clean(self, samples, *, now=None):
        frame = np.asarray(samples, dtype=np.float32).reshape(-1)
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            if timestamp - self._last_written > 0.8:
                return frame
            reference = self._reference.copy()
        n = frame.size
        if n < 16 or reference.size < n or float(np.linalg.norm(frame)) < 1e-6:
            return frame
        centered = frame - frame.mean()
        sums = np.concatenate(([0.0], np.cumsum(reference, dtype=np.float64)))
        squares = np.concatenate(([0.0], np.cumsum(reference * reference, dtype=np.float64)))
        variance = np.maximum(squares[n:] - squares[:-n] - (sums[n:] - sums[:-n]) ** 2 / n, 1e-12)
        dots = np.correlate(reference, centered, mode="valid")
        correlations = np.abs(dots) / np.sqrt(variance * max(float(np.dot(centered, centered)), 1e-12))
        best = int(np.argmax(correlations))
        if correlations[best] < 0.65:
            return frame
        # A short causal FIR estimates gain and modest room coloration. Do not
        # gate the mic merely because playback exists: retain double-talk residual.
        taps = min(8, best + 1)
        matrix = np.column_stack([reference[best - lag:best - lag + n] for lag in range(taps)])
        matrix = matrix - matrix.mean(axis=0)
        coefficients = np.linalg.lstsq(matrix, centered, rcond=1e-3)[0]
        residual = centered - matrix @ coefficients
        self.frames_cleaned += 1
        return residual.astype(np.float32)


class DuplexEchoGuard:
    """WebRTC AEC3 with a capture-clock render FIFO and playback-tail guard.

    DSP runs on the capture callback, not the MLX worker. Speaker audio is
    queued before each actual output write, resampled to 16 kHz, and consumed
    exactly once. The canceller continues receiving silence after rendering
    ends so delayed room echo is removed rather than treated as a new speaker.
    """
    name = "webrtc_aec3_with_playback_tail"

    def __init__(self, sample_rate=16000, *, processor=None, tail_seconds=.45):
        if processor is None:
            try:
                from pywebrtc_audio import AudioProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "KIRA Live speaker mode requires pywebrtc-audio==0.2.0. "
                    "Install the current Live requirements before starting."
                ) from exc
            processor = AudioProcessor(
                sample_rate=sample_rate, echo_cancellation=True,
                noise_suppression=True, ns_level=1, auto_gain_control=False,
                stream_delay_ms=40,
            )
        self.sample_rate = sample_rate
        self.processor = processor
        self.tail_seconds = tail_seconds
        self._pending = deque()
        self._pending_samples = 0
        self._lock = threading.Lock()
        self._render_end = -1e9
        self._reference_filter = PlaybackEchoFilter(sample_rate)
        self.frames_cleaned = 0

    def recent_playback(self, *, now=None):
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            return timestamp <= self._render_end + self.tail_seconds

    def set_output_delay(self, seconds):
        with self._lock:
            # Delay is a hint; AEC3 also estimates the acoustic path itself.
            self.processor.stream_delay_ms = max(0, min(500, round(float(seconds) * 1000)))

    def add_playback(self, samples, sample_rate, *, now=None):
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not values.size:
            return
        timestamp = time.monotonic() if now is None else now
        if sample_rate != self.sample_rate:
            positions = np.arange(round(values.size * self.sample_rate / sample_rate)) * sample_rate / self.sample_rate
            values = np.interp(positions, np.arange(values.size), values).astype(np.float32)
        with self._lock:
            self._pending.append(values.copy())
            self._pending_samples += len(values)
            # Protect against producer/capture clock drift, never replay stale
            # reference indefinitely or let neural synthesis queue seconds ahead.
            while self._pending_samples > self.sample_rate * .25 and len(self._pending) > 1:
                self._pending_samples -= len(self._pending.popleft())
            self._render_end = max(timestamp, self._render_end) + len(values) / self.sample_rate
        self._reference_filter.add_playback(values, self.sample_rate, now=timestamp)

    def clean(self, samples, *, now=None):
        frame = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        far = np.zeros(len(frame), dtype=np.float32)
        with self._lock:
            offset = 0
            while offset < len(far) and self._pending:
                head = self._pending.popleft()
                length = min(len(head), len(far)-offset)
                far[offset:offset+length] = head[:length]
                offset += length
                self._pending_samples -= length
                if length < len(head):
                    self._pending.appendleft(head[length:])
            cleaned = self.processor.process(frame, far)
        # The old reference matcher is useful while AEC3 initially converges.
        # It operates on residuals, retaining unrelated near-end/double-talk.
        cleaned = self._reference_filter.clean(cleaned, now=now)
        self.frames_cleaned += 1
        return np.asarray(cleaned, dtype=np.float32)

    def finish_playback(self, *, aborted=False, now=None):
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            if aborted:
                self._pending.clear()
                self._pending_samples = 0
                self._render_end = timestamp
            else:
                self._render_end = max(timestamp, self._render_end)
