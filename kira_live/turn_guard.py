from __future__ import annotations

import re
import time
import unicodedata
import threading
from collections import deque


def normalized_utterance(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


class LiveTurnGuard:
    """Suppress ASR fragments and rapid duplicates without losing chat memory."""

    def __init__(self, duplicate_window_seconds: float = 3.0):
        self.duplicate_window_seconds = max(0.5, float(duplicate_window_seconds))
        self._last_key = ""
        self._last_at = 0.0
        self._spoken = deque(maxlen=4)
        self._spoken_lock = threading.Lock()

    def note_playback(self, text: str, *, now: float | None = None) -> None:
        timestamp = time.monotonic() if now is None else float(now)
        with self._spoken_lock:
            self._spoken.append((normalized_utterance(text), timestamp))

    def accept(self, text: str, *, now: float | None = None, during_playback: bool = False) -> tuple[bool, str]:
        key = normalized_utterance(text)
        alphanumeric = "".join(character for character in key if character.isalnum())
        if len(alphanumeric) < 2:
            return False, "low_information"
        timestamp = time.monotonic() if now is None else float(now)
        # Only consider captures that overlapped speaker output or its echo
        # tail. Never reject a later deliberate repetition in ordinary listening.
        if during_playback and len(key.split()) >= 3:
            with self._spoken_lock:
                recent_spoken = tuple(self._spoken)
            for spoken, started in recent_spoken:
                if timestamp - started <= 45 and f" {key} " in f" {spoken} ":
                    return False, "speaker_echo"
        if key == self._last_key and timestamp - self._last_at <= self.duplicate_window_seconds:
            return False, "rapid_duplicate"
        self._last_key = key
        self._last_at = timestamp
        return True, "accepted"
