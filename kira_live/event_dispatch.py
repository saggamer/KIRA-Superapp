"""Keep disk/UI callbacks off the real-time microphone and playback threads."""
from collections import deque
import threading


class LiveEventDispatcher:
    def __init__(self, callback):
        self.callback = callback
        self.pending = deque()
        self.level = None
        self.condition = threading.Condition()
        self.closed = False
        self.last_error = ''
        self.worker = threading.Thread(target=self._run, name='kira-live-events', daemon=True)
        self.worker.start()

    def submit(self, kind, payload):
        with self.condition:
            if self.closed:
                return
            if kind == 'AUDIO_LEVEL':
                self.level = (kind, payload)  # Coalesce meters, never dialogue/lifecycle events.
            else:
                self.pending.append((kind, payload))
            self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending or self.level is not None)
                if self.pending:
                    event = self.pending.popleft()
                elif self.level is not None:
                    event, self.level = self.level, None
                elif self.closed:
                    return
                else:
                    continue
            try:
                if self.callback is not None:
                    self.callback(*event)
            except Exception as exc:
                self.last_error = str(exc)  # UI failures must not kill audio capture.

    def close(self):
        with self.condition:
            self.closed = True
            self.level = None
            self.condition.notify()
        if threading.current_thread() is not self.worker:
            self.worker.join(timeout=1)
