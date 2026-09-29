"""Overlap same-thread neural synthesis with a CPU-only playback worker."""
from collections import deque
import queue
import threading


def overlap_audio(chunks, consume, cancelled, prebuffer=2):
    pending = queue.Queue(maxsize=8)
    finished = threading.Event()
    errors = []
    count = [0]

    def playback():
        buffered = deque()
        started = False
        try:
            while not cancelled.is_set():
                try:
                    # Never wait for future synthesis when audio is already
                    # buffered: that inserted a 20 ms gap between chunks.
                    buffered.append(pending.get(timeout=0 if buffered else .02))
                except queue.Empty:
                    pass
                if not started and (len(buffered) >= prebuffer or finished.is_set()):
                    started = True
                if started and buffered:
                    consume(*buffered.popleft())
                    count[0] += 1
                elif finished.is_set() and pending.empty():
                    break
        except Exception as exc:
            errors.append(exc)
            cancelled.set()

    worker = threading.Thread(target=playback, name="kira-live-playback", daemon=True)
    worker.start()
    try:
        for chunk in chunks:
            while not cancelled.is_set():
                try:
                    pending.put(chunk, timeout=.02)
                    break
                except queue.Full:
                    pass
            if cancelled.is_set():
                break
    finally:
        finished.set()
        # A long spoken reply must not be cut off by a fixed join timeout.
        # Keep cancellation bounded, but drain healthy playback completely.
        while worker.is_alive() and not cancelled.is_set():
            worker.join(timeout=.05)
        if worker.is_alive():
            worker.join(timeout=2)
    if errors:
        raise errors[0]
    return count[0]
