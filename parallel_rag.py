"""Concurrent draft/retrieval with a deterministic, fail-closed output gate."""
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass, replace
import re
import threading
import time
from typing import Callable

from evidence_rag import ABSTENTION, Answer, EvidenceStore

# At most one outstanding retrieval process-wide; no unbounded worker queue.
_RETRIEVAL_SLOT = threading.BoundedSemaphore(1)


@dataclass(frozen=True)
class ParallelResult:
    answer: Answer
    draft_status: str
    retrieval_ms: float
    generation_ms: float
    total_ms: float


def parallel_answer(store: EvidenceStore, query: str,
                    generate: Callable[[str], str], *, join_timeout: float = 0.3) -> ParallelResult:
    """Launch retrieval before invoking generation on the calling model thread.

    Drafts never stream to UI, history, voice, tools, or logs. Only an exact
    whole source sentence can replace the retrieved excerpts; other drafts are
    discarded. The generate callback must enforce its own inference timeout.
    """
    started = time.monotonic()
    retrieval_ms = 0.0
    generation_ms = 0.0
    def finish(answer, status):
        return ParallelResult(answer, status, retrieval_ms, generation_ms,
                              (time.monotonic() - started) * 1000)
    if not _RETRIEVAL_SLOT.acquire(blocking=False):
        return finish(Answer("retrieval_busy", ABSTENTION), "skipped_busy")
    future = Future()

    def retrieve():
        tick = time.monotonic()
        try:
            future.set_result((store.answer(query), (time.monotonic() - tick) * 1000))
        except Exception as exc:
            future.set_exception(exc)
        finally:
            _RETRIEVAL_SLOT.release()

    worker = threading.Thread(target=retrieve, name="kira-evidence-retrieval", daemon=True)
    try:
        worker.start()
    except Exception:
        _RETRIEVAL_SLOT.release()
        return finish(Answer("retrieval_unavailable", ABSTENTION), "skipped_unavailable")
    tick = time.monotonic()
    status = "discarded"
    draft = ""
    try:
        draft = generate(query).strip()
    except Exception:
        status = "unavailable"
    generation_ms = (time.monotonic() - tick) * 1000
    try:
        answer, retrieval_ms = future.result(timeout=join_timeout)
        if answer.evidence and not store.evidence_is_current(answer.evidence):
            return finish(Answer("evidence_changed", ABSTENTION), "discarded")
    except TimeoutError:
        return finish(Answer("retrieval_timeout", ABSTENTION), "discarded")
    except Exception:
        return finish(Answer("retrieval_unavailable", ABSTENTION + " The knowledge index could not be read."), "discarded")
    # Multiple excerpts can disagree. Preserve all of them instead of accepting
    # a draft that chooses a convenient source. No fuzzy/embedding verification.
    if len(answer.evidence) == 1 and 20 <= len(draft) <= 900:
        item = answer.evidence[0]
        sentences = re.split(r"(?<=[.!?])\s+|\n+", item.text)
        if draft in {sentence.strip() for sentence in sentences}:
            offset = item.text.index(draft)
            excerpt = replace(item, text=draft, start=item.start + offset,
                              end=item.start + offset + len(draft))
            answer = store.render((excerpt,))
            status = "source_matched"
    return finish(answer, status)
