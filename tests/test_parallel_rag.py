from pathlib import Path
import tempfile
import threading
import unittest

from evidence_rag import ABSTENTION, EvidenceStore
from parallel_rag import parallel_answer
from evidence_rag_runtime import EvidenceRAGMixin


class ParallelRAGTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "evidence.sqlite3"
        self.store = EvidenceStore(self.path)
        self.store.ingest("The Atlas warranty lasts 24 months. Repairs require a receipt.",
                          uri="policy:atlas", title="Atlas policy", approved=True)

    def test_retrieval_and_generation_overlap(self):
        retrieval_entered = threading.Event()
        generation_entered = threading.Event()
        original = self.store.answer
        def retrieve(query):
            retrieval_entered.set()
            self.assertTrue(generation_entered.wait(1), "Retrieval waited for generation")
            return original(query)
        self.store.answer = retrieve
        calls = []
        def generate(query):
            calls.append(query)
            generation_entered.set()
            self.assertTrue(retrieval_entered.wait(1), "Generation waited for retrieval")
            return "The Atlas warranty lasts 24 months."
        result = parallel_answer(self.store, "Atlas warranty", generate)
        self.assertEqual(calls, ["Atlas warranty"])
        self.assertEqual(result.draft_status, "source_matched")
        self.assertEqual(result.answer.evidence[0].text, "The Atlas warranty lasts 24 months.")
        self.assertGreater(result.total_ms, 0)

    def test_unsupported_draft_never_appears(self):
        result = parallel_answer(self.store, "Atlas warranty", lambda q: "Atlas warranty lasts 999 months.")
        self.assertEqual(result.draft_status, "discarded")
        self.assertNotIn("999", result.answer.message)
        self.assertIn("24 months", result.answer.message)

    def test_no_evidence_never_releases_draft(self):
        result = parallel_answer(self.store, "Mars population", lambda q: "Mars has 500 people.")
        self.assertEqual(result.answer.message, ABSTENTION)

    def test_model_failure_still_returns_evidence(self):
        def generate(query):
            raise RuntimeError("model offline")
        result = parallel_answer(self.store, "Atlas warranty", generate)
        self.assertEqual(result.draft_status, "unavailable")
        self.assertIn("24 months", result.answer.message)

    def test_revision_change_during_generation_refuses(self):
        retrieved = threading.Event()
        original = self.store.answer
        def retrieve(query):
            answer = original(query)
            retrieved.set()
            return answer
        self.store.answer = retrieve
        def generate(query):
            self.assertTrue(retrieved.wait(1))
            self.store.remove("policy:atlas")
            return "The Atlas warranty lasts 24 months."
        result = parallel_answer(self.store, "Atlas warranty", generate)
        self.assertEqual(result.answer.status, "evidence_changed")
        self.assertEqual(result.answer.message, ABSTENTION)

    def test_timeout_and_busy_worker_never_fall_back_to_draft(self):
        released = threading.Event()
        original = self.store.answer
        entered = threading.Event()
        def retrieve(query):
            entered.set()
            released.wait(2)
            return original(query)
        self.store.answer = retrieve
        try:
            result = parallel_answer(self.store, "Atlas warranty", lambda q: "Unsupported claim",
                                     join_timeout=0.01)
            self.assertEqual(result.answer.status, "retrieval_timeout")
            calls = []
            busy = parallel_answer(self.store, "Atlas warranty", lambda q: calls.append(q))
            self.assertEqual(busy.answer.status, "retrieval_busy")
            self.assertFalse(calls)
        finally:
            released.set()
            # Wait for the bounded worker to release its slot before the next test.
            from parallel_rag import _RETRIEVAL_SLOT
            self.assertTrue(_RETRIEVAL_SLOT.acquire(timeout=2))
            _RETRIEVAL_SLOT.release()

    def test_conflicting_sources_do_not_get_hidden_by_matching_draft(self):
        self.store.ingest("The Atlas warranty lasts 12 months.", uri="policy:other",
                          title="Other policy", approved=True)
        result = parallel_answer(self.store, "Atlas warranty", lambda q: "The Atlas warranty lasts 24 months.")
        self.assertEqual(len(result.answer.evidence), 2)
        self.assertIn("12 months", result.answer.message)
        self.assertEqual(result.draft_status, "discarded")

    def test_read_only_query_does_not_initialize_missing_database(self):
        missing = Path(self.temp.name) / "absent" / "evidence.sqlite3"
        store = EvidenceStore(missing, read_only=True)
        result = parallel_answer(store, "Atlas warranty", lambda q: "Guess")
        self.assertEqual(result.answer.status, "retrieval_unavailable")
        self.assertFalse(missing.parent.exists())

    def test_model_adapter_makes_one_capped_worker_request(self):
        brain = EvidenceRAGMixin()
        calls = []
        brain._select_brain_for_mode = lambda mode, selected: selected
        brain._evict_and_load = lambda selected: calls.append(("load", selected))
        brain._tokenizer_prompt = lambda messages: "prompt"
        brain._compact_model_prompt = lambda prompt, budget: prompt
        brain._model_worker_request = lambda payload, timeout: calls.append((payload, timeout)) or "draft"
        brain._split_orchestrator_response = lambda draft: ("", draft)
        brain._strip_private_reasoning = lambda draft: draft
        self.assertEqual(brain._rag_model_draft("Atlas warranty", "kira"), "draft")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][0]["max_tokens"], 160)
        self.assertEqual(calls[1][1], 12)

    def test_model_adapter_does_not_retry_worker_failure(self):
        brain = EvidenceRAGMixin()
        calls = []
        brain._select_brain_for_mode = lambda mode, selected: selected
        brain._evict_and_load = lambda selected: None
        brain._tokenizer_prompt = lambda messages: "prompt"
        brain._compact_model_prompt = lambda prompt, budget: prompt
        def failed(payload, timeout):
            calls.append(payload)
            raise TimeoutError()
        brain._model_worker_request = failed
        with self.assertRaises(TimeoutError):
            brain._rag_model_draft("Atlas warranty", "orchestrator")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
