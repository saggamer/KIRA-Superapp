import json
from pathlib import Path
import queue
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

from evidence_rag import ABSTENTION, EvidenceStore
from evidence_rag_runtime import EvidenceRAGMixin


class EvidenceRAGTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "evidence.sqlite3"
        self.store = EvidenceStore(self.path)

    def add(self, text="The Atlas warranty lasts 24 months.", **kwargs):
        return self.store.ingest(text, uri="https://example.test/warranty", title="Warranty",
                                 approved=kwargs.pop("approved", True), **kwargs)

    def test_exact_quotes_and_provenance_survive_restart(self):
        digest = self.add()
        answer = EvidenceStore(self.path).answer("What is the Atlas warranty?")
        self.assertEqual(answer.status, "source_excerpts")
        self.assertEqual(answer.evidence[0].sha256, digest)
        self.assertEqual(answer.evidence[0].text, "The Atlas warranty lasts 24 months.")
        self.assertIn("https://example.test/warranty", answer.message)
        self.assertIn("Expires:", answer.message)

    def test_unapproved_expired_and_unrelated_sources_abstain(self):
        self.add(approved=False)
        self.assertEqual(self.store.answer("Atlas warranty").message, ABSTENTION)
        self.add(now=time.time() - 86401, ttl_days=1)
        self.assertEqual(self.store.answer("Atlas warranty").message, ABSTENTION)
        self.add()
        self.assertEqual(self.store.answer("Mars population").message, ABSTENTION)
        self.assertEqual(self.store.answer("Atlas warranty cancellation fees").message, ABSTENTION)

    def test_replacement_and_deletion_remove_old_evidence(self):
        self.add()
        self.add("The Atlas warranty lasts 36 months.")
        answer = self.store.answer("Atlas warranty")
        self.assertIn("36 months", answer.message)
        self.assertNotIn("24 months", answer.message)
        self.store.remove("https://example.test/warranty")
        self.assertEqual(self.store.answer("Atlas warranty").message, ABSTENTION)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM chunk_search").fetchone()[0], 0)

    def test_model_cannot_emit_fabricated_claims(self):
        self.add()
        evidence_id = self.store.retrieve("Atlas warranty")[0].id
        answer = self.store.answer("Atlas warranty", selector=lambda prompt: json.dumps({
            "evidence_ids": [evidence_id], "answer": "Warranty is 999 months"}))
        self.assertNotIn("999", answer.message)
        self.assertIn("24 months", answer.message)
        for response in ('{"evidence_ids":["fake"]}', "Warranty is 999 months",
                         '{"evidence_ids":[]}', '{"evidence_ids":[{}]}', '[]'):
            with self.subTest(response=response):
                self.assertEqual(self.store.answer("Atlas warranty", selector=lambda p: response).message,
                                 ABSTENTION)

    def test_selector_failure_abstains(self):
        self.add()
        def broken(prompt):
            raise TimeoutError()
        self.assertEqual(self.store.answer("Atlas warranty", selector=broken).message, ABSTENTION)

    def test_conflicting_sources_remain_attributed(self):
        self.add()
        self.store.ingest("The Atlas warranty lasts 12 months.", uri="policy:other",
                          title="Other policy", approved=True)
        answer = self.store.answer("Atlas warranty")
        self.assertEqual(len(answer.evidence), 2)
        self.assertIn("24 months", answer.message)
        self.assertIn("12 months", answer.message)

    def test_tampered_document_or_index_is_rejected(self):
        self.add()
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE documents SET body='Atlas warranty 999 months'")
        self.assertEqual(self.store.answer("Atlas warranty").message, ABSTENTION)
        self.add()
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE chunk_search SET body='Atlas warranty 999 months'")
        self.assertEqual(self.store.answer("Atlas warranty").message, ABSTENTION)

    def test_chunks_preserve_original_offsets(self):
        text = ("Atlas warranty lasts 24 months.\n" * 100)
        self.add(text)
        for item in self.store.retrieve("Atlas warranty"):
            self.assertEqual(text[item.start:item.end], item.text)
            self.assertLessEqual(len(item.text), 900)

    def test_queries_are_bounded_and_fts_syntax_is_inert(self):
        self.add()
        for query in ("", "what is the", "* OR NOT ()", "x" * 4001,
                      " ".join("term" + str(i) for i in range(65))):
            self.assertEqual(self.store.answer(query).message, ABSTENTION)
        self.assertTrue(self.store.retrieve('"Atlas" warranty'))

    def test_invalid_ingest_leaves_existing_revision_intact(self):
        self.add()
        for ttl in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.add(ttl_days=ttl)
        self.assertIn("24 months", self.store.answer("Atlas warranty").message)

    def brain(self):
        brain = EvidenceRAGMixin()
        brain.app_root = self.temp.name
        brain.response_queue = queue.Queue()
        brain.messages = []
        brain._append_chat_message = lambda *args: brain.messages.append(args)
        return brain

    def test_desktop_routing_and_fail_closed(self):
        self.add()
        brain = self.brain()
        with patch.dict("os.environ", {"KIRA_RAG_DB": str(self.path), "KIRA_STRICT_RAG": "1"}):
            self.assertTrue(brain._handle_evidence_rag_prompt("Atlas warranty", "chat", "c1"))
            self.assertIn("24 months", brain.response_queue.get()["content"])
            self.assertFalse(brain._handle_evidence_rag_prompt("edit code", "vibe_coding", "c1"))
        with patch.dict("os.environ", {"KIRA_RAG_DB": str(self.path), "KIRA_STRICT_RAG": "0"}):
            self.assertTrue(brain._handle_evidence_rag_prompt("/rag Atlas warranty", "chat", "c1"))
            self.assertFalse(brain._handle_evidence_rag_prompt("normal chat", "chat", "c1"))
        self.path.write_bytes(b"invalid sqlite database")
        with patch.dict("os.environ", {"KIRA_RAG_DB": str(self.path)}):
            brain._handle_evidence_rag_prompt("/rag Atlas warranty", "chat", "c1")
            messages = []
            while not brain.response_queue.empty():
                messages.append(brain.response_queue.get()["content"])
            self.assertIn("could not be read", messages[-1])

    def test_real_router_runs_parallel_rag_before_tool_routing(self):
        from interface import KiraBrain
        brain = KiraBrain.__new__(KiraBrain)
        brain.response_queue = queue.Queue()
        brain.app_root = self.temp.name
        brain._append_chat_message = lambda *a: None
        calls = []
        brain._rag_model_draft = lambda question, selected: calls.append((question, selected)) or "The Atlas warranty lasts 24 months."
        self.add()
        with patch.dict("os.environ", {"KIRA_RAG_DB": str(self.path)}):
            brain._route_and_generate({"prompt": "/rag Atlas warranty", "chat_id": "c1"})
        self.assertIn("24 months", brain.response_queue.get()["content"])
        self.assertEqual(calls, [("Atlas warranty", "orchestrator")])

    def test_strict_chat_uses_selected_model_once(self):
        from interface import KiraBrain
        self.add()
        for selected in ("kira", "orchestrator"):
            brain = KiraBrain.__new__(KiraBrain)
            brain.response_queue = queue.Queue()
            brain.app_root = self.temp.name
            brain._append_chat_message = lambda *a: None
            calls = []
            brain._rag_model_draft = lambda question, selected: calls.append(selected) or "The Atlas warranty lasts 24 months."
            with patch.dict("os.environ", {"KIRA_RAG_DB": str(self.path), "KIRA_STRICT_RAG": "1"}):
                brain._route_and_generate({"prompt": "Atlas warranty", "chat_id": "c1",
                                           "mode": "chat", "selected_model": selected})
            self.assertIn("24 months", brain.response_queue.get()["content"])
            self.assertEqual(calls, [selected])


if __name__ == "__main__":
    unittest.main()
