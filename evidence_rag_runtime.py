"""Desktop adapter for the strict evidence answer path."""
import os
from pathlib import Path

from evidence_rag import ABSTENTION, EvidenceStore
from parallel_rag import parallel_answer


class EvidenceRAGMixin:
    def _rag_model_draft(self, query, selected_model):
        target = self._select_brain_for_mode("chat", selected_model)
        if target not in {"kira", "orchestrator"}:
            raise ValueError("This model profile has no single-pass draft adapter")
        self._evict_and_load(target)
        messages = [
            {"role": "system", "content": (
                "Give a concise factual answer in one sentence. If uncertain, say you don't know. "
                "Do not use tools or emit commands. A separate evidence check will validate the answer.")},
            {"role": "user", "content": query},
        ]
        # One worker request, with no recovery/regeneration pass in this path.
        draft = self._model_worker_request({
            "type": "generate",
            "prompt": self._compact_model_prompt(
                self._tokenizer_prompt(messages),
                int(getattr(self, "model_worker_max_prompt_tokens", 1800))),
            "temperature": 0.0,
            "max_tokens": min(160, int(getattr(self, "model_worker_max_output_tokens", 1100))),
        }, timeout=12)
        _, draft = self._split_orchestrator_response(draft)
        return self._strip_private_reasoning(draft).strip()

    def _handle_evidence_rag_prompt(self, raw_prompt, mode, chat_id, selected_model="orchestrator"):
        parts = raw_prompt.split(maxsplit=1)
        explicit = bool(parts and parts[0].lower() == "/rag")
        strict = os.environ.get("KIRA_STRICT_RAG", "").lower() in {"1", "true", "yes", "on"}
        if not explicit and not (strict and mode == "chat"):
            return False
        query = (parts[1] if len(parts) > 1 else "") if explicit else raw_prompt
        result = None
        parallel = None
        if not query.strip():
            message = "Use /rag followed by a question about your approved knowledge sources."
        else:
            try:
                path = os.environ.get("KIRA_RAG_DB") or (
                    Path(self.app_root) / "kira_agentic_workspace" / "knowledge" / "evidence.sqlite3")
                store = EvidenceStore(path, read_only=True)
                if os.environ.get("KIRA_RAG_EXECUTION", "parallel").lower() == "extractive":
                    result = store.answer(query)
                else:
                    parallel = parallel_answer(
                        store, query, lambda question: self._rag_model_draft(question, selected_model))
                    result = parallel.answer
                message = result.message
            except Exception:
                # A corrupt/unavailable index must never fall back to model guesses.
                message = ABSTENTION + " The knowledge index could not be read."
        self._append_chat_message(chat_id, "user", raw_prompt, "User")
        self._append_chat_message(chat_id, "assistant", message, "KIRA Evidence")
        self.response_queue.put({"type": "message", "content": message,
                                 "model_used": "KIRA Evidence", "chat_id": chat_id,
                                 "rag_status": result.status if result else "unavailable",
                                 "rag_execution": "parallel" if parallel else "extractive",
                                 "rag_metrics": ({"retrieval_ms": parallel.retrieval_ms,
                                                  "generation_ms": parallel.generation_ms,
                                                  "total_ms": parallel.total_ms,
                                                  "draft_status": parallel.draft_status} if parallel else {})})
        return True
