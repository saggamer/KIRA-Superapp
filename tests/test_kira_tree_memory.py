import tempfile
from pathlib import Path
import unittest
from concurrent.futures import ThreadPoolExecutor

from kira_live.tree_memory import KiraTreeMemory
from actions import KiraActionsMixin


class KiraTreeMemoryTests(unittest.TestCase):
    def test_checkpoint_survives_restart_and_is_scoped_to_chat(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tree.sqlite3"
            memory = KiraTreeMemory(path)
            run_id = memory.start_tree("chat-a", "Create a report about glaciers")
            memory.record_step(
                "chat-a", run_id, 1, "[WEB_RESEARCH]glaciers[/WEB_RESEARCH]",
                "WEB_RESEARCH: Source: https://example.org/glaciers\nVerified retreat rate: 2m/year",
            )
            memory.finish_tree("chat-a", run_id, "Research gathered.", status="completed")

            restored = KiraTreeMemory(path)
            self.assertEqual(restored.latest_tree("chat-a"), run_id)
            self.assertIsNone(restored.latest_tree("chat-b"))
            checkpoint = restored.tree_context("chat-a", run_id)
            self.assertIn("glaciers", checkpoint)
            self.assertIn("https://example.org/glaciers", checkpoint)
            self.assertIn("evidence #2", checkpoint)
            self.assertIn("Run outcome [completed]", checkpoint)
            self.assertNotIn("glaciers", restored.tree_context("chat-b", run_id))

            restored.start_tree("chat-a", "New request awaiting its first branch")
            self.assertEqual(restored.latest_tree("chat-a"), run_id)

            actions = KiraActionsMixin()
            actions.tree_memory = restored
            self.assertIn("glaciers", actions._tree_memory_tool("chat-a"))
            self.assertIn("Verified retreat rate", actions._tree_memory_tool("chat-a", "EVENT: 2"))
            self.assertIn("No execution tree", actions._tree_memory_tool("chat-b"))

    def test_large_observation_is_bounded_but_exact_event_is_recoverable(self):
        with tempfile.TemporaryDirectory() as folder:
            memory = KiraTreeMemory(Path(folder) / "tree.sqlite3")
            run_id = memory.start_tree("chat", "Inspect build")
            full = "Build started\n" + "x" * 18000 + "\nERROR: build failed\nhttps://example.org/log"
            status = memory.record_step("chat", run_id, 1, "[PROJECT_COMMAND]test[/PROJECT_COMMAND]", full)
            checkpoint = memory.tree_context("chat", run_id, max_chars=1900)
            self.assertEqual(status, "failed")
            self.assertLessEqual(len(checkpoint), 1900)
            self.assertIn("failed", checkpoint)
            self.assertIn("https://example.org/log", checkpoint)
            self.assertIn("Pruned", checkpoint)
            self.assertEqual(list(memory.events("chat"))[1].content, full)
            page = memory.tree_event("chat", run_id, 2, offset=17000)
            self.assertIn("https://example.org/log", page)
            self.assertLessEqual(len(page), 3700)

    def test_permission_and_secret_are_not_promoted_to_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            memory = KiraTreeMemory(Path(folder) / "tree.sqlite3")
            run_id = memory.start_tree("chat", "Write file")
            status = memory.record_step(
                "chat", run_id, 1, "[WRITE_FILE]...[/WRITE_FILE]",
                "Permission required: api_key=sk-example-super-secret-value",
            )
            checkpoint = memory.tree_context("chat", run_id)
            self.assertEqual(status, "pending_permission")
            self.assertIn("pending_permission", checkpoint)
            self.assertNotIn("sk-example", checkpoint)
            self.assertNotIn("sk-example", list(memory.events("chat"))[1].content)

    def test_source_text_cannot_turn_an_observation_into_a_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            memory = KiraTreeMemory(Path(folder) / "tree.sqlite3")
            run_id = memory.start_tree("chat", "Research error budgets")
            self.assertEqual(
                memory.record_step("chat", run_id, 1, "[WEB_SEARCH]...[/WEB_SEARCH]",
                                   "WEB_SEARCH: Article title: Why projects failed\nResult: useful source"),
                "observed",
            )

    def test_browser_tool_error_is_a_failed_tree_step(self):
        with tempfile.TemporaryDirectory() as folder:
            memory = KiraTreeMemory(Path(folder) / "tree.sqlite3")
            run_id = memory.start_tree("chat", "Open a page")
            status = memory.record_step(
                "chat", run_id, 1, "[BROWSER_TOOL]ACTION: navigate[/BROWSER_TOOL]",
                "BROWSER_TOOL ERROR: HTTP 404 at https://example.org/missing",
            )
            self.assertEqual(status, "failed")
            self.assertIn("Step 1 [failed] BROWSER_TOOL", memory.tree_context("chat", run_id))

    def test_concurrent_branches_get_distinct_event_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            memory = KiraTreeMemory(Path(folder) / "tree.sqlite3")
            run_id = memory.start_tree("chat", "Read two sources")
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(
                    lambda number: memory.record_step(
                        "chat", run_id, number, "[WEB_FETCH][/WEB_FETCH]", f"source {number}"
                    ),
                    range(1, 13),
                ))
            sequences = [event.sequence for event in memory.events("chat")]
            self.assertEqual(sequences, list(range(1, 14)))


if __name__ == "__main__":
    unittest.main()
