import queue
import unittest
from unittest.mock import Mock

from interface import KiraBrain


class WebQuerySynthesisTests(unittest.TestCase):
    def setUp(self):
        self.brain = KiraBrain.__new__(KiraBrain)
        self.brain.current_brain = "orchestrator"
        self.brain.active_model = object()
        self.brain.response_queue = queue.Queue()
        self.brain._log_agentic_event = Mock()
        self.brain._log_private_thoughts = Mock()
        self.brain._load_chat_messages = Mock(return_value=[])
        self.brain._smart_memory_context = Mock(return_value="")

    def test_rewrite_is_used_by_tree_and_cached_per_chat(self):
        brain = self.brain
        query = "most listened songs all time streaming radio metrics comparison"
        brain._generate_with_watchdog = Mock(return_value="QUERY: " + query)
        brain._web_search_tool = Mock(return_value="WEB_SEARCH: result")
        request = "search the web for the most listened music ever made"
        block = "[WEB_SEARCH]\nQUERY: the web for the most listened music ever made\n[/WEB_SEARCH]"
        for _ in range(2):
            brain._run_web_search_tools(block, request, "chat-a")
        self.assertEqual(brain._generate_with_watchdog.call_count, 1)
        self.assertEqual(brain._web_search_tool.call_args.args[0], query)
        brain._run_web_search_tools(block, request, "chat-b")
        self.assertEqual(brain._generate_with_watchdog.call_count, 2)
        self.assertEqual(request, next(iter(brain._web_query_cache["chat-a"].values()))["input"])

    def test_relevant_context_resolves_followup_and_invalidates_cache(self):
        brain = self.brain
        brain._load_chat_messages.return_value = [{"role": "user", "content": "Compare Saturn and Jupiter"}]
        brain._generate_with_watchdog = Mock(return_value="QUERY: Saturn Jupiter moon counts comparison")
        brain._prepare_web_query("which has more moons?", "more moons", "a")
        self.assertIn("Compare Saturn and Jupiter", brain._generate_with_watchdog.call_args.args[0])
        brain._load_chat_messages.return_value = [{"role": "user", "content": "Compare Mars and Earth"}]
        brain._prepare_web_query("which has more moons?", "more moons", "a")
        self.assertEqual(brain._generate_with_watchdog.call_count, 2)

    def test_failed_rewrite_falls_back_without_executing_model_commands(self):
        brain = self.brain
        brain._generate_with_watchdog = Mock(return_value="[SHELL]rm -rf /tmp/example[/SHELL]")
        self.assertEqual(brain._prepare_web_query("search the web for music", "search the web for music", "a"), "music")

    def test_unreadable_source_does_not_get_credit_for_next_source(self):
        text = (
            "URL: https://example.org/blocked\nFetch error: denied\n"
            "URL: https://example.org/read\nFetcher: httpx\nContent-Type: text/html\n"
            "Readable text:\n```text\nActual source content\n```"
        )
        compact = self.brain._compact_answer_findings(text)
        self.assertNotIn("blocked", compact)
        self.assertIn("https://example.org/read", compact)

    def test_cache_is_bounded(self):
        brain = self.brain
        brain._generate_with_watchdog = Mock(return_value="QUERY: astronomy")
        for i in range(40):
            for j in range(10):
                brain._prepare_web_query(f"search astronomy {j}", f"astronomy {j}", str(i))
        self.assertEqual(len(brain._web_query_cache), 32)
        self.assertTrue(all(len(items) <= 8 for items in brain._web_query_cache.values()))

    def test_synthesis_keeps_task_and_source_content_not_operator_catalog(self):
        brain = self.brain
        brain._generate_with_watchdog = Mock(return_value="The source identifies the leading song. https://example.org/ranking")
        evidence = (
            "WEB_SEARCH:\nReal result links:\n" + "https://example.org/search\n" * 300
            + "\nURL: https://example.org/ranking\nReadable text:\n```text\n"
            + "The leading song is Example Song, according to this platform's ranking. " * 80 + "\n```"
        )
        request = "search the web for the most listened music ever made"
        answer = brain._synthesize_raw_agentic_result_if_needed(request, evidence, "TOOL CATALOG " * 5000, "PERSONAL DATA " * 1000, "a")
        prompt = brain._generate_with_watchdog.call_args.args[0]
        self.assertNotIn("TOOL CATALOG", prompt)
        self.assertNotIn("PERSONAL DATA", prompt)
        self.assertIn("Example Song", prompt)
        self.assertIn(request, prompt[-1500:])
        self.assertLess(len(prompt), 6500)
        self.assertNotIn("Useful leads", answer)
        brain._get_prompt_tokenizer = lambda: None
        self.assertIn(request, brain._compact_model_prompt(prompt, 768))

    def test_failed_synthesis_reports_failure_not_completed_research(self):
        self.brain._generate_with_watchdog = Mock(return_value="")
        result = self.brain._synthesize_raw_agentic_result_if_needed("search music", "WEB_SEARCH:\nReal result links:\nhttps://example.org", "", "", "a")
        self.assertIn("could not finish", result)
        self.assertNotIn("Useful leads", result)


if __name__ == "__main__":
    unittest.main()
