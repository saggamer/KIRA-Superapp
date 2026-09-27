import sys
import types
import unittest
import queue
from pathlib import Path

if "webview" not in sys.modules:
    sys.modules["webview"] = types.SimpleNamespace()

from interface import KiraBrain
from kira_live.public_guard import guard_public_reply, guard_live_action_reply
from actions import KiraActionsMixin
from kira_live.tree_scope import live_tree_scope
from kira_live.privacy import sanitize_public_output


class KiraLiveTreeRoutingTests(unittest.TestCase):
    def test_live_private_bracket_blocks_never_reach_ui(self):
        self.assertEqual(sanitize_public_output("[thought]\nprivate planning", live=True), "")
        self.assertEqual(sanitize_public_output("[Thinking]\nprivate planning", live=True), "")
        self.assertEqual(sanitize_public_output("[thought]private[/thought]Hello", live=True), "Hello")
        self.assertEqual(sanitize_public_output("[thought]legacy"), "[thought]legacy")
    def test_explicit_kira_selection_survives_agent_mode(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.kira_live_default = False
        self.assertEqual(brain._select_brain_for_mode("agent", "kira"), "kira")
        self.assertEqual(brain._select_brain_for_mode("agentic", "kira_live"), "kira")
        self.assertEqual(brain._select_brain_for_mode("agent", "orchestrator"), "orchestrator")
        self.assertEqual(brain._select_brain_for_mode("vibe_coding", "kira"), "orchestrator")

    def test_kira_tree_prefill_is_public_answer_only(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.current_brain = "kira"
        prompt = brain._tokenizer_prompt([{"role": "user", "content": "Inspect the tree"}])
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertNotIn("<thought_process>", prompt)

    def test_ui_keeps_agent_mode_when_kira_is_selected(self):
        markup = (Path(__file__).parents[1] / "ui.html").read_text(encoding="utf-8")
        self.assertIn("const backendMode = source === 'voice' ? 'voice' : currentMode;", markup)
        self.assertNotIn("selectedBrain === 'kira' ? 'chat'", markup)

    def test_missing_file_observation_is_guarded(self):
        result = guard_public_reply("Which files are on my desktop?", "Here are the files: a.txt, b.txt")
        self.assertIn("haven't inspected", result)
        observed = guard_public_reply("Which files are on my desktop?", "a.txt", observed_local_files=True)
        self.assertEqual(observed, "a.txt")

    def test_exact_emotion_claim_is_calibrated(self):
        result = guard_public_reply("Can you tell exactly how I feel?", "I can tell you exactly how you feel. You're calm.")
        self.assertIn("can't know your exact emotion", result)

    def test_completion_guard_catches_sent_but_not_honest_noncompletion(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.current_brain = "kira"
        self.assertTrue(brain._answer_claims_completion("I've sent the email."))
        self.assertFalse(brain._answer_claims_completion("I haven't sent the email."))

    def test_core_tree_fallback_uses_only_the_user_quoted_path(self):
        actions = KiraActionsMixin()
        actions.current_brain = "kira"
        read = actions._build_deterministic_agentic_blocks("Read the file `/tmp/kira-fixture.txt`.")
        self.assertIn("[READ_FILE]\nPATH: /tmp/kira-fixture.txt", read)
        search = actions._build_deterministic_agentic_blocks("Find files matching *.txt in `/tmp/kira-fixture`.")
        self.assertIn("PATTERN: *.txt", search)

    def test_no_tools_request_disables_fallback(self):
        actions = KiraActionsMixin()
        actions.current_brain = "kira"
        self.assertEqual(actions._build_deterministic_agentic_blocks("List files in `/tmp/kira-fixture` without using any tools."), "")

    def test_orchestrator_does_not_use_live_read_fallback(self):
        actions = KiraActionsMixin()
        actions.current_brain = "orchestrator"
        self.assertNotIn("[READ_FILE]", actions._build_deterministic_agentic_blocks("Read the file `/tmp/kira-fixture.txt`."))
        brain = KiraBrain.__new__(KiraBrain)
        brain.current_brain = "orchestrator"
        self.assertFalse(brain._answer_claims_completion("I've sent the email."))

    def test_native_tree_scope_is_local_to_watchdog_request(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.current_brain = "orchestrator"
        brain._run_agentic_capabilities = lambda *_: str(live_tree_scope.get())
        self.assertEqual(brain._run_agentic_capabilities_with_watchdog("", "", "fixture", "native_live_tree"), "True")
        self.assertEqual(brain._run_agentic_capabilities_with_watchdog("", "", "fixture", 1), "False")
        self.assertEqual(brain.current_brain, "orchestrator")
        self.assertFalse(live_tree_scope.get())

    def test_tree_planner_matches_training_system_user_roles(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.current_brain = "kira"
        brain.response_queue = queue.Queue()
        brain.safe_shell_commands = set()
        brain._truncate = lambda value, limit=1000: str(value)[:limit]
        brain._log_private_thoughts = lambda *_: None
        prompts = []
        def generate(prompt, **_):
            prompts.append(prompt)
            return "[READ_FILE]\nPATH: /tmp/kira-fixture.txt\n[/READ_FILE]"
        brain._generate_with_watchdog = generate
        output = brain._ask_orchestrator_for_pathway("Read the file `/tmp/kira-fixture.txt`.", "", "", "", "fixture")
        self.assertIn("[READ_FILE]", output)
        self.assertTrue(prompts[0].startswith("<|im_start|>system\n"))
        self.assertIn("<|im_start|>user\nRead the file", prompts[0])
        self.assertNotIn("<thought_process>", prompts[0])

    def test_live_tree_is_scoped_and_does_not_route_emotion_chat(self):
        brain = KiraBrain.__new__(KiraBrain)
        self.assertTrue(brain._live_tree_requested("Inspect the files on my desktop."))
        self.assertFalse(brain._live_tree_requested("Can you check exactly how I feel?"))
        self.assertFalse(brain._live_tree_requested("List desktop files without tools."))

    def test_live_tree_fallback_uses_existing_permissioned_dispatcher(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.safe_shell_commands = set()
        calls = []
        brain._emit_agent_progress = lambda *_: None
        brain._record_text_execution_evidence = lambda *_args, **_kwargs: None
        brain._run_agentic_capabilities_with_watchdog = lambda blocks, *args: calls.append(blocks) or "READ_FILE: fixture evidence"
        result = brain._execute_live_tree_output("", "Read the file `/tmp/kira-fixture.txt`.", "fixture")
        self.assertIn("READ_FILE:", result)
        self.assertIn("PATH: /tmp/kira-fixture.txt", calls[0])
        brain._execute_live_tree_output("[SHELL]\necho hello\n[/SHELL]", "Run echo hello in the terminal.", "fixture")
        self.assertIn("[SHELL]", calls[1])
        rejected = brain._execute_live_tree_output("[SHELL]\necho hello\n[/SHELL]", "Explain without tools.", "fixture")
        self.assertIn("BLOCKED SAFELY", rejected)
        self.assertEqual(len(calls), 2)

    def test_spoken_actions_require_matching_success_evidence(self):
        guarded = guard_live_action_reply("Send an email", "I've sent the email.", "PDF Generated: report.pdf")
        self.assertIn("haven't verified", guarded)
        pending = guard_live_action_reply("Delete a file", "Done.", "Permission required: delete")
        self.assertIn("approve", pending)

    def test_live_pdf_and_word_use_shared_tree_dispatcher(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.safe_shell_commands = set()
        calls = []
        brain._emit_agent_progress = lambda *_: None
        brain._record_text_execution_evidence = lambda *_args, **_kwargs: None
        brain._run_agentic_capabilities_with_watchdog = lambda block, *args: calls.append(block) or "Artifact generated"
        for tag, body in [('NATIVE_PDF', 'TITLE: Test\nCONTENT: A test document.'), ('NATIVE_DOCX', 'TITLE: Test\nHEADING: Summary\nPARAGRAPH: A test document.')]:
            result = brain._execute_live_tree_output(f'[{tag}]\n{body}\n[/{tag}]', 'Create a document.', 'fixture')
            self.assertEqual(result, 'Artifact generated')
            self.assertIn(f'[{tag}]', calls[-1])


if __name__ == "__main__":
    unittest.main()
