import queue
import sys
import types
import tempfile
import re
from pathlib import Path
import unittest

sys.modules.setdefault('webview', types.SimpleNamespace())
from interface import KiraBrain
from orchestrator_v15.effort import LEVELS, profile_thinking

SPEC = '''[NATIVE_PDF]
TITLE: Orchestrator v1.5 Information
HEADING: Purpose
PARAGRAPH: Orchestrator v1.5 is an everyday assistant in Kira Superapp. It uses available tools to research public information, inspect local data and create documents from supplied context.
HEADING: Tool use
PARAGRAPH: Tool results are checked before completion is reported. Browser workflows use observed page controls, and connected MCP tools follow the existing execution checks.
[/NATIVE_PDF]'''


class PdfAndEffortTests(unittest.TestCase):
    def test_no_browse_or_images_does_not_require_visual_research(self):
        from orchestrator_v15.kira_document_candidate import KiraDocumentCandidate
        for request in ('Create a PDF. Do not browse or add images.',
                        'Make a PDF without images.', 'Generate a PDF. Do not generate images.'):
            candidate = KiraDocumentCandidate(request, kind='pdf')
            self.assertFalse(candidate.needs_images)
            self.assertFalse(candidate.wants_generated_images)
            self.assertIsNone(candidate.preflight(candidate.normalize(SPEC)))
            brain = self.brain()
            brain._build_web_evidence_blocks = lambda *_a: self.fail('Offline PDF must not trigger web research')
            self.assertFalse(brain._prompt_needs_web_evidence_for_artifact(request))
            self.assertFalse(brain._pptx_visuals_requested(request))
            self.assertEqual(brain._defer_artifact_until_web_evidence(SPEC, request), SPEC)

    def test_duplicated_closing_tag_is_repaired_without_changing_content(self):
        from orchestrator_v15.kira_document_candidate import KiraDocumentCandidate
        candidate = KiraDocumentCandidate('Create a PDF', kind='pdf')
        valid = candidate.normalize(SPEC)
        malformed = valid.replace('[NATIVE_PDF]', '[/NATIVE_PDF]', 1)
        self.assertEqual(candidate.normalize(malformed), valid)

    def test_native_text_pdf_passes_for_requested_title_not_instruction_words(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = self.brain()
            brain.app_root = str(Path(__file__).parents[1])
            brain.agentic_workspace = directory
            brain._record_artifact = lambda *_a, **_kw: None
            request = ('Create a PDF titled Orchestrator v1.5 Information from supplied facts in this chat. '
                       'Use two short sections. Do not browse or add images.')
            from orchestrator_v15.kira_document_candidate import KiraDocumentCandidate
            spec = KiraDocumentCandidate(request, kind='pdf').normalize(SPEC)
            result = brain._generate_native_pdf(spec, raw_prompt=request)
            self.assertTrue(result.startswith('PDF generated:'), result)
            path = re.search(r'`([^`]+\.pdf)`', result).group(1)
            self.assertTrue(Path(path).is_file())
            wrong = brain._pdf_quality_report(path, 'Create a PDF titled Marine Biology.')
            self.assertFalse(wrong['ok'])
            self.assertIn('PDF content does not match the requested topic', wrong['reasons'])

    def brain(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.v15_model_tool_planning = True
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda *_a: None
        brain._truncate = lambda text, length=1000: str(text)[:length]
        brain._load_chat_messages = lambda *_a: []
        brain._smart_memory_context = lambda *_a, **_kw: ''
        brain._log_agentic_event = lambda *_a: None
        return brain

    def test_standard_v15_pdf_refusal_is_repaired_in_every_effort_level(self):
        for level in LEVELS:
            with self.subTest(level=level):
                brain = self.brain()
                replies = iter(['I cannot create PDFs.', SPEC, 'Done.'])
                payloads, actions = [], []
                def model(payload, timeout=60):
                    self.assertEqual(payload['type'], 'propose_v15')
                    payloads.append(payload)
                    return {'proposals': [next(replies)]*payload['branches']}
                brain._model_worker_request = model
                brain._run_agentic_capabilities_with_watchdog = lambda action, *_a: (
                    actions.append(action) or 'PDF generated: `/tmp/verified-document.pdf`')
                thinking = profile_thinking('orchestrator_v15', level)
                answer = brain._run_v15_profile('Create a PDF with its information from this chat.', 'pdf-test',
                                                thinking['effort'], structured_thinking=thinking['structured'],
                                                orchestral_candidate=thinking['orchestral'])
                self.assertEqual(len(actions), 1)
                self.assertIn('CONTENT:\n# Purpose', actions[0])
                self.assertIn('/tmp/verified-document.pdf', answer)
                self.assertIn('Native PDF and Word generation are available', payloads[0]['messages'][0]['content'])
                self.assertEqual(payloads[0]['structured_thinking'], level != 'low')
                self.assertEqual(payloads[0]['branches'], LEVELS[level].branches)
                self.assertEqual(payloads[0]['tool_channel'], 'presentation')

    def test_browser_effort_levels_reach_actual_worker_settings(self):
        for level in LEVELS:
            with self.subTest(level=level):
                brain = self.brain()
                payloads, actions = [], []
                navigate = '[BROWSER_TOOL]\nACTION: navigate\nURL: https://example.test\n[/BROWSER_TOOL]'
                def model(payload, timeout=60):
                    if payload['type'] == 'plan_v15_tools':
                        return '{"route":"browser","purpose":"Read the requested page."}'
                    payloads.append(payload)
                    answer = 'The page describes a weekly workshop.' if actions else navigate
                    return {'proposals': [answer]*payload['branches']}
                brain._model_worker_request = model
                brain._run_agentic_capabilities_with_watchdog = lambda action, *_a: (
                    actions.append(action) or 'BROWSER_TOOL:\n{"url":"https://example.test","title":"Workshop","text":"A weekly workshop.","observationId":"o1"}')
                thinking = profile_thinking('orchestrator_v15', level)
                answer = brain._run_v15_profile('Read https://example.test and summarize its key details.', 'effort-test',
                                                level, structured_thinking=thinking['structured'],
                                                orchestral_candidate=thinking['orchestral'])
                self.assertEqual(actions, [navigate])
                self.assertIn('workshop', answer)
                self.assertEqual(payloads[-1]['branches'], LEVELS[level].branches)
                self.assertEqual(payloads[-1]['structured_thinking'], level in {'medium','high'})
                self.assertEqual(payloads[-1]['diagonal_orchestral_candidate'], level == 'diagonal')

    def test_horizon_keeps_its_artifact_controller_and_legacy_aliases_work(self):
        horizon = profile_thinking('orchestrator_v15_ppt_preview', 'low')
        self.assertEqual(horizon['effort'], 'diagonal')
        self.assertTrue(horizon['artifacts'])
        self.assertTrue(profile_thinking('orchestrator_v15_orchestral_preview','medium')['orchestral'])


if __name__=='__main__':unittest.main()
