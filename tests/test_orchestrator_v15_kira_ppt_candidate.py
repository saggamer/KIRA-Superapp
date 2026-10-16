import unittest
import queue
import sys
import types

sys.modules.setdefault('webview', types.SimpleNamespace())

from orchestrator_v15.kira_ppt_candidate import KiraPPTCandidate
from interface import KiraBrain
from actions import KiraActionsMixin


class KiraPPTCandidateTests(unittest.TestCase):
    def test_separate_candidate_enforces_distinct_slide_titles(self):
        candidate = KiraPPTCandidate('Create a three slide presentation on project delivery')
        deck = ('[NATIVE_PPTX]\nTITLE: Project delivery\nTHEME: light blue\n'
                'SLIDE: Delivery\nBODY: Schedule and ownership\n'
                'SLIDE: Delivery\nBODY: Risks and mitigation\n[/NATIVE_PPTX]')
        self.assertIn('distinct', candidate.preflight(deck))

    def test_candidate_guidance_uses_kira_tools(self):
        candidate = KiraPPTCandidate('Create a presentation with images and sources')
        guidance = candidate.guidance()
        self.assertIn('WEB_IMAGE_SEARCH', guidance)
        self.assertIn('IMAGE_ANALYZE', guidance)
        self.assertIn('NATIVE_PPTX', guidance)

    def test_redundant_cover_slide_is_removed(self):
        candidate = KiraPPTCandidate('Create a three slide PowerPoint about a weekend plan')
        spec = ('[NATIVE_PPTX]\nTITLE: Weekend Plan\nTHEME: light blue\n'
                'SLIDE: Cover\nBODY: Weekend Plan\n'
                'SLIDE: Saturday outdoors\nBODY: Visit the park.\n'
                'SLIDE: Sunday indoors\nBODY: Visit the museum.\n[/NATIVE_PPTX]')
        normalized = candidate.normalize(spec)
        self.assertNotIn('SLIDE: Cover', normalized)
        self.assertIsNone(candidate.preflight(normalized))

    def test_nested_model_slide_format_is_flattened(self):
        candidate = KiraPPTCandidate('Create a three slide PowerPoint about a weekend plan')
        spec = ('[NATIVE_PPTX]\nTITLE: Weekend Plan\nTHEME: Default\n'
                'SLIDE:\n  TITLE: Saturday outdoors\n  BODY:\n    - Visit the park\n'
                'SLIDE:\n  TITLE: Sunday indoors\n  BODY:\n    - Visit the museum\n[/NATIVE_PPTX]')
        normalized = candidate.normalize(spec)
        self.assertIn('SLIDE: Saturday outdoors', normalized)
        self.assertIn('BULLET: Visit the museum', normalized)
        self.assertIsNone(candidate.preflight(normalized))

    def test_nested_visual_path_and_wrapping_prose_are_normalized(self):
        candidate = KiraPPTCandidate('Create a three slide PowerPoint with two images')
        answer = ('The deck is ready.\n[NATIVE_PPTX]\nTITLE: Coastal walk\nTHEME: light blue\n'
                  'SLIDE:\n  TITLE: Plan the route\n  BODY: Follow the shoreline.\n'
                  '  VISUAL:\n    PATH: /tmp/coast.png\n'
                  'SLIDE:\n  TITLE: Pause by the sea\n  BODY: Take a rest.\n'
                  '  VISUAL:\n    PATH: /tmp/bench.png\n[/NATIVE_PPTX]')
        normalized = candidate.normalize(answer)
        self.assertTrue(normalized.startswith('[NATIVE_PPTX]'))
        self.assertIn('IMAGE: /tmp/coast.png', normalized)
        self.assertIn('IMAGE: /tmp/bench.png', normalized)
        self.assertNotIn('VISUAL:', normalized)

    def test_exact_three_slide_request_rejects_extra_content_slide(self):
        candidate = KiraPPTCandidate('Create a three slide PowerPoint about the coast')
        spec = ('[NATIVE_PPTX]\nTITLE: Coast\nTHEME: light blue\n'
                'SLIDE: First\nBODY: Coast\nSLIDE: Second\nBODY: Path\n'
                'SLIDE: Third\nBODY: Bench\n[/NATIVE_PPTX]')
        self.assertIn('exactly 2 content slides', candidate.preflight(spec))

    def test_indented_image_field_is_renderer_ready(self):
        candidate = KiraPPTCandidate('Create a three slide PowerPoint about nature')
        spec = ('[NATIVE_PPTX]\nTITLE: Nature\nTHEME: light blue\n'
                'SLIDE: Coast\nBODY: The coast is open.\n  IMAGE: /tmp/coast.png\n'
                'SLIDE: Forest\nBODY: The forest is shaded.\n[/NATIVE_PPTX]')
        self.assertIn('\nIMAGE: /tmp/coast.png\n', candidate.normalize(spec))

    def test_kira_profile_executes_verified_deck(self):
        brain = KiraBrain.__new__(KiraBrain)
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda _model: None
        brain._load_chat_messages = lambda _chat_id: []
        brain._truncate = lambda value, length=1000: str(value)[:length]
        brain._extract_agentic_blocks = lambda _value: []
        spec = ('[NATIVE_PPTX]\nTITLE: Weekend Plan\nTHEME: light blue\n'
                'SLIDE: Saturday outdoors\nBODY: Visit the park.\n'
                'SLIDE: Sunday indoors\nBODY: Visit the museum.\n[/NATIVE_PPTX]')
        replies = [spec, 'The deck has four content slides, plus a cover.']
        brain._model_worker_request = lambda *_args, **_kwargs: {
            'proposals': [replies.pop(0)], 'token_counts': [100], 'memo_tokens': 0}
        executed = []
        brain._run_agentic_capabilities_with_watchdog = lambda action, *_args: (
            executed.append(action) or 'PPTX generated: `/tmp/weekend.pptx`\nQuality verified: 3 slides.')
        answer = brain._run_v15_profile('Create a three slide PowerPoint about a weekend plan.',
                                        'ppt-preview-test', 'diagonal', ppt_candidate=True)
        self.assertIn('/tmp/weekend.pptx', answer)
        self.assertNotIn('four content slides', answer)
        self.assertEqual(executed, [spec])

    def test_specialist_quality_respects_requested_slide_count(self):
        actions = KiraActionsMixin.__new__(KiraActionsMixin)
        spec = 'TITLE: Weekend\nSLIDE: Saturday\nBODY: Park\nSLIDE: Sunday\nBODY: Museum'
        reasons = actions._artifact_spec_quality_reasons(
            spec, 'pptx', 'Create a three slide PowerPoint about a weekend plan')
        self.assertFalse(any('fewer than' in reason for reason in reasons))
        bullet_plan = spec + '\nBULLET: Plan: Visit the local park'
        self.assertFalse(any('plan:' in reason for reason in
                             actions._artifact_spec_quality_reasons(
                                 bullet_plan, 'pptx', 'Create a three slide PowerPoint')))

    def test_image_prompt_cannot_be_slide_content(self):
        candidate = KiraPPTCandidate('Create a three slide PowerPoint about a coastal walk')
        spec = ('[NATIVE_PPTX]\nTITLE: Coastal walk\nTHEME: light blue\n'
                'SLIDE: The route\nBODY: Generate an image of a sunlit coastal path.\n'
                'SLIDE: A rest stop\nBODY: Pause near the water.\n[/NATIVE_PPTX]')
        self.assertIn('image-making instruction', candidate.preflight(spec))
        actions = KiraActionsMixin.__new__(KiraActionsMixin)
        reasons = actions._artifact_spec_quality_reasons(
            spec, 'pptx', 'Create a three slide PowerPoint about a coastal walk')
        self.assertIn('image-making instructions leaked into slide content', reasons)

    def test_explicit_one_image_does_not_require_two(self):
        actions = KiraActionsMixin.__new__(KiraActionsMixin)
        self.assertEqual(actions._pptx_min_visuals(
            'Create a three slide PowerPoint with one edited image'), 1)
        self.assertEqual(actions._pptx_min_visuals(
            'Create an image-rich PowerPoint'), 2)


if __name__ == '__main__':
    unittest.main()
