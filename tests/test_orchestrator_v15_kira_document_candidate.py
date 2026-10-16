import queue
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault('webview', types.SimpleNamespace())

from interface import KiraBrain
from orchestrator_v15.kira_document_candidate import KiraDocumentCandidate, document_kind_for_request
from orchestrator_v15.kira_ppt_candidate import KiraPPTCandidate


class KiraDocumentCandidateTests(unittest.TestCase):
    def test_document_routing(self):
        self.assertEqual(document_kind_for_request('Create a PDF briefing'), 'pdf')
        self.assertEqual(document_kind_for_request('Create a PDF presentation'), 'pdf')
        self.assertEqual(document_kind_for_request('Make a Word document'), 'docx')
        self.assertIsNone(document_kind_for_request('Read this PDF'))

    def test_pdf_field_normalization(self):
        candidate = KiraDocumentCandidate('Create a PDF guide', kind='pdf')
        spec = '[NATIVE_PDF]\nTITLE: Local Guide\nHEADING: Overview\nPARAGRAPH: Useful context.\nBULLET: First step.\n[/NATIVE_PDF]'
        normalized = candidate.normalize(spec)
        self.assertIn('CONTENT:\n# Overview', normalized)
        self.assertIsNone(candidate.preflight(normalized))

    def test_prose_before_file_gets_specific_repair_contract(self):
        candidate = KiraDocumentCandidate('Create a PDF guide', kind='pdf')
        correction = candidate.preflight('I will make the PDF now.')
        self.assertIn('[NATIVE_PDF]', correction)
        self.assertIn('[/NATIVE_PDF]', correction)

    def test_local_image_path_quoting_is_removed_for_renderers(self):
        path = '/tmp/sample photo.png'
        pdf = KiraDocumentCandidate('Create a PDF with an image', kind='pdf')
        ppt = KiraPPTCandidate('Create a presentation with an image')
        self.assertIn(f'IMAGE: {path}', pdf.normalize(
            f'[NATIVE_PDF]\nTITLE: Guide\nCONTENT:\nIMAGE: `{path}`\n[/NATIVE_PDF]'))
        self.assertIn(f'IMAGE: {path}', ppt.normalize(
            f'[NATIVE_PPTX]\nTITLE: Guide\nSLIDE: First\nBODY: Text\nIMAGE: `{path}`\n[/NATIVE_PPTX]'))

    def test_image_must_be_analyzed_before_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'example image.png'
            path.write_bytes(b'fixture')
            candidate = KiraDocumentCandidate(
                f'Create an image-rich PDF using `{path}` offline', kind='pdf')
            spec = f'[NATIVE_PDF]\nTITLE: Image guide\nCONTENT:\n# Overview\nExample.\nIMAGE: {path}\n[/NATIVE_PDF]'
            self.assertIn('Analyze', candidate.preflight(spec))
            action = candidate.next_image_analysis_action()
            self.assertIn(str(path), action)
            candidate.record(action, f'IMAGE_ANALYSIS:\nPath: `{path}`\nSubject: test image')
            self.assertIsNone(candidate.preflight(spec))

    def test_two_images_need_two_content_slides(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f'example{index}.png' for index in (1, 2)]
            candidate = KiraPPTCandidate(
                'Create a three slide PowerPoint with two images',
                context='\n'.join(map(str, paths)))
            candidate.image_research_observed = True
            candidate.image_analysis_observed = True
            candidate.analyzed_image_paths.update(map(str, paths))
            spec = ('[NATIVE_PPTX]\nTITLE: Examples\nTHEME: light blue\n'
                    f'SLIDE: First example\nBODY: First.\nIMAGE: {paths[0]}\nIMAGE: {paths[1]}\n'
                    'SLIDE: Second example\nBODY: Second.\n[/NATIVE_PPTX]')
            self.assertIn('distinct content slides', candidate.preflight(spec))

    def test_downloaded_images_are_analyzed_in_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f'visual_{index}.png' for index in (1, 2)]
            for path in paths:
                path.write_bytes(b'fixture')
            candidate = KiraPPTCandidate('Create slides with two images')
            result = ('WEB_IMAGE_SEARCH:\nDownloaded images: 2\n' +
                      '\n'.join(f'- Path: `{path}`' for path in paths))
            candidate.record('[WEB_IMAGE_SEARCH]\nQUERY: visuals\n[/WEB_IMAGE_SEARCH]', result)
            first = candidate.next_image_analysis_action()
            self.assertIsNotNone(first)
            first_path = next(path for path in paths if str(path) in first)
            candidate.record(first, f'IMAGE_ANALYSIS:\nPath: `{first_path}`\nSubject: fixture')
            second = candidate.next_image_analysis_action()
            self.assertIsNotNone(second)
            self.assertNotEqual(first, second)
            self.assertIsNone(candidate.next_image_analysis_action())

    def test_profile_executes_pdf_and_word_actions(self):
        for kind, spec, result, suffix in (
            ('pdf', '[NATIVE_PDF]\nTITLE: Guide\nCONTENT:\n# Start\nUseful guidance.\n[/NATIVE_PDF]',
             'PDF generated: `/tmp/guide.pdf`', '.pdf'),
            ('docx', '[NATIVE_DOCX]\nTITLE: Guide\nHEADING: Start\nPARAGRAPH: Useful guidance.\n[/NATIVE_DOCX]',
             'DOCX generated: `/tmp/guide.docx`\nQuality verified', '.docx'),
        ):
            with self.subTest(kind=kind):
                brain = KiraBrain.__new__(KiraBrain)
                brain.response_queue = queue.Queue()
                brain._evict_and_load = lambda _model: None
                brain._load_chat_messages = lambda _chat_id: []
                brain._truncate = lambda value, length=1000: str(value)[:length]
                brain._extract_agentic_blocks = lambda _value: []
                replies = [spec, 'Done.']
                brain._model_worker_request = lambda *_args, **_kwargs: {
                    'proposals': [replies.pop(0)], 'token_counts': [100], 'memo_tokens': 0}
                actions = []
                brain._run_agentic_capabilities_with_watchdog = lambda action, *_args: (
                    actions.append(action) or result)
                answer = brain._run_v15_profile(f'Create a {kind.upper()} guide',
                                                f'{kind}-preview-test', 'diagonal', ppt_candidate=True)
                self.assertIn(suffix, answer)
                self.assertEqual(actions, [spec])


if __name__ == '__main__':
    unittest.main()
