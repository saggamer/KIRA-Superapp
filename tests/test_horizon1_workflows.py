"""Regression cases for observable image edits and cross-format completion."""
from pathlib import Path
import os
import queue
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault('webview', types.SimpleNamespace())

from PIL import Image, ImageChops

from orchestrator_v15.kira_image_candidate import KiraImageCandidate, is_image_request
from orchestrator_v15.kira_document_candidate import KiraDocumentCandidate
from orchestrator_v15.kira_ppt_candidate import KiraPPTCandidate
from orchestrator_v15.z_image_tool import edit_image


class HorizonWorkflowTests(unittest.TestCase):
    def test_standalone_image_routing(self):
        for request in ('Edit /tmp/photo.png to add a bench', 'Generate a watercolor image',
                        'Retouch this photo', 'Remove the background from this image',
                        'Make this photo brighter', 'Add a tree to this image'):
            self.assertTrue(is_image_request(request))
        for request in ('Remove the background from this image', 'Make this photo brighter',
                        'Add a tree to this image'):
            self.assertTrue(KiraImageCandidate(request).wants_edited_images)
        for request in ('Read this image', 'Explain image editing', 'What is a photo?'):
            self.assertFalse(is_image_request(request))

    def test_only_output_analysis_completes_standalone_edit(self):
        controller = KiraImageCandidate('Edit /tmp/source.png to add a bench')
        analysis = '[IMAGE_ANALYZE]\nPATH: /tmp/source.png\n[/IMAGE_ANALYZE]'
        controller.record(analysis, 'IMAGE_ANALYSIS:\nPath: `/tmp/source.png`')
        self.assertFalse(controller.verified_path)
        edit = '[IMAGE_EDIT]\nPATH: /tmp/source.png\nPROMPT: Add a bench\nREGION: right\n[/IMAGE_EDIT]'
        self.assertIsNone(controller.preflight(edit))
        controller.record(edit, 'IMAGE_EDITED:\nPath: `/tmp/output.png`')
        self.assertIn('Analyze', controller.preflight(edit))
        controller.record('[IMAGE_ANALYZE]\nPATH: /tmp/output.png\n[/IMAGE_ANALYZE]',
                          'IMAGE_ANALYSIS error: unavailable')
        self.assertFalse(controller.verified_path)
        controller.record('[IMAGE_ANALYZE]\nPATH: /tmp/output.png\n[/IMAGE_ANALYZE]',
                          'IMAGE_ANALYSIS:\nPath: `/tmp/output.png`')
        self.assertEqual(controller.verified_path, '/tmp/output.png')

    def test_failed_generation_and_download_cannot_complete_image_task(self):
        controller = KiraImageCandidate('Generate an image')
        controller.record('[WEB_IMAGE_DOWNLOAD]\nURL: https://example.invalid/a.png\n[/WEB_IMAGE_DOWNLOAD]',
                          'Path: `/tmp/source.png`')
        controller.record('[IMAGE_GENERATE]\nPROMPT: A forest\n[/IMAGE_GENERATE]',
                          'IMAGE_GENERATE error: unavailable')
        controller.record('[IMAGE_ANALYZE]\nPATH: /tmp/source.png\n[/IMAGE_ANALYZE]',
                          'IMAGE_ANALYSIS:\nPath: `/tmp/source.png`')
        self.assertFalse(controller.verified_path)

    def test_edit_contract_rejects_conflicting_mask_and_bad_strength(self):
        controller = KiraImageCandidate('Edit /tmp/source.png', context='Mask: /tmp/mask.png')
        for fields in ('MASK: /tmp/mask.png\nREGION: right', 'STRENGTH: nan',
                       'STRENGTH: 1', 'STRENGTH: bad'):
            action = f'[IMAGE_EDIT]\nPATH: /tmp/source.png\nPROMPT: Warm the colors\n{fields}\n[/IMAGE_EDIT]'
            self.assertIsNotNone(controller.preflight(action))

    def test_explicit_image_counts_and_deterministic_analysis_order(self):
        for count in ('3', 'three'):
            controller = KiraPPTCandidate(f'Create slides with {count} original generated images')
            self.assertEqual(controller.required_images, 3)
        self.assertEqual(KiraImageCandidate('Generate one image').required_images, 1)

    def test_edit_result_must_be_embedded_in_ppt_pdf_and_word(self):
        for kind in ('pptx', 'pdf', 'docx'):
            request = 'Create a three slide presentation and edit the image /tmp/source.png offline'
            controller = (KiraPPTCandidate(request) if kind == 'pptx' else
                          KiraDocumentCandidate(request, kind=kind))
            controller.record('[IMAGE_EDIT]\nPATH: /tmp/source.png\nPROMPT: Warm colors\n[/IMAGE_EDIT]',
                              'IMAGE_EDITED:\nPath: `/tmp/edited.png`')
            controller.analyzed_image_paths.update(('/tmp/source.png', '/tmp/edited.png'))
            controller.image_analysis_observed = True
            if kind == 'pptx':
                spec = ('[NATIVE_PPTX]\nTITLE: Coast\nTHEME: light blue\nSLIDE: Coastal route\n'
                        'BODY: Follow the path.\nIMAGE: /tmp/source.png\nSLIDE: Preparation\n'
                        'BODY: Bring water.\n[/NATIVE_PPTX]')
            elif kind == 'pdf':
                spec = '[NATIVE_PDF]\nTITLE: Coast\nCONTENT:\n# Route\nFollow the path.\nIMAGE: /tmp/source.png\n[/NATIVE_PDF]'
            else:
                spec = '[NATIVE_DOCX]\nTITLE: Coast\nPARAGRAPH: Follow the path.\nIMAGE: /tmp/source.png\n[/NATIVE_DOCX]'
            with self.subTest(kind=kind):
                self.assertIn('IMAGE_EDIT result', controller.preflight(spec))
                self.assertIsNone(controller.preflight(spec.replace('IMAGE: /tmp/source.png', 'IMAGE: /tmp/edited.png')))

    def test_pdf_and_word_reject_visible_image_instructions(self):
        for kind, field in (('pdf', 'CONTENT'), ('docx', 'PARAGRAPH')):
            controller = KiraDocumentCandidate('Create a guide offline', kind=kind)
            tag = controller.output_tag
            spec = f'[{tag}]\nTITLE: Guide\n{field}: Generate an image of a forest with no text.\n[/{tag}]'
            self.assertIn('image-making instruction', controller.preflight(spec))

    def test_local_edit_preserves_canvas_alpha_and_pixels_outside_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, mask, output = (root / name for name in ('source.png', 'mask.png', 'output.png'))
            original = Image.new('RGBA', (80, 160), (10, 30, 90, 123))
            original.save(source)
            selection = Image.new('L', original.size, 0)
            selection.paste(255, (40, 60, 70, 100))
            selection.save(mask)
            Image.new('RGB', (64, 128), 'red').save(output)
            with patch('orchestrator_v15.z_image_tool._run_image_model', return_value=output):
                edit_image('Add a red object', source, root, root, mask=mask)
            with Image.open(output) as result:
                self.assertEqual(result.size, original.size)
                self.assertEqual(result.getpixel((0, 0)), original.getpixel((0, 0)))
                self.assertEqual(result.getpixel((50, 80)), (255, 0, 0, 123))
                difference = ImageChops.difference(result.convert('RGB'), original.convert('RGB'))
                self.assertEqual(difference.getbbox(), (40, 60, 70, 100))

    def test_invalid_rasters_and_empty_or_wrong_size_masks_fail_before_diffusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, mask = root / 'source.png', root / 'mask.png'
            Image.new('RGB', (80, 160), 'blue').save(source)
            for size, value in (((80, 160), 0), ((160, 80), 255)):
                Image.new('L', size, value).save(mask)
                with patch('orchestrator_v15.z_image_tool._run_image_model') as run:
                    with self.assertRaises(ValueError):
                        edit_image('Change colors', source, root, root, mask=mask)
                    run.assert_not_called()
            source.write_bytes(b'broken')
            with self.assertRaises(OSError):
                edit_image('Change colors', source, root, root)

    def test_horizon_profile_executes_standalone_generation_and_analysis(self):
        sys.modules.setdefault('webview', types.SimpleNamespace())
        from interface import KiraBrain
        brain = KiraBrain.__new__(KiraBrain)
        brain.v15_model_tool_planning = False  # This fixture tests execution after tool selection.
        brain.response_queue = queue.Queue()
        brain._evict_and_load = lambda _model: None
        brain._load_chat_messages = lambda _chat_id: []
        brain._truncate = lambda value, length=1000: str(value)[:length]
        brain._extract_agentic_blocks = lambda _value: []
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'output.png'
            replies = iter(('[IMAGE_GENERATE]\nPROMPT: A quiet forest\n[/IMAGE_GENERATE]', 'Done.'))
            brain._model_worker_request = lambda *_args, **_kwargs: {
                'proposals': [next(replies)], 'token_counts': [50], 'memo_tokens': 0}
            actions = []
            def execute(action, *_args):
                actions.append(action)
                if action.startswith('[IMAGE_GENERATE]'):
                    Image.new('RGB', (80, 80), 'green').save(image)
                    return f'IMAGE_GENERATED:\nPath: `{image}`'
                return f'IMAGE_ANALYSIS:\nPath: `{image}`\nSubject: Forest'
            brain._run_agentic_capabilities_with_watchdog = execute
            answer = brain._run_v15_profile('Generate one image of a forest', 'horizon-test', 'diagonal', ppt_candidate=True)
            self.assertIn(str(image), answer)
            self.assertIn('analyzed', answer)
            self.assertEqual(len(actions), 2)

    def test_missing_pdf_image_blocks_completion_even_if_text_passes(self):
        from tests.test_tree_web_artifacts import HarnessActions
        with tempfile.TemporaryDirectory() as directory:
            bridge = HarnessActions(Path(directory))
            spec = ('[NATIVE_PDF]\nTITLE: Coastal guide\nCONTENT:\n# Coastal route\n'
                    'Follow the marked coastal route and bring water.\nIMAGE: /missing/coast.png\n[/NATIVE_PDF]')
            with patch.object(bridge, '_pdf_quality_report', return_value={'ok': True, 'reasons': []}), patch.object(
                    bridge, '_materialize_pptx_image', return_value=(None, 'Image unavailable')), patch.object(
                    bridge, '_record_artifact') as record:
                result = bridge._generate_native_pdf(spec)
            self.assertIn('did not pass verification', result)
            self.assertNotIn('PDF generated:', result)
            record.assert_not_called()

    def test_unicode_pdf_text_is_preserved_or_fails_explicitly(self):
        from tests.test_tree_web_artifacts import HarnessActions
        import fitz
        with tempfile.TemporaryDirectory() as directory:
            bridge = HarnessActions(Path(directory))
            spec = ('[NATIVE_PDF]\nTITLE: Evening café\nCONTENT:\n# Opening hours\n'
                    'The evening café opens at 18:40. Admission is €5 per person. '
                    'Bring a coat for the outdoor viewing session. Visibility depends on clouds, '
                    'so check the conditions before leaving and allow time to arrive before the first session.\n[/NATIVE_PDF]')
            result = bridge._generate_native_pdf(spec)
            if 'PDF Error:' in result:
                self.assertIn('Unicode font', result)
            else:
                self.assertIn('PDF generated:', result)
                path = result.split('`')[1]
                with fitz.open(path) as document:
                    text = ''.join(page.get_text() for page in document)
                self.assertIn('€5', text)
                self.assertIn('café', text)

    def test_exif_inputs_are_normalized_and_temporary_copies_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            (model / 'transformer').mkdir(parents=True)
            (model / 'transformer/model.safetensors.index.json').write_text('{}')
            exe = root / '.venv-z-image-turbo/bin/mlxgen'
            exe.parent.mkdir(parents=True)
            exe.write_text('')
            source = root / 'portrait.jpg'
            exif = Image.Exif()
            exif[274] = 6
            Image.new('RGB', (80, 160), 'blue').save(source, exif=exif)
            normalized = []
            def run(command, **_kwargs):
                staged = Path(command[command.index('--image') + 1])
                self.assertNotEqual(staged, source)
                with Image.open(staged) as picture:
                    self.assertEqual(picture.size, (160, 80))
                normalized.append(staged)
                self.assertEqual(command[command.index('--width') + 1], '160')
                self.assertEqual(command[command.index('--height') + 1], '80')
                Image.new('RGB', (160, 80), 'red').save(command[command.index('--output') + 1])
                return types.SimpleNamespace(returncode=0)
            with patch.dict(os.environ, {'KIRA_Z_IMAGE_MODEL_DIR': str(model)}), patch(
                    'orchestrator_v15.z_image_tool.subprocess.run', side_effect=run):
                result = edit_image('Warm the colors', source, root, root)
            self.assertTrue(normalized)
            self.assertTrue(all(not path.exists() for path in normalized))
            with Image.open(result) as picture:
                self.assertEqual(picture.size, (160, 80))

    def test_authored_training_targets_pass_and_evaluation_topics_are_separate(self):
        from scripts.prepare_horizon1_crossformat import authored_rows
        from scripts.evaluate_horizon1_workflows import cases
        rows = authored_rows()
        self.assertEqual(len(rows), 54)
        training_requests = '\n'.join(row['messages'][1]['content'] for row in rows)
        for case_name, controller, *_rest in cases():
            self.assertNotIn(controller.request, training_requests, case_name)

    def test_light_deck_has_readable_accent_and_no_unrequested_producer_footer(self):
        from tests.test_tree_web_artifacts import HarnessActions
        from pptx import Presentation
        with tempfile.TemporaryDirectory() as directory:
            bridge = HarnessActions(Path(directory))
            result = bridge._generate_native_pptx(
                'TITLE: Coastal walk\nTHEME: light blue\nSLIDE: The coastal route\nBODY: '
                'Follow the marked coastal path and allow time to pause by the shoreline. '
                'Check access conditions locally before leaving and bring water for the walk.\n'
                'SLIDE: Prepare for weather\nBODY: '
                'Choose layers for changing conditions, check the local forecast, and agree on '
                'a return time. Keep the route within the distance your group can comfortably walk.')
            self.assertIn('PPTX generated:', result)
            presentation = Presentation(result.split('`')[1])
            self.assertFalse(any('Orchestrator V1' in shape.text for slide in presentation.slides
                                 for shape in slide.shapes if shape.has_text_frame))
            title = next(shape for shape in presentation.slides[1].shapes
                         if shape.has_text_frame and shape.text == 'The coastal route')
            foreground = title.text_frame.paragraphs[0].font.color.rgb
            def luminance(rgb):
                parts = [v / 255 for v in rgb]
                parts = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in parts]
                return sum(value * weight for value, weight in zip(parts, (0.2126, 0.7152, 0.0722)))
            contrast = (luminance((248, 250, 252)) + 0.05) / (luminance(foreground) + 0.05)
            self.assertGreaterEqual(contrast, 4.5)


if __name__ == '__main__':
    unittest.main()
