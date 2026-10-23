import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from actions import KiraActionsMixin
from orchestrator_v15.kira_document_candidate import KiraDocumentCandidate
from orchestrator_v15.kira_ppt_candidate import KiraPPTCandidate
from orchestrator_v15.z_image_tool import edit_image, generate_image


class ZImageArtifactPreviewTests(unittest.TestCase):
    def test_kira_action_returns_generated_path_as_tool_evidence(self):
        bridge = KiraActionsMixin.__new__(KiraActionsMixin)
        bridge.app_root = '/tmp/kira-app'
        bridge.agentic_workspace = '/tmp/kira-artifacts'
        block = '[IMAGE_GENERATE]\nPROMPT: A botanical conservatory at sunrise\nSEED: 7\n[/IMAGE_GENERATE]'
        with patch('orchestrator_v15.z_image_tool.generate_image',
                   return_value=Path('/tmp/kira-artifacts/generated_images/visual.png')) as generate:
            result = bridge._run_image_generation_blocks(block)
        self.assertIn('IMAGE_GENERATED:', result)
        self.assertIn('/tmp/kira-artifacts/generated_images/visual.png', result)
        generate.assert_called_once_with('A botanical conservatory at sunrise',
                                         '/tmp/kira-artifacts', '/tmp/kira-app', seed='7')

    def test_generated_image_enters_analysis_and_artifact_evidence(self):
        for controller in (
            KiraPPTCandidate('Create a three slide PowerPoint with one original generated image'),
            KiraDocumentCandidate('Create a PDF with one original generated image', kind='pdf'),
            KiraDocumentCandidate('Create a Word document with one original generated image', kind='docx'),
        ):
            with self.subTest(kind=type(controller).__name__ + getattr(controller, 'kind', 'pptx')):
                action = ('[IMAGE_GENERATE]\nPROMPT: A clear illustration of a forest path, '
                          'no labels or lettering\n[/IMAGE_GENERATE]')
                self.assertIsNone(controller.preflight(action))
                with tempfile.TemporaryDirectory() as directory:
                    image = Path(directory) / 'forest.png'
                    image.write_bytes(b'fixture')
                    controller.record(action, f'IMAGE_GENERATED:\nPath: `{image}`\nModel: Z-Image Turbo Q4 (MLX)')
                    analyze = controller.next_image_analysis_action()
                    self.assertIn(str(image), analyze)
                    controller.record(analyze, f'IMAGE_ANALYSIS:\nPath: `{image}`\nSubject: forest path')
                    self.assertIn(str(image), controller.analyzed_image_paths)
                    if isinstance(controller, KiraPPTCandidate):
                        artifact = ('[NATIVE_PPTX]\nTITLE: Forest guide\nTHEME: light green\n'
                                    f'SLIDE: A clear route\nBODY: Walk through the tree cover.\nIMAGE: {image}\n'
                                    'SLIDE: Prepare for weather\nBODY: Carry layers and water.\n[/NATIVE_PPTX]')
                    elif controller.kind == 'pdf':
                        artifact = (f'[NATIVE_PDF]\nTITLE: Forest guide\nCONTENT:\n# Route\n'
                                    f'A walk beneath the trees.\nIMAGE: {image}\n[/NATIVE_PDF]')
                    else:
                        artifact = (f'[NATIVE_DOCX]\nTITLE: Forest guide\nHEADING: Route\n'
                                    f'PARAGRAPH: A walk beneath the trees.\nIMAGE: {image}\n[/NATIVE_DOCX]')
                    self.assertIsNone(controller.preflight(artifact))

    def test_generation_uses_fixed_local_command_and_checks_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            (model / 'transformer').mkdir(parents=True)
            (model / 'transformer' / 'model.safetensors.index.json').write_text('{}')
            exe = root / '.venv-z-image-turbo' / 'bin' / 'mlxgen'
            exe.parent.mkdir(parents=True)
            exe.write_text('')

            def fake_run(command, **kwargs):
                Image.new('RGB', (128, 96), '#888').save(command[command.index('--output') + 1])
                return type('Result', (), {'returncode': 0})()

            with patch.dict(os.environ, {'KIRA_Z_IMAGE_MODEL_DIR': str(model)}), patch(
                    'orchestrator_v15.z_image_tool.subprocess.run', side_effect=fake_run) as run:
                image = generate_image('An original landscape with no words', root, root)
            self.assertTrue(image.is_file())
            command = run.call_args.args[0]
            self.assertEqual(command[:2], [str(exe), 'generate'])
            self.assertEqual(command[command.index('--steps') + 1], '8')
            self.assertEqual(command[command.index('--model') + 1], str(model))

    def test_edit_uses_observed_source_and_mask_without_strength(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            (model / 'transformer').mkdir(parents=True)
            (model / 'transformer' / 'model.safetensors.index.json').write_text('{}')
            exe = root / '.venv-z-image-turbo' / 'bin' / 'mlxgen'
            exe.parent.mkdir(parents=True)
            exe.write_text('')
            source, mask = root / 'source.png', root / 'mask.png'
            Image.new('RGB', (128, 96), '#ccc').save(source)
            Image.new('L', (128, 96), 255).save(mask)

            def fake_run(command, **kwargs):
                Image.new('RGB', (128, 96), '#888').save(command[command.index('--output') + 1])
                return type('Result', (), {'returncode': 0})()

            with patch.dict(os.environ, {'KIRA_Z_IMAGE_MODEL_DIR': str(model)}), patch(
                    'orchestrator_v15.z_image_tool.subprocess.run', side_effect=fake_run) as run:
                edited = edit_image('Add a bench', source, root, root, mask=mask)
            command = run.call_args.args[0]
            self.assertTrue(edited.is_file())
            self.assertEqual(command[command.index('--image') + 1], str(source.resolve()))
            self.assertEqual(command[command.index('--mask-path') + 1], str(mask.resolve()))
            self.assertNotIn('--image-strength', command)

    def test_edit_controller_requires_observed_path(self):
        candidate = KiraPPTCandidate('Create a presentation and edit the image with one visual')
        action = '[IMAGE_EDIT]\nPATH: /tmp/unseen.png\nPROMPT: Add a bench\n[/IMAGE_EDIT]'
        self.assertIn('Edit only', candidate.preflight(action))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.png'
            edited = Path(directory) / 'edited.png'
            source.write_bytes(b'fixture')
            edited.write_bytes(b'fixture')
            candidate.evidence_text += f'\nPath: `{source}`'
            action = f'[IMAGE_EDIT]\nPATH: {source}\nPROMPT: Add a bench\nREGION: lower_right\n[/IMAGE_EDIT]'
            self.assertIsNone(candidate.preflight(action))
            candidate.record(action, f'IMAGE_EDITED:\nPath: `{edited}`\nSource: `{source}`')
            self.assertIn(str(edited), candidate.next_image_analysis_action())

    def test_region_edit_builds_soft_mask_for_mlx_inpainting(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            (model / 'transformer').mkdir(parents=True)
            (model / 'transformer' / 'model.safetensors.index.json').write_text('{}')
            exe = root / '.venv-z-image-turbo' / 'bin' / 'mlxgen'
            exe.parent.mkdir(parents=True)
            exe.write_text('')
            source = root / 'source.png'
            Image.new('RGB', (128, 96), '#ccc').save(source)

            def fake_run(command, **kwargs):
                Image.new('RGB', (128, 96), '#888').save(command[command.index('--output') + 1])
                return type('Result', (), {'returncode': 0})()

            with patch.dict(os.environ, {'KIRA_Z_IMAGE_MODEL_DIR': str(model)}), patch(
                    'orchestrator_v15.z_image_tool.subprocess.run', side_effect=fake_run) as run:
                edit_image('Add a wooden bench', source, root, root, region='lower_right')
            command = run.call_args.args[0]
            self.assertIn('--mask-path', command)
            self.assertNotIn('--image-strength', command)
            with Image.open(command[command.index('--mask-path') + 1]) as mask:
                self.assertEqual(mask.size, (128, 96))
                self.assertLess(mask.getpixel((8, 8)), mask.getpixel((110, 70)))

    def test_artifact_cannot_skip_requested_image_generation_or_edit(self):
        generated = KiraPPTCandidate('Create a three slide presentation with one original generated image')
        edited = KiraPPTCandidate('Create a three slide presentation and edit the image')
        spec = ('[NATIVE_PPTX]\nTITLE: Coast\nTHEME: light blue\n'
                'SLIDE: The route\nBODY: Follow the shoreline.\n'
                'SLIDE: A pause\nBODY: Rest by the water.\n[/NATIVE_PPTX]')
        self.assertIn('IMAGE_GENERATE', generated.preflight(spec))
        self.assertIn('IMAGE_EDIT', edited.preflight(spec))

    def test_object_addition_requires_localized_region(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.png'
            from PIL import Image
            Image.new('RGB', (128, 96), '#ccc').save(source)
            candidate = KiraPPTCandidate(
                f'Create a presentation and edit the image `{source}` to add a bench',
                context=f'Path: `{source}`')
            action = f'[IMAGE_EDIT]\nPATH: {source}\nPROMPT: Add a wooden bench beside the path\n[/IMAGE_EDIT]'
            self.assertIn('REGION or', candidate.preflight(action))
            self.assertIsNone(candidate.preflight(action.replace('[/IMAGE_EDIT]',
                                                                  'REGION: lower_right\n[/IMAGE_EDIT]')))

    def test_slide_cannot_attribute_added_bench_to_unedited_image(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'original.png'
            edited = Path(directory) / 'edited.png'
            source.write_bytes(b'fixture')
            edited.write_bytes(b'fixture')
            candidate = KiraPPTCandidate(
                f'Create a three slide PowerPoint with two images; edit `{source}` to add a bench',
                context=f'Path: `{source}`')
            edit_action = (f'[IMAGE_EDIT]\nPATH: {source}\nPROMPT: Add a small wooden bench '
                           'beside the path\n[/IMAGE_EDIT]')
            candidate.record(edit_action, f'IMAGE_EDITED:\nPath: `{edited}`\nSource: `{source}`')
            candidate.analyzed_image_paths.update({str(source), str(edited)})
            candidate.image_analysis_observed = True
            spec = ('[NATIVE_PPTX]\nTITLE: Coastal walk\nTHEME: light blue\n'
                    f'SLIDE: Route\nBODY: Follow the coast.\nIMAGE: {edited}\n'
                    f'SLIDE: Bench rest stop\nBODY: The bench provides a pause.\nIMAGE: {source}\n'
                    '[/NATIVE_PPTX]')
            self.assertIn('edited image', candidate.preflight(spec))

    def test_rejects_prompt_with_embedded_command_or_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                generate_image('hello\n--output /tmp/other.png', directory, directory)
            with self.assertRaises(FileNotFoundError):
                generate_image('A simple illustration', directory, directory)


if __name__ == '__main__':
    unittest.main()
