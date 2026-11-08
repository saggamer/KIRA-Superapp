from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from actions import KiraActionsMixin
from orchestrator_v15.kira_image_candidate import KiraImageCandidate
from orchestrator_v15 import horizon_image_tool as tool


class HorizonAppTests(unittest.TestCase):
    def test_generation_uses_horizon_expert_without_attachment_or_prompt_rewrite(self):
        import queue
        app=SimpleNamespace(_append_chat_message=Mock(),_stop_model_worker=Mock(),_record_artifact=Mock(),
            _extract_image_sources_from_prompt=lambda request:[],response_queue=queue.Queue(),agentic_workspace='/tmp',app_root='/app')
        for prompt in ['Generate an image of a blue whale teapot.','A blue ceramic whale teapot on a stone table.']:
            with patch.object(tool,'generate_image',return_value=Path('/tmp/generated.png')) as generate:
                self.assertTrue(tool.handle_direct_generate(app,prompt,[],'chat',prompt))
                self.assertEqual(generate.call_args.args[0],prompt)
        self.assertFalse(tool.handle_direct_generate(app,'Edit this photo.',[{'path':'/tmp/photo.png'}],'chat','prompt'))
        self.assertFalse(tool.handle_direct_generate(app,'Create a PDF report',[],'chat','prompt'))

    def test_generation_action_uses_horizon_and_labels_its_expert(self):
        app=SimpleNamespace(_image_edit_backend='horizon-1',agentic_workspace='/tmp',app_root='/app',
            _extract_blocks=lambda text,tag:['body'],_parse_fields=lambda body:{'prompt':'A blue teapot.'},
            _replace_block_once=lambda text,tag,body,result:result,_stop_model_worker=Mock())
        with patch.object(tool,'generate_image',return_value=Path('/tmp/result.png')) as generate, \
                patch('orchestrator_v15.z_image_tool.generate_image') as other:
            result=KiraActionsMixin._run_image_generation_blocks(app,'action')
        self.assertIn('Horizon 1 MLX creativity',result);generate.assert_called_once();other.assert_not_called()

    def test_generation_process_passes_no_source_image(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);python=root/'python';python.touch()
            def run(command,**kwargs):
                self.assertEqual(command[command.index('--mode')+1],'generate')
                self.assertNotIn('--source',command);self.assertNotIn('--constrain-predicted-region',command)
                self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'],'1')
                Image.new('RGB',(512,512),'blue').save(command[command.index('--output')+1])
                return SimpleNamespace(returncode=0)
            with patch.object(tool,'paths',return_value=(root/'bundle',root/'base')),patch.object(tool,'runtime_python',return_value=python), \
                    patch.object(tool,'validate_bundle'),patch.object(tool,'validate_base'),patch.object(tool.subprocess,'run',side_effect=run):
                self.assertTrue(tool.generate_image('A blue teapot.',root,root).is_file())

    def test_native_edit_preserves_user_instruction_without_v15_planning(self):
        import queue
        app=SimpleNamespace(_append_chat_message=Mock(),_stop_model_worker=Mock(),_record_artifact=Mock(),
            _extract_image_sources_from_prompt=lambda request:[],response_queue=queue.Queue(),
            agentic_workspace='/tmp',app_root='/app')
        prompt='Put a butterfly on the dog.'
        with patch.object(tool,'edit_image',return_value=Path('/tmp/result.png')) as edit:
            self.assertTrue(tool.handle_direct_edit(app,prompt,[{'path':'/tmp/source.png'}],'chat',prompt))
        self.assertEqual(edit.call_args.args[0],prompt)
        app._record_artifact.assert_called_once()
        self.assertFalse(tool.handle_direct_edit(app,'Make a PDF with this photo',[], 'chat','prompt'))

    def test_native_edit_accepts_non_edit_keyword_instructions(self):
        import queue
        app=SimpleNamespace(_append_chat_message=Mock(),_stop_model_worker=Mock(),_record_artifact=Mock(),
            _extract_image_sources_from_prompt=lambda request:[],response_queue=queue.Queue(),agentic_workspace='/tmp',app_root='/app')
        for prompt in ['Let the blanket hang over the bed.','Have there be a lobster on the bed.']:
            with self.subTest(prompt=prompt),patch.object(tool,'edit_image',return_value=Path('/tmp/result.png')) as edit:
                self.assertTrue(tool.handle_direct_edit(app,prompt,[{'path':'/tmp/source.png'}],'chat',prompt))
                self.assertEqual(edit.call_args.args[0],prompt)

    def test_learned_region_needs_no_guessed_rectangle(self):
        controller=KiraImageCandidate('Edit /tmp/source.png: add a butterfly to the photo')
        controller.image_backend='horizon-1'
        action='[IMAGE_EDIT]\nPATH: /tmp/source.png\nPROMPT: Add a butterfly to the dog.\n[/IMAGE_EDIT]'
        self.assertIsNone(controller.preflight(action))
        self.assertIn('predicts',controller.preflight(action.replace('[/IMAGE_EDIT]','REGION: center\n[/IMAGE_EDIT]')))

    def test_selected_horizon_dispatches_and_releases_planner(self):
        app=SimpleNamespace(_image_edit_backend='horizon-1',agentic_workspace='/tmp',app_root='/app',
            _extract_blocks=lambda text,tag:['body'],_parse_fields=lambda body:{'path':'/tmp/source.png','prompt':'Add a butterfly.'},
            _resolve_path=lambda path:path,_replace_block_once=lambda text,tag,body,result:result,
            _stop_model_worker=Mock())
        with patch.object(tool,'edit_image',return_value=Path('/tmp/result.png')) as edit, \
                patch('orchestrator_v15.z_image_tool.edit_image') as other:
            result=KiraActionsMixin._run_image_edit_blocks(app,'action')
        self.assertIn('Horizon 1 MLX',result);self.assertIn('result.predicted-mask.png',result)
        edit.assert_called_once();other.assert_not_called();app._stop_model_worker.assert_called_once()
        self.assertIsNone(edit.call_args.kwargs['strength'])

    def test_process_uses_local_weights_offline_and_validates_output(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.png';Image.new('RGB',(32,32),'white').save(source)
            python=root/'python';python.touch()
            def run(command,**kwargs):
                self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'],'1')
                self.assertIn('--base-path',command);self.assertIn('--constrain-predicted-region',command)
                output=Path(command[command.index('--output')+1]);Image.new('RGB',(512,512),'red').save(output)
                return SimpleNamespace(returncode=0)
            with patch.object(tool,'paths',return_value=(root/'bundle',root/'base')), \
                    patch.object(tool,'runtime_python',return_value=python), \
                    patch.object(tool,'validate_bundle'),patch.object(tool,'validate_base'), \
                    patch.object(tool.subprocess,'run',side_effect=run):
                self.assertTrue(tool.edit_image('Add red.',source,root,root).is_file())

    def test_missing_weights_fail_before_process_start(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.png';Image.new('RGB',(32,32)).save(source)
            with patch.object(tool,'validate_bundle',side_effect=FileNotFoundError()),patch.object(tool.subprocess,'run') as run:
                with self.assertRaisesRegex(FileNotFoundError,'horizon-only'):tool.edit_image('Add red.',source,root,root)
                run.assert_not_called()


if __name__=='__main__':unittest.main()
