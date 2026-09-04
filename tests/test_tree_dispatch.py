import ast
import inspect
import textwrap
import unittest
from unittest.mock import Mock

from actions import KiraActionsMixin


class TreeDispatchTests(unittest.TestCase):
    def setUp(self):
        self.actions = KiraActionsMixin()
        self.actions._load_chat_messages = Mock(return_value=[])

    def test_conversion_is_generation_not_existing_file_open(self):
        for prompt in (
            'turn this skit into a docx file and open it in word .',
            'turn this skit into a word file',
            'save this as a PDF',
            'put this in a document',
        ):
            with self.subTest(prompt=prompt):
                self.assertTrue(self.actions._wants_artifact_generation(prompt, 'chat'))
                self.assertEqual(self.actions._handle_artifact_followup(prompt, 'chat'), '')
        self.actions._load_chat_messages.assert_not_called()
        self.assertFalse(self.actions._wants_artifact_generation('open the document', 'chat'))

    def test_explicit_format_does_not_read_history_but_implicit_format_does(self):
        self.assertEqual(self.actions._infer_artifact_kind('create a docx', 'chat'), 'docx')
        self.assertTrue(self.actions._wants_artifact_generation('create a docx', 'chat'))
        self.actions._load_chat_messages.assert_not_called()
        self.actions._load_chat_messages.return_value = [{'role': 'user', 'content': 'create a ppt'}]
        self.assertTrue(self.actions._wants_artifact_generation('generate it', 'chat'))
        self.actions._load_chat_messages.assert_called_once()

    def prepare_dispatch(self):
        names = (
            '_unwrap_hybrid_actions', '_defer_artifact_until_web_evidence', '_run_os_surface_tools',
            '_run_read_tools', '_run_find_in_computer_blocks', '_run_utility_tools',
            '_run_web_search_tools', '_run_specialist_tools', '_generate_native_pdf',
            '_generate_native_pptx_blocks', '_generate_native_docx_blocks',
            '_queue_pdf_to_pptx_blocks', '_run_open_tools', '_run_scheduler_tools',
            '_run_permissioned_blocks', '_execute_agentic_code',
        )
        for name in names:
            setattr(self.actions, name, Mock(side_effect=lambda text, *args: text))
        self.actions._extract_bracket_command = Mock(return_value='')

    def test_dispatch_skips_absent_branches_and_retains_permission_handler(self):
        self.prepare_dispatch()
        source = '[ native_docx ]TITLE: Test[/ native_docx ]'
        self.actions._run_agentic_capabilities_inner(source, 'create docx', 'chat')
        self.actions._generate_native_docx_blocks.assert_called_once()
        self.actions._generate_native_pptx_blocks.assert_not_called()
        self.actions._run_os_surface_tools.assert_not_called()
        self.actions._run_scheduler_tools.assert_not_called()
        self.actions._run_permissioned_blocks.assert_called_once()

    def test_dispatch_reindexes_when_prior_handler_changes_output(self):
        self.prepare_dispatch()
        self.actions._run_web_search_tools.side_effect = lambda text, *args: text+'\n[NATIVE_PPTX]TITLE: Test[/NATIVE_PPTX]'
        self.actions._run_agentic_capabilities_inner('test', 'create ppt', 'chat')
        self.actions._generate_native_pptx_blocks.assert_called_once()

    def test_os_dispatch_tag_inventory_matches_handler(self):
        handler = ast.parse(textwrap.dedent(inspect.getsource(KiraActionsMixin._run_os_surface_tools)))
        tags = {node.args[1].value for node in ast.walk(handler)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == '_extract_blocks' and isinstance(node.args[1], ast.Constant)}
        dispatcher = ast.parse(textwrap.dedent(inspect.getsource(KiraActionsMixin._run_agentic_capabilities_inner)))
        gates = [node for node in ast.walk(dispatcher) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'has_tags']
        os_gate = next(node for node in gates if node.args[0].value == 'BRANCH_REGISTRY')
        self.assertEqual(tags, {arg.value for arg in os_gate.args})


if __name__ == '__main__':
    unittest.main()
