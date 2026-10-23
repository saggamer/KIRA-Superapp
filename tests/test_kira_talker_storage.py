from pathlib import Path
import tempfile
import unittest

from kira_live.mlx_voice import local_model_view


class TalkerStorageTests(unittest.TestCase):
    def test_external_disk_metadata_is_hidden_without_modifying_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            codec = source / 'speech_tokenizer'
            codec.mkdir()
            weights = codec / 'model.safetensors'
            weights.write_bytes(b'original tensor data')
            sidecar = codec / '._model.safetensors'
            sidecar.write_bytes(b'filesystem metadata')
            view_path, view = local_model_view(str(source))
            self.addCleanup(view.cleanup)
            visible = list((Path(view_path) / 'speech_tokenizer').glob('*.safetensors'))
            self.assertEqual([item.name for item in visible], ['model.safetensors'])
            self.assertTrue(visible[0].is_symlink())
            self.assertEqual(visible[0].read_bytes(), weights.read_bytes())
            view.cleanup()
            self.assertTrue(weights.is_file())
            self.assertTrue(sidecar.is_file())

    def test_clean_model_needs_no_view(self):
        with tempfile.TemporaryDirectory() as directory:
            path, view = local_model_view(directory)
            self.assertEqual(path, directory)
            self.assertIsNone(view)
