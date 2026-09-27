import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kira_live.weights import KIRA_LIVE_WEIGHT_SPECS, resolve_composite_weights


class PackagedWeightsTests(unittest.TestCase):
    def make_bundle(self, root):
        entries = {}
        for spec in KIRA_LIVE_WEIGHT_SPECS:
            directory = root / 'weights' / spec.role
            directory.mkdir(parents=True)
            config = {'architectures': [spec.architecture], 'model_type': spec.model_type,
                      spec.hidden_width_path[0]: {spec.hidden_width_path[1]: spec.hidden_width}}
            (directory / 'config.json').write_text(json.dumps(config))
            (directory / 'model.safetensors').write_bytes(b'test fixture, not actual tensors')
            entries[spec.role] = {'model_id': spec.model_id, 'revision': spec.revision,
                                  'path': f'weights/{spec.role}'}
        (root / 'bundle_manifest.json').write_text(json.dumps({'weights': entries}))
        return entries

    def test_bundled_islands_do_not_require_donor_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); self.make_bundle(root)
            with patch.dict(os.environ, {'KIRA_LIVE_BUNDLE_ROOT': str(root), 'HF_HUB_CACHE': str(root/'empty-cache')}):
                weights=resolve_composite_weights()
                for weight in weights.by_role():
                    self.assertEqual(weight.snapshot, root/'weights'/weight.spec.role)

    def test_revision_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); entries=self.make_bundle(root)
            entries['listener']['revision']='wrong'
            (root/'bundle_manifest.json').write_text(json.dumps({'weights':entries}))
            with patch.dict(os.environ, {'KIRA_LIVE_BUNDLE_ROOT': str(root)}):
                with self.assertRaisesRegex(ValueError, 'provenance'): resolve_composite_weights()

    def test_incomplete_bundle_does_not_fall_back_to_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); self.make_bundle(root)
            (root/'weights/listener/model.safetensors').unlink()
            with patch.dict(os.environ, {'KIRA_LIVE_BUNDLE_ROOT': str(root)}):
                with self.assertRaises(FileNotFoundError): resolve_composite_weights()
