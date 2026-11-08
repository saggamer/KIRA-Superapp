"""Check that the optional cloud bundle never carries known sensitive text."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_gemma_2ba_colab_export import build, reasons, sha


class ColabExportTests(unittest.TestCase):
    def test_drops_whole_sensitive_records_and_preserves_token_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / 'source', root / 'output'
            source.mkdir()
            (source / 'manifest.json').write_text('{}')
            clean = {'id': 'clean', 'source': 'synthetic',
                     'messages': [{'role': 'user', 'content': 'Click Search'}],
                     'token_ids': [1, 2, 3], 'answer_start': 2}
            unsafe = {**clean, 'id': 'unsafe', 'messages': [
                {'role': 'user', 'content': 'Send to person@example.com'}]}
            for split in ('train', 'valid', 'agentnet_valid', 'temporal_valid'):
                with (source / f'{split}.jsonl').open('w') as out:
                    out.write(json.dumps(clean) + '\n')
                    out.write(json.dumps(unsafe) + '\n')
            report = build(source, output)
            self.assertEqual(report['files']['train']['rows'], 1)
            self.assertEqual(json.loads((output / 'train.jsonl').read_text())['token_ids'], [1, 2, 3])
            self.assertEqual(report['files']['train']['sha256'], sha(output / 'train.jsonl'))
            self.assertIn('email', reasons(unsafe))


if __name__ == '__main__':
    unittest.main()
