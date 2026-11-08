import base64
import sys
import tempfile
import types
import unittest
from pathlib import Path
from PIL import Image

sys.modules.setdefault('webview', types.SimpleNamespace())
from interface import KiraBrain, LazyKiraAPI


class ImagePreviewTests(unittest.TestCase):
    def test_workspace_image_returns_local_inline_data(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = KiraBrain.__new__(KiraBrain)
            brain.agentic_workspace = directory
            path = Path(directory) / 'generated image.png'
            Image.new('RGB', (48, 32), 'blue').save(path)
            result = brain.get_image_preview(str(path))
            self.assertTrue(result['ok'])
            self.assertEqual((result['width'], result['height']), (48, 32))
            self.assertEqual(result['name'], path.name)
            self.assertEqual(base64.b64decode(result['data_url'].split(',', 1)[1]), path.read_bytes())
            api = LazyKiraAPI()
            api._brain = brain
            self.assertEqual(api.get_image_preview(str(path)), result)

    def test_outside_paths_symlinks_and_nonimages_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'workspace'
            root.mkdir()
            brain = KiraBrain.__new__(KiraBrain)
            brain.agentic_workspace = str(root)
            outside = Path(directory) / 'private.png'
            Image.new('RGB', (10, 10)).save(outside)
            (root / 'link.png').symlink_to(outside)
            (root / 'fake.png').write_text('<script>not an image</script>')
            (root / 'vector.svg').write_text('<svg/>')
            for path in [outside, root / 'link.png', root / '../private.png', root / 'fake.png',
                         root / 'vector.svg', root / 'missing.png']:
                self.assertFalse(brain.get_image_preview(str(path))['ok'], str(path))

    def test_oversized_images_do_not_cross_the_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            brain = KiraBrain.__new__(KiraBrain)
            brain.agentic_workspace = directory
            path = Path(directory) / 'huge.png'
            with path.open('wb') as handle:
                handle.truncate(16 * 1024 * 1024 + 1)
            result = brain.get_image_preview(str(path))
            self.assertFalse(result['ok'])
            self.assertIn('limit', result['error'])


if __name__ == '__main__':
    unittest.main()
