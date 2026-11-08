import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from horizon_mlx.__main__ import main


class HorizonGenerationCLITests(unittest.TestCase):
    def test_source_free_cli_selects_creativity_and_writes_no_edit_masks(self):
        from PIL import Image
        recorded={}
        class FakeModel:
            def __init__(self,bundle,base_path,mode):recorded['mode']=mode
            def generate(self,prompt,seed,steps,guidance,width,height):
                recorded['prompt']=prompt
                return Image.new('RGB',(width,height),'blue'),{'mode':'text-to-image','source_image_supplied':False}
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'generated.png'
            args=['horizon-image','--bundle',folder,'--prompt','A blue ceramic teapot.','--output',str(output)]
            with patch('sys.argv',args),patch.dict(sys.modules,{'horizon_mlx.runtime':SimpleNamespace(HorizonMLX=FakeModel)}),contextlib.redirect_stdout(io.StringIO()):main()
            self.assertEqual(recorded,{'mode':'generate','prompt':'A blue ceramic teapot.'})
            metadata=json.loads(output.with_suffix('.json').read_text())
            self.assertNotIn('source_sha256',metadata)
            self.assertFalse(output.with_suffix('.predicted-mask.png').exists())
            self.assertTrue(metadata['output_sha256'])

    def test_generation_rejects_image_and_region_flags_before_model_loading(self):
        for extra in [['--mode','generate','--source','source.png'],['--constrain-predicted-region'],['--mode','edit']]:
            with self.subTest(extra=extra),patch('sys.argv',['horizon-image','--bundle','bundle','--prompt','A blue teapot.','--output','test.png',*extra]),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:main()
                self.assertEqual(error.exception.code,2)


if __name__=='__main__':unittest.main()
