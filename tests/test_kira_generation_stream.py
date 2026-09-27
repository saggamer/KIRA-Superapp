import unittest
from kira_live.generation_stream import bind_generation_stream

generation_stream = 'original'
def fixture(*, tokens=3):
    yield generation_stream, tokens


class StreamBindingTests(unittest.TestCase):
    def test_invocation_local_stream_preserves_global_and_defaults(self):
        bound = bind_generation_stream(fixture, 'worker')
        self.assertEqual(list(bound()), [('worker', 3)])
        self.assertEqual(list(fixture()), [('original', 3)])
        self.assertEqual(list(bound(tokens=5)), [('worker', 5)])
