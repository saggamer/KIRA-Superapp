"""Control methods tested without loading audio devices or MLX weights."""
import ast
from pathlib import Path
import threading
import types
import unittest


def native_controls():
    path = Path(__file__).parents[1] / "kira_live/native_session.py"
    tree = ast.parse(path.read_text())
    native = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "KiraNativeLiveSession")
    methods = [node for node in native.body if isinstance(node, ast.FunctionDef) and node.name in {"set_microphone_muted", "interrupt_response"}]
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(path), "exec"), scope)
    return scope


class LiveControlsTests(unittest.TestCase):
    def fixture(self):
        counts = {"reset": 0, "interrupt": 0, "playback": 0, "finished": 0}
        def count(name):
            return lambda: counts.__setitem__(name, counts[name] + 1)
        events = []
        session = types.SimpleNamespace(_capture_lock=threading.Lock(), _mic_muted=False,
            duplex=types.SimpleNamespace(segmenter=types.SimpleNamespace(reset=count("reset")), generation=types.SimpleNamespace(interrupt=count("interrupt"))),
            stop_playback=count("playback"), control=types.SimpleNamespace(response_finished=count("finished")),
            _emit=lambda kind: events.append(kind))
        return session, counts, events

    def test_mute_and_resume_flush_capture_without_reloading_models(self):
        session, counts, events = self.fixture()
        method = native_controls()["set_microphone_muted"]
        self.assertEqual(method(session, True), {"ok": True, "muted": True})
        self.assertEqual(method(session, False), {"ok": True, "muted": False})
        self.assertEqual(counts["reset"], 2)
        self.assertEqual(events, ["MIC_MUTED", "MIC_RESUMED"])

    def test_interrupt_cancels_tokens_and_playback(self):
        session, counts, events = self.fixture()
        self.assertTrue(native_controls()["interrupt_response"](session)["ok"])
        self.assertEqual(counts["interrupt"], 1)
        self.assertEqual(counts["playback"], 1)
        self.assertEqual(counts["finished"], 1)
        self.assertEqual(events, ["RESPONSE_INTERRUPTED"])


if __name__ == "__main__":
    unittest.main()
