"""Exercise model setup without network access or loading weights."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "kira_download_models", Path(__file__).resolve().parents[1] / "scripts/download_models.py"
)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class DownloadAllTests(unittest.TestCase):
    def test_live_pins_donors_and_preserves_existing_addons(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "kira_live").mkdir()
            keys = ("thinker_ple_moe", "live_emotion_stage1", "live_semantic_bridge")
            manifest = {"training_status": {key: {"checkpoint": f"training_runs/{key}/model.safetensors"} for key in keys}}
            (root / "kira_live/model_manifest.json").write_text(json.dumps(manifest))
            existing = root / "training_runs/thinker_ple_moe/model.safetensors"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"local training checkpoint")
            source = root / "cached"
            source.write_bytes(b"published checkpoint")
            from unittest.mock import Mock
            hf_download = Mock(return_value=str(source))
            validate = Mock()
            weights = SimpleNamespace(
                KIRA_LIVE_WEIGHT_SPECS=[SimpleNamespace(role="thinker", model_id="owner/model", revision="pinned")],
                resolve_weight=Mock(),
            )
            with patch.object(downloader, "APP_ROOT", root), \
                    patch.object(downloader, "download") as download, \
                    patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=hf_download),
                        "kira_live.weights": weights,
                        "kira_live.checkpoint_bundle": SimpleNamespace(resolve_live_checkpoint=validate)}):
                downloader.download_live()
            download.assert_called_once_with("owner/model", revision="pinned")
            self.assertEqual(existing.read_bytes(), b"local training checkpoint")
            self.assertEqual(hf_download.call_count, 3)
            for call in hf_download.call_args_list:
                self.assertEqual(call.kwargs["revision"], "9b7d5c2f39daafd0bd9949414f06223d0d1b4f6f")
            validate.assert_called_once_with()

    def test_all_runs_each_group(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(downloader, "APP_ROOT", Path(folder)), \
                patch.object(downloader, "download", return_value=Path(folder)) as download, \
                patch.object(downloader, "validate_orchestrator"), \
                patch.object(downloader, "download_live") as live, \
                patch.object(downloader, "download_voice") as voice, \
                patch.object(downloader, 'download_horizon') as horizon, \
                patch("sys.argv", ["download_models.py", "--all"]):
            downloader.main()
            self.assertEqual(download.call_count, 2)
            live.assert_called_once_with()
            voice.assert_called_once_with()
            horizon.assert_called_once_with()

    def test_single_model_does_not_fetch_live_or_voice(self):
        with patch.object(downloader, "download", return_value=Path("cached")) as download, \
                patch.object(downloader, "download_live") as live, \
                patch.object(downloader, "download_voice") as voice, \
                patch("sys.argv", ["download_models.py", "--whisper-only"]):
            downloader.main()
            download.assert_called_once_with(downloader.WHISPER_REPO)
            live.assert_not_called()
            voice.assert_not_called()

    def test_all_rejects_partial_flags_before_network(self):
        for flag in ("--orchestrator-only", "--whisper-only", '--horizon-only'):
            with self.subTest(flag=flag), patch("sys.argv", ["download_models.py", "--all", flag]), \
                    patch.object(downloader, "download") as download, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    downloader.main()
                self.assertEqual(error.exception.code, 2)
                download.assert_not_called()

    def test_interrupted_kokoro_download_is_not_installed(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(downloader, "APP_ROOT", Path(folder)), \
                patch.object(downloader, "download"), \
                patch.object(downloader.urllib.request, "urlopen", side_effect=OSError("offline")):
            with self.assertRaises(OSError):
                downloader.download_voice()
            self.assertEqual(list((Path(folder) / "models/kokoro").iterdir()), [])

    def test_failed_download_exits_nonzero(self):
        with patch("sys.argv", ["download_models.py", "--whisper-only"]), \
                patch.object(downloader, "download", side_effect=OSError("offline")), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                downloader.main()
            self.assertEqual(error.exception.code, 1)

    def test_horizon_only_excludes_other_models(self):
        with patch('sys.argv',['download_models.py','--horizon-only']), \
                patch.object(downloader,'download_horizon') as horizon, \
                patch.object(downloader,'download') as download:
            downloader.main()
            horizon.assert_called_once_with();download.assert_not_called()

    def test_default_download_also_includes_horizon(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(downloader,'APP_ROOT',Path(folder)), \
                patch('sys.argv',['download_models.py']),patch.object(downloader,'download',return_value=Path(folder)), \
                patch.object(downloader,'validate_orchestrator'),patch.object(downloader,'download_horizon') as horizon:
            downloader.main();horizon.assert_called_once_with()

    def test_horizon_pins_both_downloads_and_checks_results(self):
        from unittest.mock import Mock
        from horizon_mlx import catalog
        with tempfile.TemporaryDirectory() as folder, patch.object(downloader,'APP_ROOT',Path(folder)), \
                patch('horizon_mlx.install.ensure_runtime') as runtime, \
                patch.object(catalog,'validate_bundle',side_effect=[FileNotFoundError(),None]), \
                patch.object(catalog,'validate_base',side_effect=[FileNotFoundError(),None]), \
                patch.object(downloader,'download') as download:
            downloader.download_horizon()
            self.assertEqual(download.call_args_list[0].args[0],catalog.BUNDLE_REPO)
            self.assertEqual(download.call_args_list[0].kwargs['revision'],catalog.BUNDLE_REVISION)
            self.assertEqual(download.call_args_list[1].args[0],catalog.BASE_REPO)
            self.assertEqual(download.call_args_list[1].kwargs['revision'],catalog.BASE_REVISION)
            runtime.assert_called_once()


if __name__ == "__main__":
    unittest.main()
