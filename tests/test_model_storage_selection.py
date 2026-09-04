import json
import os
import sys


APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from interface import KiraBrain


def _brain_without_init():
    return object.__new__(KiraBrain)


def _make_model(path, *, allocate=True):
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    weight = path / "model.safetensors"
    if allocate:
        weight.write_bytes(b"x" * (1024 * 1024 + 1))
    else:
        with open(weight, "wb") as handle:
            handle.truncate(1024 * 1024 + 1)
    return path


def test_first_existing_path_skips_dataless_model(tmp_path):
    brain = _brain_without_init()
    cloud_model = _make_model(tmp_path / "cloud", allocate=False)
    local_model = _make_model(tmp_path / "local", allocate=True)

    selected = brain._first_existing_path([str(cloud_model), str(local_model)])

    assert selected == str(local_model)


def test_sharded_model_residency_uses_index(tmp_path):
    brain = _brain_without_init()
    model = tmp_path / "sharded"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    shard = model / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"x" * (1024 * 1024 + 1))
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": shard.name}}),
        encoding="utf-8",
    )

    assert brain._is_loadable_model_path(str(model))
    assert brain._is_model_storage_resident(str(model))
