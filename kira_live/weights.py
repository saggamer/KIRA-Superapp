from __future__ import annotations

"""Resolve and validate the three pinned KIRA Live weight islands.

This module never downloads a model.  It makes the native runtime fail closed
when a checkpoint is missing or its architecture no longer matches the bridge
contract.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WeightSpec:
    role: str
    model_id: str
    revision: str
    architecture: str
    model_type: str
    hidden_width: int
    hidden_width_path: tuple[str, ...]

    @property
    def cache_name(self) -> str:
        return "models--" + self.model_id.replace("/", "--")


@dataclass(frozen=True)
class ResolvedWeightIsland:
    spec: WeightSpec
    snapshot: Path
    config: dict[str, Any]

    @property
    def model_file(self) -> Path:
        files = self.model_files
        if not files:
            raise FileNotFoundError(f"No safetensors weights in {self.snapshot}")
        return files[0]

    @property
    def model_files(self) -> tuple[Path, ...]:
        single = self.snapshot / "model.safetensors"
        if single.is_file():
            return (single,)
        return tuple(sorted(self.snapshot.glob("model.safetensors-*.safetensors")))


@dataclass(frozen=True)
class CompositeWeightPaths:
    listener: ResolvedWeightIsland
    thinker: ResolvedWeightIsland
    talker: ResolvedWeightIsland

    def by_role(self) -> tuple[ResolvedWeightIsland, ...]:
        return (self.listener, self.thinker, self.talker)


KIRA_LIVE_WEIGHT_SPECS = (
    WeightSpec(
        role="listener",
        model_id="Qwen/Qwen3-ASR-0.6B-hf",
        revision="7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
        architecture="Qwen3ASRForConditionalGeneration",
        model_type="qwen3_asr",
        hidden_width=1024,
        hidden_width_path=("text_config", "hidden_size"),
    ),
    WeightSpec(
        role="thinker",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        architecture="Qwen3_5ForConditionalGeneration",
        model_type="qwen3_5",
        hidden_width=1024,
        hidden_width_path=("text_config", "hidden_size"),
    ),
    WeightSpec(
        role="talker",
        model_id="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        revision="52f4770fd9726457eae3d3b6aa92047a25a10776",
        architecture="Qwen3TTSForConditionalGeneration",
        model_type="qwen3_tts",
        hidden_width=2048,
        hidden_width_path=("talker_config", "hidden_size"),
    ),
)


def default_hub_cache() -> Path:
    if value := os.environ.get("HF_HUB_CACHE"):
        return Path(value).expanduser()
    if value := os.environ.get("HF_HOME"):
        return Path(value).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _nested(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Checkpoint config is missing {'.'.join(path)}")
        value = value[key]
    return value


def resolve_weight(spec: WeightSpec, cache_root: Path | None = None) -> ResolvedWeightIsland:
    root = Path(cache_root) if cache_root is not None else default_hub_cache()
    snapshot = root / spec.cache_name / "snapshots" / spec.revision
    # Complete Hub releases carry all three islands alongside the KIRA runtime.
    # Explicit cache roots retain their old behavior. An incomplete bundle
    # fails closed instead of silently substituting a different cached donor.
    if cache_root is None:
        bundle_root = Path(os.environ.get("KIRA_LIVE_BUNDLE_ROOT", Path(__file__).resolve().parents[1]))
        descriptor = bundle_root / "bundle_manifest.json"
        if descriptor.is_file():
            bundled = json.loads(descriptor.read_text(encoding="utf-8"))
            entry = bundled.get("weights", {}).get(spec.role, {})
            if entry.get("model_id") != spec.model_id or entry.get("revision") != spec.revision:
                raise ValueError(f"Bundled {spec.role} provenance does not match its pinned revision")
            relative = Path(str(entry.get("path", "")))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe bundled {spec.role} path")
            snapshot = bundle_root / relative
    config_path = snapshot / "config.json"
    single_model = snapshot / "model.safetensors"
    sharded_models = tuple(snapshot.glob("model.safetensors-*.safetensors"))
    if not config_path.is_file() or not (single_model.is_file() or sharded_models):
        raise FileNotFoundError(
            f"Pinned {spec.role} checkpoint is incomplete: {snapshot}. "
            "Download the exact revision before starting KIRA Live."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = tuple(config.get("architectures", ()))
    if spec.architecture not in architectures:
        raise ValueError(
            f"{spec.role} architecture mismatch: expected {spec.architecture}, got {architectures}"
        )
    if config.get("model_type") != spec.model_type:
        raise ValueError(
            f"{spec.role} model type mismatch: expected {spec.model_type}, "
            f"got {config.get('model_type')}"
        )
    actual_width = int(_nested(config, spec.hidden_width_path))
    if actual_width != spec.hidden_width:
        raise ValueError(
            f"{spec.role} hidden width mismatch: expected {spec.hidden_width}, got {actual_width}"
        )
    return ResolvedWeightIsland(spec=spec, snapshot=snapshot.resolve(), config=config)


def resolve_composite_weights(cache_root: Path | None = None) -> CompositeWeightPaths:
    resolved = {spec.role: resolve_weight(spec, cache_root) for spec in KIRA_LIVE_WEIGHT_SPECS}
    return CompositeWeightPaths(
        listener=resolved["listener"],
        thinker=resolved["thinker"],
        talker=resolved["talker"],
    )
