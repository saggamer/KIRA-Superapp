from __future__ import annotations

"""One validated, revision-pinned KIRA Live checkpoint *bundle*.

The three architectures remain independently executable; this is a logical
checkpoint descriptor, not a claim of merged tensors or joint training.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from .weights import CompositeWeightPaths, resolve_composite_weights


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "kira_live" / "model_manifest.json"


@dataclass(frozen=True)
class KiraLiveCheckpointBundle:
    weights: CompositeWeightPaths
    thinker_addons: Path
    emotion_head: Path
    emotion_normalization: Path
    coreml_emotion: Path | None


def _accepted_file(training: dict, key: str) -> Path:
    entry = training.get(key, {})
    decision = str(entry.get("decision", ""))
    if not decision.startswith("accepted"):
        raise RuntimeError(f"KIRA Live {key} checkpoint is not accepted.")
    relative = Path(str(entry.get("checkpoint", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"KIRA Live {key} checkpoint path is unsafe.")
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"KIRA Live {key} checkpoint is missing: {path}")
    return path


def resolve_live_checkpoint() -> KiraLiveCheckpointBundle:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    training = manifest.get("training_status", {})
    thinker_addons = _accepted_file(training, "thinker_ple_moe")
    emotion_head = _accepted_file(training, "live_emotion_stage1")
    normalization = ROOT / "training_runs/kira_live_emotion_v1/feature_normalization.npz"
    if not normalization.is_file():
        raise FileNotFoundError(f"KIRA Live emotion normalization is missing: {normalization}")
    coreml_relative = training["live_emotion_stage1"].get("coreml_package", "")
    coreml = ROOT / coreml_relative if coreml_relative else None
    if coreml is not None and not coreml.exists():
        coreml = None  # MLX emotion inference remains the supported fallback.
    return KiraLiveCheckpointBundle(
        weights=resolve_composite_weights(),
        thinker_addons=thinker_addons,
        emotion_head=emotion_head,
        emotion_normalization=normalization,
        coreml_emotion=coreml,
    )
