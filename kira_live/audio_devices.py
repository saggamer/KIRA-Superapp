from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping


def pick_audio_device(
    devices: Sequence[Mapping[str, object]],
    default_index: int | None,
    channel_key: str,
) -> int:
    """Return a concrete Core Audio device instead of PortAudio's implicit -1."""
    try:
        selected = int(default_index) if default_index is not None else -1
    except (TypeError, ValueError):
        selected = -1
    if 0 <= selected < len(devices):
        if int(devices[selected].get(channel_key, 0) or 0) > 0:
            return selected
    for index, device in enumerate(devices):
        if int(device.get(channel_key, 0) or 0) > 0:
            return index
    raise RuntimeError(f"No audio device exposes {channel_key}.")
