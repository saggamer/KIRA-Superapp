from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping


def pick_audio_device(
    devices: Sequence[Mapping[str, object]],
    default_index: int,
    channel_key: str,
) -> int:
    """Return a concrete Core Audio device instead of PortAudio's implicit -1."""
    if 0 <= int(default_index) < len(devices):
        if int(devices[int(default_index)].get(channel_key, 0)) > 0:
            return int(default_index)
    for index, device in enumerate(devices):
        if int(device.get(channel_key, 0)) > 0:
            return index
    raise RuntimeError(f"No audio device exposes {channel_key}.")

