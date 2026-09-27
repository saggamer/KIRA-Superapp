# KIRA Live 1: speaker feedback repair

The previous correlation/FIR filter handled simple echoes but could miss
room reflections. The stronger playback gate also ended as soon as the output
stream closed, even while speaker audio was still ringing in the room.
That residual speech could trigger interruption and become a new user turn.

The native Live session now uses pywebrtc-audio 0.2.0 (WebRTC AEC3). Rendered
audio is resampled to the capture clock and consumed once in a bounded FIFO.
The output device latency is passed as an AEC delay hint. Echo cancellation
continues after playback with a 450 ms tail gate, while near-end speech can
still interrupt. Automatic gain control is disabled so it cannot amplify
residual echo. A recently spoken phrase is rejected only if it is at least
three words long and captured during playback or its tail.

The multilingual vocabulary repair is included: missing Listener byte-BPE
merges are split into valid Thinker pieces without decoding and re-prompting
the user transcript. This avoids the int(None) crash for the Hindi greeting.

Install the supplied Live requirements before starting speaker mode. Missing
AEC dependencies now produce an actionable error instead of silently using
the old insufficient filter. Model weights are unchanged. Orchestrator V1's
audio and reasoning paths are unchanged.

Verification: regression tests cover one-time render consumption, resampling,
tail expiry, abort cleanup, real-user text, and multilingual round trips.
The offline benchmark simulates delayed room reflections and double-talk;
it is not evidence of flawless performance in every microphone/speaker room.

AEC dependency: https://github.com/strands-labs/pywebrtc-audio (Apache-2.0).
