# KIRA Live 1 voice compatibility and latency update

## Changes

- In-memory 8-bit quantization of 250 Talker linear layers, group size 64.
  Saved BF16 model files, Thinker/expert weights, embeddings and waveform codec
  remain unchanged. No training, new checkpoint or parameter-count change.
- Optimized Talker starts playback after one chunk; original precision retains
  the two-chunk buffer. Whole conversational text still drives one synthesis
  stream, not separate word generations.
- UI/disk callbacks run off capture/playback threads. Audio meters coalesce and
  no longer write every level to the disk event log.
- Explicit low-latency input/output streams, safe missing-device-default handling,
  and complete draining of long playback instead of a silent 60-second cutoff.
- Live details show capture-to-voice timing, stage timings and overrun counters.
  These reset with a new chat. Barge-in, noise thresholds and echo filtering were
  retained, not weakened for a faster benchmark.
- Launchers select the resident SDK environment at
  `$HOME/.local/share/kira-superapp/runtime`, outside cloud-managed
  Documents. The former environment timed out after 40 seconds reading a UI
  dependency. Both launcher scripts and the installer support the resident path.
  Resident app imports completed in 19.927 seconds on the first cold run and
  2.190 seconds after cache warming; model loading is measured separately below.

## Measured small tests

Same-phrase baseline versus updated Talker, two runs each:

| Measurement | Original BF16/two-chunk buffer | 8-bit/one-chunk buffer |
| --- | --- | --- |
| Speech startup | 2.158 / 1.435 seconds | 0.441 / 0.607 seconds |
| Synthesis time / audio duration | 1.424 / 1.166 | 0.668 / 0.731 |
| Listener word error rate | 0 / 0 | 0 / 0 |

Two longer supportive replies: startup 0.363 / 0.618 seconds, synthesis ratios
0.635 / 0.687, and word error rate 0 in both. Timing used a simulated consumer;
maximum inter-chunk gaps were below 0.1 ms in these short tests. The output WAVs
concatenate audio and do not themselves record speaker/device underruns.
Cold comparisons also include lazy weight materialization differences; the second
same-phrase run is the more useful warm comparison. This is not a broad TTS quality
or emotion-expression evaluation.

One full production native-session turn, in the resident environment, passed:

- Native model loading: 4.581 seconds, separate from turn timing.
- Capture complete to first audio: 4.240 seconds.
- Listener/frontend: 0.700 seconds; CoreML emotion: 0.038 seconds.
- Response preparation: 2.884 seconds; speech startup: 0.612 seconds.
- Spoken reply loopback word error rate: 0; completed without a Live error.

This native probe uses the real session and an isolated test memory database,
but a timed simulated speaker. Timing excludes the configured 520 ms endpoint
silence and physical output latency. Text generation still dominates; the
model is not lag-free. The separate 24-token Thinker probes were capped and
their unfinished text is not evidence of a conversational quality gain.

MacBook Air microphone 16 kHz mono and speakers 24 kHz mono formats passed
PortAudio checks. No real microphone audio was recorded and no physical
speaker/room-echo or human barge-in calibration was performed.

81 focused audio, conversation, action-contract and expert-layout regressions
passed in the resident app runtime before publication. Headless Chrome
verified start/mute/resume/interrupt/transcript/end, timing display, readiness
gate and desktop/mobile layout with a mocked UI backend.

Reports are under `training_runs/kira_voice_latency_2026_09_29_baseline`,
`kira_voice_latency_2026_09_29_int8`, `kira_voice_latency_2026_09_29_emotional`,
and `kira_native_voice_latency_2026_09_29`. This runtime update is prepared for
GitHub and Hugging Face; rejected experiment checkpoints are not included.
The accepted daily-driver checkpoints and Orchestrator V1 behavior are preserved.

## Use and fallback

Launch `RUN_KIRA_OS.command` or `RUN_KIRA_OS_REAL.command` from this checkout.
For original Talker precision, launch with
`KIRA_LIVE_TALKER_BITS=0 ./RUN_KIRA_OS.command`.
`KIRA_PYTHON` can explicitly select another interpreter. Actual microphone
permission, interruption and echo should be checked in Live before considering
this a complete hardware-quality sign-off.
