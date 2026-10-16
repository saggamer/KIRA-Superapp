# KIRA Live 1

## Status

KIRA Live 1 is available as a daily chat choice in Superapp and through the
native KIRA Live voice button. Orchestrator V1 remains a separate model for
computer workflows; Orchestrator V1.5 training is paused. The voice path has
small end-to-end benchmarks, but extended hardware sessions and commercial
model comparisons remain unverified.

## Objective

Build a low-latency, local, full-duplex voice agent for Apple Silicon that can:

- listen continuously without running a large ASR model continuously;
- transcribe voiced segments and support interruption while KIRA is speaking;
- interpret tone and likely emotion as uncertain acoustic evidence;
- route every language-model token through a small mixture of residual experts;
- retain strong tool use and permission-aware agent behavior;
- preserve every sanitized chat event locally and retrieve exact evidence across
  extremely long horizontal workflows;
- stream expressive speech locally;
- run through MLX and Core ML without sending microphone audio to a server.

## Model selection

There is no official Qwen3.5 0.5B checkpoint. The closest official checkpoint
is Qwen3.5-0.8B. KIRA Live therefore starts from:

| Role | Model | Runtime |
|---|---|---|
| Thinker | `Qwen/Qwen3.5-0.8B` | MLX-VLM |
| Speech recognition | `Qwen/Qwen3-ASR-0.6B-hf` | Core ML or MLX |
| Speech generation | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Core ML or MLX |
| Always-on voice activity | TEN VAD | Core ML |
| Emotion/tone encoder | KIRA acoustic-emotion-small | Core ML |

Training data does not increase a model's parameter count. The new parameters
come from the audio projector, emotion head, token router, and expert adapters.

## Architecture

```text
Microphone (16 kHz)
        |
        v
Core ML always-on front end
  - VAD (20 ms frames)
  - acoustic emotion/tone encoder (voiced windows only)
        |
        +--------------------------+
        |                          |
        v                          v
Qwen3-ASR 0.6B             smoothed emotion state
        |                          |
        +------------+-------------+
                     v
            audio/text projector
                     |
                     v
           Qwen3.5 0.8B thinker
       + 2/3/4-gram PLE + token-level top-2 router +
       |                            |
       v                            v
 conversation/emotion       tools/memory/workflow/safety experts
       |                            |
       +-------------+--------------+
                     v
            text + tool decisions
                     |
          +----------+-----------+
          |                      |
          v                      v
 KIRA guarded tools       Qwen3-TTS 12 Hz
                                 |
                                 v
                       streaming local audio
```

The original research target was one jointly differentiable architecture with
three pretrained weight islands and learned hidden-state bridges. The deployed
runtime is narrower: Listener semantic tokens reach the Thinker through a
validated token map, and the Thinker's public answer and Listener-derived
style drive the Talker. The islands do not hold a conversation with one
another, but they are not yet jointly optimized by end-to-end gradients.

MoE routing exists only inside the Thinker. The Listener and Talker do not use
language experts. An always-on 20 ms microphone/VAD path maintains rolling
acoustic 2/3/4-grams during both token generation and playback. User speech
immediately cancels the Thinker generation ticket and Talker playback before
the new utterance is fused back into the Thinker.

## Residual adapter MoE

Converting every dense feed-forward layer into a new sparse MoE would invalidate
most pretrained weights and require pretraining-scale compute. KIRA Live instead
adds ten low-rank residual experts to selected upper and middle layers:

1. conversation; 2. emotion/prosody; 3. agent planning; 4. tool use;
5. episodic memory; 6. workflow state; 7. safety/permissions;
8. failure recovery; 9. speech style; 10. verification/completion.

A learned router selects the top two adapters for each token. Auxiliary
load-balancing and router z-loss prevent one expert from absorbing all traffic.
The deterministic router in `kira_live/router.py` is the fallback and test
oracle; it is replaced by a learned gate during adapter-MoE training.

The base 0.8B parameters remain frozen for the first training stage. This makes
the experiment reversible, reduces catastrophic forgetting, and allows the same
checkpoint to be compared with and without Live experts.

## Multi-token prediction

Qwen3.5-0.8B already declares a one-layer multi-token-prediction component.
KIRA additionally inserts trainable hashed 2/3/4-gram PLE tables at layers 3,
7, 11, 15, 19, and 23. The n-gram orders share each layer's table and use a
learned gate, avoiding a threefold parameter increase. MTP remains optional for
verified speculative decoding and cannot bypass tool or permission checks.

## Persistent memory

The exact sanitized text event stream is append-only in local SQLite. The model
receives a bounded pack of recent exact turns, retrieved early episodes, and a
dependency-aware workflow ledger. Six low-rank memory-fusion layers convert the
retrieved evidence into 256 recurrent memory slots. This keeps storage complete
without pretending that an unlimited prompt can be recomputed on every token.

Raw microphone audio is not part of persistent memory. Credential-like values
are redacted, and remembered text cannot grant permission for a new action.

## Realtime state machine

`kira_live/session.py` implements:

```text
idle -> listening -> transcribing -> thinking -> speaking
          ^                                  |       |
          +----------------------------------+-------+
                         completion / barge-in
```

- VAD remains active while TTS plays.
- A confident user-voice start interrupts TTS immediately.
- Pre-roll prevents the beginning of an utterance being clipped.
- End-of-turn uses VAD hysteresis and silence duration, not a single frame.
- ASR and emotion inference only receive voiced windows.
- Raw audio is not persisted by default.

## Emotion policy

Emotion recognition is inherently uncertain and can vary with language,
culture, disability, microphone, room, and speaker. KIRA must never state an
inferred emotion as a fact.

The acoustic head produces a probability distribution, valence, arousal, and a
calibrated confidence score. `EmotionTracker` smooths predictions over time and
decays them toward neutral. Low-confidence estimates become “do not infer an
emotion” context. Users must be able to disable emotion adaptation separately
from speech recognition.

Training and evaluation must use speaker-independent splits and report results
by language, accent, gender, noise condition, and microphone class. Personal
voice samples require affirmative consent and must not be silently retained.

## Training stages

### Stage 0 — frozen-system baseline

- Run Qwen3.5-0.8B, ASR, emotion head, and TTS as separate components.
- Measure end-of-turn latency, first text token, first audio chunk, interruption
  latency, word error rate, emotion calibration, and tool success.
- Establish a working live loop before joint training.

### Stage 1 — acoustic emotion head

- Train the small Core ML-compatible encoder on consented public emotion and
  speaking-style corpora.
- Use speaker-disjoint validation and balanced augmentations for room noise,
  compression, gain, pitch, and tempo.
- Optimize classification, valence/arousal regression, calibration, and an
  explicit unknown/uncertain target.

### Stage 2 — expert adapters, memory fusion, and router

- Freeze the Qwen backbone.
- Train ten LoRA-style residual experts, the persistent-memory fusion adapters,
  and a top-2 gate in MLX.
- Mix natural conversation, emotion-aware response pairs, the RED v4 agentic
  corpus, permission/recovery data, and neutral counterexamples.
- Apply expert load-balancing, router z-loss, and behavior-preservation KL loss.

### Stage 3 — audio projector

- Project ASR/audio-encoder states into the Thinker's 1,024-dimensional hidden
  space.
- Train on aligned audio/transcript/instruction examples while retaining text
  targets.
- Keep transcript supervision so direct-audio behavior remains inspectable.

### Stage 4 — Talker conditioning

- Condition Qwen3-TTS on response text plus bounded style controls derived from
  KIRA's response policy—not a raw copy of the user's inferred emotion.
- Distill streaming chunks and train interruption-safe synthesis.
- Do not clone a user's voice without explicit enrollment and deletion controls.

### Stage 5 — optional joint tuning

- Unfreeze only selected upper Thinker layers, the audio projector, router, and
  Talker conditioning interface.
- Maintain separate checkpoints for every stage and reject regressions in tool
  safety, instruction following, calibration, or latency.

## Apple deployment split

### Core ML

- always-on VAD;
- acoustic emotion/tone encoder;
- optional ASR encoder or full ASR after conversion validation;
- stateful streaming caches on macOS 15 or newer;
- Neural Engine preferred for the always-on path.

### MLX

- Qwen3.5 Thinker and multi-token prediction;
- adapter-MoE training and inference;
- early Talker/ASR experimentation when Core ML conversion is incomplete;
- quantization and Apple-GPU profiling.

Core ML export must compare numerical outputs with its source checkpoint on a
fixed parity set before an exported model is accepted.

## Acceptance gates

KIRA Live 1 is not ready for integration until it meets all of these:

- no raw-audio persistence unless explicitly enabled;
- median barge-in stop below 150 ms after VAD onset;
- emotion expected calibration error at or below the chosen baseline;
- no demographic slice with an unexplained large quality drop;
- no safety or permission regression against the KIRA hard race;
- no false completion after a failed tool call;
- stable 30-minute full-duplex session without an audio or model-worker leak;
- first text and first audio latency measured on the minimum supported Mac;
- graceful fallback when emotion, ASR, TTS, or the learned expert gate is absent.

## Implemented foundation

- `kira_live/config.py` — versioned model and realtime configuration;
- `kira_live/emotion.py` — calibrated smoothing, decay, and uncertainty-aware
  prompt context;
- `kira_live/router.py` — deterministic top-2 expert-routing reference;
- `kira_live/session.py` — live session and barge-in state machine;
- `kira_live/workflow_memory.py` — append-only exact chat events and workflow ledger;
- `kira_live/model_manifest.json` — model/runtime/trainable-component contract;
- `kira_live/weights.py` — exact-revision resolver and architecture/width validation
  for the Listener, Thinker, and Talker checkpoints;
- `kira_live/composite_runtime.py` — a prompt-free execution contract that only
  permits hidden states, acoustic unit IDs, codec states, memory, and emotion
  features across weight-island boundaries;
- `kira_live/mlx_talker.py` — native MLX execution of the real 28-layer Qwen3-TTS
  Talker decoder and five-layer, 16-group codec predictor from KIRA's
  1,024-wide hidden conditioning;
- `kira_live/mlx_codec_decoder.py` — native MLX execution of all 271 real
  Qwen 12 Hz speech-tokenizer decoder tensors, producing 24 kHz waveform audio;
- `kira_live/mlx_listener.py` — native MLX execution of all 611 Qwen3-ASR
  tensors: its acoustic tower, multimodal projector, and 28-layer semantic
  decoder. The deployed path remaps its decoded semantic token IDs into the
  Thinker vocabulary; acoustic states feed emotion conditioning;
- `kira_live/mlx_thinker.py` — the real Qwen3.5 decoder entered through its
  native input-embedding boundary, without a text prompt handoff;
- `kira_live/native_session.py` — fail-closed microphone, continuous VAD,
  cancellation, three-island inference, waveform, and playback ownership;
- `tests/test_kira_live.py` — initial configuration, emotion, routing, decay,
  and interruption tests.

The pinned Listener, Thinker, and Talker checkpoints are validated as one
logical checkpoint bundle. They execute in one live session, but remain three
independently trained architectures, not one end-to-end differentiable model
or a monolithic tensor file. The active Listener→Thinker bridge remaps
semantic token pieces, while Listener acoustics supply emotion features.
The Thinker→Talker bridge supplies public answer text and style to Qwen3-TTS.
That text is the Talker's speech content, not a second LLM conversation prompt.
Research-only learned hidden-state bridges are not used in live generation
because they worsened spoken quality. Merely packing the current weights into
one file would not change these computations or improve latency.

The live conversation path is enabled in Superapp. The active
Listener-to-Thinker path remaps semantic token IDs between the two pinned
tokenizers and embeds them with the Thinker; acoustic Listener features still
feed emotion conditioning. The older hidden-state bridge is retained for
research but is not used in live responses because it degraded answer quality.
The native session owns continuous microphone input, interruption, and 24 kHz
speech playback. A three-prompt spoken-input benchmark produced distinct,
relevant answers, recalled a previous chat fact, and generated an intelligible
spoken reply. Typed and spoken turns use the same chat ID: exact turns remain
on disk, with recent and relevant older events packed into the Thinker's
bounded context. This is not literal unlimited model context. The small check does
not establish sustained-session reliability or parity with commercial live
models. See `training_runs/kira_live_voice_repetition_check/benchmark.json`.

The first Qwen3.5-0.8B MLX LoRA smoke and public-agentic continuation experiments
have completed. Neither adapter is promoted: the aggressive continuation
repeated/truncated workflow fields, while the conservative fresh adapter did
not improve exact tool arguments. The frozen donor remains the runtime baseline
until a checkpoint passes the generation gate. The next unit is a larger,
checkpoint-gated curriculum plus the Core ML VAD/emotion runner and streaming
ASR adapter. See `docs/KIRA_LIVE_1_TRAINING_LOG.md` for measured results.

The layer-level Qwen4-derived conversion is specified in
[`KIRA_LIVE_1_QWEN4_LAYER_PLAN.md`](KIRA_LIVE_1_QWEN4_LAYER_PLAN.md).
