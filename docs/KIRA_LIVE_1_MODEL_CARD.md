---
license: apache-2.0
language: [en]
tags: [mlx, speech, conversational, research-preview]
---

# KIRA Live 1

KIRA Live 1 is a local conversational assistant for Apple Silicon Macs. It
supports text and voice conversations, expressive speech, chat memory, and
permission-aware actions through KIRA Superapp's THE TREE.

## What you can do

- Talk naturally, pause the microphone, and interrupt spoken replies.
- Continue a conversation using its saved chat history.
- Request file inspection, web research, and app actions through THE TREE.
- Request PDF and Word documents in KIRA Superapp, with returned execution
  evidence rather than an unverified completion claim.

## Getting started

This is a **complete three-weight research-preview package for the KIRA runtime**, not a
drop-in Transformers model or a hosted inference endpoint. Apple Silicon and
the KIRA Live Python environment are required.
Use a **KIRA Live-enabled Superapp checkout**. Older Superapp revisions may not
include the Live button; downloading this package alone does not add that UI.

1. Use [KIRA Superapp](https://github.com/saggamer/KIRA-Superapp).
2. Download this repository into the Superapp directory, preserving its paths.
3. Install the supplied `requirements.txt`. The three pinned weight sets are
   included under `weights/`; no separate donor-weight download is needed.
4. Launch `.venv-kira-live/bin/python interface.py`, select **Talk with KIRA Live 1**,
   tap the microphone, and wait for **Listening** before speaking.

The package includes the Listener, Thinker and Talker weight sets (including
the speech codec), accepted KIRA checkpoints, required normalization data,
runtime code and SHA-256 checksums. They run together through KIRA's native
runtime, not as a single fused tensor file. This is not a claim of joint
end-to-end training. Chat databases, private recordings, training
examples, credentials, and rejected experiments are excluded.

For standalone voice conversation without Superapp tool execution:

```bash
python -m pip install -r requirements.txt
python run_kira_live.py --verify
python run_kira_live.py --probe
python run_kira_live.py
```

The first command needs network access for Python dependencies. After setup,
bundled model inference is offline. Allow microphone access when prompted.
Standalone mode is conversational; THE TREE tools require Superapp integration.

The current runtime includes local WebRTC acoustic echo cancellation for
speaker playback, an echo-tail guard, and the multilingual token-mapping
crash repair. Install the current requirements before using hands-free voice.
These are runtime fixes; the accepted model weights have not been retrained.

## Limitations

This preview can still make mistakes. Voice latency and interruption depend on
the microphone, speakers, room noise and machine load. Tool selection, document
content, permissions and long workflows need user review. No claim is made of
human-level emotional intelligence or superiority over commercial Live models.
Speech pipeline tests are not a guarantee of flawless hardware conversations.

## License and credits

KIRA's supplied code is Apache-2.0. Third-party dependencies retain their own
licenses and attribution requirements. The included donor weights retain their
original authorship and Apache-2.0 terms. Exact source repositories and revisions
are recorded in `bundle_manifest.json`; see `THIRD_PARTY_NOTICES.md` and `LICENSE`.
