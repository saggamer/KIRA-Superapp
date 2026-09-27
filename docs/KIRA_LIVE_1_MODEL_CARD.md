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

This is a **research-preview checkpoint package for the KIRA runtime**, not a
drop-in Transformers model or a hosted inference endpoint. Apple Silicon and
the KIRA Live Python environment are required.
Use a **KIRA Live-enabled Superapp checkout**. Older Superapp revisions may not
include the Live button; downloading this package alone does not add that UI.

1. Use [KIRA Superapp](https://github.com/saggamer/KIRA-Superapp).
2. Download this repository into the Superapp directory, preserving its paths.
3. Install the runtime requirements and fetch the pinned dependencies listed
   in `kira_live/model_manifest.json` and `kira_live/weights.py`.
4. Launch `.venv-kira-live/bin/python interface.py`, select **Talk with KIRA Live 1**,
   tap the microphone, and wait for **Listening** before speaking.

The package includes accepted KIRA checkpoints, required normalization data,
runtime code and SHA-256 checksums. Chat databases, private recordings, training
examples, credentials, and rejected experiments are excluded.

## Limitations

This preview can still make mistakes. Voice latency and interruption depend on
the microphone, speakers, room noise and machine load. Tool selection, document
content, permissions and long workflows need user review. No claim is made of
human-level emotional intelligence or superiority over commercial Live models.
Speech pipeline tests are not a guarantee of flawless hardware conversations.

## License and credits

KIRA's supplied code is Apache-2.0. Third-party dependencies retain their own
licenses and attribution requirements. Dependency weights are fetched from
their original publishers rather than copied into this release. Review the
included manifest and the publishers' terms before redistribution.
