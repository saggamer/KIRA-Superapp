# App improvements published on 23 October 2026

This source update includes runtime and launcher compatibility fixes, external
SSD speech loading, persistent browser research history, bounded browser/MCP
skills, document and image-edit validation, and the current website updates.

The evidence/accurate-data RAG integration, Horizon 1 standalone image workflow,
Horizon research and training, and private runtime data are excluded. Mixed
changes in the main app and README were filtered without deleting local work.

Validation ran against a separate copy of the exact source selected for the
commit, with the existing local Live checkpoints available separately:

- Python: 537 tests, no failures, four skipped.
- Diagonal Browser: all 14 integration tests passed.
- Live UI, clipboard, and dictation JavaScript checks passed.
- No model weights, training corpora, browsing history, credentials, or personal
  chat data are included in this commit.

KIRA was launched through `./superapp`; its native desktop window showed the
Chat workspace with Orchestrator V1 selected. Earlier real model and voice
checks are recorded in `SUPERAPP_VERIFICATION_2026_10_16.md`.
