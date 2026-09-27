# Run the current KIRA Live 1 release in Superapp

Apple Silicon macOS is required for the native MLX runtime. Orchestrator V1
remains an independent model and is not replaced by Live.

From the Superapp repository root:

```bash
python3 -m venv .venv-kira-live
.venv-kira-live/bin/python -m pip install -r requirements.txt -r requirements-kira-live.txt
.venv-kira-live/bin/hf download saggamer/KIRA-LIVE-1 --local-dir .
.venv-kira-live/bin/python run_kira_live.py --verify
.venv-kira-live/bin/python interface.py
```

The Hub download supplies all three pinned weight sets, the codec, accepted
KIRA additions, and their current inference runtime. Select **KIRA Live** in
the app and allow microphone access. Start a new Live conversation before
testing. Old chats are preserved; they are never part of the published model.

Install both requirements files in a clean environment. The local developer
environment may use uv instead of pip. No API key is needed for local speech.

This release includes the missing-token crash repair and WebRTC AEC3 speaker
echo cancellation. The latter uses an actual playback reference, an output
latency hint and a 450 ms post-playback echo tail. The old amplitude gate alone
was insufficient. Live can still fail in difficult room/device conditions;
the offline tests are not a guarantee of perfect physical-microphone behavior.

See [speaker feedback repair](KIRA_LIVE_1_ECHO_FIX.md) for the focused fix and
verification notes. Model architecture and weights were not retrained for
these software repairs.
