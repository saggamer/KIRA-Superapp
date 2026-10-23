# KIRA Superapp verification — 16 October 2026

This audit covers the desktop runtime, Live speech stack, tool regressions,
launchers, and Diagonal Browser. The supplied screenshot was unavailable, so
its contents could not be checked.

## Fixes

- `superapp` and `DOWNLOAD_MODELS.command` now resolve the same runtime as the
  desktop launchers and installer, including `KIRA_PYTHON` overrides. Launching
  from another directory works.
- The installer selects a supported Python 3.11 or 3.12 interpreter.
- The main requirements now include the MCP client and JSON Schema validator.
  Cached dependencies were installed in both existing app runtimes.
- Both runtimes now use MLX-LM 0.32.0 for Gemma 4 support and the compatible
  MLX-Audio 0.5.9 release. The installer requirements carry those versions.
  The worker's lightweight tokenizer adapter implements the newer import,
  vocabulary-size, and token-conversion contracts.
- Live speech loading ignores AppleDouble `._` metadata on external disks by
  constructing a temporary symlink view. Original models stay untouched. A
  missing speech tokenizer now fails immediately during initialization.
- The autonomy benchmark detects final-effect actions even when an answer
  contains surrounding prose or several browser directives.
- Model-storage tests now participate in unittest discovery. Branding tests
  check the product identity; curriculum tests validate newly generated rows
  rather than requiring immutable historical data to match today's policy.

## Evidence

| Boundary | Result | Evidence |
| --- | --- | --- |
| Python app/tool regressions | Pass | Final shared launcher runtime: 613 tests, one skipped, no failures |
| Default Orchestrator model → worker → public response | Pass | Fresh local Orchestrator V1 download; actual worker loaded in 0.89 seconds and returned `42` for `6 times 7` |
| Installation readiness | Pass | `scripts/verify_install.py` finds resident Orchestrator weights and an available local speech-to-text runtime |
| Browser actions and persistence | Pass | All 10 Diagonal Browser integration tests, including local navigation, forms, screenshots/recording, popup handling, and restored authentication state |
| Clipboard and dictation UI | Pass | Both Node regression scripts |
| Live UI controls | Pass | Start, mute/resume, interrupt, transcript, end, readiness gate, desktop/mobile controls; fixture backend |
| Real Live Thinker → document tool | Pass | Real generated story text serialized to DOCX; tool quality check passed with five paragraphs |
| Real Talker → audio → Listener | Pass | 5.52 seconds of generated audio, 12 chunks, 0.633 seconds to first audio after initialization, zero word error rate on the single test phrase |
| Complete native Listener → Thinker → Talker turn | Pass | Final shared runtime: real fixture transcription, generated response, CoreML emotion inference, and simulated playback; 7.62 seconds from turn start to first audio, zero playback underruns, zero loopback word errors |
| Installed dependency consistency | Pass | `uv --no-cache pip check` in both existing runtimes |
| Shell and edited website JavaScript syntax | Pass | `zsh -n`, `node --check`, and `git diff --check` |

Speech results are one synthetic loopback test, not a sustained microphone,
speaker-echo, or conversational-quality evaluation. The Live UI fixture test
does not prove native WKWebView or audio-device behavior.

Native-turn evidence: `kira_voice_tmp/verification-20261016-native-final/report.json`.
Speech loopback evidence:
`kira_voice_tmp/verification-20261016-integrated/verification.json`.

## Default model readiness

The original Orchestrator V1 discovery result was an iCloud placeholder. A
fresh 4.24 GB download into `models/orchestrator_v1_fused` completed and the real
isolated worker passed loading and arithmetic generation in the shared launcher
runtime. This is an inference smoke test, not a benchmark of every agentic task.

## Remaining verification limits

- Native microphone permissions, acoustic echo, Finder paste, and OS dialogs.
- Paid coding providers, billing, third-party accounts, and production MCP services.
- Every possible open-ended model/tool workflow or model response quality.
- Items shown only in the unavailable screenshot.

## Reproduce software checks

```sh
~/.local/share/kira-superapp/runtime/bin/python -m unittest discover -s tests -q
node tests/test_clipboard_ui.cjs
node tests/test_dictation_ui.cjs
node scripts/verify_kira_live_ui.cjs
cd diagonal_browser
npm test
```

Browser checks require local Chrome, browser dependencies, and localhost access.
The Live UI script uses Diagonal Browser's installed Playwright dependency.
Model weights, generated audio, documents, and private runtime state are ignored
by Git. Existing website edits were preserved.
