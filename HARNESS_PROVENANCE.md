# Coding harness provenance

KIRA Superapp keeps its permission queue, deterministic plan validation, transactional file writes, rollback, bounded command execution, and verification as the outer control layer.

## OpenAI Codex

- Upstream: https://github.com/openai/codex
- Runtime detected from the installed ChatGPT application bundle.
- Detected version during integration: `codex-cli 0.151.0-alpha.7.2`.
- License: Apache-2.0.
- KIRA integration: `codex exec` runs ephemerally with the native `read-only` sandbox and KIRA's strict JSON output schema. Codex can inspect the selected project and propose a complete file plan, but KIRA still validates the plan and asks the user before writing anything.

## DeepSeek Harness

- Upstream: https://github.com/deepseek-ai/deepseek-harness
- Package: `@deepseek-ai/dsh@0.1.1-rc.2`.
- License: MIT. The package license is retained at `external_harnesses/deepseek-runtime/package/LICENSE`.
- Vendored package launcher source: `external_harnesses/deepseek-runtime/package`.
- KIRA integration: when an installed `dsh` executable is detected, KIRA can run its headless profile inside a macOS read-only sandbox and pass the result through KIRA's plan validator. The source package alone is not reported as an available runtime.
- Upstream status: developer preview. KIRA never bypasses its own approval, path, command, rollback, or verification controls for this runtime.

Neither upstream project is copied into KIRA's native Python planner. They are invoked as optional real runtimes through a narrow adapter, preserving their licenses and avoiding an unmaintainable partial port.
