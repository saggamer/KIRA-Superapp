# KIRA Superapp Architecture and Development Handoff

## Product definition

KIRA Superapp is a local-first macOS agent runtime for Apple Silicon. It is not
only a chat client: one desktop interface coordinates local model inference,
tool execution, research, artifact creation, coding workflows, voice,
scheduling, memory, plugins, and MCP connector setup.

The central design rule is that a model proposes an action, KIRA validates and
executes it through a bounded runtime, and completion is reported only after
matching execution evidence exists.

## Runtime map

| Layer | Main files | Responsibility |
|---|---|---|
| Desktop shell | `ui.html`, `interface.py` | PyWebView window, chat UI, settings, task panels, progress events, and the JavaScript/Python API bridge |
| Orchestrator | `interface.py` | Prompt routing, model selection, agent loop, tool-result continuation, execution ledger, permissions, and completion-truth checks |
| Model process | `model_worker.py` | Isolated MLX model loading and generation over a JSON-lines subprocess protocol |
| Tool runtime | `actions.py` | Files, commands, web research, artifacts, MCP, apps, clipboard, scheduling, and guarded operating-system actions |
| Coding mode | `coding.py` | Project selection, provider/harness planning, plan validation, transactional edits, verification, rollback, budgets, and IDE opening |
| Specialists | `specialists.py` | Parallel research/media work and verified document, slide, and artifact branches |
| Voice | `voice.py`, `listen.py` | Dictation, streaming speech-to-text, voice activity, serialized TTS, interruption, and live-voice state |
| Memory | `memory.py` | Local chat continuity and small durable-memory retrieval with sensitive-text filtering |
| Vision/overlay | `vision.py`, `kira_screen_overlay.py` | Screen-context and overlay support |
| Distribution | `superapp`, `INSTALL_MACOS.command`, `DOWNLOAD_MODELS.command` | Environment setup, model download, verification, and launch |

## Main request flow

1. The browser UI calls the `LazyKiraAPI` bridge so the window can become
   responsive before the full backend and model are ready.
2. `KiraBrain.send_prompt` normalizes the chat, mode, model, and attachments and
   places the work on a background queue.
3. The router selects direct chat, coding mode, an artifact shortcut, or the
   Orchestrator agent loop.
4. Local inference runs in `model_worker.py`, outside the UI process. The parent
   process can time out or restart a failed worker without freezing the window.
5. Agentic output is parsed into KIRA directives. The runtime validates paths,
   commands, schemas, and permission requirements before acting.
6. Tool observations are returned to Orchestrator for another bounded step.
   KIRA defaults to three agentic steps and caps the configurable value at five.
7. The execution ledger checks whether a final answer's completion claim is
   supported by successful actions and verification evidence.
8. The public answer is cleaned, saved to local chat history, indexed for local
   memory, and emitted to the UI. Voice output is optional.

## Model architecture

- Orchestrator V1 is the primary local planning and tool-routing model.
- MLX is used for Apple-Silicon inference.
- The heavyweight model is isolated in a subprocess so VRAM cleanup, crashes,
  and generation timeouts do not directly take down the desktop UI.
- Model discovery supports the app folder, `models/`, common user folders, and
  LM Studio storage, plus an explicit `KIRA_ORCHESTRATOR_PATH` override.
- Cloud-only iCloud placeholders are rejected before load to avoid a silent
  startup hang.
- Generation is bounded by prompt, output, timeout, and restart-window limits.
- Generated text is truncated at known model-turn boundaries before it is
  returned to the parent process, preventing accidental display of a second
  generated turn when a backend misses EOS.

## Agentic capabilities

KIRA's directive runtime covers these capability groups:

- local file discovery, reading, writing, deletion, and project commands;
- web search, fetch, browser-style research, and per-chat source tracking;
- PDF, DOCX, PPTX, spreadsheet, and other artifact workflows;
- application discovery/opening, window reports, screen context, clipboard,
  IDE context, and AppleScript;
- MCP discovery, draft validation, and permission-gated application;
- scheduled reminders and recurring agentic tasks;
- specialist dispatch for research, media, slides, documents, and verification;
- coding projects with optional OpenAI, Anthropic, Gemini, DeepSeek, NVIDIA NIM,
  OpenRouter, or OpenAI-compatible endpoints.

Coding mode is deliberately transactional. A proposed plan is schema-checked,
dangerous commands and sensitive paths are rejected, original files are backed
up, verification commands run with time and output limits, and a failed apply
rolls the changed files back.

## Safety and privacy boundaries

- Read access is limited to known roots and checked against sensitive path
  markers.
- Mutating file, shell, IDE, MCP, automation, and operating-system actions are
  permission-gated.
- Shell and Git operations use explicit safe-command allowlists for read-only
  paths; destructive behavior is not delegated to model judgment alone.
- Tool output, webpages, documents, and UI text are treated as untrusted
  evidence rather than instructions.
- API keys are stored in macOS Keychain and are redacted from settings, logs,
  prompts, and project context.
- The Model Library requires a separate local password before revealing a
  stored key and automatically relocks.
- Provider spending can be tracked against monthly and per-task caps.
- Memory omits likely credentials and stores compact local summaries instead of
  sending an entire history to the model.
- Private reasoning and model boundary tokens are filtered before display and
  persistence.

## Local data

Runtime-generated state is intentionally separate from the shipped source:

- `kira_chat_history/` — saved conversations;
- `kira_agentic_workspace/` — attachments, artifacts, schedules, and runtime
  working files;
- `kira_agentic_logs/` — execution ledgers and diagnostic records;
- `kira_mcp_connectors/` — connector drafts;
- `kira_plugins/` — plugin manifests and enablement state;
- `kira_technical_architect/` — coding sessions, backups, runs, IDE bridge, and
  subagent state;
- `kira_smart_memory.json` and `kira_personalization_rag.json` — local memory
  and personalization.

These paths, model weights, generated artifacts, and credentials must remain out
of source control.

## UI surface

The single-window UI provides:

- persistent chats with rename/delete and attachment support;
- chat, agentic, and coding workspaces;
- source inspection for researched answers;
- voice dictation and live voice controls with interruption;
- scheduled tasks and reminders;
- provider/model selection, budget tracking, and a password-gated Model Library;
- MCP draft/test/apply management;
- plugin enablement;
- progress, branch, permission, and execution-result presentation.

## Current verified state

- Repository baseline: `21b7868` (`Initial KIRA Superapp release`).
- Python source compiles successfully in this workspace.
- The full Python suite passes in the existing KIRA environment: 162 tests,
  with one expected skip.
- The system Python in this workspace is 3.14 and lacks the app dependencies;
  supported clean installs should use Python 3.11 or 3.12 as documented.

## Model-improvement handoff

The previous development task produced two useful evaluation generations:

- A seven-task executable KIRA race scored RED v3 at 5/7, Orchestrator V1 at
  4/7, and base Gemma 4 E4B at 2/7.
- A RED v4 synthetic corpus was generated with 1,212 records: 850 training, 181
  validation, and 181 sealed test records. It includes 168 computer-use
  workflows and targets exact command repair, checkpoint continuation,
  evidence transfer, accessibility-first control, stale-screen recovery,
  scoped approvals, MCP repair, hostile-content resistance, and truthful
  completion.

The v4 corpus passed structural and tokenizer checks, but no RED v4 adapter has
been trained yet. The next model-development milestone is therefore:

1. preserve the v4 validation and sealed test split;
2. train a fresh adapter from the same Gemma 4 E4B base;
3. evaluate it with the unchanged hard-race harness;
4. reject the candidate if safety, permission, or false-completion behavior
   regresses, even if its aggregate score improves;
5. only then package the winning model for the Superapp runtime.

## Recommended engineering priorities

1. Add a reproducible development-environment command that does not download
   model weights and can run the complete test suite on a clean Mac.
2. Bring the hard-race harness and its deterministic fixtures into version
   control as a release gate.
3. Add clean-install CI or a documented local CI script for Python 3.11 and
   3.12.
4. Separate the large `interface.py` and `actions.py` modules into narrower
   routing, permission, web, artifact, scheduling, and persistence services
   while preserving the PyWebView API.
5. Add adversarial tests for prompt injection in every observation channel and
   recovery tests for stale UI state and interrupted model workers.
6. Train and evaluate RED v4 only after the benchmark and safety gates are
   checked into the repository.
