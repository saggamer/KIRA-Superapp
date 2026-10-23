# Orchestrator v1.5 browser and MCP skills

Kira's v1.5 profiles now load reviewed workflow packages from
`orchestrator_v15/skills/`. This is an inference-time runtime change. It does not
add parameters, train weights, supply missing RAG documents, restore old search
history, or connect a new MCP server.

## Design and sources

Official documentation reviewed on 2026-10-20:

- [ChatGPT/Codex skills](https://learn.chatgpt.com/docs/build-skills): metadata-first discovery, explicit or implicit activation, and `SKILL.md` packages.
- [OpenAI plugin skills](https://developers.openai.com/plugins/concepts/skills): skills provide workflows around MCP; the MCP server owns connections and actions.
- [Claude Agent Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview): progressive disclosure of metadata, selected instructions and optional resources.

This implementation adapts that pattern to a small local model: the host selects
up to two compatible skills using the resolved user request and existing task
routing. The model sees the short catalog and only selected instruction bodies.
It does not spend an additional generation choosing or loading a skill. These
are original Kira-specific instructions, not copies of proprietary vendor skills.

The implementation is an instructions-only subset: no automatic marketplace
installation, general YAML interpreter, script execution or arbitrary file reads.
Only four built-in package names are allowlisted. SKILL.md supports single-line
`name` and `description` frontmatter. Packages must remain inside their reviewed
directories; symlinks, malformed packages and files above 8 KiB are rejected.
Each request has at most 6,200 characters of catalog plus skill instructions.
A failed package is recorded in diagnostics and skipped, preserving the existing
controller rather than executing unchecked instructions.

## Included skills

| Skill | Workflow |
| --- | --- |
| `browser-research` | Search, open primary sources, compare evidence, cite URLs and capture real screenshots. |
| `browser-interaction` | Use observed controls, scroll, fill/select/click, verify state and avoid duplicate submissions. |
| `browser-history` | Search saved records, read returned IDs, distinguish cached from current information. |
| `mcp-app-workflow` | Discover configured servers, inspect schemas, execute supported read-only tools and distinguish inactive drafts from connections. |

Ordinary related requests activate skills automatically. Explicit examples:

```text
$browser-history find our saved Cedar research and read the relevant page.
$browser-interaction open https://example.org and scroll to the delivery section.
$mcp-app-workflow use my connected notes app to find the charger reminder.
```

An explicit browser skill selects the browser adapter, while presentation task
routing still takes precedence. The artifact controller retains its own tool
contract. Browser instructions do not override a deck workflow.

A browser task gets research guidance by default; interaction and history skills
are preferred when the request needs them. MCP requests get the MCP workflow.
An unrelated greeting gets the small catalog without any skill body. Tool
results, web pages, saved history and memory cannot register or select packages.
The resolved current user request remains the source of activation.

## Execution and autonomy

All v1.5 profiles share `_run_v15_profile`, so standard, thinking, diagonal
research preview and compatible artifact preview control paths use this layer. Instructions
remain present when browser observations are compacted. Compaction now preserves
saved-history IDs, result indices and pagination fields so the next action can
use actual returned records. Existing effort limits,
proposal evaluation, action validation and permission checks still govern every
execution. Skills do not add a new action executor.

Routine authorized browser navigation, scrolling, filling and read-only MCP
calls do not acquire new permission prompts. A user's review-before-submit
boundary still applies. Kira's live MCP bridge currently supports read-only
operations; a skill cannot enable writes or activate an unapproved connector.

## Audit and rollback

`brain.last_v15_skill_audit` records active names, full-file SHA-256 revisions,
context character count and load errors. Each model proposal payload includes
that audit metadata. It contains no private reasoning or retrieved content.

Launch with `KIRA_V15_SKILLS=off ./superapp` to disable both catalog and skill
instructions for an ablation. The default is on. Restart an already-running Kira
process after updating the runtime; a checkpoint replacement is unnecessary.

Run focused checks:

```bash
.venv-kira-live/bin/python -m unittest tests.test_orchestrator_v15_skills tests.test_orchestrator_v15_kira_profile tests.test_orchestrator_v15_mcp_policy tests.test_orchestrator_v15_mcp_live tests.test_orchestrator_v15_thinking_controller tests.test_orchestrator_v15_fast_mlx -q
python3 -m unittest tests.test_diagonal_search_history -q
node --test diagonal_browser/test/search_history.test.mjs
```

The standalone history fixture uses `python3` because a venv executable path
containing spaces cannot serve as that fixture's executable shebang on macOS.
The runtime itself invokes Python through argument arrays and has no such issue.

For a paired model-in-the-loop comparison on the mounted SanDisk checkpoints:

```bash
.venv-kira-live/bin/python scripts/benchmark_orchestrator_skills.py --output training_runs/orchestrator_v15/skills_ablation_20261020.json
```

The comparison uses the same weights, tools and scoring with skills off/on,
empty chat memory, a local browser site, a fictional saved page and a test Notes
MCP server. It does not access personal notes or public sites. These small
fixtures measure local workflow behavior; they cannot establish broad model
quality or frontier performance.
