# KIRA Coding Harness Provenance

KIRA's coding harness is an independent Python implementation. Its current loop incorporates
public architectural patterns from two open-source projects without vendoring their runtimes
or copying their source:

- OpenAI Codex (`openai/codex`, Apache-2.0): workspace-scoped execution, explicit file and
  command approvals, lifecycle events, bounded command execution, transactional changes,
  and verification before reporting completion.
- DeepSeek Harness (`deepseek-ai/deepseek-harness`, MIT): explicit agent-loop phases,
  deterministic pre-execution guards, one bounded plan repair, compact execution
  observations, cancellation, and success only after the loop reaches a verified terminal
  state.

KIRA keeps its own provider-neutral API adapters, THE TREE routing, money caps, macOS IDE
bridge, backup format, UI protocol, and permission system. Upstream licenses and attribution
remain with their respective projects.
