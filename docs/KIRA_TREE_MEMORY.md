# Kira TREE memory

THE TREE uses one local SQLite event log in `kira_agentic_logs/kira_tree_memory.sqlite3`.
Each agentic run has a unique ID and stores the user goal, redacted tool observations,
step status, and terminal outcome. Data is scoped to the chat. The execution ledger
remains the authority for claims that an action completed.

This adapts the DeepSeek Harness compaction and tool-result pruning pattern in the
vendored preset (`external_harnesses/deepseek-runtime/package/config/agent-presets/code/agent.cordis.yml`).
Kira keeps a bounded model-facing checkpoint instead of repeating the entire agent
transcript. Large observations retain a head, tail, and source or error anchors in
the checkpoint; their exact redacted content remains locally queryable by event
number. This prevents context growth across steps and makes the projection
auditable. The model receives the current goal and branch statuses with the
checkpoint. A checkpoint is untrusted historical evidence, not a new instruction
or a grant of permission.

`[TREE_MEMORY][/TREE_MEMORY]` returns the most recent run in the current chat.
`[TREE_MEMORY]EVENT: 2\nOFFSET: 0[/TREE_MEMORY]` retrieves a cited event in
bounded pages. The event number is scoped to that chat and run.
The model can use it to recover a previous execution state, then inspect current
state before continuing. `TREE_STATUS` remains the live progress view and
`BRANCH_REGISTRY` lists available branches. File, shell, app, and other mutations
still run through the existing permissioned dispatcher. The TREE memory store
does not replay actions automatically.

Run `python3 -m unittest tests.test_kira_tree_memory tests.test_tree_dispatch tests.test_kira_live_tree_routing`
to check checkpoint persistence, bounded projection, chat isolation, redaction,
and dispatch integration.
