# Orchestrator v1.5 MCP control pass — 2026-10-13

The Kira v1.5 control profile can discover configured MCP targets, create an
inactive connector draft, create a filesystem bootstrap draft, and request that
a validated draft be applied. Discovery and drafting run without a permission
prompt. Applying a connector uses Kira's existing permission queue; a queued
request is not reported as an active connection.

The profile does **not** expose a live MCP `tools/list` or `tools/call` transport.
When asked to execute a server tool, it reports that limitation instead of
creating configuration or claiming the tool ran. This is the main remaining
gap for agentic MCP work.

## Controller and data changes

- The v1.5 profile accepts `MCP_DISCOVER`, `MCP_DRAFT`, `MCP_BOOTSTRAP`, and
  `MCP_APPLY` action blocks. It repairs two narrowly identified model errors:
  `BROWSER_TOOL` wrappers around discovery or an explicit apply request, and
  `ENV_JSON: []` when the user clearly requested an empty environment object.
- A draft uses Kira's validated draft API. It rejects malformed JSON and
  plaintext environment values; environment entries must be variable
  references. A draft remains inactive.
- Applying requires an explicit user request, an observed draft path within
  Kira's MCP workspace, a supported target, and successful static validation
  of the server command. Only then does Kira queue permission.
- Bootstrap output no longer contains an executable `MCP_APPLY` block. The
  previous nested block could be reparsed by the action pipeline and queue an
  apply request from a tool result.
- The first training set added 33 fictional, project-authored MCP rows. A
  second alignment set added 96 train-only Kira-format turns. Validation and
  test splits remained byte-identical to their parent set. The new rows contain
  no live MCP secrets or server output.

The 40-update first adapter continuation preserved all 29 of its 31-case
generated-validation passes, but held-out MCP decisions stayed **6/10** and a
live isolated apply request regressed. The 80-update Kira-aligned continuation
also preserved 29/31, but tied the selected adapter on held-out MCP action
contracts (**6/10**), lost two general agentic/browser action contracts
(**3/10** versus **5/10**), and passed **7/9** isolated Kira MCP workflows.
Neither adapter was promoted. The selected control, browser, and presentation
checkpoints in `gated_control_browser_20261008/manifest.json` remain unchanged.

## Evaluation

The local Kira benchmark uses an isolated temporary MCP workspace and a
stubbed permission queue. It does not launch a server, apply a real connector
configuration, or execute a live MCP tool. Its cases cover discovery, safe
drafting, inactive-state reporting, explicit apply, impossible live calls,
bootstrap, plaintext-secret rejection, untrusted notes, and missing drafts.

See the `mcp_kira_*_20261013.json` reports in `training_runs/orchestrator_v15`
for raw model proposals, executed actions, answers, and per-case timing.

After the controller repairs, the selected checkpoint passed **9/9** of those
isolated Kira MCP workflows. This is an integrated controller result, not a
claim that the checkpoint learned the MCP contracts; its direct model score
remained **6/10**. The exact final result is in
`mcp_kira_selected_final_20261013.json`.

The selected Diagonal preview completed the final 16-task everyday browser
suite: **16/16** passed. It passed page reading, link following, form entry,
sorting, scrolling, comparison, screenshot capture, prompt-injection resistance,
missing-page reporting, public search and website navigation to the Python
documentation, checkout preparation, asking before checkout, one authorized
fixture purchase, booking preparation, and one authorized fixture booking.
The public search returned `https://docs.python.org/`; the site navigation
returned the observed title `Our Documentation | Python.org` and URL
`https://www.python.org/doc/`.

The first 15-case run scored **14/15**: the booking server recorded one POST,
but the page did not display confirmation after two fresh observations. The
model did not duplicate the booking and reported uncertainty. On the final
16-case rerun, the page did display `Booking confirmed` and the model reported
it after one submission. This is a timing-sensitive result; repeatability on
real booking sites is not established. A separate first public navigation
trial also opened the documentation site but looped without a final answer.
The controller now finishes simple title-and-URL requests directly from a
successful page observation; the isolated rerun and full final suite passed.
All purchase and booking actions were on isolated local fixture pages, not
real commerce sites. Raw actions, observations, answers, and timing are in
`mcp_final_everyday_16_20261013.json`. The suite is a project test, not a
standard public agent benchmark.
