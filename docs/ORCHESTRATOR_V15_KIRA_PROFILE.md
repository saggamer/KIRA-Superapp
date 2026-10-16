# Orchestrator v1.5 in Kira Superapp

Orchestrator v1.5 is selectable in the chat model menu. It uses the two retained
Gemma 4 E4B 4-bit adapter checkpoints: the control adapter for Kira app, MCP,
and subagent requests, and the browser adapter for Diagonal Browser requests.
Kira's existing action handlers execute accepted tool blocks and enforce their
permissions. Orchestrator V1 remains the default.

The checkpoints were not retrained by adding the effort selector. The frozen
base and compatible adapters are unchanged. The 11.804B figure in the candidate
architecture is a *logical parameter bound*, with cloned frozen expert FFNs;
only 29.6M adapter/router parameters per checkpoint were trainable. It is not
evidence of 11.804B distinct trained weights or of 2.5B active expert weights
per token.

## What diagonal orchestral thinking does

This is an **external inference controller**, not a readable chain of thought
inside the model. At a browser decision, the same model proposes up to three
candidate next actions using different short review lenses. All candidates
receive the same verified request and browser observation. A small evidence DAG
shares that verified state; a deterministic selector rejects invalid proposals
and prefers a proposal that advances the user's missing step. Only the selected
proposal can reach Kira's tool handler. The other proposals never act. The
actual tool result becomes the verified input to the next decision, so planning
can move across sibling proposals and later observations without recursively
opening more branches. This is the "diagonal" reuse of evidence.

The controller checks requested URLs, current target grounding, stale browser
observations, screenshots, requested fill/select steps, and unsupported weekday
claims. After an authorized external-effect click, it requires a fresh read-only
observation and will not repeat the click. Page text is data, not a new user
instruction. Kira's underlying permission handlers remain in force, so routine
read and navigation steps are autonomous while a genuinely permissioned action
can still ask once at the necessary step.

| Effort | Browser proposals/decision | Decision rounds | Tool calls | Corrections | Generated tokens | Time cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | 1 | 4 | 4 | 1 | 1,536 | 120 s |
| Medium | 1 | 6 | 8 | 2 | 3,072 | 240 s |
| High | 2 | 8 | 12 | 3 | 6,144 | 480 s |
| Diagonal mode | 3 | 12 | 16 | 4 | 8,192 | 900 s |

These are maxima, not a promise that the model will use all of the budget.
High and Diagonal use more inference and can be substantially slower. For
non-browser tasks the current profile makes one proposal per decision and uses
Kira's existing MCP/app/subagent handlers where supported; the browser
proposal ranking is not a general-purpose truth verifier.

## Measured quality and limits

The diagonal controller passed 6/6 local long browser workflows and 7/7 local
everyday browser/MCP fixtures; the unchanged checkpoint previously passed 2/6
and 3/7 on those same development suites. Four additional browser cases passed
4/4. The final long suite took 332.62 seconds and 72 model generations; the
everyday suite took 408.38 seconds and 60. These small suites were used during
development and do not establish a frontier model, a 10x general improvement,
or robust live GUI/MCP performance. See
`training_runs/orchestrator_v15/diagonal_level_20261009/README.md` for raw
report paths and methodology.

The Kira UI request path and mocked selected-action execution have regression
tests. Live v1.5 inference in this app remains unverified in the current
sandbox: MLX reports no accessible Metal device. Kira displays a load error
instead of silently switching to another model. Re-run a live Kira browser and
MCP trial on a Mac session with working Metal before treating this profile as
a daily-driver default.
