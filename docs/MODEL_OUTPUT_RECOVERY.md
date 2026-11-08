# Model output validation and recovery

A malformed reply such as `[SUBAGENT_TASK]APP_ACTION:APP_ACTION:...` used to
escape tool parsing: because the action block was unfinished, it could reach
chat as ordinary text. Public output is now checked before final-answer
acceptance or tool dispatch.

The output guard rejects incomplete, unknown or mixed action blocks and repeated
protocol labels. Streaming generation stops an observed repeated-label loop
early. Valid single action blocks and ordinary prose/code remain supported.
Invalid assistant replies are excluded from subsequent chat context.

Rejected or empty generations trigger continuation from the current task and
verified tool evidence with a larger response allowance. Successful actions are
not replayed. After consecutive recovery failures, an answer-only control pass
can synthesize the browser evidence. If that also fails, the user receives the
observed source progress and an explicit incomplete status, rather than raw
protocol tokens. Existing task budgets, tool validation and permission checks
remain authoritative.

The focused GitHub patch passed 96 regression tests in a clean checkout, covering
the reported 126-label loop, early stream termination, truncated action syntax,
context filtering, automatic recovery and existing browser/artifact workflows.
