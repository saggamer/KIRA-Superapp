# Scheduling and connector verification

## Implemented
- Tasks panel for timed reminders, app launches, and agent tasks.
- Pause, resume, cancel, and reschedule pending items; persistent schedule storage.
- Reminders can repeat every 24 hours or 7 days and run without model inference.
- Interrupted running tasks pause after restart to avoid automatically duplicating actions.
- Existing TREE SCHEDULE_TASK blocks accept KIND: reminder and optional REPEAT.
- Plugin IDs and entrypoint containment checked; replacement imports require re-enabling.
- MCP static executable/argument checks and atomic config merges; malformed existing configs preserved.

## Actual isolated UI test
Used scripts/isolated_ui_check.py with the real UI and Python backend, temporary HOME,
separate chat/schedule storage, and no production API keys or connector modifications.
- Created reminder through Tasks UI; paused and resumed it.
- Scheduler delivered it to chat and the reminder alert, then persisted completed status.
- Scheduled timestamp 1788439949.806787; completed timestamp 1788439951.327611
  (about 1.52 seconds later).
- Created another reminder, rescheduled it, and cancelled it through the UI.
- Confirmed persisted cancellation and revised run time.
- Inspected desktop screenshots and corrected panel spacing and alert positioning.

## Automated coverage
tests/test_scheduling_integrations.py covers scheduling persistence, state transitions,
restart recovery, reminder delivery, recurrence, delivery failures, write failures,
TREE reminder parsing, app action queue handoff, plugin containment, MCP validation,
secret rejection, and config preservation. Existing Python and clipboard/dictation UI
regression suites are also run.

## Limits
KIRA must remain open to execute on time. Pending overdue work executes on reopening;
this does not wake the Mac or write to Apple Reminders. Actual native app launch/quit
was not triggered during the isolated test. Agent tasks still depend on the model and
normal action permissions. Plugin registration is not arbitrary-code execution; MCP
validation is static configuration validation, not a protocol handshake/tool call.
