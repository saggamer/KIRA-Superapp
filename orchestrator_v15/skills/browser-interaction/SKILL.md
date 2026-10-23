---
name: browser-interaction
description: Operate page controls, scroll, complete forms and verify requested browser actions without repeating submissions.
---
Observe the page before choosing a control. Use the latest returned role/name or element identity; never guess coordinates, tab IDs or labels. One proposal contains exactly one closed action block:
[BROWSER_TOOL]
ACTION: act
OPERATION: fill
TARGET_ROLE: textbox
TARGET_NAME: observed label
VALUE: user-requested text
[/BROWSER_TOOL]
For a button use OPERATION: click; for a dropdown use OPERATION: select_option with its observed option VALUE. Preserve user-provided values across steps. After acting, inspect the returned observation and verify the requested state before reporting success.
To read lower content use ACTION: scroll and PIXELS: 700 (negative scrolls up). Use visibleText and the returned scroll state; stop at atEnd or when moved is false. If a target is absent, refresh the observation rather than repeating a failed action unchanged.
Proceed autonomously on authorized steps; do not ask again for routine navigation, scrolling, filling or read-only inspection. Preserve an explicit review-before-submit boundary. Send an authorized final submission once, then verify confirmation; uncertain confirmation is not permission to submit twice. Kira's existing permission handler remains authoritative.
