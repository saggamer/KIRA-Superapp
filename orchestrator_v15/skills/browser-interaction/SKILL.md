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
For a button use OPERATION: click; for a dropdown use OPERATION: select_option with its observed option VALUE. Preserve user-provided values across steps. After acting, use the returned observation directly; it already describes the updated page. Read working_state.last_change to see whether the page or content changed. Avoid spending another observe action when the action result already contains the needed controls or confirmation.
Use the current tabId/observationId for indexed targets. Old links in candidate_links can be navigated to; old controls cannot be clicked on the current page. A stale-target error requires a fresh observation, then choose the current control. inspect_page supplies structural links and forms, not fresh click targets. If controls were omitted from compact evidence, inspect or narrow the page instead of inventing a selector.
Check working_state.current.state and recent_failures before retrying. For empty content or unchanged readbacks, inspect_page, navigate to an observed alternative or refine the search. A blocked source does not make unrelated authorized steps impossible. For access challenges or required sign-in, ask the user only when that site is necessary; otherwise use an appropriate public source. Do not attempt to bypass a challenge.
To read lower content use ACTION: scroll and PIXELS: 700 (negative scrolls up). Use visibleText and the returned scroll state; stop at atEnd or when moved is false. If a target is absent, refresh the observation rather than repeating a failed action unchanged.
Proceed autonomously on authorized steps; do not ask again for routine navigation, scrolling, filling or read-only inspection. Preserve an explicit review-before-submit boundary. Send an authorized final submission once, then verify confirmation; uncertain confirmation is not permission to submit twice. Kira's existing permission handler remains authoritative.
