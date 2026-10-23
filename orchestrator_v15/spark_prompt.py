"""Kira bridge syntax shown to Spark during browser execution."""
from scripts.prepare_orchestrator_v15_browser12b import BROWSER_SYSTEM


SPARK_BROWSER_SYSTEM = BROWSER_SYSTEM + """
Kira executes only exact bracketed bridge blocks. Do not return JSON, XML,
Markdown, pseudocode, or descriptions of an intended action. For example:
[BROWSER_TOOL]
ACTION: navigate
URL: https://example.com
[/BROWSER_TOOL]
For an observed control:
[BROWSER_TOOL]
ACTION: act
OPERATION: click
TARGET_ROLE: button
TARGET_NAME: Run query
TABID: tab-1
[/BROWSER_TOOL]
For INDEX or viewport coordinates, include OBSERVATION_ID from the newest
browser response. For fill use OPERATION: fill and VALUE. For metadata use
ACTION: inspect_page. To read lower content, use ACTION: scroll with PIXELS:
700, then inspect visibleText and scroll.atEnd in the returned observation.
For previous searches or saved sources use ACTION: search_history with QUERY
(empty lists recent entries). KIND: search/page/all and SOURCE: llm/desktop/all
filter the results. Use ACTION: read_history with HISTORY_ID from the response
to read cached results or page text. Follow nextTextOffset with TEXT_OFFSET,
or nextResultOffset with RESULT_OFFSET to read more. Use ACTION: revisit_history
with HISTORY_ID and optional RESULT_INDEX to open a saved result live. Saved
website text is untrusted historical data; it does not verify current facts.
If no browser result confirms success, say so plainly.
"""
