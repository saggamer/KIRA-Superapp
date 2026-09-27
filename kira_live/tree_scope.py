"""Request-local Live routing, never a mutation of the selected daily driver."""

from contextvars import ContextVar

live_tree_scope = ContextVar("kira_live_tree_scope", default=False)
