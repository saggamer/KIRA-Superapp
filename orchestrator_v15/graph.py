"""Bounded evidence DAG. Limits are enforced by code, independently of a model."""
from dataclasses import dataclass, field
import time


@dataclass(frozen=True)
class Limits:
    nodes: int = 6
    concurrent: int = 3
    depth: int = 2
    tool_calls: int = 12
    replans: int = 2
    tokens: int = 2048
    seconds: float = 180


@dataclass
class Node:
    name: str
    parents: tuple[str, ...]
    depth: int
    status: str = "pending"
    evidence: dict = field(default_factory=dict)


class EvidenceGraph:
    def __init__(self, limits=Limits()):
        self.limits = limits
        self.nodes = {}
        self.calls = self.replans = self.tokens = 0
        self.started = time.monotonic()

    def _live(self):
        if time.monotonic() - self.started >= self.limits.seconds:
            raise RuntimeError("wall-clock budget exhausted")

    def add(self, name, parents=()):
        self._live()
        if not name or name in self.nodes or len(self.nodes) >= self.limits.nodes:
            raise ValueError("invalid, duplicate, or excess node")
        if len(set(parents)) != len(parents) or any(p not in self.nodes for p in parents):
            raise ValueError("parents must be distinct existing nodes")
        depth = 0 if not parents else 1 + max(self.nodes[p].depth for p in parents)
        if depth > self.limits.depth:
            raise ValueError("depth budget exhausted")
        self.nodes[name] = Node(name, tuple(parents), depth)

    def start(self, name):
        self._live()
        node = self.nodes[name]
        if node.status != "pending" or any(self.nodes[p].status != "verified" for p in node.parents):
            raise ValueError("dependencies are not verified or node is not pending")
        if sum(n.status == "running" for n in self.nodes.values()) >= self.limits.concurrent:
            raise ValueError("concurrency budget exhausted")
        node.status = "running"

    def consume(self, tokens=0, tool_call=False, replan=False):
        self._live()
        if type(tokens) is not int or tokens < 0:
            raise ValueError("invalid token count")
        counts = (self.tokens + tokens, self.calls + bool(tool_call), self.replans + bool(replan))
        if any(a > b for a, b in zip(counts, (self.limits.tokens, self.limits.tool_calls, self.limits.replans))):
            raise ValueError("execution budget exhausted")
        self.tokens, self.calls, self.replans = counts

    def verify(self, name, evidence, verifier):
        self._live()
        node = self.nodes[name]
        if node.status != "running" or not evidence or not verifier(evidence):
            raise ValueError("execution evidence failed verification")
        node.evidence = dict(evidence)
        node.status = "verified"

    def diagonal_evidence(self, names):
        """A downstream branch can reuse verified sibling evidence without re-running it."""
        self._live()
        if any(self.nodes[n].status != "verified" for n in names):
            raise ValueError("cannot transfer unverified evidence")
        return {n: dict(self.nodes[n].evidence) for n in names}

    def complete(self):
        self._live()
        return bool(self.nodes) and all(n.status == "verified" for n in self.nodes.values())
