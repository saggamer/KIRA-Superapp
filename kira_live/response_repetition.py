"""Discourage copying long prior replies without banning ordinary words."""
import re


def phrase_continuations(previous, order=8):
    table = {}
    for reply in previous:
        for index in range(max(0, len(reply) - order + 1)):
            key = tuple(reply[index:index + order - 1])
            table.setdefault(key, set()).add(int(reply[index + order - 1]))
    return table


def repetition_requested(text):
    return bool(re.search(r'\b(?:repeat|again|verbatim|say that|read (?:it|that) back)\b', text, re.I))


def make_reply_copy_penalty(previous):
    table = phrase_continuations(previous)
    def process(tokens, logits):
        if len(tokens) < 7 or not table:
            return logits
        next_tokens = table.get(tuple(tokens[-7:].tolist()), ())
        if next_tokens:
            import mlx.core as mx
            return logits.at[:, mx.array(sorted(next_tokens), dtype=mx.int32)].add(-4.0)
        return logits
    return process
