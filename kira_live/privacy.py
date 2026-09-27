from __future__ import annotations

"""Single public-output boundary shared by text and native live runtimes."""

from html import unescape
import re


MODEL_BOUNDARY_MARKERS = (
    "<turn|>", "<|end_of_turn|>", "<end_of_turn>", "<|im_end|>", "<eos>"
)


def sanitize_public_output(content: object, *, live: bool = False) -> str:
    """Remove private reasoning and protocol tails before UI/TTS/persistence."""
    text = unescape(str(content or ""))
    if live:
        # Live checkpoints sometimes spell private blocks as [thought] or
        # [Thinking]. An unfinished block must never reach text or speech.
        names = r"thought|thinking|think|analysis|reasoning"
        text = re.sub(rf"(?is)\[\s*(?:{names})\s*\].*?\[\s*/(?:{names})\s*\]", "", text)
        text = re.sub(rf"(?is)\[\s*(?:{names})\s*\].*$", "", text)
        text = re.sub(r"(?is)<(?:thought|thinking)>.*?(?:</(?:thought|thinking)>|$)", "", text)
    boundary = min(
        (position for marker in MODEL_BOUNDARY_MARKERS
         if (position := text.find(marker)) >= 0),
        default=-1,
    )
    if boundary >= 0:
        text = text[:boundary]
    private_names = (
        r"think|thought[_ -]?process|analysis|reasoning|chain_of_thought|"
        r"private_reasoning|scratchpad"
    )
    text = re.sub(rf"(?is)<(?:{private_names})>.*?</(?:{private_names})>", "", text)
    text = re.sub(rf"(?is)<(?:{private_names})>.*$", "", text)
    text = re.sub(rf"(?is)^.*?</(?:{private_names})>", "", text)
    text = re.sub(
        r"(?is)\[\s*(?:THOUGHT_PROCESS|PRIVATE_REASONING|CHAIN_OF_THOUGHT|"
        r"TOT_REASONING|SCRATCHPAD)\s*\].*$",
        "",
        text,
    )
    text = re.sub(r"(?is)<\|channel\>\s*(?:thought|analysis|reasoning)?.*$", "", text)
    text = re.sub(r"(?is)<\|/?(?:think|analysis|reasoning|channel|end)\|>", "", text)
    return text.strip()
