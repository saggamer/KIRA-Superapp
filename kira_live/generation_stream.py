"""Bind a generator's stream locally without mutating MLX-LM's global state."""
from types import FunctionType


def bind_generation_stream(function, stream):
    namespace = dict(function.__globals__)
    namespace["generation_stream"] = stream
    bound = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound
