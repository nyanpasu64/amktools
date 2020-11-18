from typing import TypeVar, Optional


def ceildiv(n: int, d: int) -> int:
    return -(-n // d)


T = TypeVar("T")


def coalesce(*args: Optional[T]) -> T:
    if len(args) == 0:
        raise TypeError("coalesce expected >=1 argument, got 0")
    for arg in args:
        if arg is not None:
            return arg
    raise TypeError("coalesce() called with all None")
