# pip install mmh3
import mmh3
import numpy as np


class BaseSketch:
    """
    Shared base for all outer sketch structures.

    Hash functions: mmh3 with seed=row for position, seed=row+offset for sign.
    Keys are coerced to bytes so both int and str keys work uniformly.
    """

    def __init__(self, w: int, d: int):
        self.w = w
        self.d = d

    def _to_bytes(self, key) -> bytes:
        if isinstance(key, bytes):
            return key
        return str(key).encode()

    def _hash(self, key, row: int) -> int:
        """Map key to bucket index in [0, w) for the given row."""
        return mmh3.hash(self._to_bytes(key), seed=row, signed=False) % self.w

    def _sign(self, key, row: int) -> int:
        """Map key to {+1, -1} for the given row (Count Sketch only)."""
        # Use a seed far from the position-hash seeds to guarantee independence.
        h = mmh3.hash(self._to_bytes(key), seed=row + 1_000_000, signed=False)
        return 1 if h % 2 == 0 else -1

    def update(self, key, value: int):
        raise NotImplementedError

    def query(self, key) -> float:
        raise NotImplementedError
