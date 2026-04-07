import numpy as np
from .base import BaseSketch


class CountSketch(BaseSketch):
    """
    Count Sketch (CS).

    d hash functions mapping keys to [0, w) and d sign functions mapping
    keys to {+1, -1}.  Signs cancel collisions in expectation, so errors
    are unbiased (both over- and under-estimation occur).

    update: for each row j, counters[j][hash_j(key)] += sign_j(key) * value
    query:  median over j of (counters[j][hash_j(key)] * sign_j(key))
    """

    def __init__(self, w: int = 1024, d: int = 3):
        super().__init__(w, d)
        self.counters = np.zeros((d, w), dtype=np.int32)

    def update(self, key, value: int = 1):
        for j in range(self.d):
            self.counters[j, self._hash(key, j)] += self._sign(key, j) * value

    def query(self, key) -> float:
        estimates = [
            self.counters[j, self._hash(key, j)] * self._sign(key, j)
            for j in range(self.d)
        ]
        return float(np.median(estimates))
