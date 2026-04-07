import numpy as np
from .base import BaseSketch


class CountMinSketch(BaseSketch):
    """
    Count-Min Sketch (CMS).

    d independent hash functions, each mapping keys to [0, w).
    update: increment counters[j][hash_j(key)] by value for every row j.
    query:  return min over j of counters[j][hash_j(key)].
    """

    def __init__(self, w: int = 1024, d: int = 3):
        super().__init__(w, d)
        self.counters = np.zeros((d, w), dtype=np.int32)

    def update(self, key, value: int = 1):
        for j in range(self.d):
            self.counters[j, self._hash(key, j)] += value

    def query(self, key) -> int:
        return int(min(self.counters[j, self._hash(key, j)] for j in range(self.d)))
