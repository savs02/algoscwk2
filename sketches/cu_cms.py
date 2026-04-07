import numpy as np
from .base import BaseSketch


class ConservativeUpdateCMS(BaseSketch):
    """
    Conservative Update Count-Min Sketch (CU-CMS).

    Same structure as CMS, but update only raises a counter when it is
    strictly below the current minimum estimate + value.  This prevents
    counters from being inflated by hash collisions with heavier flows,
    so estimates are tighter (less overestimation) than plain CMS.

    update: estimate = min_j counters[j][hash_j(key)]
            for each row j: counters[j][h] = max(counters[j][h], estimate + value)
    query:  same as CMS — min over rows.
    """

    def __init__(self, w: int = 1024, d: int = 3):
        super().__init__(w, d)
        self.counters = np.zeros((d, w), dtype=np.int32)

    def update(self, key, value: int = 1):
        buckets = [self._hash(key, j) for j in range(self.d)]
        estimate = min(self.counters[j, buckets[j]] for j in range(self.d))
        new_val = estimate + value
        for j in range(self.d):
            h = buckets[j]
            if self.counters[j, h] < new_val:
                self.counters[j, h] = new_val

    def query(self, key) -> int:
        return int(min(self.counters[j, self._hash(key, j)] for j in range(self.d)))
