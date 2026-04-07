# pip install mmh3 numpy
"""
Checkpoint 1 — Basic sketch accuracy on a Zipf frequency stream.

Inserts 10 000 items whose keys are drawn from a Zipf distribution (value=1
each, pure frequency counting — no histograms yet).  Queries every unique
key and compares against the ground-truth count.

Expected behaviour
------------------
CMS    : always overestimates  → positive average signed error
CU-CMS : also overestimates, but less than CMS  → smaller positive average
CS     : unbiased (errors in both directions)   → average signed error ≈ 0
"""

import numpy as np
from collections import defaultdict

from sketches.cms import CountMinSketch
from sketches.cu_cms import ConservativeUpdateCMS
from sketches.cs import CountSketch

W = 1024
D = 3
N_ITEMS = 10_000
ZIPF_ALPHA = 1.5
SEED = 42


def run():
    rng = np.random.default_rng(SEED)

    # Zipf returns integers ≥ 1; use them directly as flow keys.
    keys = rng.zipf(ZIPF_ALPHA, N_ITEMS)

    ground_truth: dict[int, int] = defaultdict(int)
    for k in keys:
        ground_truth[k] += 1

    cms = CountMinSketch(W, D)
    cu_cms = ConservativeUpdateCMS(W, D)
    cs = CountSketch(W, D)

    for k in keys:
        cms.update(k, 1)
        cu_cms.update(k, 1)
        cs.update(k, 1)

    unique_keys = list(ground_truth.keys())
    cms_err, cu_err, cs_err = [], [], []

    for k in unique_keys:
        true_val = ground_truth[k]
        cms_err.append(cms.query(k) - true_val)
        cu_err.append(cu_cms.query(k) - true_val)
        cs_err.append(cs.query(k) - true_val)

    cms_err = np.array(cms_err)
    cu_err = np.array(cu_err)
    cs_err = np.array(cs_err)

    print(f"Checkpoint 1  |  {N_ITEMS} items, Zipf α={ZIPF_ALPHA}, w={W}, d={D}")
    print(f"{'Sketch':<10} {'avg signed err':>16} {'avg abs err':>13} {'max abs err':>13}")
    print("-" * 56)
    for name, err in [("CMS", cms_err), ("CU-CMS", cu_err), ("CS", cs_err)]:
        print(
            f"{name:<10} {np.mean(err):>16.3f} {np.mean(np.abs(err)):>13.3f}"
            f" {np.max(np.abs(err)):>13.3f}"
        )

    print()
    print("Expected: CMS avg signed err > 0 (overestimates)")
    print("          CU-CMS avg signed err > 0 but smaller than CMS")
    print("          CS avg signed err ≈ 0 (errors in both directions)")


if __name__ == "__main__":
    run()
