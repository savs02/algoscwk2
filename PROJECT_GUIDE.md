# Differenced Histogram Sketch (DHS) — Implementation Guide

## Project Summary

We are building a system that detects flows whose latency distribution shape changes over time ("heavy distributional changers"). We combine two existing ideas that have never been combined before:

1. **HistSketch (ICDE 2023)**: puts a small histogram inside each sketch bucket to track per-flow distributions
2. **Sketch-based change detection (Krishnamurthy et al., IMC 2003)**: subtracts two sketches from adjacent time epochs to find flows with large changes

Our system maintains multiple snapshots of histogram-in-sketch structures over time, subtracts adjacent snapshots bin-by-bin, and flags flows whose distribution changed significantly. We compare three outer sketch structures (CMS, CU-CMS, Count Sketch) to see which handles subtraction noise best.

Language: Python (numpy for arrays, matplotlib for plots). No network simulator needed.

---

## Step 1: Basic Sketch Structures (no histograms yet)

Build three outer sketch structures as classes. Each should support `update(key, value)` and `query(key)`.

### 1a: Count-Min Sketch (CMS)
- Parameters: width `w`, depth `d`
- `d` independent hash functions mapping keys to range `[0, w)`
- `update(key, value)`: for each row `j`, increment `counters[j][hash_j(key)]` by `value`
- `query(key)`: return `min over j of counters[j][hash_j(key)]`
- All counters initialised to 0

### 1b: Conservative Update CMS (CU-CMS)
- Same structure as CMS
- `update(key, value)`: compute current estimate = `min over j of counters[j][hash_j(key)]`. Then for each row `j`, set `counters[j][hash_j(key)] = max(counters[j][hash_j(key)], estimate + value)`
- `query(key)`: same as CMS

### 1c: Count Sketch (CS)
- Parameters: width `w`, depth `d`
- `d` hash functions mapping keys to range `[0, w)` AND `d` sign functions mapping keys to `{+1, -1}`
- `update(key, value)`: for each row `j`, increment `counters[j][hash_j(key)]` by `sign_j(key) * value`
- `query(key)`: return `median over j of (counters[j][hash_j(key)] * sign_j(key))`

### Checkpoint 1
- Create a simple test: insert 10000 items from a Zipf distribution, query each, compare against ground truth
- Print average error for each sketch type
- CMS should overestimate, CS should have errors in both directions, CU-CMS should overestimate less than CMS

---

## Step 2: Histogram Buckets

Replace the single counter in each bucket with a small array of `B` counters (bins). Each bin corresponds to a range of values (e.g., latency ranges).

### 2a: Bin Configuration
- Parameter: `num_bins` (e.g., 8 or 16)
- Parameter: `bin_edges` — array of `num_bins + 1` boundary values
- Support two schemes:
  - **Uniform**: evenly spaced between min and max value
  - **Logarithmic**: log-spaced (like DDSketch), e.g., boundaries at `[0, 1, 2, 4, 8, 16, 32, 64, 128]`
- Function `get_bin(value)` returns which bin index a value falls into

### 2b: Histogram-in-Sketch Structure
- For each sketch type (CMS, CU-CMS, CS), replace `counters[j][i]` (a single integer) with `counters[j][i][b]` (an array of `num_bins` integers)
- `update(key, value)`:
  - Compute `bin_idx = get_bin(value)`
  - For CMS: for each row `j`, increment `counters[j][hash_j(key)][bin_idx]` by 1
  - For CU-CMS: same conservative update logic but applied per-bin
  - For CS: for each row `j`, increment `counters[j][hash_j(key)][bin_idx]` by `sign_j(key)`
- `query_histogram(key)`:
  - For CMS/CU-CMS: for each bin `b`, return `min over j of counters[j][hash_j(key)][b]`
  - For CS: for each bin `b`, return `median over j of (counters[j][hash_j(key)][b] * sign_j(key))`
  - Return the full histogram array of length `num_bins`

### Checkpoint 2
- Insert 10000 items with keys from Zipf and values (latencies) from a known distribution (e.g., lognormal)
- For a few specific keys, compare queried histogram against ground truth histogram
- Print per-bin errors

---

## Step 3: Temporal Snapshotting

### 3a: Epoch Manager
- Parameter: `num_snapshots` (K, e.g., 4)
- Maintain a ring buffer of K histogram-in-sketch instances
- Track current epoch index
- Function `advance_epoch()`: move to next slot in ring buffer, clear the new sketch
- Function `get_current_sketch()`: return the sketch for the active epoch
- Function `get_previous_sketch(offset)`: return the sketch from `offset` epochs ago

### 3b: Stream Processor
- Reads a stream of `(timestamp, key, value)` tuples
- Parameter: `epoch_duration` (e.g., number of items or time in seconds)
- For each item, insert into the current epoch's sketch
- When epoch boundary is reached, call `advance_epoch()`

### Checkpoint 3
- Feed a synthetic stream of 10000 items split across 4 epochs
- Verify that each epoch's sketch only contains data from that epoch
- Query a key in each epoch and confirm the histograms differ

---

## Step 4: Histogram Differencing

### 4a: Subtract Two Histogram Sketches
- Function `diff_histograms(sketch_A, sketch_B, key)`:
  - Get `hist_A = sketch_A.query_histogram(key)`
  - Get `hist_B = sketch_B.query_histogram(key)`
  - Return `hist_A[b] - hist_B[b]` for each bin `b`
  - Output: a signed array of length `num_bins`

### 4b: Change Magnitude Metrics
- Given a differenced histogram `diff`, compute:
  - **L1 norm**: `sum of |diff[b]|` for all bins — captures total change
  - **L2 norm**: `sqrt(sum of diff[b]^2)` — emphasises large bin changes
  - **Max-bin**: `max of |diff[b]|` — captures single largest bin change

### 4c: Heavy Distributional Changer Detection
- For each key in the stream, compute the differenced histogram between current and previous epoch
- Compute L1, L2, and max-bin scores
- Flag as heavy distributional changer if score exceeds a threshold
- The threshold can be a fixed value or relative to the total number of items

### Checkpoint 4
- Create two epochs: epoch 1 has flow X with lognormal(mean=5, std=1), epoch 2 has flow X with lognormal(mean=10, std=1)
- The differenced histogram should clearly show bins shifting from low to high
- Flow Y stays the same in both epochs — its diff should be near zero
- Print the differenced histograms and scores for both flows

---

## Step 5: Change Type Classification

Given a differenced histogram, classify the type of change:

- **Spike**: one or two adjacent bins have large positive values, rest near zero → sudden latency jump
- **Shift**: low bins negative, high bins positive (or vice versa) → distribution moved left or right
- **Spread**: centre bins negative, tail bins positive → distribution got wider
- **Volume change**: all bins roughly proportionally positive or negative → just more/fewer packets
- **Disappearance**: all bins negative → flow stopped

### Implementation
- Simple heuristic rules based on the shape of the diff histogram
- No ML needed — just check which pattern the diff matches best

### Checkpoint 5
- Inject each change type synthetically
- Verify the classifier labels them correctly

---

## Step 6: Synthetic Stream Generator

### 6a: Background Traffic
- Generate `num_flows` flows (e.g., 10000)
- Flow frequencies follow Zipf distribution with parameter `alpha` (e.g., 1.0 to 2.0)
- Each flow's latency follows a lognormal distribution with per-flow parameters

### 6b: Anomaly Injection
- Select a few target flows and inject changes at specific epoch boundaries:
  - **Sudden spike**: latency mean doubles in one epoch
  - **Gradual ramp**: latency mean increases by 20% each epoch
  - **Periodic burst**: latency alternates between normal and high every other epoch
  - **Spread**: latency std doubles but mean stays same
  - **Disappearance**: flow stops sending packets
- Record ground truth: which flows changed, what type, which epoch

### Checkpoint 6
- Generate a stream, feed it through the system
- Print detected changers vs ground truth changers
- Compute F1 score (precision and recall)

---

## Step 7: Full Evaluation

### 7a: CMS vs CU-CMS vs CS comparison
- Run the same stream through all three outer sketch types
- Same total memory budget for each (width * depth * num_bins * counter_size)
- Compare F1 scores for heavy distributional changer detection
- Plot: F1 score vs sketch type for each change type

### 7b: Number of bins sweep
- Fix total memory, vary `num_bins` in {4, 8, 16, 32}
- As bins increase, sketch width must decrease (fixed memory)
- Plot: F1 score vs num_bins for each sketch type

### 7c: Number of snapshots sweep
- Fix total memory, vary `num_snapshots` K in {2, 4, 8, 16}
- As K increases, per-snapshot memory decreases
- Plot: F1 score vs K

### 7d: Zipf skewness sweep
- Fix memory and sketch parameters, vary Zipf alpha in {0.5, 1.0, 1.5, 2.0}
- Plot: F1 score vs alpha for each sketch type

### 7e: Detection sensitivity
- Fix everything, vary the magnitude of injected change (e.g., latency mean multiplied by {1.1, 1.2, 1.5, 2.0, 5.0})
- Plot: F1 score vs change magnitude — shows the minimum detectable change

### 7f: Change type classification accuracy
- For each injected change type, measure classification accuracy
- Confusion matrix: predicted type vs actual type

### 7g: Uniform vs logarithmic bins
- Compare the two bin spacing schemes across all experiments
- Plot: F1 score for uniform vs log bins

### Checkpoint 7
- All plots generated and saved as PNG files
- Summary table of results printed to console

---

## File Structure

```
project/
├── sketches/
│   ├── cms.py          # Count-Min Sketch with histogram buckets
│   ├── cu_cms.py       # Conservative Update CMS with histogram buckets
│   ├── cs.py           # Count Sketch with histogram buckets
│   └── base.py         # Shared base class and bin configuration
├── temporal/
│   ├── epoch_manager.py    # Ring buffer of sketch snapshots
│   ├── differencer.py      # Histogram subtraction and scoring
│   └── change_detector.py  # Thresholding and change classification
├── data/
│   ├── stream_generator.py # Synthetic Zipf stream with anomaly injection
│   └── trace_loader.py     # Optional: load CAIDA or other real traces
├── evaluation/
│   ├── experiments.py      # All experiment configurations from Step 7
│   └── plotting.py         # Generate all plots
├── main.py                 # Entry point: run all experiments
└── README.md
```

---

## Important Notes

- Use deterministic seeding (numpy random seed) so experiments are reproducible
- All hash functions should be independent — use `mmh3` (murmurhash3) library with different seeds per row
- Memory budget should be measured as: `width * depth * num_bins * counter_bytes`
- Counter size: 4 bytes (32-bit integers) unless otherwise specified
- For Count Sketch median computation, use `numpy.median`
- Save all plots to an `outputs/` directory

---

## Design Decisions, Limitations, and Improvements

### What `main.cpp` is

`src/main.cpp` is a **consolidated improved pipeline runner**. It is not a direct replacement for `checkpoint6.cpp`. The individual checkpoint files (`checkpoint1.cpp` through `checkpoint6.cpp`) remain the authoritative baseline implementations. `main.cpp` runs the same stages in one binary, then adds extra analysis where that is useful.

---

### What improved after the baseline checkpoints

The repository now goes beyond the minimum checkpoint implementations in a few places that make the results easier to justify:

- **Stage 3** now supports timestamped `(timestamp, key, value)` streams and checks snapshot isolation more directly, instead of relying only on qualitative output.
- **Stage 4** now treats histogram differencing as a metric-driven step (`L1`, `L2`, `max-bin`) rather than implicitly using only one score.
- **Stage 5** now validates the classifier against both a changed flow and a stable background flow, so false positives are visible instead of hidden. It also includes a robustness sweep, which makes it clearer that the classifier is heuristic rather than formally guaranteed.
- **Stage 6** now separates the clean checkpoint run from broader empirical analysis. That keeps the checkpoint result intact while still showing how detector behaviour changes under different thresholds, seeds, and signal strengths.
- The anomaly generator API is also stricter now: `AnomalySpec.magnitude` must be set explicitly, which avoids silent defaults changing the experiment unintentionally.

---

### Stage 2: Sketch accuracy is now PASS/FAIL

The original Stage 2 table was purely informational. A PASS/FAIL check was added that runs at `w=64` (high collision pressure) for the top flow. Pass criteria:
- CU-CMS average absolute per-bin error < CMS average absolute per-bin error
- CMS average signed per-bin error > 0 (confirms overestimation bias)

This directly verifies the expected accuracy ordering and is the clearest place in the pipeline where CMS, CU-CMS, and CS behave differently.

---

### Stage 5: Robustness sweep includes small-N cases

The original sweep only varied width and seed, and returned 100% accuracy at all settings (not informative). The sweep now also varies packets-per-flow `N ∈ {1000, 200, 100}`:
- N=1000: 100% (clean baseline)
- N=200: ~60% — classification starts failing at moderate packet counts
- N=100: 20–40% — most heuristics fail under low-signal conditions

The classifier should be described as **rule-based heuristic classification, empirically validated on synthetic patterns** — not as a principled or guaranteed method.

---

### Stage 6: Baseline vs Improved detector (side-by-side)

`main.cpp` Stage 6 runs both detector variants against identical anomaly setups. GradualRamp stays at magnitude=1.2 (the original checkpoint value) so anomaly difficulty does not change between runs.

| Variant | Decision rule | F1 | TP/FP/FN |
|---------|--------------|-----|----------|
| Baseline | raw L1 > 300 | 0.714 | 5/0/4 |
| Improved | L1/baseline > 0.15 && L1_raw > 30 | 0.900 | 9/2/0 |

**Why the baseline misses 4 detections**: A single absolute threshold penalises lighter flows. Flow 5 (Disappearance) has ~186 packets so its raw L1 never crosses 300 even when the flow vanishes. Spread (L1 ≈ 162) and GradualRamp (L1 ≈ 270, 190) have the same problem.

**Pass/fail gating**: `main.cpp` gates on the **baseline** passing (F1 ≥ 0.7), matching the original checkpoint criterion. The improved variant is reported for comparison only.

**Why this is beyond Checkpoint 6**: the original checkpoint only asked for a single end-to-end detector run with detected changers vs ground truth and an F1 score. `main.cpp` keeps that baseline run, but then adds a second detector formulation, side-by-side comparison, ablation, threshold selection, and robustness reporting. So Stage 6 in `main.cpp` should be read as an **extension built on top of Checkpoint 6**, not as the literal checkpoint specification.

---

### Stage 6: Ablation

Three detector conditions are compared on the same stream (seed=42, w=1024):

| Condition | Rule | F1 | Notes |
|-----------|------|----|-------|
| A | raw L1 > 300 | 0.714 | Baseline; high precision, weak recall |
| B | L1/baseline > 0.15 | 0.066 | 253 FP — normalisation without a floor is unusable |
| C | L1/baseline > 0.15 && L1_raw > 30 | 0.900 | Full improved; floor is essential |

The ablation shows that the absolute floor is load-bearing: normalisation alone flags near-empty flows with tiny fluctuations as anomalies.

---

### Stage 6: Threshold selection (validation/test split)

Thresholds are tuned on validation seeds {42, 43} and reported on the held-out test seed {44}:

- Candidate normalised thresholds: 0.10, 0.12, 0.15, 0.18, 0.20, 0.25
- Best validation threshold: **0.25** (val avg F1 = 0.950 across 3 sketches × 2 seeds)
- Test result (seed 44): **F1 = 0.941**, TP=8, FP=0, FN=1

At threshold=0.25 the two Spread boundary 1/2 false positives (L1_norm ≈ 0.21) drop out, giving zero false positives on the test seed. The one missed detection is GradualRamp at boundary 2 on seed 44 (signal weaker at that draw).

---

### Sketch type differentiation

**Under same-seed epochs (normal operation):** CMS, CU-CMS, and CS produce identical F1 scores in Stages 4–6. This is expected: when the same hash seed is used across epochs, any systematic overcount bias in epoch *e* is exactly replicated in epoch *e+1* — it cancels in the diff. The three sketch types differ only in single-epoch histogram accuracy, which Stage 2 (w=64 check) directly measures and gates on.

**Under hash rotation (different seed each epoch):** the bias no longer cancels, and sketch type becomes the dominant factor. A controlled experiment (Stage 4 hash-rotation experiment section) sweeps 10 seed pairs (s, s+1) for s=0..9 using w=64, d=3, 200 Zipf flows, N=5000, and *identical data in both epochs* so all residual L1 is pure sketch bias — no sampling variance. Results (mean ± std across 10 pairs):

| Sketch | flow "1" mean ± std | flow "10" mean ± std | Interpretation |
|--------|---------------------|----------------------|----------------|
| CMS    | 3.5 ± 7.3           | 1.5 ± 1.2            | Unpredictable: sometimes 0, occasionally spikes to 25 |
| CU-CMS | **0.0 ± 0.0**       | **0.0 ± 0.0**        | Exactly zero for all 10 pairs, both flows |
| CS     | 4.0 ± 2.8           | 3.8 ± 1.8            | Consistently nonzero; tighter variance than CMS on the heavy flow, worse on the light flow |

**Why CU-CMS is zero:** with D=3 rows and W=64 and Zipf background, there is almost always at least one row with no heavy collision partner for a given bin. The min-over-rows picks that clean row and recovers the exact true count regardless of hash seed. CMS cannot do this: its overcount is strictly additive, so changing the collision partner changes the magnitude by a variable amount. CS has no systematic bias but the different sign functions across seeds produce non-cancelling per-realisation variance.

**Why CMS std > mean for flow "1":** most seed pairs produce 0 or near-0 residual (collision partners happen to give similar overcounts), but a few pairs spike to 25. This makes CMS the *least predictable* sketch under hash rotation — you cannot know in advance which pairs are "bad".

**Practical implication:** if hash seeds are fixed across epochs (the DHS pipeline default), all three sketches are equivalent for detection. If hash rotation is ever required (e.g., adversarial resilience, sketch rebuild after restart), CU-CMS is the clear choice — it achieves exactly zero residual in all tested configurations. CMS is the worst (high-variance spikes raise the noise floor unpredictably). CS sits between the two.

**Safe claim**: sketch differences are clearest in single-epoch estimation (Stage 2) and under hash rotation (Stage 4 experiment). Under same-seed normal operation, detector design dominates and sketch choice has no measurable effect on F1.

**Do not claim**: one sketch is best under all conditions, or that the CU-CMS=0 result generalises to much smaller widths or much heavier sketches without re-running the sweep.

The `BaseSketch` constructor accepts an optional `uint32_t seed` parameter (default=0, preserving all existing behaviour). This is the knob used in the hash-rotation experiment and is available for any future cross-seed study.

---

### Stage 5 classifier

The Stage 5 change-type classifier is **rule-based heuristic**: it matches diff histogram shapes against five empirical patterns (Disappearance, VolumeChange, Spike, Shift, Spread). It is validated on synthetic cases only and carries no formal guarantees outside that regime. Accuracy degrades to ~60% at N=200 packets per flow and ~20–40% at N=100.

---

### AnomalySpec API

`AnomalySpec.magnitude` no longer has a default value. All callers must set it explicitly. The per-type semantics differ:
- `SuddenSpike` / `PeriodicBurst`: anomaly-epoch mean is `exp(base_mu) * magnitude`
- `GradualRamp`: mean is multiplied by `magnitude^steps` per epoch step after `start_epoch`
- `Spread`: sigma is multiplied by `magnitude` from `start_epoch` onward
- `Disappearance`: `magnitude` is unused
