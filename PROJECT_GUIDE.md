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
- Create a simple test: insert 1000 items from a Zipf distribution, query each, compare against ground truth
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
- Insert 1000 items with keys from Zipf and values (latencies) from a known distribution (e.g., lognormal)
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
