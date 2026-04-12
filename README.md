# Differenced Histogram Sketch (DHS)

## Problem Statement

Modern network monitoring must cope with millions of flows per second. The central question for this project is:

> **Which flows changed their latency distribution between two adjacent time windows — and what kind of change was it?**

A "distributional changer" is a flow whose per-packet latency distribution shifts in shape, not just in volume. The five change types we target are:

| Type | What changes |
|------|-------------|
| Sudden Spike | Mean doubles for one epoch, then returns |
| Gradual Ramp | Mean grows multiplicatively each epoch |
| Periodic Burst | Mean alternates high/normal every other epoch |
| Spread | Standard deviation grows while mean stays fixed |
| Disappearance | Flow stops sending entirely |

The naive approach — storing a full histogram per flow — is infeasible at line rate: with 10k flows and 10 bins each, that is 400 kB of per-flow state just for one snapshot, and you need multiple snapshots. The question is whether a *sketch* (a compact, hash-based summary) can replace the exact histogram without sacrificing detection accuracy.

---

## Approach: Differenced Histogram Sketch (DHS)

We combine two existing ideas that have not previously been combined:

1. **HistSketch (ICDE 2023)** — replaces the single integer in each sketch bucket with a small histogram array of B counters. Each update increments only the bin corresponding to the packet's latency value, so a `query_histogram(key)` reconstructs an approximate per-flow latency histogram from the sketch.

2. **Sketch-based change detection (Krishnamurthy et al., IMC 2003)** — maintains K sketch snapshots over time. Subtracting two adjacent snapshots bin-by-bin gives a "differenced histogram" whose shape reveals what kind of distributional change occurred.

The DHS pipeline is:

```
Stream of (timestamp, flow_key, latency)
         ↓
  StreamProcessor  — assigns packets to epochs by timestamp
         ↓
  EpochManager     — ring buffer of K sketch snapshots
         ↓
  diff_histograms  — elementwise subtraction of adjacent epoch sketches
         ↓
  compute_scores   — L1, L2, max-bin norms of the diff
         ↓
  is_heavy_changer — threshold gate on the chosen metric
         ↓
  classify_change  — heuristic shape classifier → Spike / Shift / Spread / …
```

Three sketch types are tested as the outer structure: **Count-Min Sketch (CMS)**, **Conservative-Update CMS (CU-CMS)**, and **Count Sketch (CS)**.

---

## Repository Layout

```
src/
├── sketches/
│   ├── base_sketch.hpp      Base class; MurmurHash-based hash families; seed parameter
│   ├── bin_config.hpp       Uniform and logarithmic bin edge schemes; get_bin()
│   ├── cms.hpp              Count-Min Sketch (always overestimates per bin)
│   ├── cu_cms.hpp           Conservative-Update CMS (tighter overestimate)
│   └── cs.hpp               Count Sketch (unbiased; signed updates; median query)
│
├── temporal/
│   ├── epoch_manager.hpp    Ring buffer of K sketch slots; advance_epoch()
│   ├── stream_processor.hpp Timestamp-driven epoch boundary detection
│   ├── differencer.hpp      diff_histograms(); compute_scores(); is_heavy_changer()
│   └── change_classifier.hpp  Heuristic shape cascade → ChangeType enum
│
├── generator/
│   └── stream_generator.hpp  Synthetic multi-epoch Zipf stream; AnomalySpec injection
│
├── main.cpp                 Consolidated pipeline runner (all 6 stages; see below)
├── evaluation.cpp           Report-facing evaluation sweeps (writes CSVs to outputs/evaluation/)
├── checkpoint1.cpp  …  checkpoint6.cpp   Original per-checkpoint baselines
│
include/
└── MurmurHash3.hpp          Fast non-cryptographic 32-bit hash
```

---

## Building and Running

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target main
./build/main
```

Report-facing evaluation sweeps:
```bash
cmake --build build --target evaluation
./build/evaluation
```

This writes CSV outputs under `outputs/evaluation/` for the evaluation section
tables and plots.

Plot and summarise the evaluation outputs:
```bash
python3 analysis/plot_evaluation.py
```

Individual checkpoint binaries:
```bash
cmake --build build --target checkpoint6
./build/checkpoint6
```

All stages pass: `./build/main` exits 0 and prints `ALL STAGES PASS`.

---

## What Each Stage Does

### Stage 1 — Basic sketch accuracy (informational)
- 10,000 Zipf items, w=1024, d=3, 1-bin config (reduces to frequency estimation)
- Confirms expected ordering: CMS overestimates, CU-CMS overestimates less, CS ≈ unbiased
- **Not gated** — informational only

### Stage 2 — Histogram estimation accuracy (PASS/FAIL)
- Lognormal latencies, 8-bin uniform and logarithmic configs
- Shows per-bin truth vs estimate for the top 3 Zipf flows
- **Pass gate at w=64** (high collision pressure, differences visible):
  - CU-CMS avg absolute error per bin < CMS avg absolute error per bin
  - CMS avg signed error > 0 (confirms overestimation)

### Stage 3 — Temporal snapshotting (PASS/FAIL)
- 4 epochs × 2500 items; lognormal mean increases each epoch
- Check A: each epoch's histogram peak shifts right (non-decreasing)
- Check B: each snapshot best-matches its own epoch's ground truth by L1 distance

### Stage 4 — Histogram differencing and heavy-changer detection (PASS/FAIL)
- Flow X shifts mean 5→10; Flow Y is stable
- Thresholds derived from Y's noise floor (2× rule, data-driven)
- X must be flagged on L1, L2, and max-bin; Y must not be flagged
- **Hash-rotation experiment** (informational): 10 seed pairs swept to show when sketch type matters (see below)

### Stage 5 — Change type classification (PASS/FAIL)
- Five synthetic change types (Disappearance, VolumeChange, Spike, Shift, Spread)
- Each tested against a stable background flow Y (false-positive check)
- **Gate: must pass on seeds 42, 43, AND 44** (cross-seed validation guards against seed-42-only overfitting)

### Stage 6 — End-to-end F1 evaluation (PASS/FAIL)
- 100 Zipf flows, 4 epochs × 5000 packets, 5 injected anomalies
- **Baseline detector** (raw L1 > 300): F1 = 0.714 — gates pass/fail at F1 ≥ 0.7
- **Improved detector** (normalised L1 with validated threshold): F1 = 0.941 on held-out seed 44
- Threshold selected on validation seeds {42, 43}; reported on held-out test seed {44}
- Ablation shows absolute floor is load-bearing (without it: 253 false positives)

---

## Key Results

### Detection (Stage 6)

| Detector | Rule | Seed | TP | FP | FN | F1 |
|----------|------|------|----|----|----|----|
| Baseline | raw L1 > 300 | 42 | 5 | 0 | 4 | 0.714 |
| Improved | L1/baseline > 0.25 && L1 > 30 | 44 (held-out) | 8 | 0 | 1 | **0.941** |

The baseline misses 4 detections because a single absolute threshold penalises lighter flows. Flow 5 (Disappearance, ~186 packets) and Flow 4 (Spread, ~162 L1) never cross 300. The normalised threshold fixes this: L1 divided by the flow's own baseline count makes the criterion scale-invariant.

### Ablation (Stage 6)

| Condition | Rule | F1 | Notes |
|-----------|------|----|-------|
| A | raw L1 > 300 | 0.714 | Baseline |
| B | L1/baseline > 0.15 | 0.066 | 253 false positives — floor is essential |
| C | L1/baseline > 0.15 && L1 > 30 | 0.900 | Full improved |

### Sketch comparison under hash rotation (Stage 4 experiment)

Under normal DHS operation (fixed hash seeds across epochs), CMS, CU-CMS, and CS produce **identical F1 scores** in Stages 4–6. This is because bias cancels in the diff: any overcount introduced in epoch e is the same overcount in epoch e+1.

The sketches only diverge when epochs use **different hash seeds** (e.g., sketch rebuilt after a restart). A 10-pair sweep (w=64, 200 flows, identical data, seeds (s, s+1) for s=0..9):

| Sketch | flow "1" mean ± std L1 | flow "10" mean ± std L1 |
|--------|------------------------|-------------------------|
| CMS    | 3.5 ± 7.3 | 1.5 ± 1.2 |
| CU-CMS | **0.0 ± 0.0** | **0.0 ± 0.0** |
| CS     | 4.0 ± 2.8 | 3.8 ± 1.8 |

CU-CMS achieves exactly zero residual across all 10 pairs: with D=3 there is almost always at least one clean row (no heavy collision partner), so the min-over-rows recovers the exact count. CMS is the worst: its std > mean indicates unpredictable spikes. **Under hash rotation, prefer CU-CMS.**

### Classifier accuracy (Stage 5)

100% accuracy at N=1000 packets per flow on seeds 42, 43, 44. Accuracy degrades to ~60% at N=200 and ~20–40% at N=100. The classifier is a **rule-based heuristic** — not a learned model — and carries no formal guarantees outside the synthetic regime.

---

## What Still Needs to Be Done (for the report)

The implementation is complete and all stages pass. The remaining work is the **PDF report** and any additional experiments the report requires. Below is what each report section needs and where the code already supports it.

### Abstract (1–3 paragraphs)
Ready to write from the results above.

### Introduction (1–2 pages)
Needs:
- Motivation: why distributional change detection matters (anomaly detection, SLA monitoring, DDoS detection)
- Background on Count-Min Sketch, Conservative Update, Count Sketch
- Background on HistSketch (ICDE 2023) and Krishnamurthy et al. (IMC 2003)
- A clear statement of our contribution: combining the two ideas and evaluating the five anomaly types

### Technical Details (2+ pages)
Already covered by the implementation — the report needs to describe:
- The bin configuration choice (logarithmic vs uniform; explain why log-spaced suits latency)
- The epoch manager and stream processor design
- The diff-histogram approach and why bias cancels under same-seed
- The normalised L1 detector and why the absolute floor is required (ablation evidence)
- The hash-rotation experiment as a nuanced sketch comparison

### Evaluation (1+ pages)
The code already produces all needed numbers. The report should present:
- **F1 comparison table**: baseline vs improved, both with the honest val/test split
- **Ablation table** (A/B/C)
- **Threshold selection curve** (val_avg_F1 vs threshold)
- **Classifier accuracy vs N** (robustness sweep — already in Stage 5 output)
- **Hash-rotation table** (mean ± std across 10 seed pairs)
- **Stage 2 accuracy table** (per-bin errors at w=64)

Metrics to explain: F1 = 2PR/(P+R), precision = TP/(TP+FP), recall = TP/(TP+FN). Justify F1 ≥ 0.7 as the gate (both precision and recall matter for anomaly detection).

### Conclusions and Discussion
Points to highlight:
- Bias cancellation: under same-seed operation, sketch type is irrelevant for detection; detector design dominates
- Normalised L1 + absolute floor is the key design decision; ablation proves each component's contribution
- The classifier is heuristic and degrades at low N — an important limitation
- Hash rotation is the one scenario where sketch type matters; CU-CMS is the clear winner
- Future work: real trace data (CAIDA), adversarial inputs, wider memory sweeps, learned classifiers

### Experiments the report could add (code already supports them)

These would strengthen the evaluation section and the code infrastructure is in place:

1. **Memory vs F1 sweep** — vary w ∈ {32, 64, 128, 256, 512, 1024} and plot F1 vs memory footprint. Stage 6's `evaluate_all_sketches` already handles width as a parameter; just loop over it.

2. **Epoch size vs F1** — vary EPOCH_SIZE ∈ {500, 1000, 2000, 5000}; shows minimum viable packet budget per epoch for reliable detection. Stage 6 is parameterised.

3. **Anomaly magnitude sensitivity** — vary the SuddenSpike magnitude ∈ {1.2, 1.5, 2.0, 3.0, 5.0}; shows minimum detectable change. Stage 6 and `generate_stream` are both parameterised by magnitude.

4. **Hash-rotation at different widths** — the Stage 4 experiment currently uses w=64; running at w ∈ {32, 64, 128} would show whether CU-CMS's zero-residual property holds under heavier sketch load.

5. **Bin count sweep** — vary B ∈ {5, 10, 20} at fixed memory (reduce w proportionally); tests whether finer bins improve or hurt detection.

---

## Design Decisions and Known Limitations

**Why logarithmic bins?** Latency distributions are right-skewed (lognormal-like). Log spacing gives equal resolution per order of magnitude and prevents most packets from piling into one or two bins.

**Why absolute floor in the improved detector?** Without it, near-empty flows (1–5 packets) generate enormous normalised L1 from pure sampling noise. The ablation shows 253 false positives vs 2 with the floor.

**Why threshold selected on val seeds and tested on seed 44?** This is a minimal honest train/test split. Without it, the reported threshold (0.15) would be lower than what the data actually supports and the reported F1 would be optimistic.

**Stage 5 classifier is not robust at low N.** At N=100 packets, accuracy drops to 20–40%. The heuristic ratios (0.45 for VolumeChange, 0.70/0.35 for Spike) are calibrated for the default noise_floor=5.0 and do not automatically scale with the noise_floor parameter.

**Under same-seed epochs, all three sketches are equivalent.** This is expected and correct behaviour — it means the DHS result is not sensitive to sketch choice in normal operation, which is a robustness property, not a limitation.

---

## Running the Checkpoints

Each checkpoint binary runs independently:

```bash
cmake --build build
./build/checkpoint1   # Basic frequency estimation accuracy
./build/checkpoint2   # Histogram-in-sketch accuracy
./build/checkpoint3   # Temporal snapshotting
./build/checkpoint4   # Differencing and heavy-changer detection
./build/checkpoint5   # Change type classification
./build/checkpoint6   # End-to-end F1 on synthetic stream
./build/main          # All stages in one binary (extended analysis)
```

`checkpoint6` and `main` Stage 6 use the same baseline detector and will both report `ALL PASS`.

---

## Evaluation Experiments

All evaluation sweeps are implemented in `src/evaluation.cpp` and write CSV outputs to `outputs/evaluation/`. Run them with:

```bash
cmake --build build --target evaluation
./build/evaluation
```

The following experiments are included:

| Function | CSV output | Purpose |
|---|---|---|
| `run_threshold_selection` | `threshold_sweep.csv` | Val-seed F1 vs threshold; selects best_threshold |
| `run_baseline_vs_improved` | `baseline_vs_improved.csv`, `per_type_breakdown.csv` | Baseline vs improved detector; per-anomaly-type TP/FN |
| `run_bins_sweep` | `bins_sweep.csv` | F1 vs number of histogram bins B |
| `run_memory_sweep` | `memory_sweep.csv` | F1 vs sketch width (memory/accuracy tradeoff) |
| `run_snapshots_sweep` | `snapshots_sweep.csv` | F1 vs number of retained epoch snapshots K |
| `run_epoch_sweep` | `epoch_sweep.csv` | F1 vs epoch size (packets per epoch) |
| `run_zipf_sweep` | `zipf_sweep.csv` | F1 vs Zipf alpha (traffic skewness) |
| `run_sensitivity_sweep` | `sensitivity_sweep.csv` | F1 vs anomaly magnitude (minimum detectable change) |
| `run_bin_scheme_comparison` | `bin_scheme_comparison.csv` | Uniform vs logarithmic bin edges |
| `run_hash_rotation_experiment` | `hash_rotation.csv` | L1 residual under different-seed epochs per sketch type |
| `run_classification_evaluation` | `classifier_accuracy_vs_n.csv`, `classifier_confusion_matrix.csv` | Classifier accuracy and confusion matrix vs N |
| `run_flow_count_sweep` | `flow_count_sweep.csv` | F1 vs number of flows in the stream |
| `run_grey_failure_comparison` | `grey_failure_comparison.csv` | Detector comparison on subtle/grey failure anomalies |
