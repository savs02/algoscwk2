# Evaluation Summary

## Detector (5 held-out seeds, all 5 anomalies together)
- Baseline raw L1: F1 = 0.875 ± 0.000
- Improved normalized: F1 = 0.979 ± 0.027
- Best validation threshold: 0.50 (val mean F1 0.900)

## Per-type recall (mean across 3 sketches)
- Baseline hardest type: Disappearance (recall 0.000)
- Improved hardest type: Disappearance (recall 1.000)

## Sweeps
- Best bins setting in current sweep: 4 bins for CMS with mean F1 1.000
- Weakest snapshot setting: K=16 for CS with mean F1 0.033
- Hardest Zipf regime: alpha=0.5 with mean F1 0.912
- Lowest sensitivity point: magnitude=1.1 with mean F1 0.200

## Structure
- Mean F1 by scheme: log=1.000, uniform=0.364
- CU-CMS hash-rotation residuals remain zero in this experiment: max mean L1=0.2

## Classifier
- Lowest packet regime (100 packets) mean accuracy: 0.333
- Highest packet regime mean accuracy: 1.000
