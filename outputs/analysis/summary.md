# Evaluation Summary

## Detector (5 held-out seeds, all 5 anomalies together)
- Baseline raw L1: F1 = 0.714 ± 0.000
- Improved normalized: F1 = 0.899 ± 0.062
- Best validation threshold: 0.20 (val mean F1 0.902)

## Per-type recall (mean across 3 sketches)
- Baseline hardest type: Disappearance (recall 0.000)
- Improved hardest type: GradualRamp (recall 0.800)

- Minimum viable memory (mean F1 >= 0.9): w=512, 240.0 KB (CS, F1=0.907)
## Sweeps
- Best bins setting in current sweep: 8 bins for CMS with mean F1 0.935
- Weakest snapshot setting: K=16 for CS with mean F1 0.874
- Hardest Zipf regime: alpha=1.0 with mean F1 0.700
- Lowest sensitivity point: magnitude=1.1 with mean F1 0.795

## Structure
- Mean F1 by scheme: log=0.917, uniform=0.255
- CU-CMS hash-rotation residuals remain zero in this experiment: max mean L1=75.7

## Classifier
- Lowest packet regime (100 packets) mean accuracy: 0.333
- Highest packet regime mean accuracy: 1.000
