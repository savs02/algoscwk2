#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include "../sketches/base_sketch.hpp"

// Histogram differencing and change detection


struct ChangeScores {
    double l1;     
    double l2;       
    double max_bin;  
};

enum class ChangeMetric {
    L1,
    L2,
    MaxBin,
};

inline std::vector<double> diff_histograms(
    const BaseSketch& sketch_A,
    const BaseSketch& sketch_B,
    const std::string& key)
{
    assert(sketch_A.num_bins() == sketch_B.num_bins()
           && "sketches must have the same bin count to diff");
    assert(sketch_A.bin_config().edges() == sketch_B.bin_config().edges()
           && "sketches must have identical bin edges to diff");

    auto hist_A = sketch_A.query_histogram(key);
    auto hist_B = sketch_B.query_histogram(key);

    std::vector<double> diff(hist_A.size());
    for (size_t b = 0; b < diff.size(); ++b)
        diff[b] = hist_A[b] - hist_B[b];
    return diff;
}

// L1, L2, and max-bin scores from a dhs
inline ChangeScores compute_scores(const std::vector<double>& diff) {
    double l1 = 0.0, l2_sq = 0.0, max_bin = 0.0;
    for (double d : diff) {
        double ad = std::abs(d);
        l1      += ad;
        l2_sq   += d * d;
        max_bin  = std::max(max_bin, ad);
    }
    return {l1, std::sqrt(l2_sq), max_bin};
}

inline double score_for_metric(const ChangeScores& scores, ChangeMetric metric) {
    switch (metric) {
        case ChangeMetric::L1:     return scores.l1;
        case ChangeMetric::L2:     return scores.l2;
        case ChangeMetric::MaxBin: return scores.max_bin;
    }
    assert(false && "unknown ChangeMetric");
    return 0.0;
}

// flag as heavy 
inline bool is_heavy_changer(const ChangeScores& scores,
                             double threshold,
                             ChangeMetric metric = ChangeMetric::L1)
{
    return score_for_metric(scores, metric) > threshold;
}
