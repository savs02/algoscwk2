#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

enum class BinScheme { Uniform, Logarithmic };

// ---------------------------------------------------------------------------
// BinConfig
//
// Holds bin edges and maps a value to a bin index.
//
// Uniform:     edges evenly spaced from min_val to max_val.
// Logarithmic: edges log-spaced from min_val to max_val (requires min_val > 0).
//
// Values below edges[0] clamp to bin 0.
// Values >= edges[num_bins] clamp to bin num_bins-1.
// ---------------------------------------------------------------------------

class BinConfig {
    int                 num_bins_;
    std::vector<double> edges_;   // length = num_bins_ + 1

public:
    BinConfig(int num_bins, double min_val, double max_val,
              BinScheme scheme = BinScheme::Uniform)
        : num_bins_(num_bins), edges_(num_bins + 1)
    {
        assert(num_bins > 0       && "num_bins must be positive");
        assert(max_val > min_val  && "max_val must exceed min_val");

        if (scheme == BinScheme::Logarithmic) {
            assert(min_val > 0.0  && "Logarithmic bins require min_val > 0");
            double log_min = std::log(min_val);
            double log_max = std::log(max_val);
            double step    = (log_max - log_min) / num_bins;
            for (int i = 0; i <= num_bins; ++i)
                edges_[i] = std::exp(log_min + i * step);
        } else {
            double step = (max_val - min_val) / num_bins;
            for (int i = 0; i <= num_bins; ++i)
                edges_[i] = min_val + i * step;
        }
    }

    // Map value to bin index in [0, num_bins-1].
    int get_bin(double value) const {
        if (value <= edges_.front()) return 0;
        if (value >= edges_.back())  return num_bins_ - 1;
        // upper_bound → first edge strictly greater than value.
        // The bin is one position before that.
        auto it = std::upper_bound(edges_.begin(), edges_.end(), value);
        return static_cast<int>(std::distance(edges_.begin(), it)) - 1;
    }

    int                        num_bins() const { return num_bins_; }
    const std::vector<double>& edges()    const { return edges_; }
};
