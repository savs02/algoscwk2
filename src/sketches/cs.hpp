#pragma once
#include <vector>
#include <algorithm>
#include "base_sketch.hpp"

// ---------------------------------------------------------------------------
// Count Sketch (CS)
//
// d hash functions mapping keys to [0, w) and d sign functions mapping
// keys to {+1, -1}.  Signed increments mean collisions partially cancel
// in expectation, giving unbiased (but higher-variance) estimates.
//
// update : for each row j, counters[j][hash_j(key)] += sign_j(key) * value
// query  : median over j of (counters[j][hash_j(key)] * sign_j(key))
// ---------------------------------------------------------------------------

class CountSketch : public BaseSketch {
    std::vector<std::vector<int32_t>> counters_;

public:
    CountSketch(int w = 1024, int d = 3)
        : BaseSketch(w, d),
          counters_(d, std::vector<int32_t>(w, 0)) {}

    void update(const std::string& key, int value) override {
        for (int j = 0; j < d_; ++j)
            counters_[j][hash_pos(key, j)] +=
                static_cast<int32_t>(hash_sign(key, j) * value);
    }

    double query(const std::string& key) const override {
        std::vector<double> estimates(d_);
        for (int j = 0; j < d_; ++j)
            estimates[j] = static_cast<double>(counters_[j][hash_pos(key, j)])
                           * hash_sign(key, j);

        // Median via nth_element (O(d), in-place).
        int mid = d_ / 2;
        std::nth_element(estimates.begin(), estimates.begin() + mid, estimates.end());
        if (d_ % 2 == 1) {
            return estimates[mid];
        } else {
            // Even d: average the two middle values.
            double upper = estimates[mid];
            double lower = *std::max_element(estimates.begin(), estimates.begin() + mid);
            return (lower + upper) / 2.0;
        }
    }
};
