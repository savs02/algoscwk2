#pragma once
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>
#include "base_sketch.hpp"

// ---------------------------------------------------------------------------
// ConservativeUpdateCMS (CU-CMS) — histogram variant
//
// Per-bin conservative update: each bin is treated as an independent CMS
// counter with the conservative-update rule applied only to the bin that
// was actually hit.
//
// update : b = get_bin(value)
//          estimate_b = min_j counters[j][h_j(key)][b]
//          new_val    = estimate_b + 1
//          for each j: counters[j][h][b] = max(counters[j][h][b], new_val)
//
// Counters in unaffected bins are never touched, so cross-bin noise is
// eliminated.  Within a bin, the conservative rule prevents heavy flows from
// inflating light-flow estimates beyond what the light flow actually needs.
//
// Assumption: latency values are non-negative (asserted below).
// ---------------------------------------------------------------------------

class ConservativeUpdateCMS : public BaseSketch {
    std::vector<std::vector<std::vector<int32_t>>> counters_;

public:
    ConservativeUpdateCMS(int w, int d, const BinConfig& bin_cfg)
        : BaseSketch(w, d, bin_cfg),
          counters_(d,
              std::vector<std::vector<int32_t>>(
                  w, std::vector<int32_t>(bin_cfg.num_bins(), 0)))
    {}

    void update(const std::string& key, double value) override {
        assert(value >= 0.0 && "CU-CMS only supports non-negative latency values");
        int b = bin_cfg_.get_bin(value);

        // Conservative update for bin b only.
        int32_t estimate = std::numeric_limits<int32_t>::max();
        for (int j = 0; j < d_; ++j) {
            int32_t v = counters_[j][hash_pos(key, j)][b];
            if (v < estimate) estimate = v;
        }
        int32_t new_val = estimate + 1;
        for (int j = 0; j < d_; ++j) {
            uint32_t h = hash_pos(key, j);
            if (counters_[j][h][b] < new_val)
                counters_[j][h][b] = new_val;
        }
    }

    std::vector<double> query_histogram(const std::string& key) const override {
        int B = num_bins();
        std::vector<double> hist(B);
        for (int b = 0; b < B; ++b) {
            int32_t min_val = std::numeric_limits<int32_t>::max();
            for (int j = 0; j < d_; ++j) {
                int32_t v = counters_[j][hash_pos(key, j)][b];
                if (v < min_val) min_val = v;
            }
            hist[b] = static_cast<double>(min_val);
        }
        return hist;
    }
};
