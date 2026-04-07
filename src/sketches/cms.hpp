#pragma once
#include <vector>
#include <cstdint>
#include <limits>
#include "base_sketch.hpp"

// ---------------------------------------------------------------------------
// CountMinSketch (CMS)
//
// update : for each row j, counters[j][hash_j(key)] += value
// query  : min over j of counters[j][hash_j(key)]
//
// Always overestimates — hash collisions only add noise, never subtract it.
// ---------------------------------------------------------------------------

class CountMinSketch : public BaseSketch {
    std::vector<std::vector<int32_t>> counters_;

public:
    CountMinSketch(int w = 1024, int d = 3)
        : BaseSketch(w, d),
          counters_(d, std::vector<int32_t>(w, 0)) {}

    void update(const std::string& key, int value) override {
        for (int j = 0; j < d_; ++j)
            counters_[j][hash_pos(key, j)] += static_cast<int32_t>(value);
    }

    double query(const std::string& key) const override {
        int32_t min_val = std::numeric_limits<int32_t>::max();
        for (int j = 0; j < d_; ++j) {
            int32_t v = counters_[j][hash_pos(key, j)];
            if (v < min_val) min_val = v;
        }
        return static_cast<double>(min_val);
    }
};
