#pragma once
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>
#include "base_sketch.hpp"

// ---------------------------------------------------------------------------
// Conservative Update Count-Min Sketch (CU-CMS)
//
// Same structure as CMS, but update only raises a counter when it is
// strictly below (current_estimate + value).  Collisions with heavier flows
// cannot inflate the counter beyond what the flow actually needs, so
// estimates are tighter (less overestimation) than plain CMS.
//
// update : estimate = min_j counters[j][hash_j(key)]
//          for each j: counters[j][h] = max(counters[j][h], estimate + value)
// query  : same as CMS — min over rows.
// ---------------------------------------------------------------------------

class ConservativeUpdateCMS : public BaseSketch {
    std::vector<std::vector<int32_t>> counters_;

public:
    ConservativeUpdateCMS(int w = 1024, int d = 3)
        : BaseSketch(w, d),
          counters_(d, std::vector<int32_t>(w, 0)) {}

    void update(const std::string& key, int value) override {
        assert(value >= 0 && "CU-CMS only supports non-negative updates");
        // Compute current estimate (min across rows).
        int32_t estimate = std::numeric_limits<int32_t>::max();
        for (int j = 0; j < d_; ++j) {
            int32_t v = counters_[j][hash_pos(key, j)];
            if (v < estimate) estimate = v;
        }

        int32_t new_val = estimate + static_cast<int32_t>(value);
        for (int j = 0; j < d_; ++j) {
            uint32_t h = hash_pos(key, j);
            if (counters_[j][h] < new_val)
                counters_[j][h] = new_val;
        }
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
