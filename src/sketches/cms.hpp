#pragma once
#include <cstdint>
#include <limits>
#include <vector>
#include "base_sketch.hpp"

class CountMinSketch : public BaseSketch {
    std::vector<std::vector<std::vector<int32_t>>> counters_;

public:
    CountMinSketch(int w, int d, const BinConfig& bin_cfg, uint32_t seed = 0)
        : BaseSketch(w, d, bin_cfg, seed),
          counters_(d,
              std::vector<std::vector<int32_t>>(
                  w, std::vector<int32_t>(bin_cfg.num_bins(), 0)))
    {}

    void update(const std::string& key, double value) override {
        int b = bin_cfg_.get_bin(value);
        for (int j = 0; j < d_; ++j)
            counters_[j][hash_pos(key, j)][b] += 1;
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
