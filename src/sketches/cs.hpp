#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>
#include "base_sketch.hpp"


class CountSketch : public BaseSketch {
    std::vector<std::vector<std::vector<int32_t>>> counters_;

public:
    CountSketch(int w, int d, const BinConfig& bin_cfg, uint32_t seed = 0)
        : BaseSketch(w, d, bin_cfg, seed),
          counters_(d,
              std::vector<std::vector<int32_t>>(
                  w, std::vector<int32_t>(bin_cfg.num_bins(), 0)))
    {}

    void update(const std::string& key, double value) override {
        int b = bin_cfg_.get_bin(value);
        for (int j = 0; j < d_; ++j)
            counters_[j][hash_pos(key, j)][b] += hash_sign(key, j);
    }

    std::vector<double> query_histogram(const std::string& key) const override {
        int B = num_bins();
        std::vector<double> hist(B);
        std::vector<double> row_est(d_);

        for (int b = 0; b < B; ++b) {
            for (int j = 0; j < d_; ++j)
                row_est[j] = static_cast<double>(counters_[j][hash_pos(key, j)][b])
                             * hash_sign(key, j);

            int mid = d_ / 2;
            std::nth_element(row_est.begin(), row_est.begin() + mid, row_est.end());

            if (d_ % 2 == 1) {
                hist[b] = row_est[mid];
            } else {
                double upper = row_est[mid];
                double lower = *std::max_element(row_est.begin(), row_est.begin() + mid);
                hist[b] = (lower + upper) / 2.0;
            }
        }
        return hist;
    }
};
