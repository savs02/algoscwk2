#pragma once
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>
#include "../../include/MurmurHash3.hpp"
#include "bin_config.hpp"

// basic sketch representation

class BaseSketch {
protected:
    int       w_, d_;
    BinConfig bin_cfg_;
    uint32_t  seed_;  

    uint32_t hash_pos(const std::string& key, int row) const {
        return mmh3::hash32(key, seed_ + static_cast<uint32_t>(row))
               % static_cast<uint32_t>(w_);
    }

    int hash_sign(const std::string& key, int row) const {
        uint32_t h = mmh3::hash32(key, seed_ + static_cast<uint32_t>(row + 1'000'000));
        return (h & 1u) == 0 ? 1 : -1;
    }

public:
    BaseSketch(int w, int d, const BinConfig& bin_cfg, uint32_t seed = 0)
        : w_(w), d_(d), bin_cfg_(bin_cfg), seed_(seed) {}

    virtual ~BaseSketch() = default;

    virtual void update(const std::string& key, double value) = 0;

    virtual std::vector<double> query_histogram(const std::string& key) const = 0;

    double query(const std::string& key) const {
        auto h = query_histogram(key);
        return std::accumulate(h.begin(), h.end(), 0.0);
    }

    int              width()      const { return w_; }
    int              depth()      const { return d_; }
    int              num_bins()   const { return bin_cfg_.num_bins(); }
    const BinConfig& bin_config() const { return bin_cfg_; }
};
