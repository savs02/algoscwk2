#pragma once
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>
#include "../../include/MurmurHash3.hpp"
#include "bin_config.hpp"

// ---------------------------------------------------------------------------
// BaseSketch
//
// Shared foundation for CMS, CU-CMS, and CS.
//
// Hash families
// -------------
//   position hash : seed = row                  → maps key → [0, w)
//   sign hash     : seed = row + 1'000'000      → maps key → {+1, -1}
//
// The large seed offset keeps the two families independent (they will never
// share a seed value given d ≪ 1'000'000).
//
// update(key, value) semantics (Stage 2 onwards)
// -----------------------------------------------
//   `value` is the latency (or other metric) for one packet belonging to
//   `key`.  Internally, get_bin(value) selects which histogram bin to
//   increment.  The increment is always 1 — one call = one packet.
// ---------------------------------------------------------------------------

class BaseSketch {
protected:
    int       w_, d_;
    BinConfig bin_cfg_;
    uint32_t  seed_;   // per-sketch hash seed; default 0 preserves original behaviour

    // seed_ + row gives the MurmurHash seed for row j's position hash.
    // seed_ + row + 1,000,000 gives the seed for the sign hash (offset keeps the
    // two families independent since D << 1,000,000 in all practical settings).
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

    // Record one packet: key = flow ID, value = latency (determines bin).
    virtual void update(const std::string& key, double value) = 0;

    // Estimated histogram over all bins for this key.
    virtual std::vector<double> query_histogram(const std::string& key) const = 0;

    // Total estimated packet count = sum of histogram bins.
    double query(const std::string& key) const {
        auto h = query_histogram(key);
        return std::accumulate(h.begin(), h.end(), 0.0);
    }

    int              width()      const { return w_; }
    int              depth()      const { return d_; }
    int              num_bins()   const { return bin_cfg_.num_bins(); }
    const BinConfig& bin_config() const { return bin_cfg_; }
};
