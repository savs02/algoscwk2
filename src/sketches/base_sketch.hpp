#pragma once
#include <cstdint>
#include <string>
#include "../../include/MurmurHash3.hpp"

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
// ---------------------------------------------------------------------------

class BaseSketch {
protected:
    int w_;
    int d_;

    // Map key to bucket index in [0, w) for the given row.
    uint32_t hash_pos(const std::string& key, int row) const {
        return mmh3::hash32(key, static_cast<uint32_t>(row)) % static_cast<uint32_t>(w_);
    }

    // Map key to {+1, -1} for the given row (Count Sketch only).
    int hash_sign(const std::string& key, int row) const {
        uint32_t h = mmh3::hash32(key, static_cast<uint32_t>(row + 1'000'000));
        return (h & 1u) == 0 ? 1 : -1;
    }

public:
    BaseSketch(int w, int d) : w_(w), d_(d) {}
    virtual ~BaseSketch() = default;

    virtual void   update(const std::string& key, int value) = 0;
    virtual double query (const std::string& key) const      = 0;

    int width() const { return w_; }
    int depth() const { return d_; }
};
