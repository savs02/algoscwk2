#pragma once
// MurmurHash3 was written by Austin Appleby and is placed in the public domain.
// The author hereby disclaims copyright to this source code.
// Bundled as a single header-only file.

#include <cstdint>
#include <cstring>
#include <string>

namespace mmh3 {

namespace detail {

inline uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

inline uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6bU;
    h ^= h >> 13;
    h *= 0xc2b2ae35U;
    h ^= h >> 16;
    return h;
}

} // namespace detail

// Returns a 32-bit hash of `len` bytes starting at `key`, seeded with `seed`.
inline uint32_t hash32(const void* key, int len, uint32_t seed) {
    const uint8_t* data   = static_cast<const uint8_t*>(key);
    const int      nblocks = len / 4;

    uint32_t h1 = seed;
    constexpr uint32_t c1 = 0xcc9e2d51U;
    constexpr uint32_t c2 = 0x1b873593U;

    // --- body ---
    for (int i = 0; i < nblocks; ++i) {
        uint32_t k1;
        std::memcpy(&k1, data + i * 4, 4);

        k1 *= c1;
        k1  = detail::rotl32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1  = detail::rotl32(h1, 13);
        h1  = h1 * 5 + 0xe6546b64U;
    }

    // --- tail ---
    const uint8_t* tail = data + nblocks * 4;
    uint32_t k1 = 0;
    switch (len & 3) {
        case 3: k1 ^= static_cast<uint32_t>(tail[2]) << 16; [[fallthrough]];
        case 2: k1 ^= static_cast<uint32_t>(tail[1]) << 8;  [[fallthrough]];
        case 1: k1 ^= static_cast<uint32_t>(tail[0]);
                k1 *= c1;
                k1  = detail::rotl32(k1, 15);
                k1 *= c2;
                h1 ^= k1;
    }

    // --- finalisation ---
    h1 ^= static_cast<uint32_t>(len);
    h1  = detail::fmix32(h1);
    return h1;
}

// Convenience overload: hash a std::string.
inline uint32_t hash32(const std::string& key, uint32_t seed) {
    return hash32(key.data(), static_cast<int>(key.size()), seed);
}

} // namespace mmh3
