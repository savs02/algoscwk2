// Checkpoint 3 — Temporal snapshotting and epoch isolation.
//
// Stream: 4 epochs × 2500 items = 10 000 total.
// Each epoch uses a different lognormal mean so histograms shift predictably:
//   Epoch 0: mu=1.0 → median ≈  2.7  (low latency)
//   Epoch 1: mu=1.5 → median ≈  4.5
//   Epoch 2: mu=2.0 → median ≈  7.4
//   Epoch 3: mu=2.5 → median ≈ 12.2  (high latency)
//
// Pass criteria
// -------------
//   1. Histogram peaks shift monotonically from low bins (epoch 0) to
//      high bins (epoch 3), confirming each sketch only holds its own epoch.
//   2. Per-epoch estimated packet count for the test key is within a few
//      counts of the ground truth (CMS overestimation is expected but small).
//   3. Each stored snapshot matches its own epoch's ground-truth histogram
//      better than any other epoch's histogram.

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "sketches/bin_config.hpp"
#include "sketches/cms.hpp"
#include "temporal/epoch_manager.hpp"
#include "temporal/stream_processor.hpp"

// ---------------------------------------------------------------------------

class ZipfDistribution {
    std::discrete_distribution<int> dist_;
public:
    ZipfDistribution(int n_max, double alpha) {
        std::vector<double> weights(n_max);
        for (int i = 0; i < n_max; ++i)
            weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), alpha);
        dist_ = std::discrete_distribution<int>(weights.begin(), weights.end());
    }
    int sample(std::mt19937& rng) { return dist_(rng) + 1; }
};

// ---------------------------------------------------------------------------

int main() {
    constexpr int      K            = 4;
    constexpr int      EPOCH_SIZE   = 2500;
    constexpr int      W            = 1024;
    constexpr int      D            = 3;
    constexpr int      N_MAX_KEY    = 1000;
    constexpr uint32_t SEED         = 42;
    const std::string  TEST_KEY     = "1";

    // Bin config: 8 uniform bins over [0, 30].
    BinConfig cfg(8, 0.0, 30.0, BinScheme::Uniform);

    // Factory creates a fresh CMS sketch for each epoch slot.
    auto factory = [&]() -> EpochManager::SketchPtr {
        return std::make_unique<CountMinSketch>(W, D, cfg);
    };

    EpochManager    epoch_mgr(K, factory);
    StreamProcessor proc(epoch_mgr, EPOCH_SIZE);

    std::mt19937     rng(SEED);
    ZipfDistribution zipf(N_MAX_KEY, 1.5);

    // Shifting lognormal means across epochs.
    const std::array<double, 4> mus    = {1.0, 1.5, 2.0, 2.5};
    constexpr double             SIGMA = 0.3;

    // Per-epoch ground truth for TEST_KEY.
    std::array<std::vector<int>, K> gt;
    std::array<int, K>              gt_count = {};
    for (auto& h : gt) h.assign(cfg.num_bins(), 0);

    for (int e = 0; e < K; ++e) {
        std::lognormal_distribution<double> lat_dist(mus[e], SIGMA);
        for (int i = 0; i < EPOCH_SIZE; ++i) {
            std::string key = std::to_string(zipf.sample(rng));
            double      lat = lat_dist(rng);
            proc.process(key, lat);
            if (key == TEST_KEY) {
                gt[e][cfg.get_bin(lat)]++;
                gt_count[e]++;
            }
        }
    }

    if (proc.items_processed() != K * EPOCH_SIZE) {
        std::cerr << "FAIL — processed item count mismatch\n";
        return 1;
    }
    if (epoch_mgr.snapshots_available() != K) {
        std::cerr << "FAIL — expected " << K << " snapshots, found "
                  << epoch_mgr.snapshots_available() << "\n";
        return 1;
    }

    // After K full epochs the ring buffer layout is:
    //   get_previous_sketch(0) = epoch K-1 (most recent)
    //   get_previous_sketch(K-1-e) = epoch e
    const auto& edges = cfg.edges();
    int         B     = cfg.num_bins();

    // -----------------------------------------------------------------------
    // Print header

    std::cout << "Checkpoint 3  |  " << K << " epochs × " << EPOCH_SIZE
              << " items = " << K * EPOCH_SIZE << " total"
              << ", w=" << W << ", d=" << D << "\n";
    std::cout << "Latency: lognormal(sigma=" << SIGMA << "), mu shifts each epoch\n\n";
    for (int e = 0; e < K; ++e) {
        std::cout << "  Epoch " << e << ": mu=" << mus[e]
                  << "  median=" << std::fixed << std::setprecision(1)
                  << std::exp(mus[e]) << "\n";
    }

    // -----------------------------------------------------------------------
    // Per-epoch histogram table for TEST_KEY

    std::cout << "\nHistograms for flow \"" << TEST_KEY << "\"  [truth | estimate per epoch]\n";

    // Column widths: bin + range + K*(truth+est) columns
    std::cout << std::setw(4) << "Bin" << std::setw(16) << "Range";
    for (int e = 0; e < K; ++e)
        std::cout << std::setw(9) << ("E" + std::to_string(e) + " true")
                  << std::setw(8) << ("E" + std::to_string(e) + " est");
    std::cout << "\n" << std::string(4 + 16 + K * 17, '-') << "\n";

    // Retrieve per-epoch sketches (offset = K-1-e maps epoch e → ring slot).
    std::array<std::vector<double>, K> est;
    for (int e = 0; e < K; ++e)
        est[e] = epoch_mgr.get_previous_sketch(K - 1 - e).query_histogram(TEST_KEY);

    for (int b = 0; b < B; ++b) {
        std::ostringstream range;
        range << std::fixed << std::setprecision(1)
              << "[" << edges[b] << "," << edges[b + 1] << ")";

        std::cout << std::setw(4) << b << std::setw(16) << range.str();
        for (int e = 0; e < K; ++e) {
            std::cout << std::setw(9)  << gt[e][b]
                      << std::setw(8)  << std::fixed << std::setprecision(0)
                      << est[e][b];
        }
        std::cout << "\n";
    }

    // -----------------------------------------------------------------------
    // Epoch totals: ground truth count vs sketch total

    std::cout << "\nEpoch totals for flow \"" << TEST_KEY << "\":\n";
    std::cout << std::setw(7)  << "Epoch"
              << std::setw(10) << "True"
              << std::setw(12) << "Estimated"
              << std::setw(10) << "Error\n";
    std::cout << std::string(39, '-') << "\n";

    for (int e = 0; e < K; ++e) {
        double est_total = 0;
        for (double v : est[e]) est_total += v;
        std::cout << std::setw(7)  << e
                  << std::setw(10) << gt_count[e]
                  << std::setw(12) << std::fixed << std::setprecision(0) << est_total
                  << std::setw(10) << std::showpos << std::fixed << std::setprecision(0)
                  << (est_total - gt_count[e]) << std::noshowpos << "\n";
    }

    // -----------------------------------------------------------------------
    // Isolation checks

    std::cout << "\nIsolation check A — peak bin shifts as latency mean rises:\n";
    int prev_peak = -1;
    bool monotone = true;
    for (int e = 0; e < K; ++e) {
        int peak = static_cast<int>(
            std::max_element(est[e].begin(), est[e].end()) - est[e].begin());
        std::ostringstream range;
        range << std::fixed << std::setprecision(1)
              << "[" << edges[peak] << "," << edges[peak + 1] << ")";
        std::cout << "  Epoch " << e << " (mu=" << mus[e]
                  << ", median=" << std::fixed << std::setprecision(1)
                  << std::exp(mus[e]) << "): peak bin "
                  << peak << "  " << range.str() << "\n";
        if (peak < prev_peak) monotone = false;
        prev_peak = peak;
    }

    std::cout << "\nIsolation check B — each snapshot best matches its own epoch:\n";
    bool best_match_ok = true;
    for (int e = 0; e < K; ++e) {
        double best_l1 = std::numeric_limits<double>::infinity();
        int    best_gt_epoch = -1;
        for (int g = 0; g < K; ++g) {
            double l1 = 0.0;
            for (int b = 0; b < B; ++b)
                l1 += std::abs(est[e][b] - static_cast<double>(gt[g][b]));
            if (l1 < best_l1) {
                best_l1 = l1;
                best_gt_epoch = g;
            }
        }
        std::cout << "  Snapshot epoch " << e << " best matches ground truth epoch "
                  << best_gt_epoch << "  (L1=" << std::fixed << std::setprecision(0)
                  << best_l1 << ")\n";
        if (best_gt_epoch != e) best_match_ok = false;
    }

    bool pass = monotone && best_match_ok;
    std::cout << "\n" << (pass ? "PASS" : "FAIL")
              << " — peak bins are "
              << (monotone ? "non-decreasing" : "not monotone")
              << " and snapshot/epoch matching is "
              << (best_match_ok ? "correct" : "incorrect") << "\n";

    return pass ? 0 : 1;
}
