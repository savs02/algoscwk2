// Checkpoint 3 — Temporal snapshotting and epoch isolation.
//
// Stream: 4 time-based epochs × 2500 items = 10 000 total.
// Input to the stream processor is explicit (timestamp, key, value) tuples.
// Each epoch uses a different lognormal mean so histograms shift predictably:
//   Epoch 0: mu=1.0 → median ≈  2.7  (low latency)
//   Epoch 1: mu=1.5 → median ≈  4.5
//   Epoch 2: mu=2.0 → median ≈  7.4
//   Epoch 3: mu=2.5 → median ≈ 12.2  (high latency)
//
// Pass criteria (applied to all three sketch types: CMS, CU-CMS, CS)
// ------------------------------------------------------------------
//   1. items_processed == K * EPOCH_SIZE, snapshots_available == K.
//   2. Histogram peak bins are non-decreasing across epochs.
//   3. Each snapshot best-matches (L1) its own epoch's ground truth.

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "sketches/bin_config.hpp"
#include "sketches/cms.hpp"
#include "sketches/cu_cms.hpp"
#include "sketches/cs.hpp"
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
// Run the isolation test for one sketch type.
// Returns true if all three criteria pass.
// ---------------------------------------------------------------------------

static bool run_isolation_test(
    const std::string&                             sketch_name,
    const std::function<EpochManager::SketchPtr()>& factory,
    const std::vector<std::tuple<double, std::string, double>>& stream,
    const std::array<std::vector<int>, 4>&           gt,
    const std::array<int, 4>&                        gt_count,
    const BinConfig&                                 cfg,
    int K, int EPOCH_SIZE, double epoch_duration_seconds,
    const std::array<double,4>& mus)
{
    EpochManager    epoch_mgr(K, factory);
    StreamProcessor proc(epoch_mgr, epoch_duration_seconds);

    for (const auto& [timestamp, key, lat] : stream)
        proc.process(timestamp, key, lat);

    // Sanity checks.
    if (proc.items_processed() != K * EPOCH_SIZE) {
        std::cerr << sketch_name << ": FAIL — item count mismatch\n";
        return false;
    }
    if (epoch_mgr.snapshots_available() != K) {
        std::cerr << sketch_name << ": FAIL — snapshots_available mismatch\n";
        return false;
    }

    const std::string TEST_KEY = "1";
    const int         B        = cfg.num_bins();
    const auto&       edges    = cfg.edges();

    std::array<std::vector<double>, 4> est;
    for (int e = 0; e < K; ++e)
        est[e] = epoch_mgr.get_previous_sketch(K - 1 - e).query_histogram(TEST_KEY);

    // --- Print histogram table ---
    std::cout << "\n--- " << sketch_name << " ---\n";
    std::cout << std::setw(4) << "Bin" << std::setw(16) << "Range";
    for (int e = 0; e < K; ++e)
        std::cout << std::setw(9) << ("E"+std::to_string(e)+" true")
                  << std::setw(8) << ("E"+std::to_string(e)+" est");
    std::cout << "\n" << std::string(4 + 16 + K * 17, '-') << "\n";

    for (int b = 0; b < B; ++b) {
        std::ostringstream range;
        range << std::fixed << std::setprecision(1)
              << "[" << edges[b] << "," << edges[b+1] << ")";
        std::cout << std::setw(4) << b << std::setw(16) << range.str();
        for (int e = 0; e < K; ++e)
            std::cout << std::setw(9)  << gt[e][b]
                      << std::setw(8)  << std::fixed << std::setprecision(0) << est[e][b];
        std::cout << "\n";
    }

    // --- Check A: monotone peak bins ---
    bool monotone  = true;
    int  prev_peak = -1;
    std::cout << "\nCheck A (peak shift):\n";
    for (int e = 0; e < K; ++e) {
        int peak = static_cast<int>(
            std::max_element(est[e].begin(), est[e].end()) - est[e].begin());
        std::ostringstream r;
        r << std::fixed << std::setprecision(1)
          << "[" << edges[peak] << "," << edges[peak+1] << ")";
        std::cout << "  E" << e << " mu=" << std::fixed << std::setprecision(1) << mus[e]
                  << " peak=" << peak << " " << r.str() << "\n";
        if (peak < prev_peak) monotone = false;
        prev_peak = peak;
    }

    // --- Check B: each snapshot best-matches its own ground truth (L1) ---
    bool best_match_ok = true;
    std::cout << "Check B (L1 self-match):\n";
    for (int e = 0; e < K; ++e) {
        double best_l1      = std::numeric_limits<double>::infinity();
        int    best_gt      = -1;
        for (int g = 0; g < K; ++g) {
            double l1 = 0.0;
            for (int b = 0; b < B; ++b)
                l1 += std::abs(est[e][b] - static_cast<double>(gt[g][b]));
            if (l1 < best_l1) { best_l1 = l1; best_gt = g; }
        }
        std::cout << "  Snapshot E" << e << " best matches GT epoch "
                  << best_gt << "  L1=" << std::fixed << std::setprecision(1)
                  << best_l1 << "\n";
        if (best_gt != e) best_match_ok = false;
    }

    bool pass = monotone && best_match_ok;
    std::cout << (pass ? "PASS" : "FAIL") << " — " << sketch_name << "\n";
    return pass;
}

// ---------------------------------------------------------------------------

int main() {
    constexpr int      K                      = 4;
    constexpr int      EPOCH_SIZE             = 2500;
    constexpr double   EPOCH_DURATION_SECONDS = 2500.0;
    constexpr int      W                      = 1024;
    constexpr int      D                      = 3;
    constexpr int      N_MAX_KEY              = 1000;
    constexpr uint32_t SEED                   = 42;
    const std::string  TEST_KEY               = "1";

    BinConfig cfg(8, 0.0, 30.0, BinScheme::Uniform);

    const std::array<double, 4> mus   = {1.0, 1.5, 2.0, 2.5};
    constexpr double             SIGMA = 0.3;

    // Generate the stream once; replay it for each sketch type.
    std::mt19937     rng(SEED);
    ZipfDistribution zipf(N_MAX_KEY, 1.5);

    std::vector<std::tuple<double, std::string, double>> stream;
    stream.reserve(K * EPOCH_SIZE);

    std::array<std::vector<int>, K> gt;
    std::array<int, K>              gt_count = {};
    for (auto& h : gt) h.assign(cfg.num_bins(), 0);

    for (int e = 0; e < K; ++e) {
        std::lognormal_distribution<double> lat_dist(mus[e], SIGMA);
        for (int i = 0; i < EPOCH_SIZE; ++i) {
            double      timestamp = e * EPOCH_DURATION_SECONDS + i;
            std::string key = std::to_string(zipf.sample(rng));
            double      lat = lat_dist(rng);
            stream.push_back({timestamp, key, lat});
            if (key == TEST_KEY) {
                gt[e][cfg.get_bin(lat)]++;
                gt_count[e]++;
            }
        }
    }

    std::cout << "Checkpoint 3  |  " << K << " epochs × " << EPOCH_SIZE
              << " = " << K * EPOCH_SIZE << " items, w=" << W << ", d=" << D << "\n";
    std::cout << "Stream mode: (timestamp, key, value), epoch_duration="
              << EPOCH_DURATION_SECONDS << " seconds\n";
    std::cout << "lognormal(sigma=" << SIGMA << "), mu per epoch: ";
    for (int e = 0; e < K; ++e)
        std::cout << mus[e] << (e < K-1 ? " → " : "\n");
    std::cout << "All three sketch types must independently pass isolation.\n";

    bool all_pass = true;

    all_pass &= run_isolation_test("CMS",
        [&]{ return std::make_unique<CountMinSketch>(W, D, cfg); },
        stream, gt, gt_count, cfg, K, EPOCH_SIZE, EPOCH_DURATION_SECONDS, mus);

    all_pass &= run_isolation_test("CU-CMS",
        [&]{ return std::make_unique<ConservativeUpdateCMS>(W, D, cfg); },
        stream, gt, gt_count, cfg, K, EPOCH_SIZE, EPOCH_DURATION_SECONDS, mus);

    all_pass &= run_isolation_test("CS",
        [&]{ return std::make_unique<CountSketch>(W, D, cfg); },
        stream, gt, gt_count, cfg, K, EPOCH_SIZE, EPOCH_DURATION_SECONDS, mus);

    std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_pass ? 0 : 1;
}
