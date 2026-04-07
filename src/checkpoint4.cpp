// Checkpoint 4 — Histogram differencing and heavy-changer detection.
//
// Two epochs, each with 1000 packets per flow:
//   Flow X: lognormal(mu=log(5), sigma=1) in epoch 0  →  lognormal(mu=log(10), sigma=1) in epoch 1
//   Flow Y: lognormal(mu=log(5), sigma=1) in both epochs  (stable)
//
// The differenced histogram for X (epoch 1 − epoch 0) should show:
//   negative values in low bins  (fewer low-latency packets now)
//   positive values in high bins (more high-latency packets now)
//
// Flow Y's diff should be close to zero across all bins.
//
// Detection thresholds:
//   L1      > 300
//   L2      > 120
//   max-bin > 50
// With N=1000 packets per flow and 10 bins, statistical variation between two
// same-distribution samples produces scores far below these thresholds.
// A genuine distribution shift (median 5→10) produces scores comfortably above
// them, so flow X should be detected and flow Y should remain unflagged.
//
// All three sketch types are tested; CMS and CU-CMS should detect X cleanly;
// CS has signed noise so scores may differ but the direction should hold.

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
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
#include "temporal/differencer.hpp"

// ---------------------------------------------------------------------------

struct DetectionThresholds {
    double l1;
    double l2;
    double max_bin;
};

static void print_diff_row(const std::string& label,
                           const std::vector<double>& diff,
                           const BinConfig& cfg)
{
    const auto& edges = cfg.edges();
    int B = cfg.num_bins();

    std::cout << "\n  " << label << "\n";
    std::cout << "  " << std::setw(4) << "Bin"
              << std::setw(18) << "Range"
              << std::setw(10) << "Diff"
              << "  (bar)\n";
    std::cout << "  " << std::string(50, '-') << "\n";

    for (int b = 0; b < B; ++b) {
        std::ostringstream range;
        range << std::fixed << std::setprecision(1)
              << "[" << edges[b] << "," << edges[b+1] << ")";

        // Simple ASCII bar: '+' for positive, '-' for negative
        int bar_len = static_cast<int>(std::abs(diff[b]) / 5.0);
        bar_len = std::min(bar_len, 30);
        char bar_ch = diff[b] >= 0 ? '+' : '-';
        std::string bar(bar_len, bar_ch);

        std::cout << "  " << std::setw(4) << b
                  << std::setw(18) << range.str()
                  << std::setw(10) << std::fixed << std::setprecision(1)
                  << diff[b]
                  << "  " << bar << "\n";
    }
}

// ---------------------------------------------------------------------------

static bool run_checkpoint(
    const std::string& sketch_name,
    const std::function<EpochManager::SketchPtr()>& factory,
    const BinConfig& cfg,
    int N_PER_FLOW, double epoch_duration,
    const DetectionThresholds& thresholds,
    const std::vector<std::tuple<double, std::string, double>>& stream,
    const std::string& KEY_X, const std::string& KEY_Y)
{
    EpochManager    em(2, factory);
    StreamProcessor proc(em, epoch_duration);

    for (const auto& [timestamp, key, lat] : stream) proc.process(timestamp, key, lat);

    // After two full epochs:
    //   get_previous_sketch(0) = epoch 1 (current, most recent)
    //   get_previous_sketch(1) = epoch 0
    const BaseSketch& epoch1 = em.get_previous_sketch(0);
    const BaseSketch& epoch0 = em.get_previous_sketch(1);

    auto diff_x = diff_histograms(epoch1, epoch0, KEY_X);
    auto diff_y = diff_histograms(epoch1, epoch0, KEY_Y);

    auto scores_x = compute_scores(diff_x);
    auto scores_y = compute_scores(diff_y);

    bool x_l1  = is_heavy_changer(scores_x, thresholds.l1, ChangeMetric::L1);
    bool y_l1  = is_heavy_changer(scores_y, thresholds.l1, ChangeMetric::L1);
    bool x_l2  = is_heavy_changer(scores_x, thresholds.l2, ChangeMetric::L2);
    bool y_l2  = is_heavy_changer(scores_y, thresholds.l2, ChangeMetric::L2);
    bool x_max = is_heavy_changer(scores_x, thresholds.max_bin, ChangeMetric::MaxBin);
    bool y_max = is_heavy_changer(scores_y, thresholds.max_bin, ChangeMetric::MaxBin);

    std::cout << "\n=== " << sketch_name << " ===\n";

    print_diff_row("Flow X  (latency mean 5 → 10, should shift right)", diff_x, cfg);
    print_diff_row("Flow Y  (stable,           diff should be ≈ 0)",    diff_y, cfg);

    std::cout << "\n  Scores and detection results:\n";
    std::cout << "  " << std::setw(8) << "Flow"
              << std::setw(10) << "L1"
              << std::setw(10) << "L2"
              << std::setw(12) << "max-bin"
              << std::setw(10) << "L1?"
              << std::setw(10) << "L2?"
              << std::setw(10) << "Max?\n";
    std::cout << "  " << std::string(52, '-') << "\n";

    auto print_row = [&](const std::string& name, const ChangeScores& s,
                         bool l1_flag, bool l2_flag, bool max_flag) {
        std::cout << "  " << std::setw(8) << name
                  << std::setw(10) << std::fixed << std::setprecision(1) << s.l1
                  << std::setw(10) << s.l2
                  << std::setw(12) << s.max_bin
                  << std::setw(10) << (l1_flag ? "YES" : "no")
                  << std::setw(10) << (l2_flag ? "YES" : "no")
                  << std::setw(10) << (max_flag ? "YES" : "no") << "\n";
    };
    print_row(KEY_X, scores_x, x_l1, x_l2, x_max);
    print_row(KEY_Y, scores_y, y_l1, y_l2, y_max);

    bool pass = x_l1 && x_l2 && x_max && !y_l1 && !y_l2 && !y_max;
    std::cout << "  " << (pass ? "PASS" : "FAIL") << " — "
              << sketch_name
              << ": X flags=(" << x_l1 << "," << x_l2 << "," << x_max << ")"
              << ", Y flags=(" << y_l1 << "," << y_l2 << "," << y_max << ")\n";
    return pass;
}

// ---------------------------------------------------------------------------

int main() {
    constexpr int      N_PER_FLOW      = 1000;
    constexpr int      ITEMS_PER_EPOCH = 2 * N_PER_FLOW;  // X + Y per epoch
    constexpr double   EPOCH_DURATION  = static_cast<double>(ITEMS_PER_EPOCH);
    constexpr int      W               = 1024;
    constexpr int      D               = 3;
    constexpr uint32_t SEED            = 42;
    const std::string  KEY_X           = "X";
    const std::string  KEY_Y           = "Y";
    const DetectionThresholds thresholds{300.0, 120.0, 50.0};

    // Log-spaced bins [0.5, 200]: covers lognormal(log(5),1) and lognormal(log(10),1).
    // 5th–95th percentile of lognormal(log(5),1): [0.9, 27]; lognormal(log(10),1): [1.8, 55].
    BinConfig cfg(10, 0.5, 200.0, BinScheme::Logarithmic);

    std::mt19937 rng(SEED);
    std::lognormal_distribution<double> dist_low (std::log(5.0),  1.0);
    std::lognormal_distribution<double> dist_high(std::log(10.0), 1.0);

    // Build a single timestamped stream:
    //   epoch 0: X=low,  Y=low
    //   epoch 1: X=high, Y=low
    auto make_stream = [&](std::lognormal_distribution<double>& dist_x,
                           std::lognormal_distribution<double>& dist_y,
                           double epoch_start)
        -> std::vector<std::tuple<double, std::string, double>>
    {
        std::vector<std::tuple<double, std::string, double>> s;
        s.reserve(2 * N_PER_FLOW);
        for (int i = 0; i < N_PER_FLOW; ++i)
            s.push_back({epoch_start + i, KEY_X, dist_x(rng)});
        for (int i = 0; i < N_PER_FLOW; ++i)
            s.push_back({epoch_start + N_PER_FLOW + i, KEY_Y, dist_y(rng)});
        return s;
    };

    auto epoch0_stream = make_stream(dist_low, dist_low, 0.0);

    std::lognormal_distribution<double> dist_y_e1(std::log(5.0), 1.0);
    std::lognormal_distribution<double> dist_x_e1(std::log(10.0), 1.0);
    auto epoch1_stream = make_stream(dist_x_e1, dist_y_e1, EPOCH_DURATION);

    std::vector<std::tuple<double, std::string, double>> stream;
    stream.reserve(epoch0_stream.size() + epoch1_stream.size());
    stream.insert(stream.end(), epoch0_stream.begin(), epoch0_stream.end());
    stream.insert(stream.end(), epoch1_stream.begin(), epoch1_stream.end());

    std::cout << "Checkpoint 4  |  2 epochs × " << EPOCH_DURATION << " items"
              << ", w=" << W << ", d=" << D << "\n";
    std::cout << "Stream mode: (timestamp, key, value), epoch_duration="
              << EPOCH_DURATION << " seconds\n";
    std::cout << "Flow X: lognormal(mu=log(5),σ=1) → lognormal(mu=log(10),σ=1)\n";
    std::cout << "Flow Y: lognormal(mu=log(5),σ=1) in both epochs (stable)\n";
    std::cout << "Bins: 10 log-spaced [0.5, 200]\n";
    std::cout << "Thresholds: L1 > " << thresholds.l1
              << ", L2 > " << thresholds.l2
              << ", max-bin > " << thresholds.max_bin << "\n";

    bool all_pass = true;

    all_pass &= run_checkpoint("CMS",
        [&]{ return std::make_unique<CountMinSketch>(W, D, cfg); },
        cfg, N_PER_FLOW, EPOCH_DURATION, thresholds,
        stream, KEY_X, KEY_Y);

    all_pass &= run_checkpoint("CU-CMS",
        [&]{ return std::make_unique<ConservativeUpdateCMS>(W, D, cfg); },
        cfg, N_PER_FLOW, EPOCH_DURATION, thresholds,
        stream, KEY_X, KEY_Y);

    all_pass &= run_checkpoint("CS",
        [&]{ return std::make_unique<CountSketch>(W, D, cfg); },
        cfg, N_PER_FLOW, EPOCH_DURATION, thresholds,
        stream, KEY_X, KEY_Y);

    std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_pass ? 0 : 1;
}
