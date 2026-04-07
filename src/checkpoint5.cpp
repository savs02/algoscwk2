// Checkpoint 5 — Change type classification.
//
// Five synthetic change types are injected, one per test case.
// For each case we run two epochs through all three sketch types and verify
// that the classifier labels the changed flow correctly while leaving an
// unchanged background flow classified as None.
//
// Change types and how they are synthesised:
//   Disappearance  — flow X sends packets in epoch 0 but nothing in epoch 1
//   VolumeChange   — same distribution, 3× more packets in epoch 1 than epoch 0
//   Spike          — epoch 1 adds a concentrated burst at a single latency value
//                    (only a narrow band of latency values, rest same as epoch 0)
//   Shift          — lognormal mean doubles (5 → 10), same sigma
//   Spread         — same mean, sigma doubles (0.5 → 1.0)
//
// Each case also includes a stable background flow Y. CMS, CU-CMS, and CS
// must all classify X correctly and keep Y as None.

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
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
#include "temporal/change_classifier.hpp"

// ---------------------------------------------------------------------------

struct TestCase {
    std::string                                           name;
    ChangeType                                            expected;
    double                                                epoch_duration; // seconds per epoch
    std::vector<std::tuple<double, std::string, double>>  stream;
};

struct SuiteSummary {
    int sketch_case_total  = 0;
    int sketch_case_passed = 0;
};

// ---------------------------------------------------------------------------
// Helper: build a timestamped stream for two epochs.
//   ep1 starts at epoch_duration so the boundary fires exactly once.
//   epoch_duration must be >= max(ep0.size(), ep1.size()) to prevent
//   the stream processor from firing extra advances mid-epoch.
// ---------------------------------------------------------------------------
static std::vector<std::tuple<double, std::string, double>>
make_stream(const std::vector<std::pair<std::string, double>>& ep0,
            const std::vector<std::pair<std::string, double>>& ep1,
            double epoch_duration)
{
    std::vector<std::tuple<double, std::string, double>> s;
    s.reserve(ep0.size() + ep1.size());
    for (int i = 0; i < static_cast<int>(ep0.size()); ++i)
        s.push_back({static_cast<double>(i), ep0[i].first, ep0[i].second});
    for (int i = 0; i < static_cast<int>(ep1.size()); ++i)
        s.push_back({epoch_duration + i, ep1[i].first, ep1[i].second});
    return s;
}

// ---------------------------------------------------------------------------
// Produce N lognormal samples for key.
// ---------------------------------------------------------------------------
static std::vector<std::pair<std::string, double>>
lognormal_packets(const std::string& key, int N, double mu, double sigma,
                  std::mt19937& rng)
{
    std::lognormal_distribution<double> dist(mu, sigma);
    std::vector<std::pair<std::string, double>> out;
    out.reserve(N);
    for (int i = 0; i < N; ++i) out.push_back({key, dist(rng)});
    return out;
}

static void append_packets(std::vector<std::pair<std::string, double>>& dst,
                           const std::vector<std::pair<std::string, double>>& src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}

static std::vector<TestCase> build_test_cases(int N,
                                              uint32_t seed,
                                              const std::string& KEY_X,
                                              const std::string& KEY_Y)
{
    std::mt19937 rng(seed);
    std::vector<TestCase> cases;

    // --- Disappearance: X active in epoch 0, silent in epoch 1.
    //     Y is stable background in both epochs. ---
    {
        std::vector<std::pair<std::string, double>> ep0;
        std::vector<std::pair<std::string, double>> ep1;
        append_packets(ep0, lognormal_packets(KEY_X, N, std::log(5.0), 0.5, rng));
        append_packets(ep0, lognormal_packets(KEY_Y, N, std::log(5.0), 0.5, rng));
        append_packets(ep1, lognormal_packets(KEY_Y, N, std::log(5.0), 0.5, rng));
        double ep_dur = static_cast<double>(std::max(ep0.size(), ep1.size()));
        cases.push_back({"Disappearance", ChangeType::Disappearance, ep_dur,
                         make_stream(ep0, ep1, ep_dur)});
    }

    // --- VolumeChange: same distribution, 3× packets in epoch 1. ---
    {
        std::vector<std::pair<std::string, double>> ep0;
        std::vector<std::pair<std::string, double>> ep1;
        append_packets(ep0, lognormal_packets(KEY_X,     N, std::log(5.0), 0.5, rng));
        append_packets(ep0, lognormal_packets(KEY_Y,     N, std::log(5.0), 0.5, rng));
        append_packets(ep1, lognormal_packets(KEY_X, 3 * N, std::log(5.0), 0.5, rng));
        append_packets(ep1, lognormal_packets(KEY_Y,     N, std::log(5.0), 0.5, rng));
        double ep_dur = static_cast<double>(std::max(ep0.size(), ep1.size()));
        cases.push_back({"VolumeChange", ChangeType::VolumeChange, ep_dur,
                         make_stream(ep0, ep1, ep_dur)});
    }

    // --- Spike: epoch 0 normal; epoch 1 = epoch 0 baseline + concentrated
    //     burst at latency ~30 ms (bin 7 of 10, log-spaced), narrow sigma. ---
    {
        std::vector<std::pair<std::string, double>> ep0;
        std::vector<std::pair<std::string, double>> ep1;
        append_packets(ep0, lognormal_packets(KEY_X, N, std::log(5.0), 0.5, rng));
        append_packets(ep0, lognormal_packets(KEY_Y, N, std::log(5.0), 0.5, rng));
        append_packets(ep1, lognormal_packets(KEY_X, N, std::log(5.0), 0.5, rng));
        append_packets(ep1, lognormal_packets(KEY_X, 2 * N, std::log(30.0), 0.05, rng));
        append_packets(ep1, lognormal_packets(KEY_Y, N, std::log(5.0), 0.5, rng));
        double ep_dur = static_cast<double>(std::max(ep0.size(), ep1.size()));
        cases.push_back({"Spike", ChangeType::Spike, ep_dur,
                         make_stream(ep0, ep1, ep_dur)});
    }

    // --- Shift: mean doubles (log(5) → log(10)), same sigma ---
    {
        std::vector<std::pair<std::string, double>> ep0;
        std::vector<std::pair<std::string, double>> ep1;
        append_packets(ep0, lognormal_packets(KEY_X, N, std::log(5.0),  1.0, rng));
        append_packets(ep0, lognormal_packets(KEY_Y, N, std::log(5.0),  0.5, rng));
        append_packets(ep1, lognormal_packets(KEY_X, N, std::log(10.0), 1.0, rng));
        append_packets(ep1, lognormal_packets(KEY_Y, N, std::log(5.0),  0.5, rng));
        double ep_dur = static_cast<double>(std::max(ep0.size(), ep1.size()));
        cases.push_back({"Shift", ChangeType::Shift, ep_dur,
                         make_stream(ep0, ep1, ep_dur)});
    }

    // --- Spread: same mean, sigma doubles (0.5 → 1.0) ---
    {
        std::vector<std::pair<std::string, double>> ep0;
        std::vector<std::pair<std::string, double>> ep1;
        append_packets(ep0, lognormal_packets(KEY_X, N, std::log(5.0), 0.5, rng));
        append_packets(ep0, lognormal_packets(KEY_Y, N, std::log(5.0), 0.5, rng));
        append_packets(ep1, lognormal_packets(KEY_X, N, std::log(5.0), 1.0, rng));
        append_packets(ep1, lognormal_packets(KEY_Y, N, std::log(5.0), 0.5, rng));
        double ep_dur = static_cast<double>(std::max(ep0.size(), ep1.size()));
        cases.push_back({"Spread", ChangeType::Spread, ep_dur,
                         make_stream(ep0, ep1, ep_dur)});
    }

    return cases;
}

// ---------------------------------------------------------------------------
// Run one test case against one sketch type.
// Returns true if the classifier matches the expected type for X and keeps
// the unchanged background flow Y classified as None.
// ---------------------------------------------------------------------------
static bool run_one(const std::string& sketch_name,
                    const std::function<EpochManager::SketchPtr()>& factory,
                    const BinConfig& cfg,
                    double epoch_duration,
                    const TestCase& tc,
                    bool verbose)
{
    EpochManager    em(2, factory);
    StreamProcessor proc(em, epoch_duration);

    for (const auto& [ts, key, lat] : tc.stream)
        proc.process(ts, key, lat);

    const BaseSketch& epoch1 = em.get_previous_sketch(0);
    const BaseSketch& epoch0 = em.get_previous_sketch(1);

    const std::string KEY_X = "X";
    const std::string KEY_Y = "Y";
    auto diff   = diff_histograms(epoch1, epoch0, KEY_X);
    auto diff_y = diff_histograms(epoch1, epoch0, KEY_Y);
    auto result = classify_change(diff);
    auto result_y = classify_change(diff_y);

    bool pass_x = (result == tc.expected);
    bool pass_y = (result_y == ChangeType::None);
    bool pass = pass_x && pass_y;

    if (verbose) {
        std::cout << "    " << std::setw(8) << sketch_name
                  << "  X diff: [";
        for (int b = 0; b < static_cast<int>(diff.size()); ++b)
            std::cout << std::fixed << std::setprecision(0)
                      << (b ? ", " : "") << diff[b];
        std::cout << "]\n";
        std::cout << "    " << std::setw(8) << ""
                  << "  Y diff: [";
        for (int b = 0; b < static_cast<int>(diff_y.size()); ++b)
            std::cout << std::fixed << std::setprecision(0)
                      << (b ? ", " : "") << diff_y[b];
        std::cout << "]\n";
        std::cout << "    " << std::setw(8) << ""
                  << "  X got=" << change_type_name(result)
                  << "  expected=" << change_type_name(tc.expected)
                  << "  " << (pass_x ? "PASS" : "FAIL") << "\n";
        std::cout << "    " << std::setw(8) << ""
                  << "  Y got=" << change_type_name(result_y)
                  << "  expected=None"
                  << "  " << (pass_y ? "PASS" : "FAIL") << "\n";
    }
    return pass;
}

static SuiteSummary run_suite(const std::vector<TestCase>& cases,
                              int W,
                              int D,
                              const BinConfig& cfg,
                              bool verbose)
{
    SuiteSummary summary;

    for (const auto& tc : cases) {
        if (verbose) {
            std::cout << "--- " << tc.name
                      << " (expected: " << change_type_name(tc.expected) << ") ---\n";
        }

        auto cms_factory = [&]{ return std::make_unique<CountMinSketch>(W, D, cfg); };
        auto cu_factory  = [&]{ return std::make_unique<ConservativeUpdateCMS>(W, D, cfg); };
        auto cs_factory  = [&]{ return std::make_unique<CountSketch>(W, D, cfg); };

        bool p1 = run_one("CMS",    cms_factory, cfg, tc.epoch_duration, tc, verbose);
        bool p2 = run_one("CU-CMS", cu_factory,  cfg, tc.epoch_duration, tc, verbose);
        bool p3 = run_one("CS",     cs_factory,  cfg, tc.epoch_duration, tc, verbose);

        summary.sketch_case_total += 3;
        summary.sketch_case_passed += static_cast<int>(p1) + static_cast<int>(p2) + static_cast<int>(p3);

        if (verbose) {
            bool case_pass = p1 && p2 && p3;
            std::cout << "  => " << (case_pass ? "PASS" : "FAIL") << "\n\n";
        }
    }

    return summary;
}

// ---------------------------------------------------------------------------

int main() {
    constexpr int      N         = 1000;   // packets per flow per epoch
    constexpr int      W         = 1024;
    constexpr int      D         = 3;
    constexpr uint32_t SEED      = 42;
    const std::string  KEY_X     = "X";
    const std::string  KEY_Y     = "Y";   // background flow (unchanged)

    // 10 log-spaced bins [0.5, 200] — same as checkpoint 4.
    BinConfig cfg(10, 0.5, 200.0, BinScheme::Logarithmic);
    auto cases = build_test_cases(N, SEED, KEY_X, KEY_Y);

    // -----------------------------------------------------------------------
    // Run all cases × all sketch types.
    // -----------------------------------------------------------------------
    std::cout << "Checkpoint 5  |  Change type classification\n";
    std::cout << "w=" << W << ", d=" << D << ", N=" << N
              << ", bins=10 log-spaced [0.5, 200]\n";
    std::cout << "Each case checks changed flow X and stable background flow Y.\n\n";
    std::cout << "These labels are heuristic: the clean cases below are the\n"
              << "checkpoint pass/fail target, and the robustness sweep after\n"
              << "that is informational rather than a formal guarantee.\n\n";

    SuiteSummary clean = run_suite(cases, W, D, cfg, true);
    bool all_pass = (clean.sketch_case_passed == clean.sketch_case_total);

    std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";

    // Informational robustness sweep: smaller width and multiple seeds.
    std::cout << "\nRobustness sweep (informational only)\n";
    std::cout << "This checks the same heuristics under different seeds and a\n"
              << "collision-heavier width. It is not part of checkpoint gating.\n\n";
    std::cout << std::left
              << std::setw(8)  << "Width"
              << std::setw(8)  << "Seed"
              << std::setw(12) << "Passed"
              << std::setw(12) << "Total"
              << std::setw(12) << "Accuracy"
              << "\n";
    std::cout << std::string(52, '-') << "\n";

    for (int width : {1024, 128}) {
        for (uint32_t seed : {42u, 43u, 44u}) {
            auto sweep_cases = build_test_cases(N, seed, KEY_X, KEY_Y);
            auto summary = run_suite(sweep_cases, width, D, cfg, false);
            double accuracy = 100.0 * summary.sketch_case_passed / summary.sketch_case_total;
            std::cout << std::left
                      << std::setw(8)  << width
                      << std::setw(8)  << seed
                      << std::setw(12) << summary.sketch_case_passed
                      << std::setw(12) << summary.sketch_case_total
                      << std::setw(11) << std::fixed << std::setprecision(1) << accuracy
                      << "%\n";
        }
    }

    return all_pass ? 0 : 1;
}
