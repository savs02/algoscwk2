// Checkpoint 6 — Synthetic stream generator and F1 evaluation.
//
// Stream: 100 Zipf flows, 4 epochs × 5000 packets = 20 000 total.
// Five anomaly flows are injected (one per type, assigned to the 5 heaviest flows
// so they have enough packets to reliably detect):
//
//   Flow "1" — SuddenSpike   : mean doubles for epoch 1 only, returns to normal
//   Flow "2" — GradualRamp   : mean grows ×1.2 per epoch from epoch 1
//   Flow "3" — PeriodicBurst : alternates high/normal from epoch 1
//   Flow "4" — Spread        : sigma doubles from epoch 1 onward
//   Flow "5" — Disappearance : no packets from epoch 1 onward
//
// Ground truth: (flow_id, boundary) pairs where the distribution changes
// between adjacent epochs.
//
// Detection: for each sketch type, for every adjacent epoch pair, query each
// flow and flag as a heavy changer if L1 > threshold.
//
// Output:
//   1. Clean checkpoint run: per-sketch table of TP/FP/FN and F1 score
//   2. Informational robustness sweep across seeds / widths / thresholds
//
// As with checkpoint 5, the threshold and anomaly setup are heuristic rather
// than formally optimal. The clean run is the checkpoint pass/fail target;
// the sweep is there to show empirical behaviour under noisier settings.

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <tuple>
#include <vector>

#include "sketches/bin_config.hpp"
#include "sketches/cms.hpp"
#include "sketches/cu_cms.hpp"
#include "sketches/cs.hpp"
#include "temporal/epoch_manager.hpp"
#include "temporal/stream_processor.hpp"
#include "temporal/differencer.hpp"
#include "generator/stream_generator.hpp"

// ---------------------------------------------------------------------------

struct DetectionResult {
    int tp = 0, fp = 0, fn = 0;

    double precision() const {
        return (tp + fp > 0) ? double(tp) / (tp + fp) : 0.0;
    }
    double recall() const {
        return (tp + fn > 0) ? double(tp) / (tp + fn) : 0.0;
    }
    double f1() const {
        double p = precision(), r = recall();
        return (p + r > 0.0) ? 2.0 * p * r / (p + r) : 0.0;
    }
};

struct EvaluationSummary {
    std::string sketch_name;
    DetectionResult result;
};

// ---------------------------------------------------------------------------

static DetectionResult run_detection(
    const std::string&                              sketch_name,
    const std::function<EpochManager::SketchPtr()>& factory,
    const GeneratedStream&                          gs,
    const BinConfig&                                cfg,
    double                                          l1_threshold,
    const std::vector<AnomalySpec>&                 anomaly_specs,
    bool                                            verbose)
{
    const int K = gs.num_epochs;

    EpochManager    em(K, factory);
    StreamProcessor proc(em, gs.epoch_duration);

    for (const auto& [ts, key, lat] : gs.packets)
        proc.process(ts, key, lat);

    // Build ground-truth set: (flow_id, boundary).
    std::set<std::pair<std::string, int>> gt_set;
    for (const auto& e : gs.ground_truth)
        gt_set.insert({e.flow_id, e.boundary});

    // Detect: for each adjacent epoch pair and each known flow.
    // get_previous_sketch(0)         = most recent epoch (K-1)
    // get_previous_sketch(K-1)       = oldest epoch (0)
    // Boundary b = epoch b → epoch b+1.
    // Sketch for epoch e = get_previous_sketch(K-1 - e).

    std::set<std::pair<std::string, int>> detected_set;

    for (int b = 0; b < K - 1; ++b) {
        const BaseSketch& sk_new = em.get_previous_sketch(K - 2 - b);  // epoch b+1
        const BaseSketch& sk_old = em.get_previous_sketch(K - 1 - b);  // epoch b

        for (const auto& fid : gs.flow_keys) {
            auto diff   = diff_histograms(sk_new, sk_old, fid);
            auto scores = compute_scores(diff);
            if (is_heavy_changer(scores, l1_threshold, ChangeMetric::L1))
                detected_set.insert({fid, b});
        }
    }

    // Compute TP / FP / FN.
    DetectionResult res;
    for (const auto& d : detected_set) {
        if (gt_set.count(d)) ++res.tp;
        else                  ++res.fp;
    }
    for (const auto& g : gt_set) {
        if (!detected_set.count(g)) ++res.fn;
    }

    if (verbose) {
        std::cout << "\n  " << sketch_name << ":\n";

        // Print per-boundary per-anomaly table.
        std::cout << "  " << std::setw(10) << "Flow"
                  << std::setw(12) << "Anomaly"
                  << std::setw(10) << "Boundary"
                  << std::setw(8)  << "GT"
                  << std::setw(8)  << "Det"
                  << std::setw(8)  << "L1\n";
        std::cout << "  " << std::string(56, '-') << "\n";

        // Build anomaly type lookup.
        std::unordered_map<std::string, const char*> anom_name_map;
        for (const auto& a : anomaly_specs)
            anom_name_map[a.flow_id] = anomaly_type_name(a.type);

        // Only print anomaly flows.
        std::vector<std::string> anomaly_flow_ids;
        for (const auto& e : gs.ground_truth) {
            if (std::find(anomaly_flow_ids.begin(), anomaly_flow_ids.end(), e.flow_id)
                == anomaly_flow_ids.end())
                anomaly_flow_ids.push_back(e.flow_id);
        }

        for (const auto& fid : anomaly_flow_ids) {
            const char* atype = anom_name_map.count(fid) ? anom_name_map.at(fid) : "";
            for (int b = 0; b < K - 1; ++b) {
                const BaseSketch& sk_new = em.get_previous_sketch(K - 2 - b);
                const BaseSketch& sk_old = em.get_previous_sketch(K - 1 - b);
                auto diff   = diff_histograms(sk_new, sk_old, fid);
                auto scores = compute_scores(diff);

                bool in_gt  = gt_set.count({fid, b}) > 0;
                bool in_det = detected_set.count({fid, b}) > 0;

                if (in_gt || scores.l1 > l1_threshold * 0.5) {
                    std::cout << "  " << std::setw(10) << fid
                              << std::setw(14) << atype
                              << std::setw(10) << b
                              << std::setw(8)  << (in_gt  ? "YES" : "-")
                              << std::setw(8)  << (in_det ? "YES" : "-")
                              << std::setw(8)  << std::fixed << std::setprecision(0)
                              << scores.l1 << "\n";
                }
            }
        }

        std::cout << "  TP=" << res.tp << " FP=" << res.fp << " FN=" << res.fn
                  << "  P=" << std::fixed << std::setprecision(3) << res.precision()
                  << " R="  << res.recall()
                  << " F1=" << res.f1() << "\n";
    }

    return res;
}

static std::vector<EvaluationSummary> evaluate_all_sketches(
    int                                            W,
    int                                            D,
    const GeneratedStream&                         gs,
    const BinConfig&                               cfg,
    double                                         l1_threshold,
    const std::vector<AnomalySpec>&                anomaly_specs,
    bool                                           verbose)
{
    std::vector<EvaluationSummary> out;
    out.push_back({"CMS",
        run_detection("CMS",
            [&]{ return std::make_unique<CountMinSketch>(W, D, cfg); },
            gs, cfg, l1_threshold, anomaly_specs, verbose)});
    out.push_back({"CU-CMS",
        run_detection("CU-CMS",
            [&]{ return std::make_unique<ConservativeUpdateCMS>(W, D, cfg); },
            gs, cfg, l1_threshold, anomaly_specs, verbose)});
    out.push_back({"CS",
        run_detection("CS",
            [&]{ return std::make_unique<CountSketch>(W, D, cfg); },
            gs, cfg, l1_threshold, anomaly_specs, verbose)});
    return out;
}

// ---------------------------------------------------------------------------

int main() {
    constexpr int      NUM_FLOWS   = 100;
    constexpr int      NUM_EPOCHS  = 4;
    constexpr int      EPOCH_SIZE  = 5000;
    constexpr double   ZIPF_ALPHA  = 1.5;
    const double       BASE_MU     = std::log(5.0);
    constexpr double   BASE_SIGMA  = 0.5;
    constexpr int      W           = 1024;
    constexpr int      D           = 3;
    constexpr uint32_t SEED        = 42;
    constexpr double   L1_THRESH   = 300.0;

    BinConfig cfg(10, 0.5, 200.0, BinScheme::Logarithmic);

    // Five anomaly flows assigned to the top-5 Zipf flows (heaviest hitters).
    std::vector<AnomalySpec> anomalies = {
        {"1", AnomalyType::SuddenSpike,   1, 2.0},
        {"2", AnomalyType::GradualRamp,   1, 1.2},
        {"3", AnomalyType::PeriodicBurst, 1, 2.0},
        {"4", AnomalyType::Spread,        1, 2.0},
        {"5", AnomalyType::Disappearance, 1, 2.0},
    };

    auto gs = generate_stream(NUM_FLOWS, NUM_EPOCHS, EPOCH_SIZE,
                               ZIPF_ALPHA, BASE_MU, BASE_SIGMA,
                               anomalies, SEED);

    std::cout << "Checkpoint 6  |  Synthetic stream + F1 evaluation\n";
    std::cout << "Flows=" << NUM_FLOWS << ", epochs=" << NUM_EPOCHS
              << ", epoch_size=" << EPOCH_SIZE
              << ", Zipf(alpha=" << ZIPF_ALPHA << ")\n";
    std::cout << "Bins: 10 log-spaced [0.5, 200], w=" << W << ", d=" << D << "\n";
    std::cout << "L1 threshold: " << L1_THRESH << "\n\n";
    std::cout << "This checkpoint uses one clean synthetic setup as the pass/fail\n"
              << "target. The robustness sweep later is informational and shows\n"
              << "how sensitive detection is to seed / memory / threshold choices.\n\n";

    // Print anomaly specs.
    std::cout << "Injected anomalies:\n";
    for (const auto& a : anomalies)
        std::cout << "  Flow " << a.flow_id
                  << ": " << anomaly_type_name(a.type)
                  << " starting epoch " << a.start_epoch << "\n";

    // Print ground truth.
    std::cout << "\nGround truth (" << gs.ground_truth.size() << " entries):\n";
    for (const auto& g : gs.ground_truth)
        std::cout << "  flow=" << g.flow_id << " boundary=" << g.boundary << "\n";

    std::cout << "\nDetection results:\n";

    bool all_pass = true;
    auto clean_results = evaluate_all_sketches(W, D, gs, cfg, L1_THRESH, anomalies, true);
    for (const auto& eval : clean_results) {
        bool pass = eval.result.f1() >= 0.7;
        std::cout << "  " << eval.sketch_name << ": " << (pass ? "PASS" : "FAIL")
                  << " (F1=" << std::fixed << std::setprecision(3) << eval.result.f1() << ")\n\n";
        all_pass &= pass;
    }

    std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";

    std::cout << "\nRobustness sweep (informational only)\n";
    std::cout << "Each row reruns the full synthetic generator and detector for a\n"
              << "different seed / width / threshold combination. This is not part\n"
              << "of checkpoint gating; it is there to expose empirical stability.\n\n";
    std::cout << std::left
              << std::setw(8)  << "Width"
              << std::setw(8)  << "Seed"
              << std::setw(10) << "Thresh"
              << std::setw(10) << "Sketch"
              << std::setw(8)  << "TP"
              << std::setw(8)  << "FP"
              << std::setw(8)  << "FN"
              << std::setw(10) << "F1"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (int width : {1024, 256}) {
        for (uint32_t seed : {42u, 43u, 44u}) {
            for (double threshold : {250.0, 300.0, 350.0}) {
                auto sweep_gs = generate_stream(NUM_FLOWS, NUM_EPOCHS, EPOCH_SIZE,
                                                ZIPF_ALPHA, BASE_MU, BASE_SIGMA,
                                                anomalies, seed);
                auto sweep_results = evaluate_all_sketches(width, D, sweep_gs, cfg,
                                                           threshold, anomalies, false);
                for (const auto& eval : sweep_results) {
                    std::cout << std::left
                              << std::setw(8)  << width
                              << std::setw(8)  << seed
                              << std::setw(10) << std::fixed << std::setprecision(0) << threshold
                              << std::setw(10) << eval.sketch_name
                              << std::setw(8)  << eval.result.tp
                              << std::setw(8)  << eval.result.fp
                              << std::setw(8)  << eval.result.fn
                              << std::setw(10) << std::fixed << std::setprecision(3) << eval.result.f1()
                              << "\n";
                }
            }
        }
    }

    return all_pass ? 0 : 1;
}
