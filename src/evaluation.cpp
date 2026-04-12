#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "generator/stream_generator.hpp"
#include "sketches/bin_config.hpp"
#include "sketches/cms.hpp"
#include "sketches/cu_cms.hpp"
#include "sketches/cs.hpp"
#include "temporal/change_classifier.hpp"
#include "temporal/differencer.hpp"
#include "temporal/epoch_manager.hpp"
#include "temporal/stream_processor.hpp"

namespace fs = std::filesystem;

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

enum class SketchKind {
    CMS,
    CUCMS,
    CS,
};

static const char* sketch_name(SketchKind kind) {
    switch (kind) {
        case SketchKind::CMS:   return "CMS";
        case SketchKind::CUCMS: return "CU-CMS";
        case SketchKind::CS:    return "CS";
    }
    return "Unknown";
}

static std::unique_ptr<BaseSketch> make_sketch(
    SketchKind kind, int width, int depth, const BinConfig& cfg, uint32_t seed = 0)
{
    switch (kind) {
        case SketchKind::CMS:
            return std::make_unique<CountMinSketch>(width, depth, cfg, seed);
        case SketchKind::CUCMS:
            return std::make_unique<ConservativeUpdateCMS>(width, depth, cfg, seed);
        case SketchKind::CS:
            return std::make_unique<CountSketch>(width, depth, cfg, seed);
    }
    return nullptr;
}

struct DetectionResult {
    int tp = 0;
    int fp = 0;
    int fn = 0;
    std::map<AnomalyType, int> tp_by_type;
    std::map<AnomalyType, int> fn_by_type;

    double precision() const {
        return (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
    }

    double recall() const {
        return (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
    }

    double f1() const {
        const double p = precision();
        const double r = recall();
        return (p + r > 0.0) ? (2.0 * p * r / (p + r)) : 0.0;
    }
};

struct DetectorConfig {
    bool   normalized = false;
    double ratio_threshold = 0.0;
    double raw_threshold = 0.0;
    double absolute_floor = 0.0;
};

struct ClassificationCase {
    std::string name;
    ChangeType expected;
    double epoch_duration;
    std::vector<std::tuple<double, std::string, double>> stream;
};

static std::vector<AnomalySpec> make_default_anomalies(double spike_magnitude) {
    // Matches main.cpp Stage 6 exactly. Only SuddenSpike scales with the
    // parameter so the sensitivity sweep still has a single knob to vary.
    // GradualRamp/PeriodicBurst/Spread stay at their main.cpp defaults so
    // every other sweep matches the Stage 6 baseline.
    return {
        {"1", AnomalyType::SuddenSpike,   1, spike_magnitude},
        {"2", AnomalyType::GradualRamp,   1, 1.2},
        {"3", AnomalyType::PeriodicBurst, 1, 2.0},
        {"4", AnomalyType::Spread,        1, 2.0},
        {"5", AnomalyType::Disappearance, 1, 2.0},
    };
}

static DetectionResult run_detection(
    SketchKind kind,
    int width,
    int depth,
    int num_snapshots,
    const GeneratedStream& gs,
    const BinConfig& cfg,
    const DetectorConfig& detector,
    uint32_t sketch_seed = 0)
{
    auto factory = [&]() {
        return make_sketch(kind, width, depth, cfg, sketch_seed);
    };

    EpochManager em(num_snapshots, factory);
    std::set<std::pair<std::string, int>> gt_set;
    std::map<std::pair<std::string, int>, AnomalyType> gt_type_map;
    for (const auto& entry : gs.ground_truth) {
        gt_set.insert({entry.flow_id, entry.boundary});
        gt_type_map[{entry.flow_id, entry.boundary}] = entry.type;
    }

    std::set<std::pair<std::string, int>> detected_set;
    auto detect_boundary = [&](int boundary_idx) {
        const BaseSketch& sk_new = em.get_previous_sketch(0);
        const BaseSketch& sk_old = em.get_previous_sketch(1);
        for (const auto& fid : gs.flow_keys) {
            auto diff = diff_histograms(sk_new, sk_old, fid);
            auto scores = compute_scores(diff);

            bool detected = false;
            if (detector.normalized) {
                const double baseline = std::max(1.0, sk_old.query(fid));
                const double normalized_l1 = scores.l1 / baseline;
                detected = normalized_l1 > detector.ratio_threshold
                        && scores.l1 > detector.absolute_floor;
            } else {
                detected = scores.l1 > detector.raw_threshold;
            }

            if (detected)
                detected_set.insert({fid, boundary_idx});
        }
    };

    int current_epoch = 0;
    for (const auto& [ts, key, lat] : gs.packets) {
        const int packet_epoch = static_cast<int>(ts / gs.epoch_duration);
        while (packet_epoch > current_epoch) {
            if (current_epoch > 0)
                detect_boundary(current_epoch - 1);
            em.advance_epoch();
            ++current_epoch;
        }
        em.get_current_sketch().update(key, lat);
    }
    if (current_epoch > 0)
        detect_boundary(current_epoch - 1);

    DetectionResult result;
    for (const auto& d : detected_set) {
        auto it = gt_type_map.find(d);
        if (it != gt_type_map.end()) {
            ++result.tp;
            result.tp_by_type[it->second]++;
        } else {
            ++result.fp;
        }
    }
    for (const auto& g : gt_set) {
        if (!detected_set.count(g)) {
            ++result.fn;
            result.fn_by_type[gt_type_map[g]]++;
        }
    }
    return result;
}

static int width_for_budget(int base_width, int base_bins, int base_snapshots,
                            int bins, int snapshots)
{
    const double budget = static_cast<double>(base_width) * base_bins * base_snapshots;
    return std::max(32, static_cast<int>(std::round(budget / (bins * snapshots))));
}

static GeneratedStream make_eval_stream(int num_flows,
                                        int num_epochs,
                                        int epoch_size,
                                        double zipf_alpha,
                                        double magnitude,
                                        uint32_t seed)
{
    const double base_mu = std::log(5.0);
    const double base_sigma = 0.5;
    return generate_stream(num_flows, num_epochs, epoch_size, zipf_alpha, base_mu,
                           base_sigma, make_default_anomalies(magnitude), seed);
}

static void write_histogram_accuracy(const fs::path& out_dir) {
    constexpr int n_items = 10000;
    constexpr int n_max_key = 1000;
    constexpr double zipf_alpha = 1.5;
    constexpr double ln_mu = 1.6;
    constexpr double ln_sigma = 0.4;
    constexpr int width = 64;
    constexpr int depth = 3;
    constexpr uint32_t seed = 42;

    std::mt19937 rng(seed);
    ZipfDistribution zipf(n_max_key, zipf_alpha);
    std::lognormal_distribution<double> lat_dist(ln_mu, ln_sigma);
    BinConfig cfg(8, 0.5, 30.0, BinScheme::Logarithmic);

    std::vector<std::pair<std::string, double>> stream;
    std::map<std::string, int> freq;
    for (int i = 0; i < n_items; ++i) {
        std::string key = std::to_string(zipf.sample(rng));
        double lat = lat_dist(rng);
        stream.push_back({key, lat});
        freq[key]++;
    }

    std::string top_key;
    int top_count = -1;
    for (const auto& [key, count] : freq) {
        if (count > top_count) {
            top_count = count;
            top_key = key;
        }
    }

    std::vector<int> truth(cfg.num_bins(), 0);
    CountMinSketch cms(width, depth, cfg);
    ConservativeUpdateCMS cu(width, depth, cfg);
    CountSketch cs(width, depth, cfg);

    for (const auto& [key, lat] : stream) {
        cms.update(key, lat);
        cu.update(key, lat);
        cs.update(key, lat);
        if (key == top_key)
            truth[cfg.get_bin(lat)]++;
    }

    auto cms_h = cms.query_histogram(top_key);
    auto cu_h = cu.query_histogram(top_key);
    auto cs_h = cs.query_histogram(top_key);

    std::ofstream out(out_dir / "histogram_accuracy.csv");
    out << "bin,range_left,range_right,truth,cms,cu_cms,cs\n";
    const auto& edges = cfg.edges();
    for (int b = 0; b < cfg.num_bins(); ++b) {
        out << b << "," << edges[b] << "," << edges[b + 1] << ","
            << truth[b] << "," << cms_h[b] << "," << cu_h[b] << "," << cs_h[b]
            << "\n";
    }

    std::cout << "Histogram accuracy sample written for top flow \"" << top_key
              << "\" at w=64 -> outputs/evaluation/histogram_accuracy.csv\n";
}

static double mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

static double stdev(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;
    const double mu = mean(values);
    double acc = 0.0;
    for (double v : values) acc += (v - mu) * (v - mu);
    return std::sqrt(acc / values.size());
}

static void run_threshold_selection(const fs::path& out_dir,
                                    double& best_threshold,
                                    double& best_val_f1)
{
    // BUG FIX: must match run_baseline_vs_improved (Stage 6 setup). The
    // optimal L1/baseline ratio depends on per-flow packet counts and the
    // noise floor; tuning on a 10k-flow / 100k-packet stream and then
    // applying the result to a 100-flow / 5k-packet stream produces a
    // threshold that has no relationship to the data it is evaluated on.
    constexpr int num_flows = 100;
    constexpr int num_epochs = 4;
    constexpr int epoch_size = 5000;
    constexpr int width = 1024;
    constexpr int depth = 3;
    constexpr int bins = 10;
    constexpr int snapshots = 4;
    constexpr double zipf_alpha = 1.5;
    constexpr double abs_floor = 30.0;

    BinConfig cfg(bins, 0.5, 200.0, BinScheme::Logarithmic);
    std::vector<uint32_t> val_seeds = {42, 43};
    std::vector<double> thresholds = {0.05, 0.10, 0.15, 0.20, 0.25,
                                      0.30, 0.35, 0.40, 0.45, 0.50,
                                      0.60, 0.75, 1.00, 1.25, 1.50};

    std::ofstream sweep_out(out_dir / "threshold_sweep.csv");
    sweep_out << "threshold,sketch,seed,f1\n";
 
    best_threshold = thresholds.front();
    best_val_f1 = -1.0;
    for (double threshold : thresholds) {
        std::vector<double> f1s;
        for (uint32_t seed : val_seeds) {
            auto gs = make_eval_stream(num_flows, num_epochs, epoch_size,
                                       zipf_alpha, 2.0, seed);
            DetectorConfig detector;
            detector.normalized = true;
            detector.ratio_threshold = threshold;
            detector.absolute_floor = abs_floor;
            for (SketchKind kind : {SketchKind::CMS, SketchKind::CUCMS, SketchKind::CS}) {
                auto res = run_detection(kind, width, depth, snapshots,
                                         gs, cfg, detector);
                f1s.push_back(res.f1());
                sweep_out << threshold << "," << sketch_name(kind) << ","
                          << seed << "," << res.f1() << "\n";
            }
        }
        const double avg_f1 = mean(f1s);
        if (avg_f1 > best_val_f1) {
            best_val_f1 = avg_f1;
            best_threshold = threshold;
        }
    }

    std::cout << "Threshold sweep complete. Best validation threshold="
              << std::fixed << std::setprecision(2) << best_threshold
              << " with mean val F1=" << best_val_f1 << "\n";
}

static void run_baseline_vs_improved(const fs::path& out_dir,
                                     double best_threshold) {
    // Reproduces the Stage 6 setup exactly: 100 Zipf flows, 4 epochs x 5000
    // packets, all five anomalies present together, held-out seed 44.
    constexpr int num_flows   = 100;
    constexpr int num_epochs  = 4;
    constexpr int epoch_size  = 5000;
    constexpr int width       = 1024;
    constexpr int depth       = 3;
    constexpr int bins        = 10;
    constexpr int snapshots   = 4;
    constexpr double zipf_alpha = 1.5;
    constexpr double abs_floor  = 30.0;
    constexpr uint32_t test_seed = 44;

    const std::vector<uint32_t> test_seeds = {44, 45, 46, 47, 48};

    BinConfig cfg(bins, 0.5, 200.0, BinScheme::Logarithmic);
    auto anomalies = make_default_anomalies(2.0);

    DetectorConfig baseline;
    baseline.raw_threshold = 300.0;

    DetectorConfig improved;
    improved.normalized = true;
    improved.ratio_threshold = best_threshold;
    improved.absolute_floor = abs_floor;

    std::ofstream overall_out(out_dir / "baseline_vs_improved.csv");
    overall_out << "detector,sketch,seed,anomaly_type,tp,fp,fn,"
                   "precision,recall,f1\n";

    std::ofstream per_type_out(out_dir / "per_type_breakdown.csv");
    per_type_out << "detector,sketch,seed,anomaly_type,tp,fn,recall\n";

    struct Run { const char* name; const DetectorConfig* cfg; };
    const std::vector<Run> runs = {
        {"baseline_raw_l1",    &baseline},
        {"improved_normalized", &improved},
    };

    // Collect every anomaly type that appears in GT so per-type rows are
    // emitted even for types with 0 TPs and 0 FNs (keeps the CSV rectangular).
    std::set<AnomalyType> all_types;
    {
        auto probe = generate_stream(num_flows, num_epochs, epoch_size,
                                     zipf_alpha, std::log(5.0), 0.5,
                                     anomalies, test_seeds.front());
        for (const auto& entry : probe.ground_truth)
            all_types.insert(entry.type);
    }

    for (uint32_t seed : test_seeds) {
        auto gs = generate_stream(num_flows, num_epochs, epoch_size,
                                  zipf_alpha, std::log(5.0), 0.5,
                                  anomalies, seed);
        for (const auto& run : runs) {
            for (SketchKind kind : {SketchKind::CMS, SketchKind::CUCMS, SketchKind::CS}) {
                auto res = run_detection(kind, width, depth, snapshots,
                                         gs, cfg, *run.cfg);

                overall_out << run.name << "," << sketch_name(kind) << ","
                            << seed << ",ALL,"
                            << res.tp << "," << res.fp << "," << res.fn << ","
                            << res.precision() << "," << res.recall() << ","
                            << res.f1() << "\n";

                for (AnomalyType type : all_types) {
                    const int tp_c = res.tp_by_type.count(type) ? res.tp_by_type.at(type) : 0;
                    const int fn_c = res.fn_by_type.count(type) ? res.fn_by_type.at(type) : 0;
                    const double rec = (tp_c + fn_c > 0)
                        ? static_cast<double>(tp_c) / (tp_c + fn_c) : 0.0;
                    per_type_out << run.name << "," << sketch_name(kind) << ","
                                 << seed << "," << anomaly_type_name(type) << ","
                                 << tp_c << "," << fn_c << "," << rec << "\n";
                }
            }
        }
    }

    std::cout << "Baseline-vs-improved across " << test_seeds.size()
              << " held-out seeds written to baseline_vs_improved.csv "
              << "and per_type_breakdown.csv\n";
}

static void run_bins_sweep(const fs::path& out_dir, const DetectorConfig& detector) {
    constexpr int base_width = 1024;
    constexpr int depth = 3;
    constexpr int base_bins = 10;
    constexpr int snapshots = 4;
    constexpr int base_snapshots = 4;
    // BUG FIX: align with the rest of the evaluation (Stage 6 scale) so this
    // sweep is comparable to memory_sweep, zipf_sweep, sensitivity_sweep, etc.
    constexpr int num_flows = 100;
    constexpr int num_epochs = 4;
    constexpr int epoch_size = 5000;
    constexpr double zipf_alpha = 1.5;
    std::vector<uint32_t> seeds = {42, 43, 44};

    std::ofstream out(out_dir / "bins_sweep.csv");
    out << "sketch,bins,width,seed,tp,fp,fn,f1\n";

    for (int bins : {4, 8, 16, 32}) {
        const int width = width_for_budget(base_width, base_bins, base_snapshots,
                                           bins, snapshots);
        BinConfig cfg(bins, 0.5, 200.0, BinScheme::Logarithmic);
        for (uint32_t seed : seeds) {
            auto gs = make_eval_stream(num_flows, num_epochs, epoch_size,
                                       zipf_alpha, 2.0, seed);
            for (SketchKind kind : {SketchKind::CMS, SketchKind::CUCMS, SketchKind::CS}) {
                auto res = run_detection(kind, width, depth, snapshots, gs, cfg, detector);
                out << sketch_name(kind) << "," << bins << "," << width << ","
                    << seed << "," << res.tp << "," << res.fp << ","
                    << res.fn << "," << res.f1() << "\n";
            }
        }
    }
}

static void run_memory_sweep(const fs::path& out_dir, const DetectorConfig& detector) {
    // Fix everything except width. This isolates the memory/accuracy tradeoff
    // from the confounds that bins_sweep and snapshots_sweep introduce
    // (those reduce width as a function of bins/K to hold total memory fixed).
    constexpr int depth      = 3;
    constexpr int bins       = 10;
    constexpr int snapshots  = 4;
    constexpr int num_flows  = 100;
    constexpr int num_epochs = 4;
    constexpr int epoch_size = 5000;
    constexpr double zipf_alpha = 1.5;

    BinConfig cfg(bins, 0.5, 200.0, BinScheme::Logarithmic);
    const std::vector<uint32_t> seeds = {44, 45, 46, 47, 48};
    const std::vector<int> widths = {32, 64, 128, 256, 512, 1024};

    std::ofstream out(out_dir / "memory_sweep.csv");
    out << "sketch,width,memory_bytes,seed,tp,fp,fn,f1\n";

    for (int width : widths) {
        // Per-snapshot memory; total across K snapshots = this * snapshots.
        // counter size = 4 bytes (int32_t), matching the rest of the repo.
        const long mem_bytes =
            static_cast<long>(width) * depth * bins * 4 * snapshots;

        for (uint32_t seed : seeds) {
            auto gs = make_eval_stream(num_flows, num_epochs, epoch_size,
                                       zipf_alpha, 2.0, seed);
            for (SketchKind kind : {SketchKind::CMS, SketchKind::CUCMS, SketchKind::CS}) {
                auto res = run_detection(kind, width, depth, snapshots,
                                         gs, cfg, detector);
                out << sketch_name(kind) << "," << width << ","
                    << mem_bytes << "," << seed << ","
                    << res.tp << "," << res.fp << "," << res.fn << ","
                    << res.f1() << "\n";
            }
        }
    }
}

static void run_snapshots_sweep(const fs::path& out_dir, const DetectorConfig& detector) {
    constexpr int base_width = 1024;
    constexpr int depth = 3;
    constexpr int bins = 10;
    constexpr int base_bins = 10;
    constexpr int base_snapshots = 4;
    // BUG FIX: align with Stage 6 scale (see bins_sweep).
    //
    // CAVEAT (not a code bug, but worth flagging in the report): with
    // num_epochs fixed at 4 and detection only ever diffing adjacent epochs,
    // increasing K beyond 2 allocates ring-buffer slots that are never read.
    // The width-for-budget formula then shrinks `width` as K grows, so this
    // sweep effectively measures F1 vs width under a different label —
    // memory_sweep already does that more cleanly. To make this experiment
    // genuinely about K, num_epochs would need to grow with K and the
    // detector would need to use older snapshots.
    constexpr int num_flows = 100;
    constexpr int epoch_size = 5000;
    constexpr double zipf_alpha = 1.5;
    constexpr int num_epochs = 4;
    std::vector<uint32_t> seeds = {42, 43, 44};

    std::ofstream out(out_dir / "snapshots_sweep.csv");
    out << "sketch,snapshots,width,seed,tp,fp,fn,f1\n";

    for (int snapshots : {2, 4, 8, 16}) {
        const int width = width_for_budget(base_width, base_bins, base_snapshots,
                                           bins, snapshots);
        BinConfig cfg(bins, 0.5, 200.0, BinScheme::Logarithmic);
        for (uint32_t seed : seeds) {
            auto gs = make_eval_stream(num_flows, num_epochs, epoch_size,
                                       zipf_alpha, 2.0, seed);
            for (SketchKind kind : {SketchKind::CMS, SketchKind::CUCMS, SketchKind::CS}) {
                auto res = run_detection(kind, width, depth, snapshots, gs, cfg, detector);
                out << sketch_name(kind) << "," << snapshots << "," << width << ","
                    << seed << "," << res.tp << "," << res.fp << ","
                    << res.fn << "," << res.f1() << "\n";
            }
        }
    }
}

// NEW: epoch size sensitivity (Task 4 from PROJECT_GUIDE.md). Shows the
// minimum viable per-epoch packet budget for reliable detection. Sweeps
// epoch_size while holding num_flows, num_epochs, and the detector fixed.
static void run_epoch_sweep(const fs::path& out_dir, const DetectorConfig& detector) {
    constexpr int width = 1024;
    constexpr int depth = 3;
    constexpr int bins = 10;
    constexpr int snapshots = 4;
    constexpr int num_flows = 100;
    constexpr int num_epochs = 4;
    constexpr double zipf_alpha = 1.5;
    const std::vector<uint32_t> seeds = {42, 43, 44};
    const std::vector<int> epoch_sizes = {500, 1000, 2000, 5000, 10000};

    BinConfig cfg(bins, 0.5, 200.0, BinScheme::Logarithmic);
    std::ofstream out(out_dir / "epoch_sweep.csv");
    out << "sketch,epoch_size,seed,tp,fp,fn,f1\n";

    for (int epoch_size : epoch_sizes) {
        for (uint32_t seed : seeds) {
            auto gs = make_eval_stream(num_flows, num_epochs, epoch_size,
                                       zipf_alpha, 2.0, seed);
            for (SketchKind kind : {SketchKind::CMS, SketchKind::CUCMS, SketchKind::CS}) {
                auto res = run_detection(kind, width, depth, snapshots,
                                         gs, cfg, detector);
                out << sketch_name(kind) << "," << epoch_size << "," << seed << ","
                    << res.tp << "," << res.fp << "," << res.fn << ","
                    << res.f1() << "\n";
            }
        }
    }
}

static void run_zipf_sweep(const fs::path& out_dir, const DetectorConfig& detector) {
    constexpr int width = 1024;
    constexpr int depth = 3;
    constexpr int bins = 10;
    constexpr int snapshots = 4;
    std::vector<uint32_t> seeds = {42, 43, 44};

    BinConfig cfg(bins, 0.5, 200.0, BinScheme::Logarithmic);
    std::ofstream out(out_dir / "zipf_sweep.csv");
    out << "sketch,zipf_alpha,seed,tp,fp,fn,f1\n";

    for (double alpha : {0.5, 1.0, 1.5, 2.0}) {
        for (uint32_t seed : seeds) {
            auto gs = make_eval_stream(100, 4, 5000, alpha, 2.0, seed);
            for (SketchKind kind : {SketchKind::CMS, SketchKind::CUCMS, SketchKind::CS}) {
                auto res = run_detection(kind, width, depth, snapshots, gs, cfg, detector);
                out << sketch_name(kind) << "," << alpha << "," << seed << ","
                    << res.tp << "," << res.fp << "," << res.fn << "," << res.f1()
                    << "\n";
            }
        }
    }
}

static void run_sensitivity_sweep(const fs::path& out_dir, const DetectorConfig& detector) {
    constexpr int width = 1024;
    constexpr int depth = 3;
    constexpr int bins = 10;
    constexpr int snapshots = 4;
    constexpr double zipf_alpha = 1.5;
    std::vector<uint32_t> seeds = {42, 43, 44};
    BinConfig cfg(bins, 0.5, 200.0, BinScheme::Logarithmic);

    std::ofstream out(out_dir / "sensitivity_sweep.csv");
    out << "sketch,magnitude,seed,tp,fp,fn,f1\n";

    for (double magnitude : {1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 5.0}) {
        for (uint32_t seed : seeds) {
            auto gs = make_eval_stream(100, 4, 5000, zipf_alpha, magnitude, seed);
            for (SketchKind kind : {SketchKind::CMS, SketchKind::CUCMS, SketchKind::CS}) {
                auto res = run_detection(kind, width, depth, snapshots, gs, cfg, detector);
                out << sketch_name(kind) << "," << magnitude << "," << seed << ","
                    << res.tp << "," << res.fp << "," << res.fn << "," << res.f1()
                    << "\n";
            }
        }
    }
}

static void run_bin_scheme_comparison(const fs::path& out_dir,
                                      const DetectorConfig& detector) {
    constexpr int width = 1024;
    constexpr int depth = 3;
    constexpr int bins = 10;
    constexpr int snapshots = 4;
    constexpr double zipf_alpha = 1.5;
    std::vector<uint32_t> seeds = {42, 43, 44};

    std::ofstream out(out_dir / "bin_scheme_comparison.csv");
    out << "sketch,scheme,seed,tp,fp,fn,f1\n";

    for (uint32_t seed : seeds) {
        auto gs = make_eval_stream(100, 4, 5000, zipf_alpha, 2.0, seed);
        for (auto scheme : {BinScheme::Uniform, BinScheme::Logarithmic}) {
            BinConfig cfg = (scheme == BinScheme::Uniform)
                ? BinConfig(bins, 0.0, 200.0, BinScheme::Uniform)
                : BinConfig(bins, 0.5, 200.0, BinScheme::Logarithmic);

            for (SketchKind kind : {SketchKind::CMS, SketchKind::CUCMS, SketchKind::CS}) {
                auto res = run_detection(kind, width, depth, snapshots, gs, cfg, detector);
                out << sketch_name(kind) << ","
                    << (scheme == BinScheme::Uniform ? "uniform" : "logarithmic") << ","
                    << seed << "," << res.tp << "," << res.fp << ","
                    << res.fn << "," << res.f1() << "\n";
            }
        }
    }
}

static void run_hash_rotation_experiment(const fs::path& out_dir) {
    constexpr int width = 64;
    constexpr int depth = 3;
    constexpr int bins = 10;
    constexpr int num_flows = 10000;
    constexpr int n_items = 100000;
    constexpr double zipf_alpha = 1.5;
    const double mu = std::log(5.0);
    constexpr double sigma = 0.5;

    BinConfig cfg(bins, 0.5, 200.0, BinScheme::Logarithmic);
    std::ofstream out(out_dir / "hash_rotation.csv");
    out << "sketch,flow,mean_l1,std_l1\n";

    for (SketchKind kind : {SketchKind::CMS, SketchKind::CUCMS, SketchKind::CS}) {
        std::vector<double> flow1_l1, flow10_l1, flow100_l1, flow1000_l1, flow10000_l1;

        for (uint32_t pair_seed = 0; pair_seed < 10; ++pair_seed) {
            std::mt19937 rng(1000 + pair_seed);
            ZipfDistribution zipf(num_flows, zipf_alpha);
            std::lognormal_distribution<double> lat_dist(mu, sigma);

            auto sk_a = make_sketch(kind, width, depth, cfg, pair_seed);
            auto sk_b = make_sketch(kind, width, depth, cfg, pair_seed + 1);

            for (int i = 0; i < n_items; ++i) {
                std::string key = std::to_string(zipf.sample(rng));
                double lat = lat_dist(rng);
                sk_a->update(key, lat);
                sk_b->update(key, lat);
            }

            flow1_l1.push_back(compute_scores(diff_histograms(*sk_a, *sk_b, "1")).l1);
            flow10_l1.push_back(compute_scores(diff_histograms(*sk_a, *sk_b, "10")).l1);
            flow100_l1.push_back(compute_scores(diff_histograms(*sk_a, *sk_b, "100")).l1);
            flow1000_l1.push_back(compute_scores(diff_histograms(*sk_a, *sk_b, "1000")).l1);
            flow10000_l1.push_back(compute_scores(diff_histograms(*sk_a, *sk_b, "10000")).l1);
        }

        out << sketch_name(kind) << ",1," << mean(flow1_l1) << ","
            << stdev(flow1_l1) << "\n";
        out << sketch_name(kind) << ",10," << mean(flow10_l1) << ","
            << stdev(flow10_l1) << "\n";
        out << sketch_name(kind) << ",100," << mean(flow100_l1) << ","
            << stdev(flow100_l1) << "\n";
        out << sketch_name(kind) << ",1000," << mean(flow1000_l1) << ","
            << stdev(flow1000_l1) << "\n";
        out << sketch_name(kind) << ",10000," << mean(flow10000_l1) << ","
            << stdev(flow10000_l1) << "\n";
    }
}

static std::vector<std::pair<std::string, double>>
lognormal_packets(const std::string& key, int n, double mu, double sigma,
                  std::mt19937& rng)
{
    std::lognormal_distribution<double> dist(mu, sigma);
    std::vector<std::pair<std::string, double>> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i)
        out.push_back({key, dist(rng)});
    return out;
}

static void append_packets(std::vector<std::pair<std::string, double>>& dst,
                           const std::vector<std::pair<std::string, double>>& src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}

static std::vector<std::tuple<double, std::string, double>>
make_classification_stream(const std::vector<std::pair<std::string, double>>& ep0,
                           const std::vector<std::pair<std::string, double>>& ep1,
                           double epoch_duration)
{
    std::vector<std::tuple<double, std::string, double>> stream;
    stream.reserve(ep0.size() + ep1.size());
    for (int i = 0; i < static_cast<int>(ep0.size()); ++i)
        stream.push_back({static_cast<double>(i), ep0[i].first, ep0[i].second});
    for (int i = 0; i < static_cast<int>(ep1.size()); ++i)
        stream.push_back({epoch_duration + i, ep1[i].first, ep1[i].second});
    return stream;
}

static std::vector<ClassificationCase> build_classification_cases(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    const std::string key_x = "X";
    const std::string key_y = "Y";
    std::vector<ClassificationCase> cases;

    {
        std::vector<std::pair<std::string, double>> ep0, ep1;
        append_packets(ep0, lognormal_packets(key_x, n, std::log(5.0), 0.5, rng));
        append_packets(ep0, lognormal_packets(key_y, n, std::log(5.0), 0.5, rng));
        append_packets(ep1, lognormal_packets(key_y, n, std::log(5.0), 0.5, rng));
        const double dur = static_cast<double>(std::max(ep0.size(), ep1.size()));
        cases.push_back({"Disappearance", ChangeType::Disappearance, dur,
                         make_classification_stream(ep0, ep1, dur)});
    }
    {
        std::vector<std::pair<std::string, double>> ep0, ep1;
        append_packets(ep0, lognormal_packets(key_x, n, std::log(5.0), 0.5, rng));
        append_packets(ep0, lognormal_packets(key_y, n, std::log(5.0), 0.5, rng));
        append_packets(ep1, lognormal_packets(key_x, 3 * n, std::log(5.0), 0.5, rng));
        append_packets(ep1, lognormal_packets(key_y, n, std::log(5.0), 0.5, rng));
        const double dur = static_cast<double>(std::max(ep0.size(), ep1.size()));
        cases.push_back({"VolumeChange", ChangeType::VolumeChange, dur,
                         make_classification_stream(ep0, ep1, dur)});
    }
    {
        std::vector<std::pair<std::string, double>> ep0, ep1;
        append_packets(ep0, lognormal_packets(key_x, n, std::log(5.0), 0.5, rng));
        append_packets(ep0, lognormal_packets(key_y, n, std::log(5.0), 0.5, rng));
        append_packets(ep1, lognormal_packets(key_x, n, std::log(5.0), 0.5, rng));
        append_packets(ep1, lognormal_packets(key_x, 2 * n, std::log(30.0), 0.05, rng));
        append_packets(ep1, lognormal_packets(key_y, n, std::log(5.0), 0.5, rng));
        const double dur = static_cast<double>(std::max(ep0.size(), ep1.size()));
        cases.push_back({"Spike", ChangeType::Spike, dur,
                         make_classification_stream(ep0, ep1, dur)});
    }
    {
        std::vector<std::pair<std::string, double>> ep0, ep1;
        append_packets(ep0, lognormal_packets(key_x, n, std::log(5.0), 1.0, rng));
        append_packets(ep0, lognormal_packets(key_y, n, std::log(5.0), 0.5, rng));
        append_packets(ep1, lognormal_packets(key_x, n, std::log(10.0), 1.0, rng));
        append_packets(ep1, lognormal_packets(key_y, n, std::log(5.0), 0.5, rng));
        const double dur = static_cast<double>(std::max(ep0.size(), ep1.size()));
        cases.push_back({"Shift", ChangeType::Shift, dur,
                         make_classification_stream(ep0, ep1, dur)});
    }
    {
        std::vector<std::pair<std::string, double>> ep0, ep1;
        append_packets(ep0, lognormal_packets(key_x, n, std::log(5.0), 0.5, rng));
        append_packets(ep0, lognormal_packets(key_y, n, std::log(5.0), 0.5, rng));
        append_packets(ep1, lognormal_packets(key_x, n, std::log(5.0), 1.0, rng));
        append_packets(ep1, lognormal_packets(key_y, n, std::log(5.0), 0.5, rng));
        const double dur = static_cast<double>(std::max(ep0.size(), ep1.size()));
        cases.push_back({"Spread", ChangeType::Spread, dur,
                         make_classification_stream(ep0, ep1, dur)});
    }

    return cases;
}

static std::pair<ChangeType, ChangeType> classify_case(SketchKind kind,
                                                       const ClassificationCase& test_case,
                                                       const BinConfig& cfg)
{
    constexpr int width = 1024;
    constexpr int depth = 3;
    auto factory = [&]() { return make_sketch(kind, width, depth, cfg); };
    EpochManager em(2, factory);
    StreamProcessor proc(em, test_case.epoch_duration);

    for (const auto& [ts, key, lat] : test_case.stream)
        proc.process(ts, key, lat);

    const BaseSketch& epoch1 = em.get_previous_sketch(0);
    const BaseSketch& epoch0 = em.get_previous_sketch(1);

    auto diff_x = diff_histograms(epoch1, epoch0, "X");
    auto diff_y = diff_histograms(epoch1, epoch0, "Y");
    return {classify_change(diff_x), classify_change(diff_y)};
}

static void run_classification_evaluation(const fs::path& out_dir) {
    BinConfig cfg(10, 0.5, 200.0, BinScheme::Logarithmic);
    std::ofstream acc_out(out_dir / "classifier_accuracy_vs_n.csv");
    acc_out << "sketch,n_packets,seed,accuracy,background_none_rate\n";

    // Key: {sketch, expected, predicted}
    std::map<std::tuple<std::string, std::string, std::string>, int> confusion;
    for (int n : {100, 200, 500, 1000}) {
        for (uint32_t seed : {42u, 43u, 44u}) {
            auto cases = build_classification_cases(n, seed);
            for (SketchKind kind : {SketchKind::CMS, SketchKind::CUCMS, SketchKind::CS}) {
                int correct = 0;
                int bg_none = 0;
                for (const auto& test_case : cases) {
                    auto [pred_x, pred_y] = classify_case(kind, test_case, cfg);
                    if (pred_x == test_case.expected) ++correct;
                    if (pred_y == ChangeType::None) ++bg_none;
                    if (n == 1000) {
                        confusion[{sketch_name(kind),
                                   change_type_name(test_case.expected),
                                   change_type_name(pred_x)}]++;
                    }
                }
                const double acc = static_cast<double>(correct) / cases.size();
                const double bg_rate = static_cast<double>(bg_none) / cases.size();
                acc_out << sketch_name(kind) << "," << n << "," << seed << ","
                        << acc << "," << bg_rate << "\n";
            }
        }
    }

    std::ofstream conf_out(out_dir / "classifier_confusion_matrix.csv");
    conf_out << "sketch,expected,predicted,count\n";
    for (const auto& [key, count] : confusion)
        conf_out << std::get<0>(key) << "," << std::get<1>(key) << ","
                 << std::get<2>(key) << "," << count << "\n";
}

// Experiment 1: Flow count sweep (100 to 10k).
// Tests whether sketch equivalence holds under heavier collision load.
// Width is held fixed at 1024 so that increasing num_flows directly
// increases per-bucket collision pressure without any other confound.
static void run_flow_count_sweep(const fs::path& out_dir, const DetectorConfig& detector) {
    constexpr int width = 1024;
    constexpr int depth = 3;
    constexpr int bins = 10;
    constexpr int snapshots = 4;
    constexpr int num_epochs = 4;
    constexpr int epoch_size = 5000;
    constexpr double zipf_alpha = 1.5;
    const std::vector<uint32_t> seeds = {42, 43, 44};
    const std::vector<int> flow_counts = {100, 500, 1000, 5000, 10000};

    BinConfig cfg(bins, 0.5, 200.0, BinScheme::Logarithmic);
    std::ofstream out(out_dir / "flow_count_sweep.csv");
    out << "sketch,num_flows,seed,tp,fp,fn,f1\n";

    for (int num_flows : flow_counts) {
        for (uint32_t seed : seeds) {
            auto gs = make_eval_stream(num_flows, num_epochs, epoch_size,
                                       zipf_alpha, 2.0, seed);
            for (SketchKind kind : {SketchKind::CMS, SketchKind::CUCMS, SketchKind::CS}) {
                auto res = run_detection(kind, width, depth, snapshots,
                                         gs, cfg, detector);
                out << sketch_name(kind) << "," << num_flows << "," << seed << ","
                    << res.tp << "," << res.fp << "," << res.fn << "," << res.f1()
                    << "\n";
            }
        }
    }
}

// Experiment 3: Grey-failure head-to-head.
// Compares DHS against a scalar heavy-changer detector on streams where
// packet volume is stable but the latency distribution shifts.  A scalar
// detector (flags flow when |count_new - count_old| / count_old > 30%)
// cannot see pure distribution shifts and should score ~0 detection rate,
// while DHS (normalised L1 > threshold and L1 > floor) flags the change
// via the differenced histogram.
static void run_grey_failure_comparison(const fs::path& out_dir, double best_threshold) {
    constexpr int width = 1024, depth = 3, bins = 10;
    constexpr int num_flows = 100, num_epochs = 4, epoch_size = 5000;
    constexpr double zipf_alpha = 1.5;
    constexpr double abs_floor = 30.0;
    constexpr double scalar_threshold = 0.30;  // 30% relative count change

    BinConfig cfg(bins, 0.5, 200.0, BinScheme::Logarithmic);

    struct Scenario {
        const char* name;
        AnomalyType type;
        double      magnitude;
        const char* flow;
    };
    const std::vector<Scenario> scenarios = {
        // SuddenSpike: 10x mean latency increase, packet count unchanged.
        {"LatencySpike", AnomalyType::SuddenSpike, 10.0, "1"},
        // Spread: 3x wider latency distribution (sigma × 3), count unchanged.
        {"Spread",       AnomalyType::Spread,       3.0,  "2"},
        // GradualRamp: mean grows by 2× per epoch, count unchanged.
        // start_epoch=0 so that boundary 0→1 already shows steps=1 shift.
        {"GradualRamp",  AnomalyType::GradualRamp,  2.0,  "3"},
    };

    std::ofstream out(out_dir / "grey_failure_comparison.csv");
    out << "seed,sketch,scenario,detected_dhs,detected_scalar,l1,count_ratio\n";

    for (uint32_t seed : {42u, 43u, 44u}) {
        std::vector<AnomalySpec> specs;
        for (int i = 0; i < static_cast<int>(scenarios.size()); ++i) {
            const auto& s = scenarios[i];
            // GradualRamp needs start_epoch=0 so the epoch-0→1 boundary
            // captures steps=1 (the first actual ramp step).
            // SuddenSpike/Spread use start_epoch=1: epoch 0 is normal,
            // epoch 1 is anomalous, giving a clean epoch-0→1 contrast.
            int start = (s.type == AnomalyType::GradualRamp) ? 0 : 1;
            specs.push_back({std::string(s.flow), s.type, start, s.magnitude});
        }

        auto gs = generate_stream(num_flows, num_epochs, epoch_size, zipf_alpha,
                                  std::log(5.0), 0.5, specs, seed);

        for (SketchKind kind : {SketchKind::CMS, SketchKind::CUCMS, SketchKind::CS}) {
            // Build two independent sketches for epochs 0 and 1.
            auto sk0 = make_sketch(kind, width, depth, cfg);
            auto sk1 = make_sketch(kind, width, depth, cfg);

            for (const auto& [ts, key, lat] : gs.packets) {
                int e = static_cast<int>(ts / gs.epoch_duration);
                if (e == 0)      sk0->update(key, lat);
                else if (e == 1) sk1->update(key, lat);
            }

            for (const auto& s : scenarios) {
                const std::string fid(s.flow);
                auto diff        = diff_histograms(*sk1, *sk0, fid);
                auto scores      = compute_scores(diff);
                double cnt_new   = std::max(1.0, sk1->query(fid));
                double cnt_old   = std::max(1.0, sk0->query(fid));
                double norm_l1   = scores.l1 / cnt_old;
                double cnt_ratio = std::abs(cnt_new - cnt_old) / cnt_old;

                bool dhs_det    = norm_l1 > best_threshold && scores.l1 > abs_floor;
                bool scalar_det = cnt_ratio > scalar_threshold;

                out << seed << "," << sketch_name(kind) << "," << s.name << ","
                    << (dhs_det ? 1 : 0) << "," << (scalar_det ? 1 : 0) << ","
                    << std::fixed << std::setprecision(4)
                    << scores.l1 << "," << cnt_ratio << "\n";
            }
        }
    }
}

int main() {
    const fs::path out_dir = fs::path("outputs") / "evaluation";
    fs::create_directories(out_dir);

    std::cout << "Evaluation runner\n";
    std::cout << "Writing CSV outputs to " << out_dir.string() << "\n\n";

    write_histogram_accuracy(out_dir);

    double best_threshold = 0.25;
    double best_val_f1 = 0.0;
    run_threshold_selection(out_dir, best_threshold, best_val_f1);
    run_baseline_vs_improved(out_dir, best_threshold);

    DetectorConfig improved;
    improved.normalized = true;
    improved.ratio_threshold = best_threshold;
    improved.absolute_floor = 30.0;

    run_bins_sweep(out_dir, improved);
    run_snapshots_sweep(out_dir, improved);
    run_memory_sweep(out_dir, improved);   
    run_zipf_sweep(out_dir, improved);
    run_sensitivity_sweep(out_dir, improved);
    run_bin_scheme_comparison(out_dir, improved);
    run_hash_rotation_experiment(out_dir);
    run_classification_evaluation(out_dir);
    run_epoch_sweep(out_dir, improved);
    run_flow_count_sweep(out_dir, improved);
    run_grey_failure_comparison(out_dir, best_threshold);

    std::cout << "Generated:\n";
    std::cout << "  outputs/evaluation/histogram_accuracy.csv\n";
    std::cout << "  outputs/evaluation/threshold_sweep.csv\n";
    std::cout << "  outputs/evaluation/baseline_vs_improved.csv\n";
    std::cout << "  outputs/evaluation/bins_sweep.csv\n";
    std::cout << "  outputs/evaluation/snapshots_sweep.csv\n";
    std::cout << "  outputs/evaluation/zipf_sweep.csv\n";
    std::cout << "  outputs/evaluation/sensitivity_sweep.csv\n";
    std::cout << "  outputs/evaluation/bin_scheme_comparison.csv\n";
    std::cout << "  outputs/evaluation/hash_rotation.csv\n";
    std::cout << "  outputs/evaluation/classifier_accuracy_vs_n.csv\n";
    std::cout << "  outputs/evaluation/classifier_confusion_matrix.csv\n";
    std::cout << "  outputs/evaluation/per_type_breakdown.csv\n";
    std::cout << "  outputs/evaluation/memory_sweep.csv\n";
    std::cout << "  outputs/evaluation/epoch_sweep.csv\n";
    std::cout << "  outputs/evaluation/flow_count_sweep.csv\n";
    std::cout << "  outputs/evaluation/grey_failure_comparison.csv\n";
    return 0;
}
