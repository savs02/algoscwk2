#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Synthetic multi-epoch stream generator
//
// Produces a stream of (timestamp, flow_key, latency) tuples that can be
// replayed through the sketch/epoch pipeline.
//
// Background traffic:
//   num_flows flows, each with a base lognormal(base_mu, base_sigma) latency.
//   Per-epoch packet counts follow Zipf(alpha): flow i (1-indexed) receives a
//   fraction proportional to 1/i^alpha of the epoch_size total packets.
//
// Anomaly injection (6b anomaly types):
//   SuddenSpike   — mean doubles for exactly one epoch, then returns to normal
//   GradualRamp   — mean multiplied by 1.2^k each epoch k steps after start
//   PeriodicBurst — alternates between high (mean×2) and normal every epoch
//   Spread        — sigma doubles from start_epoch onward
//   Disappearance — flow stops sending from start_epoch onward
//
// Ground truth: the set of (flow_id, boundary_index) where the distribution
// changes between adjacent epochs.  A boundary at index b covers epoch b→b+1.
// ---------------------------------------------------------------------------

enum class AnomalyType {
    SuddenSpike,
    GradualRamp,
    PeriodicBurst,
    Spread,
    Disappearance,
};

inline const char* anomaly_type_name(AnomalyType t) {
    switch (t) {
        case AnomalyType::SuddenSpike:   return "SuddenSpike";
        case AnomalyType::GradualRamp:   return "GradualRamp";
        case AnomalyType::PeriodicBurst: return "PeriodicBurst";
        case AnomalyType::Spread:        return "Spread";
        case AnomalyType::Disappearance: return "Disappearance";
    }
    return "Unknown";
}

struct AnomalySpec {
    std::string flow_id;
    AnomalyType type;
    int         start_epoch;     // first epoch where anomaly is active
    double      magnitude = 2.0; // multiplier applied to the mean shift/spread
};

struct GroundTruthEntry {
    std::string flow_id;
    int         boundary;   // index e means epoch e → epoch e+1
};

struct GeneratedStream {
    std::vector<std::tuple<double, std::string, double>> packets;
    std::vector<std::string>   flow_keys;    // all distinct keys
    std::vector<GroundTruthEntry> ground_truth;
    int    num_epochs;
    double epoch_duration;   // timestamp units per epoch
};

// ---------------------------------------------------------------------------

// Return per-epoch lognormal parameters (mu, sigma) and packet count for
// a flow given its base parameters and any anomaly spec that applies.
// Returns {mu, sigma, n_packets}.  n_packets == 0 means no packets (disappearance).
static std::tuple<double, double, int>
flow_params_for_epoch(double base_mu, double base_sigma, int base_n,
                      const AnomalySpec* anomaly,  // nullptr = no anomaly
                      int epoch)
{
    if (anomaly == nullptr)
        return {base_mu, base_sigma, base_n};

    int steps = epoch - anomaly->start_epoch;

    switch (anomaly->type) {
        case AnomalyType::SuddenSpike:
            // Mean multiplied by magnitude for exactly one epoch.
            if (steps == 0)
                return {base_mu + std::log(anomaly->magnitude), base_sigma, base_n};
            break;

        case AnomalyType::GradualRamp:
            // Mean multiplied by 1.2^steps.
            if (steps >= 0)
                return {base_mu + steps * std::log(anomaly->magnitude), base_sigma, base_n};
            break;

        case AnomalyType::PeriodicBurst:
            // Alternates: high at start_epoch, then normal, then high, ...
            if (steps >= 0 && (steps % 2 == 0))
                return {base_mu + std::log(anomaly->magnitude), base_sigma, base_n};
            break;

        case AnomalyType::Spread:
            // Sigma doubles from start_epoch onward.
            if (steps >= 0)
                return {base_mu, base_sigma * anomaly->magnitude, base_n};
            break;

        case AnomalyType::Disappearance:
            // No packets from start_epoch onward.
            if (steps >= 0)
                return {base_mu, base_sigma, 0};
            break;
    }
    return {base_mu, base_sigma, base_n};
}

// ---------------------------------------------------------------------------

inline GeneratedStream generate_stream(
    int    num_flows,
    int    num_epochs,
    int    epoch_size,   // total packets per epoch across all flows
    double zipf_alpha,
    double base_mu,
    double base_sigma,
    const std::vector<AnomalySpec>& anomaly_specs,
    uint32_t seed)
{
    assert(num_flows  > 0);
    assert(num_epochs > 1);
    assert(epoch_size > 0);

    // --- Build flow list (1-indexed keys "1", "2", ..., num_flows) ---
    std::vector<std::string> flow_keys;
    flow_keys.reserve(num_flows);
    for (int i = 1; i <= num_flows; ++i)
        flow_keys.push_back(std::to_string(i));

    // --- Zipf weights and per-epoch base packet counts ---
    std::vector<double> weights(num_flows);
    double weight_sum = 0.0;
    for (int i = 0; i < num_flows; ++i) {
        weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), zipf_alpha);
        weight_sum += weights[i];
    }
    // Base packet counts: round-robin remainder distribution to hit epoch_size.
    std::vector<int> base_counts(num_flows);
    {
        int allocated = 0;
        for (int i = 0; i < num_flows; ++i) {
            base_counts[i] = static_cast<int>(weights[i] / weight_sum * epoch_size);
            allocated += base_counts[i];
        }
        // Distribute leftover to the heaviest flows.
        int leftover = epoch_size - allocated;
        for (int i = 0; i < leftover; ++i) base_counts[i]++;
    }

    // --- Build anomaly lookup by flow_id ---
    std::unordered_map<std::string, const AnomalySpec*> anomaly_map;
    for (const auto& a : anomaly_specs)
        anomaly_map[a.flow_id] = &a;

    // --- Compute ground truth ---
    // A boundary b (epoch b → b+1) is in GT for flow f if the distribution or
    // packet count changes between epochs b and b+1 for flow f.
    GeneratedStream result;
    result.num_epochs     = num_epochs;
    result.epoch_duration = static_cast<double>(epoch_size);
    result.flow_keys      = flow_keys;

    for (int f = 0; f < num_flows; ++f) {
        const std::string& key  = flow_keys[f];
        auto it = anomaly_map.find(key);
        const AnomalySpec* anom = (it != anomaly_map.end()) ? it->second : nullptr;

        for (int b = 0; b < num_epochs - 1; ++b) {
            auto [mu0, s0, n0] = flow_params_for_epoch(base_mu, base_sigma,
                                                        base_counts[f], anom, b);
            auto [mu1, s1, n1] = flow_params_for_epoch(base_mu, base_sigma,
                                                        base_counts[f], anom, b + 1);
            bool changed = (std::abs(mu0 - mu1) > 1e-9)
                        || (std::abs(s0  - s1)  > 1e-9)
                        || (n0 == 0) != (n1 == 0)
                        || ((n0 > 0) && (n1 > 0) && std::abs(double(n0) - n1) / n0 > 0.01);
            if (changed)
                result.ground_truth.push_back({key, b});
        }
    }

    // --- Generate packets ---
    std::mt19937 rng(seed);
    double epoch_duration = static_cast<double>(epoch_size);

    for (int e = 0; e < num_epochs; ++e) {
        double epoch_start = e * epoch_duration;

        // Collect all packets for this epoch, then shuffle for realism.
        std::vector<std::pair<std::string, double>> epoch_pkts;
        epoch_pkts.reserve(epoch_size);

        for (int f = 0; f < num_flows; ++f) {
            const std::string& key = flow_keys[f];
            auto it = anomaly_map.find(key);
            const AnomalySpec* anom = (it != anomaly_map.end()) ? it->second : nullptr;

            auto [mu, sigma, n] = flow_params_for_epoch(base_mu, base_sigma,
                                                         base_counts[f], anom, e);
            if (n <= 0) continue;

            std::lognormal_distribution<double> dist(mu, sigma);
            for (int i = 0; i < n; ++i)
                epoch_pkts.push_back({key, dist(rng)});
        }

        // Shuffle within epoch so keys are interleaved.
        std::shuffle(epoch_pkts.begin(), epoch_pkts.end(), rng);

        // Assign timestamps within [epoch_start, epoch_start + epoch_size).
        for (int i = 0; i < static_cast<int>(epoch_pkts.size()); ++i) {
            result.packets.push_back({
                epoch_start + i,
                epoch_pkts[i].first,
                epoch_pkts[i].second
            });
        }
    }

    return result;
}
