// Checkpoint 2 — Histogram-in-sketch accuracy on a Zipf + lognormal stream.
//
// Stream: 10 000 packets. Key = flow ID drawn from Zipf(alpha=1.5).
//         Latency = drawn from lognormal(mu=1.6, sigma=0.4).
//         Lognormal median ≈ exp(1.6) ≈ 5.0, values mostly in [1, 20].
//
// Bin configs tested:
//   Uniform:     8 bins, [0, 30]  → bin width = 3.75
//   Logarithmic: 8 bins, [0.5, 30]
//
// For each config: select the 3 highest-frequency keys, print ground truth,
// estimated histograms, and per-bin errors.
//
// Expected: CMS and CU-CMS overestimate per bin (non-negative errors).
//           CS errors are signed (both over and under per bin).
//           CU-CMS errors smaller than CMS.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "sketches/bin_config.hpp"
#include "sketches/cms.hpp"
#include "sketches/cu_cms.hpp"
#include "sketches/cs.hpp"

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

// w=128 intentionally: with w=1024 per-bin collision noise rounds to 0 at
// this stream size, hiding the over/under-estimation signal we want to verify.
// The full evaluation sweeps w; 128 is just enough to make errors visible here.
static void run_checkpoint(const BinConfig& cfg, const std::string& scheme_name,
                           const std::vector<std::pair<std::string, double>>& stream,
                           const std::map<std::string, std::vector<int>>& ground_truth,
                           const std::vector<std::string>& top_keys)
{
    int W = 128, D = 3;
    CountMinSketch        cms(W, D, cfg);
    ConservativeUpdateCMS cu_cms(W, D, cfg);
    CountSketch           cs(W, D, cfg);

    for (const auto& [key, latency] : stream) {
        cms.update(key, latency);
        cu_cms.update(key, latency);
        cs.update(key, latency);
    }

    int B = cfg.num_bins();
    const auto& edges = cfg.edges();

    std::cout << "\n=== " << scheme_name << " bins ===\n";

    for (const auto& key : top_keys) {
        const auto& truth     = ground_truth.at(key);
        auto        cms_hist  = cms.query_histogram(key);
        auto        cu_hist   = cu_cms.query_histogram(key);
        auto        cs_hist   = cs.query_histogram(key);

        int true_count = 0;
        for (int c : truth) true_count += c;
        std::cout << "\nFlow \"" << key << "\"  (true packet count = " << true_count << ")\n";

        // Header
        std::cout << std::setw(5)  << "Bin"
                  << std::setw(18) << "Range"
                  << std::setw(8)  << "Truth"
                  << std::setw(8)  << "CMS"
                  << std::setw(8)  << "CU"
                  << std::setw(8)  << "CS"
                  << std::setw(10) << "CMS err"
                  << std::setw(12) << "CU-CMS err"
                  << std::setw(10) << "CS err"
                  << "\n";
        std::cout << std::string(87, '-') << "\n";

        for (int b = 0; b < B; ++b) {
            // Format bin range string.
            std::ostringstream range;
            range << std::fixed << std::setprecision(1)
                  << "[" << edges[b] << ", " << edges[b + 1] << ")";

            std::cout << std::setw(5)  << b
                      << std::setw(18) << range.str()
                      << std::setw(8)  << truth[b]
                      << std::setw(8)  << std::fixed << std::setprecision(0) << cms_hist[b]
                      << std::setw(8)  << cu_hist[b]
                      << std::setw(8)  << cs_hist[b]
                      << std::setw(10) << std::showpos << (cms_hist[b]  - truth[b])
                      << std::setw(12) << (cu_hist[b]   - truth[b])
                      << std::setw(10) << (cs_hist[b]   - truth[b])
                      << std::noshowpos << "\n";
        }
    }
}

// ---------------------------------------------------------------------------

int main() {
    constexpr int      N_ITEMS    = 10'000;
    constexpr int      N_MAX_KEY  = 1000;   // fewer unique keys → more load per key → visible collision noise
    constexpr double   ZIPF_ALPHA = 1.5;
    constexpr double   LN_MU     = 1.6;   // lognormal parameters (log-space)
    constexpr double   LN_SIGMA  = 0.4;
    constexpr uint32_t SEED       = 42;

    std::mt19937 rng(SEED);
    ZipfDistribution               zipf(N_MAX_KEY, ZIPF_ALPHA);
    std::lognormal_distribution<double> latency_dist(LN_MU, LN_SIGMA);

    // Build the stream and per-key ground truth histograms under both configs.
    // We record raw (key, latency) pairs so we can replay for each bin scheme.
    std::vector<std::pair<std::string, double>> stream;
    stream.reserve(N_ITEMS);
    std::map<std::string, int> freq;

    for (int i = 0; i < N_ITEMS; ++i) {
        std::string key     = std::to_string(zipf.sample(rng));
        double      latency = latency_dist(rng);
        stream.push_back({key, latency});
        freq[key]++;
    }

    // Pick the 3 highest-frequency keys.
    std::vector<std::pair<int, std::string>> sorted_freq;
    for (const auto& [k, c] : freq)
        sorted_freq.push_back({c, k});
    std::sort(sorted_freq.rbegin(), sorted_freq.rend());

    std::vector<std::string> top_keys;
    for (int i = 0; i < std::min(3, static_cast<int>(sorted_freq.size())); ++i)
        top_keys.push_back(sorted_freq[i].second);

    std::cout << "Checkpoint 2  |  " << N_ITEMS << " items"
              << ", Zipf alpha=" << ZIPF_ALPHA
              << ", lognormal(mu=" << LN_MU << ", sigma=" << LN_SIGMA << ")\n";
    std::cout << "Top keys by frequency: ";
    for (const auto& k : top_keys)
        std::cout << "\"" << k << "\" (" << freq[k] << ")  ";
    std::cout << "\n";

    // --- Uniform bins ---
    BinConfig uniform_cfg(8, 0.0, 30.0, BinScheme::Uniform);
    {
        std::map<std::string, std::vector<int>> gt;
        for (const auto& k : top_keys) gt[k].assign(8, 0);
        for (const auto& [key, lat] : stream)
            if (gt.count(key)) gt[key][uniform_cfg.get_bin(lat)]++;
        run_checkpoint(uniform_cfg, "Uniform [0, 30]", stream, gt, top_keys);
    }

    // --- Logarithmic bins ---
    BinConfig log_cfg(8, 0.5, 30.0, BinScheme::Logarithmic);
    {
        std::map<std::string, std::vector<int>> gt;
        for (const auto& k : top_keys) gt[k].assign(8, 0);
        for (const auto& [key, lat] : stream)
            if (gt.count(key)) gt[key][log_cfg.get_bin(lat)]++;
        run_checkpoint(log_cfg, "Logarithmic [0.5, 30]", stream, gt, top_keys);
    }

    return 0;
}
