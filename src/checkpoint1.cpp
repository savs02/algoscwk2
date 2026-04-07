// Checkpoint 1 — Basic sketch accuracy on a Zipf frequency stream.
//
// Inserts 10 000 items whose keys are drawn from a Zipf distribution
// (value = 1 each, pure frequency counting — no histograms yet).
// Queries every unique key against ground truth and prints:
//   avg signed error, avg absolute error, max absolute error.
//
// Expected behaviour
// ------------------
//   CMS    : always overestimates  → positive avg signed error
//   CU-CMS : also overestimates, but less than CMS
//   CS     : unbiased              → avg signed error ≈ 0, errors both ways

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "sketches/cms.hpp"
#include "sketches/cu_cms.hpp"
#include "sketches/cs.hpp"

// ---------------------------------------------------------------------------
// Zipf distribution sampler
//
// Precomputes a discrete_distribution over keys 1..n_max weighted by k^-alpha.
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
    int sample(std::mt19937& rng) { return dist_(rng) + 1; } // 1-indexed
};

// ---------------------------------------------------------------------------

int main() {
    constexpr int    W          = 1024;
    constexpr int    D          = 3;
    constexpr int    N_ITEMS    = 10'000;
    constexpr int    N_MAX_KEY  = 10'000; // max possible key value
    constexpr double ZIPF_ALPHA = 1.5;
    constexpr uint32_t SEED     = 42;

    std::mt19937 rng(SEED);
    ZipfDistribution zipf(N_MAX_KEY, ZIPF_ALPHA);

    // Generate keys and build ground truth.
    std::map<std::string, int> ground_truth;
    std::vector<std::string>   stream;
    stream.reserve(N_ITEMS);

    for (int i = 0; i < N_ITEMS; ++i) {
        std::string key = std::to_string(zipf.sample(rng));
        stream.push_back(key);
        ground_truth[key]++;
    }

    // Initialise sketches.
    CountMinSketch        cms(W, D);
    ConservativeUpdateCMS cu_cms(W, D);
    CountSketch           cs(W, D);

    // Insert all items (value = 1 per item, frequency counting only).
    for (const auto& key : stream) {
        cms.update(key, 1);
        cu_cms.update(key, 1);
        cs.update(key, 1);
    }

    // Evaluate: query every unique key.
    std::vector<double> cms_err, cu_err, cs_err;
    cms_err.reserve(ground_truth.size());
    cu_err.reserve(ground_truth.size());
    cs_err.reserve(ground_truth.size());

    for (const auto& [key, true_val] : ground_truth) {
        cms_err.push_back(cms.query(key)    - true_val);
        cu_err.push_back(cu_cms.query(key)  - true_val);
        cs_err.push_back(cs.query(key)      - true_val);
    }

    auto avg = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto avg_abs = [](const std::vector<double>& v) {
        double s = 0;
        for (double x : v) s += std::abs(x);
        return s / v.size();
    };
    auto max_abs = [](const std::vector<double>& v) {
        double m = 0;
        for (double x : v) m = std::max(m, std::abs(x));
        return m;
    };

    std::cout << "Checkpoint 1  |  " << N_ITEMS << " items"
              << ", Zipf alpha=" << ZIPF_ALPHA
              << ", w=" << W << ", d=" << D << "\n";
    std::cout << std::left
              << std::setw(12) << "Sketch"
              << std::right
              << std::setw(18) << "avg signed err"
              << std::setw(15) << "avg abs err"
              << std::setw(15) << "max abs err"
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (auto& [name, err] : std::vector<std::pair<std::string, std::vector<double>>>{
            {"CMS",    cms_err},
            {"CU-CMS", cu_err},
            {"CS",     cs_err}}) {
        std::cout << std::left  << std::setw(12) << name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(18) << avg(err)
                  << std::setw(15) << avg_abs(err)
                  << std::setw(15) << max_abs(err)
                  << "\n";
    }

    std::cout << "\nExpected: CMS avg signed err > 0 (overestimates)\n"
              << "          CU-CMS avg signed err > 0 but smaller than CMS\n"
              << "          CS avg signed err ~ 0 (errors in both directions)\n";
    return 0;
}
