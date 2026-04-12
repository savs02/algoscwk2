#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>


"""
Change type classification from a differenced histogram.

Five types (plus None when no significant change is detected):

  Disappearance  — all bins are negative (flow stopped)
  VolumeChange   — bins all same sign and roughly proportional
  Spike          — one or two adjacent bins dominate (large positive), rest near zero
  Shift          — low bins negative AND high bins positive (or vice versa)
  Spread         — centre bins negative, tail bins positive (distribution widened)
"""


enum class ChangeType {
    None,
    Disappearance,
    VolumeChange,
    Spike,
    Shift,
    Spread,
};

inline const char* change_type_name(ChangeType t) {
    switch (t) {
        case ChangeType::None:         return "None";
        case ChangeType::Disappearance:return "Disappearance";
        case ChangeType::VolumeChange: return "VolumeChange";
        case ChangeType::Spike:        return "Spike";
        case ChangeType::Shift:        return "Shift";
        case ChangeType::Spread:       return "Spread";
    }
    return "Unknown";
}

inline ChangeType classify_change(const std::vector<double>& diff,
                                  double noise_floor = 5.0)
{
    const int B = static_cast<int>(diff.size());
    assert(B >= 2 && "need at least 2 bins to classify");

    double total_abs = 0.0;
    double max_abs   = 0.0;
    for (double d : diff) {
        total_abs += std::abs(d);
        max_abs    = std::max(max_abs, std::abs(d));
    }

    if (total_abs < 3.0 * noise_floor * B)
        return ChangeType::None;

    int n_pos = 0, n_neg = 0;
    double pos_mass = 0.0, neg_mass = 0.0;
    for (double d : diff) {
        if (d >  noise_floor) {
            ++n_pos;
            pos_mass += d;
        }
        if (d < -noise_floor) {
            ++n_neg;
            neg_mass += -d;
        }
    }

    if (n_pos == 0 && n_neg > 0)
        return ChangeType::Disappearance;

    {
        int same_sign_bins = std::max(n_pos, n_neg);
        int opposite_bins  = std::min(n_pos, n_neg);
        bool broad_support = same_sign_bins >= std::max(3, B / 3);
        bool same_sign     = opposite_bins <= 1 && broad_support;
        bool not_spiky     = (total_abs > 0.0) && (max_abs / total_abs <= 0.45);

        if (same_sign && not_spiky)
            return ChangeType::VolumeChange;
    }

    {
        int peak_bin = static_cast<int>(
            std::max_element(diff.begin(), diff.end(),
                [](double a, double b){ return std::abs(a) < std::abs(b); })
            - diff.begin());

        double spike_mass = std::abs(diff[peak_bin]);
        if (peak_bin > 0)     spike_mass += std::max(0.0, diff[peak_bin - 1]);
        if (peak_bin < B - 1) spike_mass += std::max(0.0, diff[peak_bin + 1]);

        if (pos_mass > 0.0
            && spike_mass / pos_mass >= 0.70
            && std::abs(diff[peak_bin]) / total_abs >= 0.35)
        {
            return ChangeType::Spike;
        }
    }

    {
        std::vector<int> sign_seq(B, 0);
        for (int b = 0; b < B; ++b)
            sign_seq[b] = (diff[b] > noise_floor) ? 1 : (diff[b] < -noise_floor) ? -1 : 0;

        int flips = 0;
        int last_nonzero = 0;
        bool seen_nonzero = false;
        for (int b = 0; b < B; ++b) {
            if (sign_seq[b] == 0) continue;
            if (seen_nonzero && sign_seq[b] != last_nonzero) ++flips;
            last_nonzero = sign_seq[b];
            seen_nonzero = true;
        }

        if (flips == 1 && n_pos >= 1 && n_neg >= 1)
            return ChangeType::Shift;
    }

    {
        int tail = std::max(1, B / 4);   
        double tail_sum   = 0.0;
        double centre_sum = 0.0;
        for (int b = 0; b < B; ++b) {
            if (b < tail || b >= B - tail)
                tail_sum   += diff[b];
            else
                centre_sum += diff[b];
        }
        if (tail_sum > noise_floor && centre_sum < -noise_floor)
            return ChangeType::Spread;
    }

    if (n_pos > 0 && n_neg > 0)
        return ChangeType::Shift;

    return ChangeType::None;
}
