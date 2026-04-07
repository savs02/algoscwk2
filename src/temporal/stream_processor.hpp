#pragma once
#include <cassert>
#include <string>
#include "epoch_manager.hpp"

// ---------------------------------------------------------------------------
// StreamProcessor
//
// Feeds a stream of (key, value) packets into an EpochManager, advancing
// the epoch automatically based on a fixed item count (epoch_duration_).
//
// Advance policy: lazy / start-of-epoch.
//   The epoch rolls over on the FIRST call after the current epoch has
//   collected epoch_duration_ items, not on the last call of the epoch.
//   This guarantees that when you query get_current_sketch() at any point
//   after a full epoch completes, that sketch contains exactly the data
//   from that epoch and nothing more.
//
//   Timeline (epoch_duration=3):
//     process calls:  1  2  3  | 4  5  6  | 7  8  9  | ...
//     active epoch:   0  0  0  | 1  1  1  | 2  2  2  | ...
//     advance at:               ^           ^
//   (advance fires on call 4 before inserting that call's packet)
// ---------------------------------------------------------------------------

class StreamProcessor {
    EpochManager& epoch_mgr_;
    int           epoch_duration_;
    int           items_in_epoch_;
    int           items_processed_;

public:
    StreamProcessor(EpochManager& epoch_mgr, int epoch_duration)
        : epoch_mgr_(epoch_mgr),
          epoch_duration_(epoch_duration),
          items_in_epoch_(0),
          items_processed_(0)
    {
        assert(epoch_duration > 0 && "epoch_duration must be positive");
    }

    void process(const std::string& key, double value) {
        // Advance before inserting if the current epoch is full.
        if (items_in_epoch_ >= epoch_duration_) {
            epoch_mgr_.advance_epoch();
            items_in_epoch_ = 0;
        }
        epoch_mgr_.get_current_sketch().update(key, value);
        ++items_in_epoch_;
        ++items_processed_;
    }

    int items_in_epoch() const { return items_in_epoch_; }
    int epoch_duration() const { return epoch_duration_; }
    int items_processed() const { return items_processed_; }
};
