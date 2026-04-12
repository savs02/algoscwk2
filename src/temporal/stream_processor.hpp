#pragma once
#include <cassert>
#include <limits>
#include <string>
#include "epoch_manager.hpp"

"""
StreamProcessor

Feeds a stream of packets into an EpochManager, advancing the epoch
automatically when a boundary is reached.  Two epoch-boundary modes are
supported; choose one at construction time and use the matching overload
of process().  Mixing the two overloads on the same instance is undefined.

Item-count mode (default)
  StreamProcessor proc(epoch_mgr, epoch_duration);
  proc.process(key, value);

  Advances after every `epoch_duration` items.  Advance is lazy
  (fires on the FIRST call of the new epoch) so a completed epoch's sketch
  is stable and queryable between epochs.

Timestamp mode 
  StreamProcessor proc(epoch_mgr, epoch_duration_seconds);
  proc.process(timestamp, key, value);

  Advances when `timestamp >= epoch_start + epoch_duration`.  The epoch
  start is anchored to the first timestamp seen.  Consecutive advances
  are possible if a large gap exists between packets.
  """


class StreamProcessor {
public:
    enum class EpochMode { ItemCount, Time };

private:
    EpochManager& epoch_mgr_;
    EpochMode     mode_;
    double        epoch_duration_;  
    int           items_in_epoch_;
    int           items_processed_;
    double        epoch_start_time_; 

public:
    StreamProcessor(EpochManager& epoch_mgr, int epoch_duration)
        : epoch_mgr_(epoch_mgr),
          mode_(EpochMode::ItemCount),
          epoch_duration_(static_cast<double>(epoch_duration)),
          items_in_epoch_(0),
          items_processed_(0),
          epoch_start_time_(std::numeric_limits<double>::quiet_NaN())
    {
        assert(epoch_duration > 0 && "epoch_duration must be positive");
    }

    StreamProcessor(EpochManager& epoch_mgr, double epoch_duration_seconds)
        : epoch_mgr_(epoch_mgr),
          mode_(EpochMode::Time),
          epoch_duration_(epoch_duration_seconds),
          items_in_epoch_(0),
          items_processed_(0),
          epoch_start_time_(std::numeric_limits<double>::quiet_NaN())
    {
        assert(epoch_duration_seconds > 0.0 && "epoch_duration must be positive");
    }

  
    void process(const std::string& key, double value) {
        assert(mode_ == EpochMode::ItemCount
               && "use process(timestamp, key, value) in timestamp mode");
        if (items_in_epoch_ >= static_cast<int>(epoch_duration_)) {
            epoch_mgr_.advance_epoch();
            items_in_epoch_ = 0;
        }
        epoch_mgr_.get_current_sketch().update(key, value);
        ++items_in_epoch_;
        ++items_processed_;
    }

    void process(double timestamp, const std::string& key, double value) {
        assert(mode_ == EpochMode::Time
               && "use process(key, value) in item-count mode");
        assert(timestamp >= 0.0 && "timestamps must be non-negative");

        if (items_processed_ == 0)
            epoch_start_time_ = timestamp;

        assert(timestamp >= epoch_start_time_ && "timestamps must be non-decreasing");

        while (timestamp >= epoch_start_time_ + epoch_duration_) {
            epoch_mgr_.advance_epoch();
            epoch_start_time_ += epoch_duration_;
            items_in_epoch_ = 0;
        }

        epoch_mgr_.get_current_sketch().update(key, value);
        ++items_in_epoch_;
        ++items_processed_;
    }

    int    items_in_epoch()  const { return items_in_epoch_; }
    double epoch_duration()  const { return epoch_duration_; }
    int    items_processed() const { return items_processed_; }
    EpochMode mode() const { return mode_; }
    double epoch_start_time() const { return epoch_start_time_; }
};
