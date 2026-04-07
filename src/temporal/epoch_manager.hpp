#pragma once
#include <cassert>
#include <functional>
#include <memory>
#include <vector>
#include "../sketches/base_sketch.hpp"

// ---------------------------------------------------------------------------
// EpochManager
//
// Maintains a ring buffer of K histogram-sketch snapshots.  Each slot holds
// the data from one time epoch.  When advance_epoch() is called, the next
// slot is claimed and filled with a fresh sketch (the factory function is
// called to construct it), overwriting the oldest epoch once the buffer is
// full.
//
// Slot layout after N advances (N < K):
//   current_slot_          → epoch N  (most recent, being written)
//   (current_slot_-1) % K  → epoch N-1
//   ...
//   (current_slot_-N) % K  → epoch 0  (oldest still in buffer)
//
// Once N ≥ K the oldest epoch is silently overwritten — this is intentional
// ring-buffer behaviour.
//
// Usage
// -----
//   auto factory = [&]{ return std::make_unique<CountMinSketch>(w, d, cfg); };
//   EpochManager em(4, factory);
//   em.get_current_sketch().update(key, value); // fill epoch 0
//   em.advance_epoch();                          // move to epoch 1
//   em.get_previous_sketch(1).query_histogram(key); // read epoch 0
// ---------------------------------------------------------------------------

class EpochManager {
public:
    using SketchPtr     = std::unique_ptr<BaseSketch>;
    using SketchFactory = std::function<SketchPtr()>;

private:
    int                    K_;
    std::vector<SketchPtr> slots_;
    int                    current_slot_;
    int                    epochs_elapsed_;  // total advance_epoch() calls
    int                    snapshots_available_;
    SketchFactory          factory_;

public:
    EpochManager(int num_snapshots, SketchFactory factory)
        : K_(num_snapshots),
          slots_(num_snapshots),
          current_slot_(0),
          epochs_elapsed_(0),
          snapshots_available_(1),
          factory_(std::move(factory))
    {
        assert(num_snapshots >= 2 && "need at least 2 snapshots to compare epochs");
        for (auto& s : slots_) s = factory_();
    }

    // Non-copyable: owns unique_ptrs and holds a factory closure.
    EpochManager(const EpochManager&)            = delete;
    EpochManager& operator=(const EpochManager&) = delete;

    // Move to the next ring-buffer slot and install a fresh empty sketch there.
    void advance_epoch() {
        current_slot_ = (current_slot_ + 1) % K_;
        slots_[current_slot_] = factory_();
        ++epochs_elapsed_;
        if (snapshots_available_ < K_) ++snapshots_available_;
    }

    // The sketch for the active (current) epoch — call update() on this.
    BaseSketch& get_current_sketch() {
        return *slots_[current_slot_];
    }

    const BaseSketch& get_current_sketch() const {
        return *slots_[current_slot_];
    }

    // The sketch from `offset` epochs ago.
    //   offset = 0 → current epoch (same as get_current_sketch, read-only)
    //   offset = 1 → previous epoch
    //   offset ∈ [0, snapshots_available()-1]
    const BaseSketch& get_previous_sketch(int offset) const {
        assert(offset >= 0 && offset < snapshots_available_
               && "offset refers to an epoch not yet available");
        int slot = (current_slot_ - offset + K_) % K_;
        return *slots_[slot];
    }

    int num_snapshots()  const { return K_; }
    int epochs_elapsed() const { return epochs_elapsed_; }
    int snapshots_available() const { return snapshots_available_; }
    int current_slot()   const { return current_slot_; }
};
