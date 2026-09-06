#pragma once
#include <functional>
#include <memory>
#include <vector>

// Optional: splits conv1d()/convTranspose1d()'s tile loop across N
// participants -- the calling thread plus N-1 persistent worker tasks --
// via arduino-audio-tools's portable Task abstraction
// (https://github.com/pschatzmann/arduino-audio-tools/wiki/Multicore-Processing)
// -- one implementation instead of hand-rolling FreeRTOS/std::thread
// separately. Included directly (not the Concurrency.h/AudioTools.h
// umbrella, which additionally drags in QueueRTOS.h/BufferRTOS.h/
// MutexRTOS.h/SynchronizedNBufferRTOS.h on ESP32 -- none of which this
// needs) so the dependency surface stays small. __has_include-gated, no
// library.properties dependency, same non-hard-requirement pattern as
// Ops.h's TINYTTS_HAVE_ESP_DSP: a build without arduino-audio-tools
// installed (or a desktop build that hasn't defined IS_MIN_DESKTOP) still
// compiles and runs correctly, just single-core.
//
// N beyond 2 only really means something on desktop (real std::thread
// cores beyond the calling one); ESP32 has exactly 2 physical cores, so
// asking for more there just adds FreeRTOS tasks time-sliced on the same
// two cores -- not real parallelism, just overhead. Nothing here stops a
// caller from doing that, but the library default (see ops::numWorkers()
// in Ops.h) stays 2 for that reason.
#if defined(ESP32) && __has_include(<AudioTools/Concurrency/RTOS/Task.h>)
#include <AudioTools/Concurrency/RTOS/Task.h>
#define TINYTTS_HAVE_TASK 1
#elif defined(TINYTTS_DESKTOP_TASK) && __has_include(<AudioTools/Concurrency/Desktop/Task.h>)
// TINYTTS_DESKTOP_TASK must be defined by the consumer (a build-time flag,
// e.g. -DTINYTTS_DESKTOP_TASK), together with IS_MIN_DESKTOP -- pulling in
// AudioToolsConfig.h here (which in turn includes
// AudioTools/PlatformConfig/desktop.h) is what actually defines USE_CPP_TASK
// (needed for Desktop/Task.h's body to compile at all -- it's an empty file
// otherwise) and provides delay() as a free function on desktop (Arduino
// has no delay() of its own; the Emulation/Time.h shim does), which
// TileSplitter's job-handoff wait below depends on.
#include <AudioToolsConfig.h>
#include <AudioTools/Concurrency/Desktop/Task.h>
#define TINYTTS_HAVE_TASK 1
#endif

#ifdef TINYTTS_HAVE_TASK
#include <atomic>
#endif

namespace tinytts {

/**
 * @brief Splits a [0, num_tiles) tile-index range across the calling thread
 * and up to `num_workers - 1` persistent worker tasks, then joins. Falls
 * back to a single, in-place call covering the whole range when
 * TINYTTS_HAVE_TASK isn't defined (or num_workers <= 1), so callers don't
 * need their own #ifdef at the call site.
 *
 * `fn(tile0, tile1, participant)` must only touch state private to
 * `participant` (0..N-1, N = however many participants this call actually
 * used -- see run()) -- conv1d()'s per-tile weight-decode scratch buffer
 * (`wtile`) is exactly the kind of state that must NOT be shared between
 * participants: give each one its own local copy. See Ops.h's
 * conv1d()/convTranspose1d() for the two real call sites and why
 * convTranspose1d() additionally needs each participant to accumulate into
 * its own private output buffer, indexed by `participant` (its inner loop
 * scatters read-modify-write additions into the shared output, unlike
 * conv1d()'s disjoint per-output-channel writes) rather than writing into a
 * shared `y` directly.
 *
 * One instance is meant to be reused for the process/device lifetime (e.g.
 * a function-local `static TileSplitter`, same pattern as Ops.h's existing
 * `static InternalVector<float> wtile`) -- creating a FreeRTOS task (or
 * std::thread) per call would be real, avoidable overhead given how many
 * tiles run per synthesis. Worker tasks are created lazily on first use and
 * persist thereafter -- num_workers can't be changed after that first call
 * (see ops::setNumWorkers()'s doc in Ops.h for the practical implication:
 * set it before the first synthesize() call, not mid-stream).
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class TileSplitter {
 public:
  // num_workers: total participants including the calling thread (2 =
  // today's original behavior: one extra worker). stackSizeWords/priority
  // match audio_tools::Task's constructor -- see that class's own doc for
  // what each means on ESP32 vs desktop (stackSizeWords is words, not
  // bytes). The first extra worker is pinned to whichever of ESP32's two
  // real cores the CALLING thread is NOT already on -- determined at
  // runtime via xPortGetCoreID(), not hardcoded to core 1. An earlier
  // version of this file hardcoded core 1 "assuming the calling thread
  // runs on core 0 as Arduino sketches normally do" -- that assumption was
  // simply wrong and went unnoticed until real-hardware measurement showed
  // zero speedup from 2 workers (see docs/performance.md): Arduino-ESP32's
  // own core/main.cpp pins the sketch's main task (loopTask, where
  // setup()/loop()/every speak() call actually runs) to
  // ARDUINO_RUNNING_CORE, which defaults to 1, not 0 -- so the "worker on
  // core 1" was landing on the exact same physical core as the caller the
  // whole time, just adding task-switch overhead with no real parallelism.
  // Any further workers (num_workers > 2) are left unpinned (-1,
  // scheduler's choice) since ESP32 has no third core to pin to --
  // meaningful parallelism beyond 2 is a desktop-only case anyway.
  explicit TileSplitter(int num_workers = 2, int stackSizeWords = 4096, int priority = 1) {
#ifdef TINYTTS_HAVE_TASK
    int extra = num_workers > 1 ? num_workers - 1 : 0;
    workers_.reserve(extra);
#if defined(ESP32)
    int other_core = 1 - xPortGetCoreID();
#else
    int other_core = 1;  // desktop: std::thread-backed Task, no real core-affinity API used here
#endif
    for (int i = 0; i < extra; i++) workers_.push_back(std::unique_ptr<WorkerSlot>(
        new WorkerSlot(stackSizeWords, priority, i == 0 ? other_core : -1)));
#else
    (void)num_workers;
    (void)stackSizeWords;
    (void)priority;
#endif
  }

  /// Total participants this instance can use (calling thread + workers) --
  /// what was requested at construction, not adjusted for a given
  /// `num_tiles` (run() itself never uses more participants than there are
  /// tiles to hand out).
  int numWorkers() const {
#ifdef TINYTTS_HAVE_TASK
    return (int)workers_.size() + 1;
#else
    return 1;
#endif
  }

  /// How many participants a run(num_tiles, ...) call would actually use --
  /// min(numWorkers(), num_tiles), never more, since a participant with no
  /// tiles to work on is pointless. Callers that need to size a
  /// per-participant resource up front (e.g. convTranspose1d()'s private
  /// accumulators, one per participant) should use this rather than
  /// numWorkers() directly, so their sizing can never disagree with what
  /// run() itself actually does.
  int participantsFor(int num_tiles) const {
    int n = numWorkers();
    return (n > 1 && num_tiles < n) ? num_tiles : n;
  }

  /// Splits [0, num_tiles) into up to participantsFor(num_tiles) contiguous
  /// chunks (as evenly as possible; earlier chunks absorb any remainder),
  /// runs each worker's chunk on its own task and the last chunk on the
  /// calling thread, then blocks until all workers finish. With no
  /// available Task (or participantsFor(num_tiles) == 1), just calls
  /// fn(0, num_tiles, 0) inline on the calling thread.
  void run(int num_tiles, const std::function<void(int tile0, int tile1, int participant)>& fn) {
#ifdef TINYTTS_HAVE_TASK
    int n = participantsFor(num_tiles);
    if (n > 1) {
      int base = num_tiles / n, rem = num_tiles % n;
      int pos = 0;
      for (int w = 0; w < n - 1; w++) {
        int len = base + (w < rem ? 1 : 0);
        WorkerSlot& slot = *workers_[w];
        if (!slot.started) {
          slot.task.begin([&slot]() { workerLoop(slot); });
          slot.started = true;
        }
        slot.fn = &fn;
        slot.tile0 = pos;
        slot.tile1 = pos + len;
        slot.participant = w;
        slot.job_done.store(false, std::memory_order_relaxed);
        slot.job_ready.store(true, std::memory_order_release);
        pos += len;
      }
      int my_len = num_tiles - pos;  // last chunk absorbs whatever's left
      fn(pos, pos + my_len, n - 1);

      for (int w = 0; w < n - 1; w++) {
        // Lock-free, but yielding: a bare spin here can starve this core's
        // FreeRTOS idle task long enough to trip the per-core Task
        // Watchdog on ESP32 (looks like an unrelated intermittent crash)
        // -- delay(1) yields each iteration instead. Not a mutex/SpinLock:
        // this is a one-shot job handoff, not mutual exclusion over
        // shared state.
        while (!workers_[w]->job_done.load(std::memory_order_acquire)) delay(1);
      }
      return;
    }
#endif
    fn(0, num_tiles, 0);
  }

 private:
#ifdef TINYTTS_HAVE_TASK
  struct WorkerSlot {
    audio_tools::Task task;
    std::atomic<bool> job_ready{false};
    std::atomic<bool> job_done{false};
    const std::function<void(int, int, int)>* fn = nullptr;
    int tile0 = 0;
    int tile1 = 0;
    int participant = 0;
    bool started = false;

    WorkerSlot(int stackSizeWords, int priority, int core) : task("tinytts_tile", stackSizeWords, priority, core) {}
  };

  static void workerLoop(WorkerSlot& slot) {
    if (!slot.job_ready.load(std::memory_order_acquire)) {
      delay(1);
      return;
    }
    slot.job_ready.store(false, std::memory_order_relaxed);
    (*slot.fn)(slot.tile0, slot.tile1, slot.participant);
    slot.job_done.store(true, std::memory_order_release);
  }

  // unique_ptr per slot: WorkerSlot holds a std::atomic, which is neither
  // copyable nor movable, so the vector itself can only ever hold stable
  // heap addresses, never relocate the slots in place.
  std::vector<std::unique_ptr<WorkerSlot>> workers_;
#endif
};

}  // namespace tinytts
