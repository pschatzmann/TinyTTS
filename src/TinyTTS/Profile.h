#pragma once
#include <cstdint>

#ifdef ARDUINO
#include <Arduino.h>
#else
#include <chrono>
#endif

namespace tinytts {

/**
 * @brief Optional per-op time totals, for finding where synthesis time goes
 * on slow targets. Define TINYTTS_PROFILE before including TinyTTS.h:
 * TinyTTSCore::synthesize() then resets the totals at the start and prints
 * them (via timingLog(), like the per-stage lines) at the end. Without the
 * define, TINYTTS_PROFILE_SCOPE() compiles to nothing.
 *
 * Totals are inclusive: a scope nested inside another (e.g. `engine`
 * inside `conv1d int8 accel`, which is itself inside `conv1d`) counts in
 * both. Timing uses micros() on Arduino, so each scope adds a few
 * microseconds of its own -- noticeable only for `engine`, which is timed
 * once per accelerator call.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
namespace profile {

enum Slot {
  kG2P,
  kDictionaryModel,
  kLinear,
  kLinearAccel,
  kConv1d,
  kConv1dInt8Accel,
  kConv1dFloatAccel,
  kConv1dQuantize,
  kConvTranspose1d,
  kConvTranspose1dAccel,
  kEngine,
  kAttentionScores,
  kElementwise,
  kSlotCount
};

inline const char* slotName(int slot) {
  static const char* const kNames[kSlotCount] = {
      "g2p",
      "  dictionary model (neural G2P)",
      "linear",
      "  linear accel",
      "conv1d",
      "  conv1d int8 accel",
      "  conv1d float accel",
      "  conv1d int8 quantize",
      "convTranspose1d",
      "  convTranspose1d accel",
      "engine compute() (in the accel ops)",
      "attention scores/values",
      "elementwise (relu/leaky/add/norm/softmax)",
  };
  return kNames[slot];
}

struct Total {
  uint64_t us = 0;
  uint32_t calls = 0;
};

inline Total* totals() {
  static Total t[kSlotCount];
  return t;
}

inline uint32_t nowMicros() {
#ifdef ARDUINO
  return micros();
#else
  using namespace std::chrono;
  return (uint32_t)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
#endif
}

inline void reset() {
  for (int i = 0; i < kSlotCount; i++) totals()[i] = Total();
}

class Scope {
 public:
  explicit Scope(Slot slot) : slot_(slot), start_(nowMicros()) {
#if defined(TINYTTS_PROFILE_TRACE) && defined(ARDUINO)
    if (slot_ != kEngine && slot_ != kElementwise) {
      Serial.print("> ");
      Serial.println(slotName(slot_));
    }
#endif
  }
  ~Scope() {
#if defined(TINYTTS_PROFILE_TRACE) && defined(ARDUINO)
    if (slot_ != kEngine && slot_ != kElementwise) {
      Serial.print("< ");
      Serial.print(slotName(slot_));
      Serial.print(" ");
      Serial.println((unsigned long)(nowMicros() - start_));
    }
#endif
    Total& t = totals()[slot_];
    t.us += (uint32_t)(nowMicros() - start_);  // wrap-safe for scopes < 71 min
    t.calls++;
  }

 private:
  Slot slot_;
  uint32_t start_;
};

}  // namespace profile
}  // namespace tinytts

#ifdef TINYTTS_PROFILE
#define TINYTTS_PROFILE_CONCAT2(a, b) a##b
#define TINYTTS_PROFILE_CONCAT(a, b) TINYTTS_PROFILE_CONCAT2(a, b)
#define TINYTTS_PROFILE_SCOPE(slot) \
  ::tinytts::profile::Scope TINYTTS_PROFILE_CONCAT(tinytts_profile_scope_, __LINE__)(::tinytts::profile::slot)
#else
#define TINYTTS_PROFILE_SCOPE(slot) ((void)0)
#endif
