/**
 * Timing-only benchmark: no I2S/codec, no audio hardware needed at all --
 * just runs speak() with a no-op AudioChunkFn sink and reports the
 * built-in per-stage timing ([TinyTTS] encoder/duration_predictor/flow/
 * vocoder lines, always-on via TinyTTSCore.h's timingLog()) over Serial.
 * Matches the same benchmark methodology docs/performance.md used for the
 * ESP32-P4 measurement -- a real board, no display/audio peripheral
 * dependency, just synthesis timing.
 *
 * Board settings: PSRAM=opi, FlashSize=16M, PartitionScheme=custom (this
 * sketch's own partitions.csv), USBMode=hwcdc, CDCOnBoot=cdc.
 */
#include "TinyTTS.h"
#include "TinyTTS/data/default_weights_data.h"
#include "TinyTTS/data/default_cmudict_slim_data.h"
#include "TinyTTS/data/default_dictionary_model_data.h"

tinytts::TinyTTS tts;

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  delay(2000);

  Serial.printf("PSRAM: %u bytes free of %u\n", ESP.getFreePsram(), ESP.getPsramSize());

  tts.setWeights(default_weights, default_weights_len);
  tts.setDictionary(default_cmudict_slim, default_cmudict_slim_len);
  tts.setDictionaryModel(default_dictionary_model, default_dictionary_model_len);
#ifdef TINYTTS_BENCH_INT8
  tts.setDecoderPrecision(tinytts::ops::DecoderPrecision::kInt8Activations);
  Serial.println("Decoder precision: INT8 activations (fused, per-window quant)");
#else
  Serial.println("Decoder precision: float32 (default)");
#endif
  if (!tts.begin([](const float*, size_t) { /* discard: timing only */ })) {
    Serial.println("TinyTTS.begin() failed -- did you call all setters above?");
    while (true) {}
  }

  Serial.printf("PSRAM after begin(): %u bytes free of %u\n", ESP.getFreePsram(), ESP.getPsramSize());
  Serial.println("Ready.");
}

void loop() {
  Serial.println("=== speak(\"Hello world!\") ===");
  tts.speak("Hello world!");
  delay(2000);  // matches tts_i2s_output's own pattern -- avoids a task-watchdog reset on repeated back-to-back calls
}
