#pragma once

#ifdef ARDUINO
#include <Arduino.h>
#include <FS.h>
#include <Print.h>
#ifdef ESP32
#  include <tflm_esp32.h>
#endif
#else
// Host/test builds: the same tflite:: types straight from a vendored copy
// of the real TFLite Micro source (test/vendor/, see test/main.cpp's build
// command) instead of tflm_esp32's precompiled-for-Xtensa binary -- lets
// the interpreter-building/allocation logic below (buildInterpreter() et
// al.) be compiled and tested here too, not just on real ESP32-S3 hardware.
#include <cstdlib>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#endif
