#pragma once

/**
 * @brief Storage attribute for the large generated model-data arrays in
 * TinyTTS/data/ (see research/export_headers.py): PROGMEM on Arduino, so
 * cores with a separate constant-data area put them there (e.g. the Tang
 * Nano 20K core, whose PROGMEM places data in memory-mapped SPI flash
 * instead of its 64KB SRAM), and empty elsewhere -- on ESP32/RP2040 PROGMEM
 * is itself empty, so nothing changes there. Define TINYTTS_PROGMEM before
 * including the data headers to override it.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
#ifndef TINYTTS_PROGMEM
#ifdef ARDUINO
#include <Arduino.h>
#endif
#ifdef PROGMEM
#define TINYTTS_PROGMEM PROGMEM
#else
#define TINYTTS_PROGMEM
#endif
#endif
