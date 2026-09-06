# Status / known limitations

- **Out-of-dictionary words fall back to crude character-level phonemes unless
  `setDictionaryModel()` is called.** The neural GRU fallback (`DictionaryModel`/
  `default_dictionary_model`) gives much better results for proper nouns and unusual words,
  but costs ~0.95MB more flash, so it's opt-in rather than always-on. The CMU dictionary
  covers the large majority of real English words either way.
- **Splitting text across multiple `speak()` calls must happen on word boundaries, never
  mid-word** -- see `docs/text-input.md` for what input shapes are safe (individual words,
  full sentences, multiple sentences, and multi-call word-boundary splits all work; a
  multi-call split that cuts a word in half mispronounces it).
- **This project tests and tunes for one board** (the `partitions.csv` shipped alongside
  the example, the board settings in `docs/requirements.md`, etc.), even though none of the
  model code itself is chip-specific. `DataBuffer`'s `File`-loading path uses ESP32's own
  PSRAM allocator directly, `library.properties` scopes the library to the `esp32`
  architecture family, and the model's multi-megabyte weight/dictionary data needs a module
  with real PSRAM -- so this isn't a "runs on any microcontroller" library, just one with no
  inference-runtime dependency within that family.
