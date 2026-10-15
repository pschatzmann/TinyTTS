# Model data files

The binary model data TinyTTS can load at runtime. Copy these files to LittleFS/FFat/SD and pass
them to the `File` setters (`tts.setWeights(file)`, ...), or use them with the desktop CLI
(`tinytts --weights data/ljspeech16k_weights.bin ...`). Each file is byte-identical to the
payload of the compiled-in header with the same name in `src/TinyTTS/data/`. Use the file *or*
the header, whichever suits your board (see `docs/model-data.md`).

## Model weights (`setWeights`): pick one

| File | Sample rate | Voice | Size | Source |
|---|---:|---|---:|---|
| `default_weights.bin` | 44100 Hz | upstream release (male) | 2.07 MB | released `backtracking/tiny-tts` checkpoint |
| `ljspeech16k_weights.bin` | 16000 Hz | LJSpeech (female) | 2.07 MB | fine-tuned with `research/training/` (step 40,000), see `docs/training.md` |

The output sample rate is stored in the weights file itself. Read `tts.getAudioSampleRate()`
**after** `tts.begin()` to configure your audio output. On an ESP32-S3,
`ljspeech16k_weights.bin` synthesizes about 2.2 times faster than `default_weights.bin`; see
the README's timing table.

## Dictionary and G2P data (English, shared by all weights)

| File | Setter | Size | Notes |
|---|---|---:|---|
| `default_cmudict.bin` | `setDictionary` | 3.47 MB | Full CMU dictionary; works on its own |
| `default_cmudict_slim.bin` | `setDictionary` | 1.12 MB | Slimmed dictionary; use instead of the full one, only together with `default_dictionary_model.bin` |
| `default_dictionary_model.bin` | `setDictionaryModel` | 0.99 MB | Neural G2P fallback for words not in the dictionary |

The recommended combination is weights + `default_cmudict_slim.bin` +
`default_dictionary_model.bin`.

## Regenerating

- `default_*`: `research/export_weights_and_vectors.py`, `export_cmudict.py` and
  `export_dictionary_model.py` (see `docs/research.md`).
- Trained weights: `research/export_weights_and_vectors.py --checkpoint <G_N.pth> --out-dir <dir>`
  (see `docs/training.md`).
- After regenerating, refresh the matching header with `research/export_headers.py` (for
  trained weights: `--weights <file> --name <name>`) so the file and the header stay identical.
