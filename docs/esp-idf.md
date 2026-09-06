# Building as an ESP-IDF component

TinyTTS also builds as a plain ESP-IDF component -- point `EXTRA_COMPONENT_DIRS` (or an
`idf_component.yml` dependency) at this repo, `#include <TinyTTS.h>`, and link against it
like any other component.

There's no Arduino dependency in that path, so use the portable API instead of the
`Print`/`File` conveniences: `setWeights(data, len)`/`setDictionary(data, len)` and
`begin(const AudioChunkFn&)` (see `TinyTTS.h`) -- the same calls this project's own host
tests use.
