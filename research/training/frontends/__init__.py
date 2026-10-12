"""
Language frontends: text -> (phone_ids, tone_ids, lang_ids), already mapped
into upstream's shared multi-lingual symbol table and blank-interleaved
(commons.insert_blanks), i.e. exactly what the model's text encoder consumes.

Each language lives in its own module exposing `text_to_ids(text)`; the key is
the upstream language code from tiny_tts.text.symbols.language_id_map ("EN",
"DE", "FR", ...). Modules are imported lazily, so preparing an English dataset
doesn't require any other language's G2P dependencies.

Adding a language: add `<code>.py` here with a `text_to_ids()` producing
symbols from tiny_tts.text.symbols (unknown ones map to "UNK") and tones
offset by language_tone_start_map[code], and register it in _FRONTENDS. The
C++ runtime then needs a matching G2P too (TextG2P.h is English-only) -- see
docs/training.md.
"""
import importlib

_FRONTENDS = {
    "EN": "en",
}


def get_frontend(lang):
    """Returns the frontend module for an upstream language code (e.g. "EN")."""
    lang = lang.upper()
    if lang not in _FRONTENDS:
        raise ValueError(f"no frontend for language {lang!r}; available: {sorted(_FRONTENDS)}")
    return importlib.import_module(f"{__name__}.{_FRONTENDS[lang]}")


def available_languages():
    return sorted(_FRONTENDS)
