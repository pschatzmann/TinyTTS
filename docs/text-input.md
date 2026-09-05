# Feeding text to `speak()`

What you can pass to `TinyTTS::speak()` (or `TinyTTSCore::synthesize()` directly), and the
one real constraint on how you split it up.

## Within a single `speak()` call

- **Individual words**, **a sentence**, or **multiple sentences** all work, with no length
  limit. `TextG2P::process()` just tokenizes on whitespace and handles punctuation (`.`,
  `,`, `!`, `?`, `-`, `'`) as it goes -- there's nothing sentence-specific about it, it's one
  continuous token stream regardless of how many sentences are in it.
- There used to be a 32-phoneme ceiling (`duration_predictor`'s fixed TFLite Micro input
  window), but `TinyTTS::runDurationPredictor()` now slides overlapping windows across
  arbitrarily long text and keeps only each window's real-context "trusted middle" (see
  that method's doc in `src/TinyTTS.h`) -- so a single `speak()` call handles text of any
  length correctly, not just the first 32 phonemes of it.

## Splitting text across multiple `speak()` calls

If you have a long stream of text arriving incrementally (e.g. from a network connection)
and want to start synthesizing before it's all in, you can call `speak()` more than once --
but there's exactly one rule:

**Split on whitespace/word boundaries. Never split in the middle of a word.**

### Splitting at word boundaries: safe, but not seamless

`speak("Hello ")` followed by `speak("world.")` works correctly -- each word is looked up
whole, so pronunciation is right. But each `speak()` call is an **independent utterance**:
`TextG2P::process()` pads the start and end of *every* call (matching the reference
model's own `pad_start_end` behavior), and there's no prosodic state shared between calls.
The result is two separate audio clips placed back-to-back, not one continuous sentence --
expect a small discontinuity (a slight pause/reset in intonation) at the seam.

If you want one seamless utterance and you already have the whole text available,
concatenate it into a single string and call `speak()` once instead -- that's exactly as
safe as it is for a single sentence, now that the length ceiling is gone.

### Splitting mid-word: not safe

`speak("Hel")` followed by `speak("lo world.")` will **mispronounce** the split word.
Word resolution (CMU dictionary lookup -> `DictionaryModel` neural fallback -> crude
character-level fallback, see `TextG2P::resolveWord()`) happens independently within each
`speak()` call, with no buffering to reassemble a word that was cut in half. "HEL" and "LO"
each get looked up as if they were complete, real words -- neither is in the dictionary, so
both fall through to a fallback path and produce the wrong sound.

If your input arrives in arbitrary byte/character chunks that might land mid-word (e.g.
streamed text), buffer up to the last whitespace character and only pass the
whitespace-terminated portion to `speak()`, holding the remainder (the partial trailing
word) until more text arrives to complete it.

## Summary

| Input shape | Safe? | Notes |
|---|---|---|
| Individual word | Yes | |
| One sentence | Yes | |
| Multiple sentences, one `speak()` call | Yes | No length limit |
| Text split at word boundaries, multiple `speak()` calls | Yes, but not seamless | Small discontinuity at each call boundary |
| Text split mid-word, multiple `speak()` calls | **No** | Mispronounces the split word |
