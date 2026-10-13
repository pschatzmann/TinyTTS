# Phonemes: TinyTTS's symbol table vs. TinyTTSTools

How TinyTTS represents phonemes, and how that maps to the phoneme ids of
[TinyTTSTools](https://github.com/pschatzmann/TinyTTSTools) (its
[`docs/PHONEMES.md`](https://github.com/pschatzmann/TinyTTSTools/blob/main/docs/PHONEMES.md)).
Both start from the same ARPAbet set of the CMU Pronouncing Dictionary, but encode stress and
pauses differently.

## How TinyTTS encodes a phoneme

The model doesn't take ARPAbet strings. Every position in its input is three ids:

| Input | What it is | English values |
|---|---|---|
| phone id | index into upstream's shared multi-lingual symbol table (`tiny_tts/text/symbols.py`, 219 entries) | see the table below |
| tone id | stress, offset by the language's tone start | 7-10 (raw tone 0-3 + 7) |
| language id | `language_id_map` | 2 (`EN`) |

**Phone ids.** The symbol table is one sorted list over all of upstream's languages (Chinese
pinyin, Japanese, English, ...), so many English phones share an entry with a pinyin or
Japanese symbol of the same spelling (`s`, `t`, `sh`, ...). The language id tells the
model which one is meant. English phones are stored lowercase (`aa`, `hh`), except
ARPAbet `V`, which stays uppercase `V`: lowercase `v` is already pinyin's ü.

**Tone ids (stress).** The CMU dictionary's stress digit becomes the tone, so stress isn't a
separate phone:

| CMU phone | Raw tone | Tone id passed to the model |
|---|---:|---:|
| consonant (no digit), e.g. `T` | 0 | 7 |
| vowel, stress 0 (unstressed), e.g. `AH0` | 1 | 8 |
| vowel, stress 1 (primary), e.g. `AH1` | 2 | 9 |
| vowel, stress 2 (secondary), e.g. `AH2` | 3 | 10 |

The dictionary files (`cmudict.bin`, `cmudict_slim.bin`) and the neural G2P model's output
table store (phone id, raw tone) pairs; `TextG2P::resolveWord()` adds the offset 7
(`kToneOffsetEn`). The conversion is `parse_phone()`/`map_phoneme()` in
`research/export_cmudict.py`, which matches upstream's `npm-package/index.js`.

**Sequence assembly** (`TextG2P::process()`, matching upstream's `phonemes_to_ids` +
`commons.insert_blanks`, then `TinyTTSCore::synthesize()`): one `_` (phone 0, tone 0,
language 2) at the start and end, then a blank (phone 0, tone 0, language 0) between every
two symbols and at both ends. Punctuation becomes its own symbol (tone 7).

**Durations** aren't fixed per phoneme: the duration predictor decides how many frames each
symbol gets (512 samples, 11.6 ms per frame for the released 44.1 kHz model), scaled by
`setLengthScale()`.

## Mapping table

TinyTTSTools ids 0-42 (its ARPAbet band) against TinyTTS's phone id and raw tone. "Count" is
how often the phone occurs in the 123,463-word CMU dictionary TinyTTS ships.

### Pauses

| TinyTTSTools | TinyTTS | Notes |
|---|---|---|
| 0 `SIL` | `.` (212), or `!` (208), `?` (209) | No silence phone. Pauses come from punctuation symbols, which the duration predictor turns into a pause. |
| 1 `SP` | `,` (211) | Symbol `SP` (217) exists in the table, but the English pipeline never produces it, so don't use it. |
| -- | `_` (0) | Padding/blank, inserted automatically between symbols and at both ends. Not a pause. |
| -- | `'` (213), `-` (214), `…` (210) | Other punctuation symbols the English pipeline produces. |
| -- | `UNK` (218) | Anything that can't be mapped. |

### Vowels

| TinyTTSTools id | ARPAbet | IPA | TinyTTS symbol | Phone id | Raw tone | Count |
|---:|---|---|---|---:|---|---:|
| 2 | AA | ɑ | `aa` | 21 | 1-3 | 22,957 |
| 3 | AE | æ | `ae` | 22 | 1-3 | 20,066 |
| 4 | AH | ʌ | `ah` | 23 | 2 or 3 (stressed) | 65,905 (all stresses) |
| 5 | AH0 | ə | `ah` | 23 | 1 (unstressed) | (included above) |
| 6 | AO | ɔ | `ao` | 27 | 1-3 | 10,460 |
| 7 | AW | aʊ | `aw` | 28 | 1-3 | 3,139 |
| 8 | AY | aɪ | `ay` | 29 | 1-3 | 10,169 |
| 9 | EH | ɛ | `eh` | 39 | 1-3 | 24,855 |
| 10 | ER | ɝ | `er` | 43 | 2 or 3 (stressed) | 26,913 (all stresses) |
| 11 | ER0 | ɚ | `er` | 43 | 1 (unstressed) | (included above) |
| 12 | EY | eɪ | `ey` | 44 | 1-3 | 12,272 |
| 13 | IH | ɪ | `ih` | 59 | 1-3 | 44,459 |
| 14 | IY | i | `iy` | 65 | 1-3 | 31,303 |
| 15 | OW | oʊ | `ow` | 80 | 1-3 | 17,870 |
| 16 | OY | ɔɪ | `oy` | 81 | 1-3 | 1,165 |
| 17 | UH | ʊ | `uh` | 99 | 1-3 | 2,136 |
| 18 | UW | u | `uw` | 103 | 1-3 | 8,876 |

### Consonants (raw tone always 0)

| TinyTTSTools id | ARPAbet | IPA | TinyTTS symbol | Phone id | Count |
|---:|---|---|---|---:|---:|
| 19 | B | b | `b` | 30 | 19,747 |
| 20 | D | d | `d` | 34 | 29,553 |
| 21 | G | ɡ | `g` | 46 | 12,653 |
| 22 | K | k | `k` | 68 | 39,539 |
| 23 | P | p | `p` | 82 | 18,082 |
| 24 | T | t | `t` | 89 | 44,302 |
| 25 | DH | ð | `dh` | 35 | 521 |
| 26 | F | f | `f` | 45 | 12,655 |
| 27 | HH | h | `hh` | 49 | 8,445 |
| 28 | S | s | `s` | 87 | 45,516 |
| 29 | SH | ʃ | `sh` | 88 | 7,945 |
| 30 | TH | θ | `th` | 90 | 2,708 |
| 31 | V | v | `V` | 14 | 9,702 |
| 32 | Z | z | `z` | 111 | 25,307 |
| 33 | ZH | ʒ | `zh` | 112 | 493 |
| 34 | CH | tʃ | `ch` | 33 | 4,569 |
| 35 | JH | dʒ | `jh` | 67 | 5,670 |
| 36 | L | l | `l` | 70 | 45,741 |
| 37 | R | ɹ | `r` | 85 | 42,056 |
| 38 | W | w | `w` | 108 | 7,987 |
| 39 | Y | j | `y` | 110 | 4,504 |
| 40 | M | m | `m` | 71 | 27,242 |
| 41 | N | n | `n` | 73 | 55,416 |
| 42 | NG | ŋ | `ng` | 74 | 9,032 |

These are the 39 phones of `en_symbols`; the CMU dictionary uses exactly these, all of
which map (no `UNK`).

## Differences

| | TinyTTSTools | TinyTTS |
|---|---|---|
| Id space | Own `Phone` enum: 0-42 ARPAbet, 43-120 IPA extension | Upstream's shared multi-lingual table (219 entries), plus a language id |
| Stress | `PhonemeModifier` (`MOD_STRESS_PRIMARY`/`SECONDARY`, `"`/`%` prefix in text) | Tone id, from the CMU stress digit |
| Unstressed vowels | `AH0` (schwa) and `ER0` get their own ids; other vowels have no unstressed marker | Every vowel carries stress 0/1/2 as tone; `AH0`/`ER0` are `ah`/`er` with tone 1 |
| Silence | `SIL`, `SP` phones with fixed durations | Punctuation symbols; durations predicted |
| Durations | Fixed per phoneme (60-400 ms) | Predicted per symbol by the model |
| Other modifiers (length, nasalized, aspirated, ...) | One per segment | None. The English model has no input for them. |
| Non-English phones (ids 43-120) | Supported | No counterpart: the C++ pipeline only produces English (language 2) input, and the released model hasn't been checked with other languages' symbols |

## Converting

**TinyTTSTools → TinyTTS** (e.g. to feed TinyTTSTools dictionary output into TinyTTS):

1. Silence: `SIL` → `.`, `SP` → `,`.
2. Phone: look up the phone id in the table above (`AH0` → `ah`, `ER0` → `er`).
3. Raw tone: consonant → 0. Vowel: `MOD_STRESS_PRIMARY` → 2, `MOD_STRESS_SECONDARY` → 3,
   no stress modifier (and always for `AH0`/`ER0`) → 1. `AH`/`ER` without a stress modifier
   are stressed by definition in TinyTTSTools, so use 2.
4. Drop all other modifiers.
5. Tone id = raw tone + 7, language id = 2; then add the padding and blanks as above.

**TinyTTS → TinyTTSTools** (e.g. to show what TinyTTS will say as ARPAbet/IPA):

1. Skip `_` (0); map `.`/`!`/`?` → `SIL`, `,` → `SP`.
2. Raw tone = tone id - 7. `ah` with raw tone 1 → `AH0` (5); `er` with raw tone 1 → `ER0`
   (11); any other vowel → its base id, with `MOD_STRESS_PRIMARY` for raw tone 2 and
   `MOD_STRESS_SECONDARY` for 3.
3. Consonants → their base id.

The C++ API doesn't take phoneme input yet: `speak()` only accepts text, and the ids are
produced inside `TextG2P`. Feeding phonemes directly means calling the stages in
`TinyTTSCore` yourself.
