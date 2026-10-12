"""
English frontend: upstream's own pipeline, unchanged -- normalize_text ->
grapheme_to_phoneme (CMU dict, g2p_en fallback) -> phonemes_to_ids("EN") ->
insert_blanks. Same sequence as tiny_tts/infer.py's synthesize(), and the one
research/g2p_reference.json (and so the C++ TextG2P.h) was validated against.
"""
import nltk

import upstream  # noqa: F401  (makes tiny_tts importable)

# g2p_en (the out-of-dictionary fallback) POS-tags with NLTK; NLTK >= 3.9
# renamed the tagger resource g2p_en's own auto-download still fetches the
# old name of, so fetch the new one here (once, into ~/nltk_data).
for _res in ("taggers/averaged_perceptron_tagger_eng", "corpora/cmudict"):
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_res.split("/")[1], quiet=True)

from tiny_tts.nn import commons  # noqa: E402
from tiny_tts.text import phonemes_to_ids  # noqa: E402
from tiny_tts.text.english import grapheme_to_phoneme, normalize_text  # noqa: E402
from tiny_tts.utils import ADD_BLANK  # noqa: E402

LANG = "EN"


def text_to_ids(text):
    phones, tones, _ = grapheme_to_phoneme(normalize_text(text))
    phone_ids, tone_ids, lang_ids = phonemes_to_ids(phones, tones, LANG)
    if ADD_BLANK:
        phone_ids = commons.insert_blanks(phone_ids, 0)
        tone_ids = commons.insert_blanks(tone_ids, 0)
        lang_ids = commons.insert_blanks(lang_ids, 0)
    return phone_ids, tone_ids, lang_ids
