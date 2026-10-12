"""
Makes the upstream model package (research/tiny-tts, a gitignored clone of
https://github.com/tronghieuit/tiny-tts, `develop` branch -- see
docs/research.md) importable, the same way research/export_*.py do.

Every training script imports this first, so the model classes, symbol table
and hyperparameters (tiny_tts.models / tiny_tts.text.symbols /
tiny_tts.utils.config) are always upstream's own -- never copies. That is what
keeps a trained checkpoint loadable by export_weights_and_vectors.py and the
C++ runtime unchanged.
"""
import os
import sys

RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPSTREAM_DIR = os.path.join(RESEARCH_DIR, "tiny-tts")

if not os.path.isdir(os.path.join(UPSTREAM_DIR, "tiny_tts")):
    raise SystemExit(
        f"upstream clone not found at {UPSTREAM_DIR}\n"
        "run: git clone --depth 1 -b develop https://github.com/tronghieuit/tiny-tts.git "
        f"{UPSTREAM_DIR}"
    )
if UPSTREAM_DIR not in sys.path:
    sys.path.insert(0, UPSTREAM_DIR)
