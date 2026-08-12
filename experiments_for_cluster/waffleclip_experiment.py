"""WaffleCLIP baseline (Roth et al., ICCV 2023, "Waffling around for Performance: Visual
Classification with Random Words and Broad Concepts"), ported from the official repo:
https://github.com/ExplainableML/WaffleCLIP

Ported verbatim from `waffle_tools.py` (mode='waffle') and `base_main.py`:
  - `wordify`, `make_descriptor_sentence`, `modify_descriptor` -- identical helper functions.
  - `structured_descriptor_builder` template and default hyperparameters
    (label_before_text='A photo of a ', descriptor_separator=', ', label_after_text='.',
    apply_descriptor_modification=True, waffle_count=15 -- see base_main.py argparse defaults
    and replicate_key_runs.sh, which always calls `--mode=waffle --waffle_count=15`).
  - The 'waffle' mode's random descriptor generation: per waffle_count iteration, one
    "base_word" built from `avg_num_words` real dictionary words (from `word_list.pkl`,
    truncated to the dataset's average word length) and one "noise_word" built from random
    characters shaped like the dataset's average class-name (space/length pattern) -- giving
    2*waffle_count=30 descriptors total. `word_list.pkl` is the exact 4330-word asset shipped
    in the official repo, copied here as `waffleclip_word_list.pkl`. The official code derives
    its random-character pool from words appearing in a GPT descriptor file it has on disk for
    other baselines; we don't have that file, so `CHARACTER_LIST` below is the literal pool
    their own code produces when run against their shipped `descriptors/descriptors_imagenet.json`.
  - Crucially, the official 'waffle' mode draws ONE random set of words/characters and reuses
    it for every class (`match_key = np.random.choice(key_list); gpt_descriptions = {k:
    gpt_descriptions[match_key] ...}` then text-substitutes the class name in) -- the paper's
    point is that the randomness need not be class-specific. We replicate that: the random
    draw happens once per script run (seeded, default seed=1 matching `--seed 1`), not per word.
  - Default aggregation: `base_main.py` only mean-pools description *embeddings*
    (`descr_means`) when `--merge_predictions` is passed; `replicate_key_runs.sh` never passes
    it, so the default (and the one used for all reported numbers) is score-level averaging
    (`aggregate_similarity(..., aggregation_method='mean')` over per-descriptor image-text
    similarity). We use `run_multi_descriptor_experiment`, which mirrors this by mean-averaging
    per-descriptor CLIPSeg probability maps instead of blending embeddings.
"""
import argparse
import os
import pickle

import numpy as np

from experiment_common import RESULTS_DIR, run_multi_descriptor_experiment
import expanded_benchmark_helpers as bm_hp

OUT_PATH = os.path.join(RESULTS_DIR, "waffleclip_experiment.csv")
WORD_LIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waffleclip_word_list.pkl")

WAFFLE_COUNT = 15  # base_main.py --waffle_count default, used verbatim in replicate_key_runs.sh
SEED = 1  # base_main.py --seed default

# label_before_text / descriptor_separator / label_after_text defaults (base_main.py argparse)
LABEL_BEFORE_TEXT = "A photo of a "
DESCRIPTOR_SEPARATOR = ", "
LABEL_AFTER_TEXT = "."
PRE_DESCRIPTOR_TEXT = ""

# Exact character pool produced by the official waffle_tools.py 'waffle' mode when run against
# their shipped descriptors/descriptors_imagenet.json (see docstring above).
CHARACTER_LIST = list("\"'()-/0123456789:ABCDEFGIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwxyz")


# ---- waffle_tools.py helpers (verbatim) ----
def wordify(string):
    return string.replace('_', ' ')


def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') \
            or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"


def modify_descriptor(descriptor, apply_changes=True):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor


def structured_descriptor_builder(item, cls):
    return (f"{PRE_DESCRIPTOR_TEXT}{LABEL_BEFORE_TEXT}{wordify(cls)}{DESCRIPTOR_SEPARATOR}"
            f"{modify_descriptor(item)}{LABEL_AFTER_TEXT}")


_shared_fragments = None  # (base_words, noise_words), drawn once per run -- shared across all classes


def _build_shared_fragments():
    global _shared_fragments
    if _shared_fragments is not None:
        return _shared_fragments

    rng = np.random.RandomState(SEED)
    with open(WORD_LIST_PATH, "rb") as f:
        raw_word_list = pickle.load(f)

    # key_list: stand-in for the dataset's full class-name list (official code averages these
    # stats over the entire benchmark's class names, then shares the result across classes).
    key_list = bm_hp.COCO_80

    avg_num_words = int(max(round(np.mean([len(wordify(x).split(' ')) for x in key_list])), 1))
    avg_word_length = int(round(np.mean([np.mean([len(y) for y in wordify(x).split(' ')]) for x in key_list])))
    word_list = [str(x)[:avg_word_length] for x in raw_word_list]

    num_spaces = int(round(np.mean([x.count(' ') for x in key_list]))) + 1
    num_chars = int(np.ceil(np.mean([max(len(y) for y in x.split(' ')) for x in key_list])))
    num_chars += num_spaces - num_chars % num_spaces

    sample_key = ''
    for s in range(num_spaces):
        for _ in range(num_chars // num_spaces):
            sample_key += 'a'
        if s < num_spaces - 1:
            sample_key += ' '

    base_words, noise_words = [], []
    for _ in range(WAFFLE_COUNT):
        base_word = ''
        for a in range(avg_num_words):
            base_word += str(rng.choice(word_list))
            if a < avg_num_words - 1:
                base_word += ' '
        base_words.append(base_word)

        noise_word = ''
        for c in sample_key:
            if c != ' ':
                noise_word += rng.choice(CHARACTER_LIST)
            else:
                noise_word += ', '
        noise_words.append(noise_word)

    _shared_fragments = (base_words, noise_words)
    return _shared_fragments


def waffle_descriptors(word):
    base_words, noise_words = _build_shared_fragments()
    return [structured_descriptor_builder(item, word) for item in base_words] + \
           [structured_descriptor_builder(item, word) for item in noise_words]


def baseline_descriptors(word):
    return [f"{LABEL_BEFORE_TEXT}{wordify(word)}{LABEL_AFTER_TEXT}"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--waffle-count", type=int, default=WAFFLE_COUNT)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()
    WAFFLE_COUNT = args.waffle_count
    run_multi_descriptor_experiment(
        args.out, "waffleclip", baseline_descriptors, waffle_descriptors,
        args.limit_categories, args.limit_images,
    )
