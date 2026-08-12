"""LLM-descriptor baseline (Menon & Vondrick, ICLR 2023, "Visual Classification via
Description from Large Language Models" / "classify by description"), ported from the
official repo: https://github.com/sachit-menon/classify_by_description_release

Ported verbatim:
  - `generate_prompt()`  <- descriptor_strings.py / generate_descriptors.py (identical two-shot
    exemplar prompt, "lemur" + "television" examples).
  - `stringtolist()`     <- descriptor_strings.py (parses "- descriptor" bullet lines out of
    the LLM completion).
  - `make_descriptor_sentence` / `modify_descriptor` <- descriptor_strings.py / loading_helpers.py
    (identical in both files).
  - Sentence template <- loading_helpers.py:load_gpt_descriptions with
    `hparams['category_name_inclusion'] = 'prepend'` (the default): each of the ~7 GPT
    descriptors becomes its own sentence
    `f"{before_text}{word}{between_text}{modify_descriptor(item)}{after_text}"`, with the
    non-ImageNet-dataset defaults `before_text='', between_text=', ', after_text=''`
    (load.py:57-66) -- i.e. `"{word}, which has {descriptor}."`-style, one sentence per
    descriptor, NOT one sentence with all descriptors crammed together.
  - Baseline label encoding <- load.py:177, `label_before_text + wordify(l) + label_after_text`;
    for non-ImageNet datasets both default to `''`, so the baseline is just the bare class word.
  - Aggregation <- load.py:181-184 `aggregate_similarity(..., 'mean')`: each descriptor sentence
    is embedded and scored independently, and the *scores* are averaged, not the embeddings.
    We use `run_multi_descriptor_experiment`, which mirrors this by mean-averaging the
    per-descriptor CLIPSeg probability maps.

Deliberate substitution: the paper calls OpenAI `text-davinci-003` (temperature=0, max_tokens=100)
to generate descriptors offline once per class; that model is retired and requires a paid API
key we don't have configured here. We send the exact same prompt/temperature/max_tokens to a
local Ollama model instead, and parse the response with the same `stringtolist`. If no LLM is
reachable we fall back to the WordNet gloss as a single descriptor -- a fallback the original
paper does not have (it assumes the GPT call always succeeds), added purely so the script can
run without network access to an LLM.
"""
import argparse
import os

import requests

from experiment_common import RESULTS_DIR, run_multi_descriptor_experiment
import expanded_benchmark_helpers as bm_hp

OUT_PATH = os.path.join(RESULTS_DIR, "llm_description_experiment.csv")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

BEFORE_TEXT = ""
BETWEEN_TEXT = ", "
AFTER_TEXT = ""


# ---- descriptor_strings.py (verbatim) ----
def generate_prompt(category_name: str) -> str:
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""


def stringtolist(description):
    return [descriptor[2:] for descriptor in description.split('\n') if (descriptor != '') and (descriptor.startswith('- '))]


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


def query_llm_descriptors(word):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": generate_prompt(word),
            "options": {"temperature": 0.0, "num_predict": 100},
            "stream": False,
        }, timeout=30)
        r.raise_for_status()
        text = "-" + r.json().get("response", "")  # prompt ends in a bare "-", completion picks up after it
        descriptors = stringtolist(text)
        if descriptors:
            return descriptors
    except Exception:
        pass
    synset = bm_hp.find_best_synset(word)
    if synset and synset.definition():
        return [synset.definition()]
    return [word]


def llm_descriptors(word):
    items = query_llm_descriptors(word)
    return [f"{BEFORE_TEXT}{wordify(word)}{BETWEEN_TEXT}{modify_descriptor(item)}{AFTER_TEXT}" for item in items]


def baseline_descriptors(word):
    return [wordify(word)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()
    run_multi_descriptor_experiment(
        args.out, "llm_description", baseline_descriptors, llm_descriptors,
        args.limit_categories, args.limit_images,
    )
