"""
The 5 query-transform methods compared in the paper (Baseline / Ours / SHiNe /
WaffleCLIP / LLM-descriptor), each computing a text embedding for a linguistic
variant word against a given pluggable model
(models.SUPPORTED_MODELS: clipseg, groupvit, sclip).

Local copy for this standalone model folder (see ../README.md) -- source of
truth is ../../experiments_for_cluster/approaches.py, kept in sync manually.

SHiNe / WaffleCLIP / LLM-descriptor are ports of their respective papers'
official repos (see each function's own docstring below for exact
provenance). This module always has a real `model` to embed against.

"Ours" is NOT a verbatim port -- helper_functions.ours_blend_info() builds
every variant's neighbor pool from the CANONICAL word's own synonym set
(so e.g. airplane's hypernym "heavier-than-air craft" would still draw
neighbors from "aeroplane"/"plane"). This version re-disambiguates the QUERY
word itself against WordNet and builds its centroid from ITS OWN synonym set
-- "linguistic variations of the linguistic variations" (see conversation
decision log). For query_word == canonical_word (the "orig" variant) the two
are identical.
"""
import os
import pickle

import numpy as np
import requests
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from expanded_benchmark_helpers import (  # noqa: E402  (local copy, sibling file)
    to_display_form,
    find_best_synset,
    build_word_sets_from_synset,
    cosine_sim,
    COCO_80,
)

DEFAULT_ALPHA = 0.52  # alpha_tuning_dev/alpha_best_value.md sweet spot (fine sweep best 0.52)
DEFAULT_TOPK = 5


# =============================================================================
# Shared embedding cache + WordNet helpers
# =============================================================================

_embed_cache = {}


def embed(model, word, desc=True):
    """
    model.get_text_embedding(word, desc=desc), cached per (model, word, desc)
    for this process. desc=True is used throughout (matches
    helper_functions.py's convention): every string handed to embed() here is
    already the full intended prompt (a bare category/variant word, or a full
    descriptor/candidate sentence for waffleclip/shine/llm_descriptor) -- it
    should never additionally get wrapped in "a photo of a {}".
    """
    key = (id(model), word, desc)
    if key not in _embed_cache:
        _embed_cache[key] = model.get_text_embedding(word, desc=desc).cpu()
    return _embed_cache[key].clone()


def eligible_synset(word):
    """A word is usable for WordNet-based blending if it resolves to a
    synset with at least one other synonym to build a centroid from."""
    s = find_best_synset(word)
    if s and len(s.lemma_names()) >= 2:
        return s
    return None


def _top_k_neighbors(model, word, candidates, k=DEFAULT_TOPK):
    query_emb = embed(model, word).squeeze()
    scored = []
    for cand in candidates:
        if cand.lower() == word.lower():
            continue
        cand_emb = embed(model, cand)
        sim = cosine_sim(query_emb, cand_emb)
        scored.append((cand, cand_emb, sim))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:k]


def compute_centroid(neighbors):
    return torch.stack([e.squeeze() for _, e, _ in neighbors]).mean(dim=0)


def compute_weighted_centroid(neighbors):
    sims = torch.tensor([max(s, 0.0) for _, _, s in neighbors])
    if sims.sum() <= 0:
        weights = torch.ones(len(neighbors)) / len(neighbors)
    else:
        weights = sims / sims.sum()
    embs = torch.stack([e.squeeze() for _, e, _ in neighbors])
    return (weights.unsqueeze(1) * embs).sum(dim=0)


def blend_embedding(query_emb, centroid, alpha):
    """(1-alpha)*query + alpha*centroid. Preserves raw embedding norm --
    required for CLIPSeg's FiLM decoder; harmless for models that normalize
    internally (GroupViT, SCLIP re-normalizes in predict_with_embeddings)."""
    q = query_emb.squeeze()
    return (1 - alpha) * q + alpha * centroid.squeeze()


# =============================================================================
# Baseline: bare query-word embedding, no transform.
# =============================================================================

def baseline_embedding(canonical_word, query_word, model):
    return embed(model, query_word)


# =============================================================================
# Ours: recursive alpha-blending toward the QUERY word's own WordNet-synonym
# centroid.
# =============================================================================

def ours_blend_info(query_word, model, k=DEFAULT_TOPK):
    synset = eligible_synset(query_word)
    if synset is None:
        return None
    w_s = [to_display_form(w) for w in build_word_sets_from_synset(synset)["W_S"]]
    candidates = [w for w in w_s if w.lower() != query_word.lower()]
    if not candidates:
        return None

    query_emb = embed(model, query_word)
    neighbors = _top_k_neighbors(model, query_word, candidates, k=k)
    if not neighbors:
        return None

    return {
        "query_emb": query_emb,
        "centroid_uw": compute_centroid(neighbors),
        "centroid_w": compute_weighted_centroid(neighbors),
        "neighbors": [w for w, _, _ in neighbors],
    }


def ours_embedding(canonical_word, query_word, model, alpha=DEFAULT_ALPHA, weighted=False, k=DEFAULT_TOPK):
    """Returns None if `query_word` has no usable synset/neighbors (caller
    should skip this (variant, approach) row, not fail the whole run)."""
    info = ours_blend_info(query_word, model, k=k)
    if info is None:
        return None
    centroid = info["centroid_w"] if weighted else info["centroid_uw"]
    return blend_embedding(info["query_emb"], centroid, alpha)


# =============================================================================
# SHiNe (Liu et al., CVPR 2024) -- verbatim port from helper_functions.py.
# =============================================================================

class SignatureComposer:
    def __init__(self, prompter='isa'):
        if prompter not in ['a', 'avg', 'concat', 'isa']:
            raise NameError(f"{prompter} prompter is not supported")
        self._prompter = prompter
        self._composers = {
            'a': self._compose_a,
            'avg': self._compose_avg,
            'concat': self._compose_concat,
            'isa': self._compose_isa,
        }

    def _compose_a(self, signature_list):
        return [f'a {cname}' for cname in signature_list]

    def _compose_avg(self, signature_list):
        return [[f'a {catName}' for catName in signature] for signature in signature_list]

    def _compose_concat(self, signature_list):
        return ['a ' + signature[0] + ''.join([f' {parentName}' for parentName in signature[1:]])
                for signature in signature_list]

    def _compose_isa(self, signature_list):
        return ['a ' + signature[0] + ''.join([f', which is a {parentName}' for parentName in signature[1:]])
                for signature in signature_list]

    def compose(self, signature_list):
        return self._composers[self._prompter](signature_list)


class Themer:
    def __init__(self, method='mean', thresh=1, alpha=0.5):
        if method not in ['mean', 'peigen', 'mixed', 'all_eigens']:
            raise NameError(f"{method} is not supported")
        self.method = method
        self.T = thresh
        self.alpha = alpha

    def _get_principal_eigenvector(self, stacked_feats):
        if stacked_feats.shape[0] == 1:
            return stacked_feats[0]
        stacked_feats32 = stacked_feats.to(torch.float32)
        U, S, V = torch.svd(stacked_feats32)
        return V[:, 0]

    def _get_all_eigenvector(self, stacked_feats):
        if stacked_feats.shape[0] == 1:
            return stacked_feats[0]
        stacked_feats32 = stacked_feats.to(torch.float32)
        U, S, V = torch.svd(stacked_feats32)
        normalized_weights_s = S / torch.sum(S)
        weighted_avg_v = torch.zeros_like(V[:, 0])
        for i in range(V.size(1)):
            weighted_avg_v += normalized_weights_s[i] * V[:, i]
        return weighted_avg_v

    def _get_mean_vector(self, stacked_feats):
        return torch.mean(stacked_feats, dim=0)

    def _get_mixed_vector(self, stacked_feats):
        mean_v = self._get_mean_vector(stacked_feats)
        peigen_v = self._get_principal_eigenvector(stacked_feats)
        return self.alpha * mean_v + (1 - self.alpha) * peigen_v

    def get_theme(self, stacked_feats):
        if self.method == 'peigen' and stacked_feats.shape[0] > self.T:
            return self._get_principal_eigenvector(stacked_feats)
        elif self.method == 'all_eigens' and stacked_feats.shape[0] > self.T:
            return self._get_all_eigenvector(stacked_feats)
        elif self.method == 'mixed' and stacked_feats.shape[0] > self.T:
            return self._get_mixed_vector(stacked_feats)
        else:
            return self._get_mean_vector(stacked_feats)


SHINE_COMPOSER = SignatureComposer(prompter='isa')
SHINE_THEME_MAKER = Themer(method='mean', thresh=1, alpha=0.5)


def shine_candidate_sentences(word):
    """Builds SHiNe's per-class candidate sentences from WordNet hypernym
    paths (synset.hypernym_paths() stands in for SHiNe's curated hierarchy
    trees: one or more root-to-leaf ancestor chains per word)."""
    synset = find_best_synset(word)
    if not synset:
        return SHINE_COMPOSER.compose([[word]])
    paths = synset.hypernym_paths()
    signatures = []
    for path in paths:
        ancestors = [to_display_form(s.lemma_names()[0]) for s in reversed(path[:-1])]
        signatures.append([word] + ancestors)
    return SHINE_COMPOSER.compose(signatures)


def shine_embedding(query_word, model):
    sentences = shine_candidate_sentences(query_word)
    embs = torch.stack([embed(model, s).squeeze() for s in sentences])
    return SHINE_THEME_MAKER.get_theme(embs)


# =============================================================================
# WaffleCLIP (Roth et al., ICCV 2023) -- verbatim port from helper_functions.py.
# `class_name_list` (defaults to COCO_80 for parity with the original
# derivation) should be our benchmark's real category list when called from
# an orchestration script -- it's only used to derive length statistics
# (avg words/class, avg word length) for the random base/noise tokens, not
# looked up per-category.
# =============================================================================

WAFFLE_WORD_LIST_PATH = os.path.join(THIS_DIR, "waffleclip_word_list.pkl")
WAFFLE_COUNT = 15
WAFFLE_SEED = 1

WAFFLE_LABEL_BEFORE_TEXT = "A photo of a "
WAFFLE_DESCRIPTOR_SEPARATOR = ", "
WAFFLE_LABEL_AFTER_TEXT = "."
WAFFLE_PRE_DESCRIPTOR_TEXT = ""

WAFFLE_CHARACTER_LIST = list("\"'()-/0123456789:ABCDEFGIJKLMNOPQRSTUVWZabcdefghijklmnopqrstuvwxyz")


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
    return (f"{WAFFLE_PRE_DESCRIPTOR_TEXT}{WAFFLE_LABEL_BEFORE_TEXT}{wordify(cls)}{WAFFLE_DESCRIPTOR_SEPARATOR}"
            f"{modify_descriptor(item)}{WAFFLE_LABEL_AFTER_TEXT}")


_waffle_shared_fragments = {}  # keyed by id(class_name_list) -- one derivation per distinct class list


def _build_waffle_shared_fragments(class_name_list):
    cache_key = id(class_name_list)
    if cache_key in _waffle_shared_fragments:
        return _waffle_shared_fragments[cache_key]

    rng = np.random.RandomState(WAFFLE_SEED)
    with open(WAFFLE_WORD_LIST_PATH, "rb") as f:
        raw_word_list = pickle.load(f)

    key_list = class_name_list

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
                noise_word += rng.choice(WAFFLE_CHARACTER_LIST)
            else:
                noise_word += ', '
        noise_words.append(noise_word)

    _waffle_shared_fragments[cache_key] = (base_words, noise_words)
    return _waffle_shared_fragments[cache_key]


def waffle_descriptors(word, class_name_list=None):
    base_words, noise_words = _build_waffle_shared_fragments(class_name_list or COCO_80)
    return [structured_descriptor_builder(item, word) for item in base_words] + \
           [structured_descriptor_builder(item, word) for item in noise_words]


def waffleclip_embedding(query_word, model, class_name_list=None):
    descriptors = waffle_descriptors(query_word, class_name_list=class_name_list)
    embs = torch.stack([embed(model, s).squeeze() for s in descriptors])
    return embs.mean(dim=0)


# =============================================================================
# LLM-descriptor (Menon & Vondrick, ICLR 2023) -- verbatim port from
# helper_functions.py. Uses a local Ollama server; falls back to the WordNet
# gloss when unreachable.
# =============================================================================

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")


def generate_llm_prompt(category_name):
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
    return [d[2:] for d in description.split('\n') if d != '' and d.startswith('- ')]


def query_llm_descriptors(word):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": generate_llm_prompt(word),
            "options": {"temperature": 0.0, "num_predict": 100, "stop": ["\nQ:"]},
            "stream": False,
            "raw": True,
        }, timeout=30)
        r.raise_for_status()
        text = "-" + r.json().get("response", "")
        descriptors = stringtolist(text)
        if descriptors:
            return descriptors
    except Exception as e:
        print(f"  [llm_description] Ollama unreachable/failed for '{word}' ({e}); falling back to WordNet gloss")
    synset = find_best_synset(word)
    if synset and synset.definition():
        return [synset.definition()]
    return [word]


def llm_descriptor_sentences(word):
    items = query_llm_descriptors(word)
    return [f"{wordify(word)}{WAFFLE_DESCRIPTOR_SEPARATOR}{modify_descriptor(item)}" for item in items]


def llm_descriptor_embedding(query_word, model):
    sentences = llm_descriptor_sentences(query_word)
    embs = torch.stack([embed(model, s).squeeze() for s in sentences])
    return embs.mean(dim=0)


# =============================================================================
# Unified dispatcher.
# =============================================================================

ALL_APPROACHES = ["baseline", "ours", "shine", "waffleclip", "llm_descriptor"]


def approach_embedding(approach, canonical_word, query_word, model, alpha=DEFAULT_ALPHA, weighted=False,
                        k=DEFAULT_TOPK, class_name_list=None):
    """
    Single entrypoint: computes `approach`'s embedding for `query_word`
    (a linguistic variant of `canonical_word`) against `model`. Returns None
    if the approach couldn't resolve an embedding (currently only "ours",
    when query_word has no usable WordNet synset) -- callers should skip
    that (variant, approach) row rather than treat it as an error.
    """
    if approach == "baseline":
        return baseline_embedding(canonical_word, query_word, model)
    if approach == "ours":
        return ours_embedding(canonical_word, query_word, model, alpha=alpha, weighted=weighted, k=k)
    if approach == "shine":
        return shine_embedding(query_word, model)
    if approach == "waffleclip":
        return waffleclip_embedding(query_word, model, class_name_list=class_name_list)
    if approach == "llm_descriptor":
        return llm_descriptor_embedding(query_word, model)
    raise ValueError(f"Unknown approach '{approach}'. Available: {ALL_APPROACHES}")
