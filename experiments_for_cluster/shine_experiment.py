"""SHiNe baseline (Liu et al., CVPR 2024, "Semantic Hierarchy Nexus for Open-vocabulary
Object Detection"), ported from the official repo: https://github.com/naver/shine

Ported verbatim (structure/math unchanged, only fp16 CLIP-memory casts dropped since we run
CLIPSeg's text tower in fp32/mps rather than fp16 CUDA CLIP):
  - `SignatureComposer`   <- shine/tools/composer.py (and shine_cls/utils/composer.py, identical)
  - `Themer`              <- shine_cls/utils/themer.py

Official pipeline (build_inat_sing.py + zeroshot.py:do_shine):
  1. For each class, walk its hierarchy to build one "signature" (word, parent, grandparent, ...)
     per ancestor path.
  2. Compose each signature into a hierarchy-aware sentence. The paper's main results
     (see replicate_key_runs.sh / scripts_build_nexus/*/build_*_isa.sh) use `--prompter isa`:
     "a {word}, which is a {parent}, which is a {grandparent}, ..."
  3. Embed every candidate sentence with CLIP, then fuse them into one "nexus" classifier
     vector with a `Themer` (default aggregation in zeroshot.py:do_shine is "mean").

We don't have SHiNe's curated iNat/FSOD/ImageNet hierarchy-tree JSONs, so the hierarchy is
built from WordNet instead: `synset.hypernym_paths()` gives one or more root-to-leaf ancestor
chains per query word (multiple chains exist when a synset has multiple parents), which plays
the same role as SHiNe's per-class "candidate_sentences" list.
"""
import argparse
import os

import torch

from experiment_common import RESULTS_DIR, run_query_transform_experiment
import expanded_benchmark_helpers as bm_hp

OUT_PATH = os.path.join(RESULTS_DIR, "shine_experiment.csv")


# ---- shine/tools/composer.py::SignatureComposer (verbatim) ----
class SignatureComposer:
    def __init__(self, prompter='a'):
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
        return [f'a {cname}'
                for cname in signature_list]

    def _compose_avg(self, signature_list):
        return [[f'a {catName}' for catName in signature]
                for signature in signature_list]

    def _compose_concat(self, signature_list):
        return ['a ' + signature[0] + ''.join([f' {parentName}' for parentName in signature[1:]])
                for signature in signature_list]

    def _compose_isa(self, signature_list):
        return ['a ' + signature[0] + ''.join([f', which is a {parentName}' for parentName in signature[1:]])
                for signature in signature_list]

    def compose(self, signature_list):
        return self._composers[self._prompter](signature_list)


# ---- shine_cls/utils/themer.py::Themer (SVD/mean math verbatim; fp16 CLIP casts dropped) ----
class Themer:
    def __init__(self, method, thresh=1, alpha=0.5):
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
        peigen_v = V[:, 0]
        return peigen_v

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


COMPOSER = SignatureComposer(prompter='isa')       # matches replicate_key_runs.sh / build_*_isa.sh
THEME_MAKER = Themer(method='mean', thresh=1, alpha=0.5)  # matches zeroshot.py:do_shine default


def shine_candidate_sentences(word):
    synset = bm_hp.find_best_synset(word)
    if not synset:
        return COMPOSER.compose([[word]])

    paths = synset.hypernym_paths()  # each: [root, ..., synset] -- multiple when multi-parent
    signatures = []
    for path in paths:
        ancestors = [bm_hp.to_display_form(s.lemma_names()[0]) for s in reversed(path[:-1])]
        signatures.append([word] + ancestors)
    return COMPOSER.compose(signatures)


def shine_embedding(word):
    sentences = shine_candidate_sentences(word)
    embs = torch.stack([bm_hp.get_text_embedding_cached(s).squeeze() for s in sentences])
    return THEME_MAKER.get_theme(embs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()
    run_query_transform_experiment(args.out, "shine", shine_embedding, args.limit_categories, args.limit_images)
