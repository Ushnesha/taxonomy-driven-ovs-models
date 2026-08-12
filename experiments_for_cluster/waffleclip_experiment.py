import argparse
import os

import numpy as np
import torch

from experiment_common import RESULTS_DIR, run_query_transform_experiment
import expanded_benchmark_helpers as bm_hp

OUT_PATH = os.path.join(RESULTS_DIR, "waffleclip_experiment.csv")
M = 5
ALPHABET = list("abcdefghijklmnopqrstuvwxyz")


def waffle_embedding(word, m=M, seed=42):
    seed_offset = sum(ord(c) for c in word) % 10000
    rng = np.random.RandomState(seed + seed_offset)
    embs = []
    for _ in range(m):
        length = rng.randint(5, 12)
        waffle = "".join(rng.choice(ALPHABET, size=length))
        prompt = f"{word}, which has {waffle}"
        embs.append(bm_hp.get_text_embedding_cached(prompt).squeeze())
    return torch.stack(embs).mean(dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--m", type=int, default=M)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()
    run_query_transform_experiment(
        args.out, "waffleclip", lambda w: waffle_embedding(w, m=args.m),
        args.limit_categories, args.limit_images,
    )
