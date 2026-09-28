"""Does CAT-Seg actually SEE the embedding we inject? Run this before any sweep.

predict_with_embeddings() works by overwriting rows of CAT-Seg's cached text-embedding
tensor, running a forward pass, then restoring them. If that swap does not reach the
model -- wrong attribute path, a cache that is rebuilt per call, only some template rows
written -- then every approach (baseline / ours / shine / waffleclip / llm_descriptor)
produces the SAME mask and the whole comparison is silently meaningless. It does not
crash; it just returns identical numbers.

This script fails loudly instead. Three checks, in increasing strictness:

  1. SANITY    -- the real class embedding produces a non-empty, non-total mask.
  2. INJECTION -- a nonsense embedding in the same row produces a DIFFERENT mask.
                  If this fails, nothing downstream means anything.
  3. BLENDING  -- alpha=0 (pure query) and alpha=1 (pure synonym centroid) produce
                  different masks. This is what the alpha sweep actually varies, so
                  if check 2 passes but this fails, the sweep would return a flat
                  curve and read as "blending does nothing".

It also prints the text-embedding cache shape (classes x prompt_templates x dim). If
templates > 1, catseg.py's all-template row swap is load-bearing; with the upstream
single-row swap the injection would have been diluted to 1/templates.

Uses a real benchmark image and the same vocabulary the experiments build, so it
exercises the real path rather than a toy one.

Usage:
    python3 probe_injection.py --config ./CAT-Seg/configs/vitl_336.yaml \\
                               --weights /scratch/$USER/catseg_ckpt/catseg_model_large.pth
"""
import argparse
import os
import sys

import numpy as np
import torch

import benchmark_data as bd
import approaches as ap
from catseg import CATSegModel


def describe(name, mask):
    total = mask.size
    on = int(mask.sum())
    print(f"  {name:<28} {on:>9,} / {total:,} px  ({100.0*on/total:5.2f}%)")
    return on


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--category", default=None,
                   help="which benchmark category to probe (default: first one with "
                        "positive images AND a resolvable WordNet synonym set)")
    p.add_argument("--alpha", type=float, default=0.69)
    p.add_argument("--max-tries", type=int, default=12,
                   help="how many candidate categories to try before giving up on "
                        "finding one the model actually predicts")
    args = p.parse_args()

    bm = bd.load_benchmark(args.benchmark_dir)
    class_json = os.path.join("/tmp", f"catseg_probe_classes_{os.getpid()}.json")
    bd.build_catseg_class_json(bm, class_json)

    # Build an ORDERED candidate list rather than taking the first category.
    #
    # The first alphabetically is 'Bannister' -- a WordNet proper noun whose "synonyms"
    # are 'Roger Bannister' the runner. With threshold=0.0 (pure argmax over 326 classes)
    # a rare class can legitimately win NO pixels, and then every mask is empty, 0 == 0,
    # and the injection/blending checks fail VACUOUSLY. That looks identical to a broken
    # injection but says nothing at all.
    #
    # So: prefer categories the model will actually predict. Positive-set size is a good
    # proxy for "common, large, frequently-annotated object", and lowercase names skew to
    # ordinary nouns rather than proper-noun artifacts. Then try candidates in order until
    # one yields a non-empty mask.
    def _rank(c):
        return (c[:1].isupper(), -len(bm.positive_set.get(c, [])))

    if args.category:
        candidates = [args.category]
    else:
        candidates = sorted(
            (c for c in bm.categories
             if bm.positive_set.get(c) and ap.eligible_synset(c) is not None),
            key=_rank,
        )[:args.max_tries]
        if not candidates:
            candidates = sorted((c for c in bm.categories if bm.positive_set.get(c)),
                                key=_rank)[:args.max_tries]
    print(f"candidate categories (most-annotated first): {candidates[:6]}"
          f"{' ...' if len(candidates) > 6 else ''}")

    print("\nbuilding CAT-Seg ...", flush=True)
    model = CATSegModel(config=args.config, weights=args.weights, class_name_path=class_json)
    if model.baseline:
        sys.exit("ABORT: model fell back to BASELINE (CLIPSeg) mode -- check --config/--weights.")

    # Force the cache to exist so its shape can be reported before any probing.
    model._warm_cache()
    shape = tuple(model._predictor_module.cache.shape)
    print(f"\ncache shape (classes x templates x dim) = {shape}")
    n_templates = shape[1] if len(shape) == 3 else None
    if n_templates and n_templates > 1:
        print(f"  -> {n_templates} templates ensembled; the all-template row swap in")
        print(f"     catseg.py IS load-bearing (upstream's [idx, 0, :] would have")
        print(f"     injected only 1/{n_templates} of the signal).")
    else:
        print("  -> single template; the all-template swap is harmless insurance.")

    noise = model.get_text_embedding(
        "a chaotic explosion of static and noise with no object", desc=True)

    # Walk the candidates until one produces a non-empty mask. An all-zero mask makes
    # the later comparisons meaningless, so it is a bad probe subject, not a failure.
    cat = image = m_real = None
    n_real = 0
    for c in candidates:
        img_id = bm.positive_set[c][0]
        entry = bm.img_by_id.get(img_id)
        if entry is None:
            continue
        try:
            img = bd.fetch_image(entry)
        except Exception as e:
            print(f"  skip {c}: fetch error {e}")
            continue
        m = model.predict_with_embeddings(img, {c: ap.embed(model, c)})[c]
        on = int(m.sum())
        print(f"  trying {c!r:<22} img={img_id:<16} {on:>9,}/{m.size:,} px")
        if 0 < on < m.size:
            cat, image, m_real, n_real = c, img, m, on
            break
    if cat is None:
        print("\nSTOP. No candidate category produced a non-empty mask. Either the model")
        print("is not loading its weights (check the Detectron2 'not found' warnings above)")
        print("or predict_with_embeddings is broken. Re-run with --category <a big, common")
        print("class you know is in the image> to confirm before digging further.")
        sys.exit(1)

    # ap.embed(desc=True) is what baseline_embedding uses, so 'real' here is exactly
    # the baseline approach's embedding -- and therefore exactly alpha=0 below.
    real = ap.embed(model, cat)
    print(f"\nprobing with category = {cat!r}   size = {image.size}")
    print("\nmask sizes")
    describe(f"real '{cat}'", m_real)
    m_noise = model.predict_with_embeddings(image, {cat: noise})[cat]
    describe("nonsense embedding", m_noise)

    ok_sanity = 0 < n_real < m_real.size
    ok_inject = not np.array_equal(m_real, m_noise)

    # ---- blending: what the alpha sweep varies ----
    info = ap.ours_blend_info(cat, model)
    ok_blend = None
    if info is None:
        print(f"\n  (no WordNet synonym set for '{cat}' -- skipping the blending check; "
              f"rerun with --category on a word that has synonyms)")
    else:
        e0 = ap.blend_embedding(info["query_emb"], info["centroid_w"], 0.0)
        e1 = ap.blend_embedding(info["query_emb"], info["centroid_w"], 1.0)
        ea = ap.blend_embedding(info["query_emb"], info["centroid_w"], args.alpha)
        m0 = model.predict_with_embeddings(image, {cat: e0})[cat]
        m1 = model.predict_with_embeddings(image, {cat: e1})[cat]
        ma = model.predict_with_embeddings(image, {cat: ea})[cat]
        print(f"\n  neighbours used: {info['neighbors']}")
        describe("alpha=0.0 (pure query)", m0)
        describe(f"alpha={args.alpha} (blended)", ma)
        describe("alpha=1.0 (pure centroid)", m1)
        ok_blend = not np.array_equal(m0, m1)

    print("\n" + "=" * 62)
    print(f"  1. SANITY    real mask non-empty and not the whole image : {ok_sanity}")
    print(f"  2. INJECTION real vs nonsense differ                     : {ok_inject}")
    print(f"  3. BLENDING  alpha=0 vs alpha=1 differ                   : {ok_blend}")
    print("=" * 62)
    if not ok_inject:
        print("\nSTOP. The injected embedding is not reaching the forward pass, so every")
        print("approach would score identically. Do not run the sweep. Inspect")
        print("catseg.py's _locate_predictor_module() path against the real object graph.")
        sys.exit(1)
    if ok_blend is False:
        print("\nSTOP. Injection works but blending does not change the output -- the alpha")
        print("sweep would return a flat curve. Check the cache shape above.")
        sys.exit(1)
    print("\nAll checks passed. Proceed to the dry run.")


if __name__ == "__main__":
    main()
