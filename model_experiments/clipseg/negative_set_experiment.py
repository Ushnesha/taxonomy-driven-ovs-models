"""Negative-set experiment for CLIPSeg: same (variant, approach) grid as
positive_set_experiment.py, run instead on each category's negative images (images that do
NOT contain it). Reports false-positive rate (fraction of image pixels predicted positive)
instead of IoU, since IoU against an empty ground truth is 0 by construction -- this is the
check that blending/query transforms aren't hallucinating masks. Checkpointed the same way
as positive_set_experiment.py.

Negative sets are large (a category's negative set is "every other image in the
benchmark") -- always pass --limit-images for local verification; the cluster run can drop
the limit.

Usage:
    python3 negative_set_experiment.py --alpha 0.71 --limit-categories 5 --limit-images 3
    python3 negative_set_experiment.py --alpha 0.71   # full-scale
"""
import argparse
import os
import random

import benchmark_data as bd
import approaches as ap
from clipseg import CLIPSegModel

SAMPLE_SEED = 42  # --limit-images takes a SEEDED RANDOM sample, not the first N: the
                  # benchmark JSON's image order is not random, so img_ids[:N] biases the
                  # subset. Seeded per (seed, category) so it is reproducible and each
                  # category's draw is independent of the others.

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIELDNAMES = ["category", "variant", "variant_word", "img_id", "approach", "fpr"]


def predict_masks_for_tags(model, image, tag_to_embedding, threshold=0.5):
    if not tag_to_embedding:
        return {}
    return model.predict_with_embeddings(image, tag_to_embedding, threshold=threshold)


def run(alpha, weighted, out_path, benchmark_dir, exclude_sources=(), limit_categories=None, limit_images=None, device=None, sample_seed=SAMPLE_SEED):
    bm = bd.load_benchmark(benchmark_dir, exclude_sources=exclude_sources)
    print("Loading CLIPSeg (CIDAS/clipseg-rd64-refined)...", flush=True)
    model = CLIPSegModel(device=device)

    categories = bm.categories
    if limit_categories:
        categories = categories[:limit_categories]
    print(f"{len(categories)} eligible categories, approaches={ap.ALL_APPROACHES}")

    unresolved_counts = {}  # {(approaches that returned None,): times it happened}
    f, writer, seen = bd.resumable_csv_writer(out_path, FIELDNAMES, ["category", "img_id"])
    try:
        for cat_name in categories:
            variants = bd.get_variants(cat_name, bm.word_sets)
            img_ids = bm.negative_set.get(cat_name, [])
            if limit_images and len(img_ids) > limit_images:
                img_ids = sorted(random.Random(f"{sample_seed}:{cat_name}").sample(
                    img_ids, limit_images))

            for img_id in img_ids:
                key = (cat_name, str(img_id))
                if key in seen:
                    continue

                entry = bm.img_by_id.get(img_id)
                if entry is None:
                    continue
                try:
                    image = bd.fetch_image(entry)
                except Exception as e:
                    print(f"  skip {cat_name}/{img_id}: fetch error {e}")
                    continue

                rows = []
                for vname, word in variants.items():
                    # This image lacks cat_name (that's why it's in cat_name's negative
                    # set), but if it contains a sibling sharing cat_name's hyper word,
                    # the hyper query has a legitimate target here -- a correct
                    # segmentation isn't a false positive, so skip this row rather than
                    # penalize it. See benchmark_data.hyper_variant_contaminated.
                    if vname == "hyper" and bd.hyper_variant_contaminated(bm, entry, cat_name):
                        continue

                    tag_to_embedding = {}
                    unresolved = []
                    for approach in ap.ALL_APPROACHES:
                        emb = ap.approach_embedding(
                            approach, cat_name, word, model,
                            alpha=alpha, weighted=weighted, class_name_list=bm.categories,
                        )
                        if emb is None:
                            unresolved.append(approach)
                            continue
                        tag_to_embedding[f"{vname}::{approach}"] = emb
                    if unresolved:
                        # approach_embedding returns None when the query word has no usable
                        # WordNet synset -- in practice only "ours" does this. The rows for
                        # the approaches that DID resolve are real measurements and are
                        # kept: discarding them would throw away data from four working
                        # methods. The tally is reported at the end so the coverage gap is
                        # visible, and analysis restricts to the paired subset when a
                        # strict 5-way mean comparison is wanted.
                        k = tuple(unresolved)
                        unresolved_counts[k] = unresolved_counts.get(k, 0) + 1
                    if not tag_to_embedding:
                        continue

                    masks = predict_masks_for_tags(model, image, tag_to_embedding)
                    for tag, mask in masks.items():
                        vname_, approach = tag.split("::")
                        rows.append({
                            "category": cat_name, "variant": vname_, "variant_word": word,
                            "img_id": str(img_id), "approach": approach,
                            "fpr": bd.false_positive_rate(mask),
                        })

                for row in rows:
                    writer.writerow(row)
                seen.add(key)
                f.flush()
            print(f"  {cat_name}: done ({len(img_ids)} images)")
    finally:
        f.close()
    if unresolved_counts:
        total = sum(unresolved_counts.values())
        print(f"[coverage] {total} (category, image, variant) cases where an approach could "
              f"not resolve an embedding and wrote no row (other approaches still scored): "
              + ", ".join(f"{'+'.join(k)}={v}" for k, v in sorted(unresolved_counts.items())))
    print(f"detail written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True,
                         help="alpha for 'ours' blending -- use the value alpha_value_experiment.py found for CLIPSeg")
    parser.add_argument("--weighted", action="store_true",
                         help="use the cosine-similarity-weighted centroid for 'ours' instead of the unweighted mean")
    parser.add_argument("--benchmark-dir", default=None, help="defaults to ../../../benchmark (or $BENCHMARK_DIR)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=5,
                         help="negative sets are huge (every other image in the benchmark); "
                              "default caps at 5 per category for sane local/default runs -- pass -1 for full scale")
    parser.add_argument("--exclude-sources", default="",
                         help="comma-separated img_src values to drop entirely, e.g. 'ade20k'. "
                              "Use when a source's HF row indices no longer align with the "
                              "frozen benchmark. Excluding a WHOLE source is the only safe "
                              "option -- never filter per-image on a size check.")
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED,
                         help="seed for the --limit-images random sample")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    from provenance import RunMeta, count_csv_rows
    meta = RunMeta(args.out_dir, args, __file__)

    limit_images = None if args.limit_images is not None and args.limit_images < 0 else args.limit_images
    out_path = os.path.join(args.out_dir, "negative_set_experiment_clipseg.csv")
    summary_path = os.path.join(args.out_dir, "negative_set_experiment_clipseg_summary.csv")

    exclude_sources = tuple(x.strip() for x in args.exclude_sources.split(",") if x.strip())
    run(args.alpha, args.weighted, out_path, args.benchmark_dir, exclude_sources=exclude_sources,
        limit_categories=args.limit_categories, limit_images=limit_images, sample_seed=args.sample_seed, device=args.device)

    bd.summarize_csv(out_path, ["approach", "variant"], "fpr", summary_path)
    print(f"summary written to {summary_path}")
    meta.finish(detail_csv=out_path, summary_csv=summary_path,
                n_detail_rows=count_csv_rows(out_path))
