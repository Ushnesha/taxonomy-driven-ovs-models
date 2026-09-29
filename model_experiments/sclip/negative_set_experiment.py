"""Negative-set experiment for SCLIP: same (variant, approach) grid as
positive_set_experiment.py, run instead on each category's negative images (images that do
NOT contain it). Reports false-positive rate (fraction of image pixels predicted positive)
instead of IoU, since IoU against an empty ground truth is 0 by construction -- this is the
check that blending/query transforms aren't hallucinating masks. Checkpointed the same way
as positive_set_experiment.py.

Negative sets are large (a category's negative set is "every other image in the
benchmark") -- always pass --limit-images for local verification; the cluster run can drop
the limit. See positive_set_experiment.py's docstring for the SCLIP-specific performance
note (one predict_with_embeddings call per tag, not batched).

Usage:
    python3 negative_set_experiment.py --alpha 0.71 --limit-categories 5 --limit-images 3
    python3 negative_set_experiment.py --alpha 0.71   # full-scale
"""
import argparse
import os
import random

import benchmark_data as bd
import approaches as ap
from sclip import SClipModel

# SCLIP's prob_thd is a floor on the JOINT softmax over every benchmark class, not a
# per-prompt binary sigmoid like CLIPSeg's -- so its correct value depends on HOW MANY
# classes are in the vocabulary, and cannot be carried over from another configuration.
THRESHOLD = 0.0   # prob_thd. NOT SCLIP's published 0.1 -- that value is calibrated for
                  # COCO-Object's 81 classes, and this benchmark's vocabulary is 327
                  # classes / 670 queries. The winning softmax probability shrinks with
                  # the query count, so a 0.1 floor suppresses EVERY pixel. Measured with
                  # check_native_iou.py on 16 (category, image) pairs:
                  #     prob_thd  0.0    0.005  0.01   0.02   0.05   0.1
                  #     mean IoU  0.1639 0.1639 0.1589 0.1307 0.0194 0.0000
                  # 0.0 = pure argmax, which is correct here because the vocabulary has a
                  # background class (line 0 of the name file) to absorb "none of these" --
                  # suppression is that class's job, not a confidence floor's.
                  #
                  # The 0.1 runs produced walls of 0.0 IoU and ~1e-5 FPR. Any result
                  # generated before this line changed is invalid.

SAMPLE_SEED = 42  # --limit-images takes a SEEDED RANDOM sample, not the first N: the
                  # benchmark JSON's image order is not random, so img_ids[:N] biases the
                  # subset. Seeded per (seed, category) so it is reproducible and each
                  # category's draw is independent of the others.

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIELDNAMES = ["category", "variant", "variant_word", "img_id", "approach", "fpr"]


def predict_masks_for_tags(model, cat_name, image, tag_to_embedding, threshold=None):
    """See positive_set_experiment.py's version of this function -- SCLIP needs one call
    per tag, all keyed by the real class name `cat_name`."""
    if threshold is None:
        threshold = THRESHOLD
    masks = {}
    for tag, emb in tag_to_embedding.items():
        result = model.predict_with_embeddings(image, {cat_name: emb}, threshold=threshold)
        masks[tag] = result[cat_name]
    return masks


def run(alpha, weighted, out_path, benchmark_dir, exclude_sources=(), limit_categories=None,
        limit_images=None, device=None, sample_seed=SAMPLE_SEED,
        category_shard=None, also_seen_from=()):
    bm = bd.load_benchmark(benchmark_dir, exclude_sources=exclude_sources)

    sclip_name_path = os.path.join(os.path.dirname(out_path) or ".", "sclip_name_path.txt")
    bd.build_sclip_name_path(bm, sclip_name_path)

    print("Loading SCLIP (ViT-B/16)...", flush=True)
    model = SClipModel(device=device, name_path=sclip_name_path)

    categories = bm.categories
    if limit_categories:
        categories = categories[:limit_categories]
    if category_shard:
        i, n = category_shard
        # Interleaved, not contiguous: each shard then gets a similar mix of
        # easy/hard and large/small categories, so they finish together.
        categories = categories[i::n]
        print(f"shard {i}/{n}: {len(categories)} categories")
    print(f"{len(categories)} eligible categories, approaches={ap.ALL_APPROACHES}")

    unresolved_counts = {}  # {(approaches that returned None,): times it happened}
    f, writer, seen = bd.resumable_csv_writer(
        out_path, FIELDNAMES, ["category", "img_id"], also_seen_from=also_seen_from)
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

                    masks = predict_masks_for_tags(model, cat_name, image, tag_to_embedding)
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
                         help="alpha for 'ours' blending -- use the value alpha_value_experiment.py found for SCLIP")
    parser.add_argument("--weighted", action="store_true",
                         help="use the cosine-similarity-weighted centroid for 'ours' instead of the unweighted mean")
    parser.add_argument("--benchmark-dir", default=None, help="defaults to ../../../benchmark (or $BENCHMARK_DIR)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=5,
                         help="negative sets are huge (every other image in the benchmark); "
                              "default caps at 5 per category for sane local/default runs -- pass -1 for full scale")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                         help="SCLIP prob_thd (joint-softmax floor). Default 0.1 = SCLIP's own "
                              "published value. Do NOT use CLIPSeg's 0.5.")
    parser.add_argument("--exclude-sources", default="",
                         help="comma-separated img_src values to drop entirely, e.g. 'ade20k'. "
                              "Use when a source's HF row indices no longer align with the "
                              "frozen benchmark. Excluding a WHOLE source is the only safe "
                              "option -- never filter per-image on a size check.")
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED,
                         help="seed for the --limit-images random sample")
    parser.add_argument("--category-shard", default=None,
                         help="'I/N' -- process only every N-th category starting at I, for a "
                              "SLURM job array. Each shard writes its own detail CSV "
                              "(..._shardIofN.csv); merge with merge_shards.py afterwards. "
                              "SCLIP cannot batch (one joint forward pass per variant x "
                              "approach), so sharding is the only way to use several GPUs.")
    parser.add_argument("--also-seen-from", default="",
                         help="comma-separated existing detail CSVs to treat as already-done "
                              "WITHOUT writing to them, e.g. a previous sequential run.")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()
    THRESHOLD = args.threshold

    from provenance import RunMeta, count_csv_rows
    meta = RunMeta(args.out_dir, args, __file__)

    limit_images = None if args.limit_images is not None and args.limit_images < 0 else args.limit_images
    shard = None
    if args.category_shard:
        try:
            i_s, n_s = args.category_shard.split("/")
            shard = (int(i_s), int(n_s))
        except ValueError:
            parser.error("--category-shard must look like '3/8'")
        if not (0 <= shard[0] < shard[1]):
            parser.error(f"--category-shard out of range: {args.category_shard}")
    # Distinct files per shard: two processes appending to one CSV interleave rows, and a
    # shared summary path would have every task racing to overwrite a partial view.
    suffix = f"_shard{shard[0]}of{shard[1]}" if shard else ""
    out_path = os.path.join(args.out_dir, f"negative_set_experiment_sclip{suffix}.csv")
    summary_path = os.path.join(
        args.out_dir, f"negative_set_experiment_sclip{suffix}_summary.csv")

    exclude_sources = tuple(x.strip() for x in args.exclude_sources.split(",") if x.strip())
    run(args.alpha, args.weighted, out_path, args.benchmark_dir, exclude_sources=exclude_sources,
        category_shard=shard,
        also_seen_from=tuple(x.strip() for x in args.also_seen_from.split(",") if x.strip()),
        limit_categories=args.limit_categories, limit_images=limit_images, sample_seed=args.sample_seed, device=args.device)

    bd.summarize_csv(out_path, ["approach", "variant"], "fpr", summary_path)
    print(f"summary written to {summary_path}")
    meta.finish(detail_csv=out_path, summary_csv=summary_path,
                n_detail_rows=count_csv_rows(out_path))
