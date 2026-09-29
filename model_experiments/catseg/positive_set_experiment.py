"""Positive-set experiment for CAT-Seg (paper's E3): for every eligible benchmark
category, across all 4 linguistic variants (orig/syn/hypo/hyper) and all 5 approaches
(baseline/ours/shine/waffleclip/llm_descriptor), measures IoU against ground truth on the
category's positive images (images that actually contain it). Checkpointed -- safe to
interrupt and resume.

Single-model version of ../../experiments_for_cluster/positive_set_experiment.py, self-
contained in this folder (no --model flag needed, no dependency on the rest of the repo).
--alpha should be the value derived by alpha_value_experiment.py's COCO sweep for CAT-Seg.

IMPORTANT: see catseg.py's module docstring -- the real CAT-Seg production-mode path has
not been run end-to-end locally (no CUDA/detectron2 here). --config/--weights are REQUIRED
at full scale (see argparse below) so this script can never silently fall back to CLIPSeg
and report those numbers as CAT-Seg's.

CAT-Seg performance note: like SCLIP (unlike clipseg/groupvit), its joint-softmax interface
only accepts one embedding per class per call (see catseg.py's module docstring), so
predict_masks_for_tags below issues one predict_with_embeddings call per (variant,
approach) tag instead of one batched call -- expect this to run substantially slower than
clipseg/groupvit.

Usage:
    python3 positive_set_experiment.py --alpha 0.71 \\
        --config ./CAT-Seg/configs/vitl_336.yaml --weights /path/to/model_large.pth
    python3 positive_set_experiment.py --alpha 0.71 --limit-categories 5 --limit-images 3
        # (baseline/CLIPSeg fallback allowed here since --limit implies this isn't a real run)
"""
import argparse
import os
import random

import benchmark_data as bd
import approaches as ap
from catseg import CATSegModel

SAMPLE_SEED = 42  # --limit-images takes a SEEDED RANDOM sample, not the first N: the
                  # benchmark JSON's image order is not random, so img_ids[:N] biases the
                  # subset. Seeded per (seed, category) so it is reproducible AND draws the
                  # same images as the clipseg/sclip/groupvit runs, making the cross-model
                  # comparison paired at the image level.

THRESHOLD = 0.0   # CAT-Seg classifies each pixel by ARGMAX over its whole joint softmax;
                  # its own evaluation applies no confidence floor. With a 326-class
                  # vocabulary the winning softmax probability is routinely well below 0.5,
                  # so the previous default of 0.5 would have blanked most masks and made
                  # every approach look equally bad. This is exactly the bug that invalidated
                  # a full SCLIP run (prob_thd 0.5 vs its published 0.1) -- keep 0.0 unless
                  # you have a specific reason, and record --threshold in the run log.

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIELDNAMES = ["category", "variant", "variant_word", "img_id", "approach", "iou"]


def predict_masks_for_tags(model, cat_name, image, tag_to_embedding, threshold=None):
    """CAT-Seg's predict_with_embeddings() only accepts one embedding per call, keyed by a
    real class name (its joint-softmax interface swaps one vocabulary row in per call --
    see catseg.py's module docstring), unlike CLIPSeg/GroupViT's arbitrary-tag batching. So
    this issues one call per tag, all keyed by `cat_name` (the actual class every tag's
    embedding is a variant/approach computed for), re-keying the single-entry result back
    onto the tag."""
    if threshold is None:
        threshold = THRESHOLD
    masks = {}
    for tag, emb in tag_to_embedding.items():
        result = model.predict_with_embeddings(image, {cat_name: emb}, threshold=threshold)
        masks[tag] = result[cat_name]
    return masks


def run(alpha, weighted, out_path, benchmark_dir, config=None, weights=None, catseg_path=None,
        exclude_sources=(), sample_seed=SAMPLE_SEED,
        category_shard=None, also_seen_from=(),
        limit_categories=None, limit_images=None, device=None):
    bm = bd.load_benchmark(benchmark_dir, exclude_sources=exclude_sources)

    # CAT-Seg needs a fixed vocabulary at construction time (see catseg.py's module
    # docstring) covering every category predict_with_embeddings will be asked to resolve.
    catseg_class_json = os.path.join(os.path.dirname(out_path) or ".", "catseg_class_names.json")
    bd.build_catseg_class_json(bm, catseg_class_json)

    print("Loading CAT-Seg...", flush=True)
    model = CATSegModel(device=device, config=config, weights=weights, catseg_path=catseg_path,
                         class_name_path=catseg_class_json)

    categories = bm.categories
    if limit_categories:
        categories = categories[:limit_categories]
    if category_shard:
        i, n = category_shard
        # Every n-th category, offset i. Interleaved rather than
        # contiguous so each shard gets a similar mix of easy/hard
        # and large/small categories -- contiguous blocks would make
        # one shard finish hours before another.
        categories = categories[i::n]
        print(f"shard {i}/{n}: {len(categories)} categories")
    print(f"{len(categories)} eligible categories, approaches={ap.ALL_APPROACHES}")

    f, writer, seen = bd.resumable_csv_writer(
        out_path, FIELDNAMES, ["category", "img_id"],
        also_seen_from=also_seen_from)
    try:
        for cat_name in categories:
            variants = bd.get_variants(cat_name, bm.word_sets)
            img_ids = bm.positive_set.get(cat_name, [])
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
                base_gt = bd.decode_gt_mask(entry, cat_name)
                if base_gt is None or base_gt.sum() == 0:
                    continue

                rows = []
                for vname, word in variants.items():
                    # Shared-hypernym images (e.g. "seat" for both "chair" and "sofa")
                    # score against the union of every present sibling's mask -- see
                    # benchmark_data.decode_gt_mask_for_variant's docstring.
                    gt = bd.decode_gt_mask_for_variant(bm, entry, cat_name, vname)

                    tag_to_embedding = {}
                    for approach in ap.ALL_APPROACHES:
                        emb = ap.approach_embedding(
                            approach, cat_name, word, model,
                            alpha=alpha, weighted=weighted, class_name_list=bm.categories,
                        )
                        if emb is None:
                            continue
                        tag_to_embedding[f"{vname}::{approach}"] = emb
                    if not tag_to_embedding:
                        continue

                    masks = predict_masks_for_tags(model, cat_name, image, tag_to_embedding)
                    for tag, mask in masks.items():
                        vname_, approach = tag.split("::")
                        rows.append({
                            "category": cat_name, "variant": vname_, "variant_word": word,
                            "img_id": str(img_id), "approach": approach,
                            "iou": bd.compute_iou(mask, gt),
                        })

                for row in rows:
                    writer.writerow(row)
                seen.add(key)
                f.flush()
            print(f"  {cat_name}: done ({len(img_ids)} images)")
    finally:
        f.close()
    print(f"detail written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True,
                         help="alpha for 'ours' blending -- use the value alpha_value_experiment.py found for CAT-Seg")
    parser.add_argument("--weighted", action="store_true",
                         help="use the cosine-similarity-weighted centroid for 'ours' instead of the unweighted mean")
    parser.add_argument("--benchmark-dir", default=None, help="defaults to ../../../benchmark (or $BENCHMARK_DIR)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default=None,
                         help="CAT-Seg Detectron2 config, e.g. ./CAT-Seg/configs/vitl_336.yaml. "
                              "Required (with --weights) for a full-scale run -- see catseg.py.")
    parser.add_argument("--weights", default=None, help="CAT-Seg checkpoint (.pth).")
    parser.add_argument("--catseg-path", default=None,
                         help="Path to the vendored CAT-Seg repo checkout (default: ./CAT-Seg).")
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                         help="confidence floor on CAT-Seg's joint-softmax argmax. Default 0.0 "
                              "= pure argmax, matching CAT-Seg's own evaluation.")
    parser.add_argument("--exclude-sources", default="",
                         help="comma-separated img_src values to drop entirely, e.g. 'ade20k'. "
                              "Use when a source's HF row indices no longer align with the "
                              "frozen benchmark. Excluding a WHOLE source is the only safe "
                              "option -- never filter per-image on a size check.")
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED,
                         help="seed for the --limit-images random sample; keep at 42 to draw "
                              "the same images as the other models' runs")
    parser.add_argument("--category-shard", default=None,
                         help="'I/N' -- process only every N-th category starting at I, for a "
                              "SLURM job array. Each shard writes its own detail CSV "
                              "(..._shardIofN.csv); merge them afterwards with merge_shards.py. "
                              "CAT-Seg cannot batch (one forward pass per variant x approach), "
                              "so sharding is the only way to use more than one GPU.")
    parser.add_argument("--also-seen-from", default="",
                         help="comma-separated existing detail CSVs to treat as already-done "
                              "WITHOUT writing to them. Point shards at a previous sequential "
                              "run's CSV so its finished categories are not recomputed.")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    parser.add_argument("--allow-baseline", action="store_true",
                         help="Allow CLIPSeg-fallback baseline mode even without --limit-categories/"
                              "--limit-images caps -- i.e. permit a 'full-scale' run that is NOT real "
                              "CAT-Seg. Off by default so this can never silently happen.")
    args = parser.parse_args()

    is_full_scale = not args.limit_categories and not args.limit_images
    have_real_model = args.config is not None and args.weights is not None
    if is_full_scale and not have_real_model and not args.allow_baseline:
        parser.error(
            "--config and --weights are required for a full-scale run (no --limit-categories/"
            "--limit-images) -- without them this would silently fall back to CLIPSeg and "
            "report those numbers as CAT-Seg's. Pass --limit-categories/--limit-images for a "
            "small local pipeline check instead, or --allow-baseline to override."
        )

    from provenance import RunMeta, count_csv_rows
    meta = RunMeta(args.out_dir, args, __file__)

    THRESHOLD = args.threshold

    shard = None
    if args.category_shard:
        try:
            i_s, n_s = args.category_shard.split("/")
            shard = (int(i_s), int(n_s))
        except ValueError:
            parser.error("--category-shard must look like '3/8'")
        if not (0 <= shard[0] < shard[1]):
            parser.error(f"--category-shard out of range: {args.category_shard}")
    # A distinct file per shard: two processes appending to one CSV interleave rows.
    suffix = f"_shard{shard[0]}of{shard[1]}" if shard else ""
    out_path = os.path.join(args.out_dir, f"positive_set_experiment_catseg{suffix}.csv")
    # Shard suffix here too: without it every array task would race to overwrite
    # one summary file with its own partial view. The real summary comes from
    # merge_shards.py once all shards are done.
    summary_path = os.path.join(
        args.out_dir, f"positive_set_experiment_catseg{suffix}_summary.csv")

    exclude_sources = tuple(x.strip() for x in args.exclude_sources.split(",") if x.strip())
    run(args.alpha, args.weighted, out_path, args.benchmark_dir,
        config=args.config, weights=args.weights, catseg_path=args.catseg_path,
        exclude_sources=exclude_sources, sample_seed=args.sample_seed,
        category_shard=shard,
        also_seen_from=tuple(x.strip() for x in args.also_seen_from.split(",") if x.strip()),
        limit_categories=args.limit_categories, limit_images=args.limit_images, device=args.device)

    bd.summarize_csv(out_path, ["approach", "variant"], "iou", summary_path)
    print(f"summary written to {summary_path}")
    meta.finish(detail_csv=out_path, summary_csv=summary_path,
                n_detail_rows=count_csv_rows(out_path))
