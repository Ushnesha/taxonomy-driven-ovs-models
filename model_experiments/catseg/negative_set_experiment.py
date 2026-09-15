"""Negative-set experiment for CAT-Seg: same (variant, approach) grid as
positive_set_experiment.py, run instead on each category's negative images (images that do
NOT contain it). Reports false-positive rate (fraction of image pixels predicted positive)
instead of IoU, since IoU against an empty ground truth is 0 by construction -- this is the
check that blending/query transforms aren't hallucinating masks. Checkpointed the same way
as positive_set_experiment.py.

IMPORTANT: see catseg.py's module docstring -- the real CAT-Seg production-mode path has
not been run end-to-end locally. --config/--weights are REQUIRED at full scale (no
--limit-categories and --limit-images -1) so this script can never silently fall back to
CLIPSeg and report those numbers as CAT-Seg's.

Negative sets are large (a category's negative set is "every other image in the
benchmark") -- always pass --limit-images for local verification; the cluster run can drop
the limit. See positive_set_experiment.py's docstring for the CAT-Seg-specific performance
note (one predict_with_embeddings call per tag, not batched).

Usage:
    python3 negative_set_experiment.py --alpha 0.71 --limit-categories 5 --limit-images 3
    python3 negative_set_experiment.py --alpha 0.71 --limit-images -1 \\
        --config ./CAT-Seg/configs/vitl_336.yaml --weights /path/to/model_large.pth  # full-scale
"""
import argparse
import os

import benchmark_data as bd
import approaches as ap
from catseg import CATSegModel

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIELDNAMES = ["category", "variant", "variant_word", "img_id", "approach", "fpr"]


def predict_masks_for_tags(model, cat_name, image, tag_to_embedding, threshold=0.5):
    """See positive_set_experiment.py's version of this function -- CAT-Seg needs one call
    per tag, all keyed by the real class name `cat_name`."""
    masks = {}
    for tag, emb in tag_to_embedding.items():
        result = model.predict_with_embeddings(image, {cat_name: emb}, threshold=threshold)
        masks[tag] = result[cat_name]
    return masks


def run(alpha, weighted, out_path, benchmark_dir, config=None, weights=None, catseg_path=None,
        limit_categories=None, limit_images=None, device=None):
    bm = bd.load_benchmark(benchmark_dir)

    catseg_class_json = os.path.join(os.path.dirname(out_path) or ".", "catseg_class_names.json")
    bd.build_catseg_class_json(bm, catseg_class_json)

    print("Loading CAT-Seg...", flush=True)
    model = CATSegModel(device=device, config=config, weights=weights, catseg_path=catseg_path,
                         class_name_path=catseg_class_json)

    categories = bm.categories
    if limit_categories:
        categories = categories[:limit_categories]
    print(f"{len(categories)} eligible categories, approaches={ap.ALL_APPROACHES}")

    f, writer, seen = bd.resumable_csv_writer(out_path, FIELDNAMES, ["category", "img_id"])
    try:
        for cat_name in categories:
            variants = bd.get_variants(cat_name, bm.word_sets)
            img_ids = bm.negative_set.get(cat_name, [])
            if limit_images:
                img_ids = img_ids[:limit_images]

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
                            "fpr": bd.false_positive_rate(mask),
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
    parser.add_argument("--limit-images", type=int, default=5,
                         help="negative sets are huge (every other image in the benchmark); "
                              "default caps at 5 per category for sane local/default runs -- pass -1 for full scale")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    parser.add_argument("--allow-baseline", action="store_true",
                         help="Allow CLIPSeg-fallback baseline mode even at full scale -- i.e. permit a "
                              "'full-scale' run that is NOT real CAT-Seg. Off by default so this can "
                              "never silently happen.")
    args = parser.parse_args()

    limit_images = None if args.limit_images is not None and args.limit_images < 0 else args.limit_images
    is_full_scale = not args.limit_categories and limit_images is None
    have_real_model = args.config is not None and args.weights is not None
    if is_full_scale and not have_real_model and not args.allow_baseline:
        parser.error(
            "--config and --weights are required for a full-scale run (no --limit-categories, "
            "--limit-images -1) -- without them this would silently fall back to CLIPSeg and "
            "report those numbers as CAT-Seg's. Pass --limit-categories/--limit-images for a "
            "small local pipeline check instead, or --allow-baseline to override."
        )

    out_path = os.path.join(args.out_dir, "negative_set_experiment_catseg.csv")
    summary_path = os.path.join(args.out_dir, "negative_set_experiment_catseg_summary.csv")

    run(args.alpha, args.weighted, out_path, args.benchmark_dir,
        config=args.config, weights=args.weights, catseg_path=args.catseg_path,
        limit_categories=args.limit_categories, limit_images=limit_images, device=args.device)

    bd.summarize_csv(out_path, ["approach", "variant"], "fpr", summary_path)
    print(f"summary written to {summary_path}")
