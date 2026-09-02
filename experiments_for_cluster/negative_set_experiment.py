"""
Negative-set experiment: same (model, variant, approach) grid as
positive_set_experiment.py, run instead on each category's negative images
(images that do NOT contain it). Reports false-positive rate (fraction of
image pixels predicted positive) instead of IoU, since IoU against an empty
ground truth is 0 by construction -- this is the check that blending/query
transforms aren't hallucinating masks on images where the category is absent.
Checkpointed the same way as positive_set_experiment.py.

Negative sets are large (a category's negative set is "every other image in
the benchmark", e.g. tens of thousands) -- always pass --limit-images for
local verification; the cluster run can drop the limit.

Usage:
    python3 negative_set_experiment.py --model clipseg --limit-categories 5 --limit-images 3
    python3 negative_set_experiment.py --model clipseg   # full-scale
"""
import argparse
import os

import benchmark_data as bd
import models as mdl
import approaches as ap

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIELDNAMES = ["model", "category", "variant", "variant_word", "img_id", "approach", "fpr"]


def run(model_name, approaches_to_run, alpha, weighted, out_path, benchmark_dir,
        limit_categories=None, limit_images=None, device=None):
    bm = bd.load_benchmark(benchmark_dir)

    sclip_name_path = None
    if model_name == "sclip":
        sclip_name_path = os.path.join(os.path.dirname(out_path) or ".", "sclip_name_path.txt")
        bd.build_sclip_name_path(bm, sclip_name_path)
    model = mdl.get_model(model_name, device=device, sclip_name_path=sclip_name_path)

    categories = bm.categories
    if limit_categories:
        categories = categories[:limit_categories]
    print(f"[{model_name}] {len(categories)} eligible categories, approaches={approaches_to_run}")

    f, writer, seen = bd.resumable_csv_writer(out_path, FIELDNAMES, ["model", "category", "img_id"])
    try:
        for cat_name in categories:
            variants = bd.get_variants(cat_name, bm.word_sets)
            img_ids = bm.negative_set.get(cat_name, [])
            if limit_images:
                img_ids = img_ids[:limit_images]

            for img_id in img_ids:
                key = (model_name, cat_name, str(img_id))
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
                    # set), but if it contains a sibling sharing cat_name's hyper word
                    # (e.g. "sofa" while testing "chair"'s hyper "seat"), the hyper query
                    # has a legitimate target here -- a correct segmentation isn't a false
                    # positive, so skip scoring this row rather than penalize it.
                    if vname == "hyper" and bd.hyper_variant_contaminated(bm, entry, cat_name):
                        continue

                    tag_to_embedding = {}
                    for approach in approaches_to_run:
                        emb = ap.approach_embedding(
                            approach, cat_name, word, model,
                            alpha=alpha, weighted=weighted, class_name_list=bm.categories,
                        )
                        if emb is None:
                            continue
                        tag_to_embedding[f"{vname}::{approach}"] = emb
                    if not tag_to_embedding:
                        continue

                    masks = mdl.predict_masks_for_tags(model, model_name, image, cat_name, tag_to_embedding)
                    for tag, mask in masks.items():
                        vname_, approach = tag.split("::")
                        rows.append({
                            "model": model_name, "category": cat_name, "variant": vname_,
                            "variant_word": word, "img_id": str(img_id), "approach": approach,
                            "fpr": bd.false_positive_rate(mask),
                        })

                for row in rows:
                    writer.writerow(row)
                seen.add(key)
                f.flush()
            print(f"  {cat_name}: done ({len(img_ids)} images)")
    finally:
        f.close()
    print(f"[{model_name}] detail written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=mdl.SUPPORTED_MODELS)
    parser.add_argument("--approaches", default=",".join(ap.ALL_APPROACHES),
                         help=f"comma-separated subset of {ap.ALL_APPROACHES}")
    parser.add_argument("--alpha", type=float, default=ap.DEFAULT_ALPHA)
    parser.add_argument("--weighted", action="store_true",
                         help="use the cosine-similarity-weighted centroid for 'ours' instead of the unweighted mean")
    parser.add_argument("--benchmark-dir", default=None, help="defaults to ../../benchmark (or $BENCHMARK_DIR)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=5,
                         help="negative sets are huge (every other image in the benchmark); "
                              "default caps at 5 per category for sane local/default runs -- pass -1 for full scale")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    approaches_to_run = [a.strip() for a in args.approaches.split(",") if a.strip()]
    limit_images = None if args.limit_images is not None and args.limit_images < 0 else args.limit_images
    out_path = os.path.join(args.out_dir, f"negative_set_experiment_{args.model}.csv")
    summary_path = os.path.join(args.out_dir, f"negative_set_experiment_{args.model}_summary.csv")

    run(args.model, approaches_to_run, args.alpha, args.weighted, out_path, args.benchmark_dir,
        limit_categories=args.limit_categories, limit_images=limit_images, device=args.device)

    bd.summarize_csv(out_path, ["model", "approach", "variant"], "fpr", summary_path)
    print(f"summary written to {summary_path}")
