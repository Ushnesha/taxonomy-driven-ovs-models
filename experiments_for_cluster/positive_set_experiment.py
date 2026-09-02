"""
Positive-set experiment (paper's E3, generalized beyond CLIPSeg): for one
model and every eligible benchmark category, across all 4 linguistic variants
(orig/syn/hypo/hyper) and all 5 approaches (baseline/ours/shine/waffleclip/
llm_descriptor), measures IoU against ground truth on the category's positive
images (images that actually contain it). Checkpointed -- safe to interrupt
and resume.

Usage:
    # small local subset, for verification:
    python3 positive_set_experiment.py --model clipseg --limit-categories 5 --limit-images 3

    # full-scale (cluster) run, one model at a time:
    python3 positive_set_experiment.py --model clipseg
    python3 positive_set_experiment.py --model groupvit
    python3 positive_set_experiment.py --model sclip   # needs the sclip venv, see ../../models_reference/sclip/setup_env.sh

Repeat across all 4 models (see SUPPORTED_MODELS in models.py) to fill the
paper's 4-model x 5-approach matrix; results land in one CSV per model under
--out-dir, distinguished by the "model" column.
"""
import argparse
import os

import benchmark_data as bd
import models as mdl
import approaches as ap

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIELDNAMES = ["model", "category", "variant", "variant_word", "img_id", "approach", "iou"]


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
            img_ids = bm.positive_set.get(cat_name, [])
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
                base_gt = bd.decode_gt_mask(entry, cat_name)
                if base_gt is None or base_gt.sum() == 0:
                    continue

                rows = []
                for vname, word in variants.items():
                    # Shared-hypernym images (e.g. "seat" for both "chair" and "sofa")
                    # score against the union of every present sibling's mask, not just
                    # cat_name's own -- see decode_gt_mask_for_variant's docstring.
                    gt = bd.decode_gt_mask_for_variant(bm, entry, cat_name, vname)

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
                            "iou": bd.compute_iou(mask, gt),
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
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    approaches_to_run = [a.strip() for a in args.approaches.split(",") if a.strip()]
    out_path = os.path.join(args.out_dir, f"positive_set_experiment_{args.model}.csv")
    summary_path = os.path.join(args.out_dir, f"positive_set_experiment_{args.model}_summary.csv")

    run(args.model, approaches_to_run, args.alpha, args.weighted, out_path, args.benchmark_dir,
        limit_categories=args.limit_categories, limit_images=args.limit_images, device=args.device)

    bd.summarize_csv(out_path, ["model", "approach", "variant"], "iou", summary_path)
    print(f"summary written to {summary_path}")
