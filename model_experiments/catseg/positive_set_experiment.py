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

import benchmark_data as bd
import approaches as ap
from catseg import CATSegModel

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIELDNAMES = ["category", "variant", "variant_word", "img_id", "approach", "iou"]


def predict_masks_for_tags(model, cat_name, image, tag_to_embedding, threshold=0.5):
    """CAT-Seg's predict_with_embeddings() only accepts one embedding per call, keyed by a
    real class name (its joint-softmax interface swaps one vocabulary row in per call --
    see catseg.py's module docstring), unlike CLIPSeg/GroupViT's arbitrary-tag batching. So
    this issues one call per tag, all keyed by `cat_name` (the actual class every tag's
    embedding is a variant/approach computed for), re-keying the single-entry result back
    onto the tag."""
    masks = {}
    for tag, emb in tag_to_embedding.items():
        result = model.predict_with_embeddings(image, {cat_name: emb}, threshold=threshold)
        masks[tag] = result[cat_name]
    return masks


def run(alpha, weighted, out_path, benchmark_dir, config=None, weights=None, catseg_path=None,
        limit_categories=None, limit_images=None, device=None):
    bm = bd.load_benchmark(benchmark_dir)

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
    print(f"{len(categories)} eligible categories, approaches={ap.ALL_APPROACHES}")

    f, writer, seen = bd.resumable_csv_writer(out_path, FIELDNAMES, ["category", "img_id"])
    try:
        for cat_name in categories:
            variants = bd.get_variants(cat_name, bm.word_sets)
            img_ids = bm.positive_set.get(cat_name, [])
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

    out_path = os.path.join(args.out_dir, "positive_set_experiment_catseg.csv")
    summary_path = os.path.join(args.out_dir, "positive_set_experiment_catseg_summary.csv")

    run(args.alpha, args.weighted, out_path, args.benchmark_dir,
        config=args.config, weights=args.weights, catseg_path=args.catseg_path,
        limit_categories=args.limit_categories, limit_images=args.limit_images, device=args.device)

    bd.summarize_csv(out_path, ["approach", "variant"], "iou", summary_path)
    print(f"summary written to {summary_path}")
