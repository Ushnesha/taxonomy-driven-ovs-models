"""Prompt-template ablation (paper's E2, extended): compares the bare-word baseline
against the "Image of a {word}" template, across all 4 linguistic variants and every
model in models.SUPPORTED_MODELS, isolating the effect of prompt phrasing from the
query-transform methods approaches.py covers (this isn't one of the 5 dispatched
approaches -- there's no WordNet blending or descriptor generation here, just a fixed
alternative wording of the same query word).

Runs against the same `../../benchmark/` (LVIS/ADE20K/PascalVOC) positive_set_experiment.py
and negative_set_experiment.py use, via benchmark_data.py, and applies the same
hypernym-sibling GT mask merging (benchmark_data.decode_gt_mask_for_variant) for the
"hyper" variant.

Usage:
    python3 prompt_template_experiment.py --model clipseg --limit-categories 5 --limit-images 3
    python3 prompt_template_experiment.py --model sclip   # full-scale, needs the sclip venv
                                                           # (see ../../models_reference/sclip/setup_env.sh)
"""
import argparse
import os

import benchmark_data as bd
import models as mdl

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIELDNAMES = ["model", "category", "variant", "variant_word", "img_id", "method", "iou"]


def templated_embedding(model, word):
    return model.get_text_embedding(f"Image of a {word}", desc=True)


def run(model_name, out_path, benchmark_dir, limit_categories=None, limit_images=None, device=None):
    bm = bd.load_benchmark(benchmark_dir)

    sclip_name_path = None
    if model_name == "sclip":
        sclip_name_path = os.path.join(os.path.dirname(out_path) or ".", "sclip_name_path.txt")
        bd.build_sclip_name_path(bm, sclip_name_path)
    model = mdl.get_model(model_name, device=device, sclip_name_path=sclip_name_path)

    categories = bm.categories
    if limit_categories:
        categories = categories[:limit_categories]
    print(f"[{model_name}] {len(categories)} eligible categories")

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
                    gt = bd.decode_gt_mask_for_variant(bm, entry, cat_name, vname)
                    tag_to_embedding = {
                        f"{vname}::baseline": model.get_text_embedding(word, desc=True),
                        f"{vname}::image_of_a": templated_embedding(model, word),
                    }
                    masks = mdl.predict_masks_for_tags(model, model_name, image, cat_name, tag_to_embedding)
                    for tag, mask in masks.items():
                        vname_, method = tag.split("::")
                        rows.append({
                            "model": model_name, "category": cat_name, "variant": vname_,
                            "variant_word": word, "img_id": str(img_id), "method": method,
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
    parser.add_argument("--benchmark-dir", default=None, help="defaults to ../../benchmark (or $BENCHMARK_DIR)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    out_path = os.path.join(args.out_dir, f"prompt_template_experiment_{args.model}.csv")
    summary_path = os.path.join(args.out_dir, f"prompt_template_experiment_{args.model}_summary.csv")

    run(args.model, out_path, args.benchmark_dir,
        limit_categories=args.limit_categories, limit_images=args.limit_images, device=args.device)

    bd.summarize_csv(out_path, ["model", "method", "variant"], "iou", summary_path)
    print(f"summary written to {summary_path}")
