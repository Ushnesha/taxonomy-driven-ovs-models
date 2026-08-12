import argparse
import os

from experiment_common import (
    ALPHA, BUILDERS, FETCHERS, RESULTS_DIR, all_ids_for, blend_variants_for_class, build_negative_set,
    eligible_synset, resumable_writer, run_segmentation_batch,
)
import expanded_benchmark_helpers as bm_hp

OUT_PATH = os.path.join(RESULTS_DIR, "negative_set_experiment.csv")
FIELDNAMES = ["dataset", "category", "variant", "variant_word", "img_ref", "method", "fp_rate"]


def fp_rate(mask):
    return float(mask.sum()) / mask.size


def run(alpha=ALPHA, out_path=OUT_PATH, limit_categories=None, limit_images=None):
    f, writer, seen = resumable_writer(out_path, FIELDNAMES, ["dataset", "category", "img_ref"])
    try:
        for dataset_name, builder in BUILDERS.items():
            ctx, positive_set = builder()
            all_ids = all_ids_for(dataset_name, ctx)
            negative_set = build_negative_set(positive_set, all_ids)

            categories = [c for c in sorted(positive_set) if eligible_synset(c) is not None]
            if limit_categories:
                categories = categories[:limit_categories]
            print(f"{dataset_name}: {len(categories)} categories")

            for cat_name in categories:
                synset = eligible_synset(cat_name)
                blend_info = blend_variants_for_class(cat_name, synset)
                img_refs = negative_set.get(cat_name, [])
                if limit_images:
                    img_refs = img_refs[:limit_images]

                for img_ref in img_refs:
                    key = (dataset_name, cat_name, str(img_ref))
                    if key in seen:
                        continue
                    try:
                        image, _ = FETCHERS[dataset_name](ctx, cat_name, img_ref)
                    except Exception as e:
                        print(f"  skip {dataset_name}/{cat_name}/{img_ref}: {e}")
                        continue
                    if image is None:
                        continue

                    embeddings, tags = [], []
                    for vname, info in blend_info.items():
                        embeddings.append(info["query_emb"])
                        tags.append((vname, info["word"], "baseline"))
                        if info["centroid_uw"] is not None:
                            embeddings.append(bm_hp.blend_embedding(info["query_emb"], info["centroid_uw"], alpha))
                            tags.append((vname, info["word"], "alpha_blend"))
                            embeddings.append(bm_hp.blend_embedding(info["query_emb"], info["centroid_w"], alpha))
                            tags.append((vname, info["word"], "weighted_blend"))

                    masks = run_segmentation_batch(image, embeddings)
                    for i, (vname, word, method) in enumerate(tags):
                        writer.writerow({
                            "dataset": dataset_name, "category": cat_name, "variant": vname,
                            "variant_word": word, "img_ref": str(img_ref), "method": method,
                            "fp_rate": fp_rate(masks[i]),
                        })
                    seen.add(key)
                    f.flush()
                print(f"  {cat_name}: done ({len(img_refs)} images)")
    finally:
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()
    run(alpha=args.alpha, out_path=args.out, limit_categories=args.limit_categories, limit_images=args.limit_images)
