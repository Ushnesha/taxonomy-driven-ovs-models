"""COCO-only alpha hyperparameter sweep for "Ours" (alpha-blending / weighted-centroid
blending toward the WordNet synset centroid), generalized across every model in
models.SUPPORTED_MODELS (clipseg, groupvit, sclip).

This experiment is separate from positive_set_experiment.py / negative_set_experiment.py's
baseline/ours/shine/waffleclip/llm_descriptor suite and does not touch benchmark_data.py or
approaches.py's descriptor-baseline machinery. Its only job is deriving each model's best
alpha value, and it runs on COCO specifically (not the LVIS/ADE20K/PascalVOC `../../benchmark/`
those two scripts use) because COCO is the calibration dataset: once a model's alpha is found
here (at full scale -- every eligible category, every positive image), that fixed value gets
passed as --alpha to positive_set_experiment.py / negative_set_experiment.py when running
"ours" on the other three datasets.

Ported from for_execution/alpha_value_experiment_coco.py: that script's COCO local-cache image
download, WordNet disambiguation, and coarse/fine alpha grid are preserved near-verbatim (COCO
access + WordNet/centroid math is dataset-agnostic, reused here from expanded_benchmark_helpers.py
and approaches.py rather than re-copied). Two things generalized: (1) --model support, via
models.get_model() / models.predict_masks_for_tags() in place of that script's hand-rolled
CLIPSeg-only embedding/segmentation calls; (2) hypernym-sibling GT mask merging -- COCO's 80
categories can share a hypernym too (e.g. two animals resolving to the same WordNet hypernym),
so the "hyper" variant scores against the union of every present sibling's mask, not just one
category's -- see build_hyper_siblings/get_mask_for_variant below, which mirror
benchmark_data.py's build_hyper_siblings/decode_gt_mask_for_variant for the other 3 datasets.

Performance note: for SCLIP, models.predict_masks_for_tags() issues one predict_with_embeddings
call per (variant, weighted, alpha) tag -- SCLIP's joint-softmax interface only accepts one
embedding per class per call (see models.py's module docstring), so a full alpha grid (up to
~4 variants x 2 weighted x 41 alphas = 328 tags per category per image) means 328 forward passes
per image. This is inherent to SCLIP's interface, not something this script works around --
expect the SCLIP sweep to run substantially slower than clipseg/groupvit at full scale.

Usage:
    # full scale (every eligible COCO category, every one of its positive images) -- the
    # actual intended run, once per model:
    python3 alpha_value_experiment.py --model clipseg
    python3 alpha_value_experiment.py --model groupvit
    python3 alpha_value_experiment.py --model sclip     # needs the sclip venv, see
                                                         # ../../models_reference/sclip/setup_env.sh

    # small local subset, for a quick sanity check only -- NOT how the real sweep should
    # be run/trusted:
    python3 alpha_value_experiment.py --model clipseg --n-categories 10 --n-images 5
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import requests
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import expanded_benchmark_helpers as bm_hp  # noqa: E402
import approaches as ap  # noqa: E402
import models as mdl  # noqa: E402

RESULTS_DIR = os.path.join(THIS_DIR, "results")

SEED = 42
N_CATEGORIES = -1  # -1 = every eligible COCO category (the real, intended run)
N_IMAGES = -1       # -1 = every positive image for that category (the real, intended run)
MIN_IMAGES = 5       # a category needs at least this many positive images to be eligible

ALPHA_GRID_COARSE = [round(x * 0.1, 2) for x in range(11)]          # 0.0, 0.1, ..., 1.0
ALPHA_GRID_FINE = [round(0.50 + 0.01 * i, 2) for i in range(31)]    # 0.50, 0.51, ..., 0.80


# =============================================================================
# COCO access, local-cached (avoids re-downloading val2017 images every run -- matters
# once this is actually run at full scale, unlike expanded_benchmark_helpers.download_image
# which re-fetches over HTTP every call).
# =============================================================================

def get_local_image_path(coco_dir, file_name):
    img_dir = os.path.join(coco_dir, "val2017")
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, file_name)
    if os.path.exists(path):
        return path
    img_id = int(os.path.splitext(file_name)[0])
    r = requests.get(bm_hp.COCO_URL.format(img_id), timeout=30)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def predownload_all_images(coco, coco_dir):
    img_ids = coco.getImgIds()
    print(f"Pre-downloading {len(img_ids)} val2017 images to {coco_dir}/val2017/ ...")
    for i, img_id in enumerate(img_ids):
        file_name = coco.loadImgs([img_id])[0]["file_name"]
        get_local_image_path(coco_dir, file_name)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(img_ids)} images cached locally")
    print("Pre-download complete.")


def build_positive_set(coco):
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    positive_set = defaultdict(set)
    for ann in coco.dataset["annotations"]:
        cat_name = cat_id_to_name.get(ann["category_id"])
        if cat_name:
            positive_set[cat_name].add(ann["image_id"])
    return {cat: sorted(ids) for cat, ids in positive_set.items()}


def get_single_cat_mask(coco, cat_name_to_id, img_id, cat_name):
    cat_id = cat_name_to_id.get(cat_name)
    if cat_id is None:
        return None
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)
    if not anns:
        return None
    img_info = coco.loadImgs([img_id])[0]
    h, w = img_info["height"], img_info["width"]
    gt = np.zeros((h, w), dtype=np.uint8)
    for a in anns:
        gt = np.maximum(gt, coco.annToMask(a))
    return gt


def get_image(coco, coco_dir, img_id):
    img_info = coco.loadImgs([img_id])[0]
    path = get_local_image_path(coco_dir, img_info["file_name"])
    return Image.open(path).convert("RGB")


# =============================================================================
# WordNet variants + hypernym-sibling grouping. COCO-specific: computed live over
# COCO's 80 categories (mirrors benchmark_data.build_hyper_siblings, which does the same
# thing from the other 3 datasets' precomputed word_sets_v2.json).
# =============================================================================

def get_variants(cat_name, synset):
    ws = bm_hp.build_word_sets_from_synset(synset)
    w_s = [bm_hp.to_display_form(w) for w in ws["W_S"]]
    w_s_hp = [bm_hp.to_display_form(w) for w in ws["W_S_Hp"]]
    w_s_hp_he = [bm_hp.to_display_form(w) for w in ws["W_S_Hp_He"]]

    syn_candidates = [w for w in w_s if w.lower() != cat_name.lower()]
    hypo_candidates = [w for w in w_s_hp if w not in w_s]
    hyper_candidates = [w for w in w_s_hp_he if w not in w_s_hp]

    variants = {"orig": cat_name}
    if syn_candidates:
        variants["syn"] = syn_candidates[0]
    if hypo_candidates:
        variants["hypo"] = hypo_candidates[0]
    if hyper_candidates:
        variants["hyper"] = hyper_candidates[0]
    return variants, w_s


def build_hyper_siblings(categories):
    """{category: [category, ...siblings]} -- COCO categories whose "hyper" variant word
    matches. Categories with no shared hypernym map to [category] alone (no behavior
    change for those). See get_mask_for_variant / benchmark_data.decode_gt_mask_for_variant
    for why this matters: an image containing two sibling categories should score a
    "hyper" query against the union of their masks, not just the one the query happened
    to be derived from."""
    by_word = defaultdict(list)
    cat_hyper_word = {}
    for cat in categories:
        synset = ap.eligible_synset(cat)
        hyper = None
        if synset is not None:
            variants, _ = get_variants(cat, synset)
            hyper = variants.get("hyper")
        word = hyper.lower() if hyper else None
        cat_hyper_word[cat] = word
        if word:
            by_word[word].append(cat)
    return {cat: (by_word[word] if word else [cat]) for cat, word in cat_hyper_word.items()}


def get_mask_for_variant(coco, cat_name_to_id, cat_name, img_id, variant_name, hyper_siblings, base_mask=None):
    mask = base_mask if base_mask is not None else get_single_cat_mask(coco, cat_name_to_id, img_id, cat_name)
    if variant_name != "hyper":
        return mask
    for sib in hyper_siblings.get(cat_name, [cat_name]):
        if sib == cat_name:
            continue
        sib_mask = get_single_cat_mask(coco, cat_name_to_id, img_id, sib)
        if sib_mask is not None:
            mask = sib_mask if mask is None else np.maximum(mask, sib_mask)
    return mask


# =============================================================================
# Alpha sweep
# =============================================================================

def pick_categories(positive_set, n=N_CATEGORIES, min_images=MIN_IMAGES, seed=SEED):
    eligible = [
        cat for cat in sorted(positive_set)
        if len(positive_set[cat]) >= min_images and ap.eligible_synset(cat) is not None
    ]
    if n < 0 or len(eligible) <= n:
        return eligible
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(eligible), size=n, replace=False)
    return sorted(eligible[i] for i in idx)


def pick_images(positive_set, cat_name, n=N_IMAGES, seed=SEED):
    ids = list(positive_set[cat_name])
    if n < 0 or len(ids) <= n:
        return ids
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    return ids[:n]


def blend_variants_for_class(cat_name, synset, model, k=ap.DEFAULT_TOPK):
    variants, w_s = get_variants(cat_name, synset)
    info = {}
    for vname, word in variants.items():
        query_emb = ap.embed(model, word)
        candidates = [w for w in w_s if w.lower() != word.lower()]
        neighbors = ap._top_k_neighbors(model, word, candidates, k=k) if candidates else []
        info[vname] = {
            "word": word,
            "query_emb": query_emb,
            "centroid_uw": ap.compute_centroid(neighbors) if neighbors else None,
            "centroid_w": ap.compute_weighted_centroid(neighbors) if neighbors else None,
        }
    return info


def blended_embeddings_for_category(blend_info, alpha_grid):
    embeddings, tags = [], []
    for vname, info in blend_info.items():
        for weighted, centroid in [(False, info["centroid_uw"]), (True, info["centroid_w"])]:
            if centroid is None:
                continue
            for alpha in alpha_grid:
                embeddings.append(ap.blend_embedding(info["query_emb"], centroid, alpha))
                tags.append((vname, info["word"], weighted, alpha))
    return embeddings, tags


def resumable_writer(path, fieldnames, key_fields):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    exists = os.path.exists(path)
    seen = set()
    if exists:
        with open(path) as f:
            for row in csv.DictReader(f):
                seen.add(tuple(row[k] for k in key_fields))
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
        f.flush()
    return f, writer, seen


def run_grid(grid_name, alpha_grid, model, model_name, coco, coco_dir, cat_name_to_id, positive_set,
             hyper_siblings, n_categories, n_images, out_path):
    fieldnames = ["grid", "model", "category", "variant", "variant_word", "weighted", "img_ref", "alpha", "iou"]
    f, writer, seen = resumable_writer(out_path, fieldnames, ["grid", "model", "category", "img_ref"])
    try:
        categories = pick_categories(positive_set, n=n_categories)
        print(f"[{grid_name}/{model_name}] {len(categories)} COCO categories selected: {categories}")

        for cat_name in categories:
            synset = ap.eligible_synset(cat_name)
            blend_info = blend_variants_for_class(cat_name, synset, model)
            img_refs = pick_images(positive_set, cat_name, n=n_images)

            embeddings, tags = blended_embeddings_for_category(blend_info, alpha_grid)
            if not embeddings:
                print(f"  [skip class] '{cat_name}' -- no usable variants/neighbors")
                continue
            tag_to_embedding = {str(i): emb for i, emb in enumerate(embeddings)}

            for img_ref in img_refs:
                key = (grid_name, model_name, cat_name, str(img_ref))
                if key in seen:
                    continue

                base_mask = get_single_cat_mask(coco, cat_name_to_id, img_ref, cat_name)
                if base_mask is None or base_mask.sum() == 0:
                    continue
                # Shared-hypernym images (e.g. two animals both -> the same WordNet
                # hypernym) score the "hyper" variant against the union of every present
                # sibling's mask -- see get_mask_for_variant's docstring.
                hyper_mask = get_mask_for_variant(coco, cat_name_to_id, cat_name, img_ref,
                                                   "hyper", hyper_siblings, base_mask=base_mask)

                try:
                    image = get_image(coco, coco_dir, img_ref)
                except Exception as e:
                    print(f"     [skip image] {img_ref}: fetch error {e}")
                    continue

                masks = mdl.predict_masks_for_tags(model, model_name, image, cat_name, tag_to_embedding)
                for i, (vname, word, weighted, alpha) in enumerate(tags):
                    mask = masks[str(i)]
                    gt = hyper_mask if vname == "hyper" else base_mask
                    writer.writerow({
                        "grid": grid_name, "model": model_name, "category": cat_name, "variant": vname,
                        "variant_word": word, "weighted": weighted, "img_ref": str(img_ref),
                        "alpha": alpha, "iou": bm_hp.compute_iou(mask, gt),
                    })
                seen.add(key)
                f.flush()
            print(f"  {cat_name}: done ({len(img_refs)} images)")
    finally:
        f.close()
    print(f"[{grid_name}/{model_name}] detail written to {out_path}")


def summarize(detail_path, summary_path):
    rows = []
    with open(detail_path) as f:
        for row in csv.DictReader(f):
            row["alpha"] = float(row["alpha"])
            row["iou"] = float(row["iou"])
            row["weighted"] = row["weighted"] == "True"
            rows.append(row)

    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["grid"], r["model"], r["weighted"], r["alpha"])].append(r["iou"])

    summary_rows = []
    for (grid, model_name, weighted, alpha), ious in sorted(by_key.items()):
        summary_rows.append({
            "grid": grid, "model": model_name, "weighted": weighted, "alpha": alpha,
            "n": len(ious), "mean_iou": float(np.mean(ious)),
        })

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["grid", "model", "weighted", "alpha", "n", "mean_iou"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"summary written to {summary_path}")

    print("\n=== mean IoU per (grid, model, weighted, alpha) ===")
    for grid in sorted({r["grid"] for r in summary_rows}):
        for model_name in sorted({r["model"] for r in summary_rows}):
            for weighted in [False, True]:
                subset = [r for r in summary_rows
                          if r["grid"] == grid and r["model"] == model_name and r["weighted"] == weighted]
                if not subset:
                    continue
                best = max(subset, key=lambda r: r["mean_iou"])
                label = "weighted-centroid" if weighted else "unweighted-centroid"
                print(f"  [{grid}/{model_name}/{label}] best alpha={best['alpha']:.2f}  "
                      f"mean_iou={best['mean_iou']:.4f}  (n={best['n']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=mdl.SUPPORTED_MODELS)
    parser.add_argument("--coco-dir", default=os.environ.get("COCO_DIR", os.path.join(REPO_ROOT, "datasets", "coco")),
                         help="Root dir with (or to populate with) instances_val2017.json and val2017/*.jpg. "
                              "Defaults to datasets/coco/, where instances_val2017.json is already cached.")
    parser.add_argument("--predownload-images", action="store_true",
                         help="Eagerly download every val2017 image up front instead of lazily on first "
                              "use (useful before a full-scale run).")
    parser.add_argument("--grid", choices=["coarse", "fine", "both"], default="both")
    parser.add_argument("--n-categories", type=int, default=N_CATEGORIES,
                         help="-1 = every eligible COCO category (the real, intended run)")
    parser.add_argument("--n-images", type=int, default=N_IMAGES,
                         help="-1 = every positive image per category (the real, intended run)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    coco_dir = os.path.abspath(args.coco_dir)
    os.makedirs(coco_dir, exist_ok=True)
    coco = bm_hp.load_coco(ann_path=os.path.join(coco_dir, "instances_val2017.json"))

    if args.predownload_images:
        predownload_all_images(coco, coco_dir)

    cat_name_to_id = bm_hp.get_cat_name_to_id(coco)
    positive_set = build_positive_set(coco)
    categories_all = sorted(positive_set)
    hyper_siblings = build_hyper_siblings(categories_all)

    model = mdl.get_model(args.model, device=args.device)

    detail_out = os.path.join(args.out_dir, f"alpha_value_experiment_{args.model}_detail.csv")
    summary_out = os.path.join(args.out_dir, f"alpha_value_experiment_{args.model}_summary.csv")

    if args.grid in ("coarse", "both"):
        run_grid("coarse", ALPHA_GRID_COARSE, model, args.model, coco, coco_dir, cat_name_to_id,
                  positive_set, hyper_siblings, args.n_categories, args.n_images, detail_out)
    if args.grid in ("fine", "both"):
        run_grid("fine", ALPHA_GRID_FINE, model, args.model, coco, coco_dir, cat_name_to_id,
                  positive_set, hyper_siblings, args.n_categories, args.n_images, detail_out)

    summarize(detail_out, summary_out)
