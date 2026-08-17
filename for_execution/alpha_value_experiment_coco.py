"""
Self-contained alpha hyperparameter sweep for Ours (alpha-blending / weighted-centroid
blending toward the WordNet synset centroid), on COCO + CLIPSeg only.

This is a standalone port of experiments_for_cluster/alpha_value_experiment.py, meant
to run from a fresh `pip`/`uv` virtual environment with nothing else from this repo
checked out -- see requirements.txt in this same folder. The blending math, WordNet
disambiguation, and CLIPSeg inference are copied verbatim from
expanded_benchmark_helpers.py / experiment_common.py (not reimplemented), just without
that code's other-dataset (LVIS/ADE20K/Pascal VOC) and BabelNet-fallback machinery,
which this script never needs for COCO's 80 canonical category words.

Methodology: alpha_tuning_dev/alpha_best_value.md found the sweet spot for this method
at alpha=0.50-0.55 (fine-sweep best 0.52) on a 200-pair sample across all 4 datasets.
This script re-derives that number for COCO specifically -- run small locally first to
verify, then at full scale on the cluster; once confirmed, the same alpha gets reused
for the other 3 datasets.

---- COCO dataset location ----
Point `--coco-dir` (or the COCO_DIR env var) at a directory that either already has a
local COCO val2017 checkout, or is just an empty/new directory to populate:
    <coco-dir>/annotations/instances_val2017.json
    <coco-dir>/val2017/<file_name>.jpg
If the annotations file is missing, it's downloaded and extracted automatically. Each
image is looked up locally first; if missing, it's downloaded once from the COCO CDN
and cached under <coco-dir>/val2017/ so every later run (including a full-scale
cluster run) reuses it instead of re-downloading. Pass --predownload-images to eagerly
fetch every image up front instead of lazily on first use.

Usage:
    # small local subset, for verification (default -- a few minutes on CPU):
    python3 alpha_value_experiment_coco.py --coco-dir ./coco_data

    # point at an already-downloaded COCO checkout instead of downloading one:
    python3 alpha_value_experiment_coco.py --coco-dir /path/to/existing/coco

    # full-scale COCO run (cluster):
    python3 alpha_value_experiment_coco.py --coco-dir /path/to/coco --n-categories -1 --n-images -1
"""
import argparse
import csv
import os
import threading
import zipfile
from collections import defaultdict
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from pycocotools.coco import COCO

# =============================================================================
# CLIPSeg (verbatim from expanded_benchmark_helpers.py: get_text_embedding,
# load_model, run_segmentation -- chunked/batched form matches
# experiment_common.py's run_segmentation_probs_batch/run_segmentation_batch)
# =============================================================================

_device = None
_processor = None
_model = None
_embedding_cache = {}
SEGMENTATION_CHUNK_SIZE = 16  # cap peak memory: each item duplicates the full-res image


def get_device():
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = "cuda"
        elif torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
    return _device


def load_clipseg():
    global _processor, _model
    if _processor is None:
        from transformers import AutoProcessor, CLIPSegForImageSegmentation
        print("Loading CLIPSeg (CIDAS/clipseg-rd64-refined)...", flush=True)
        _processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        _model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        _model.to(get_device())
        _model.eval()
    return _processor, _model


def get_text_embedding(word: str):
    """Bare-word CLIP embedding. Raw norm preserved (~8-10) -- CLIPSeg's FiLM decoder
    was trained on these raw vectors; never L2-normalize."""
    processor, model = load_clipseg()
    inputs = processor(text=word, return_tensors="pt", padding=True, truncation=True).to(get_device())
    with torch.inference_mode():
        emb = model.clip.get_text_features(**inputs).pooler_output  # [1, 512]
    return emb.cpu()


def get_text_embedding_cached(word: str):
    if word in _embedding_cache:
        return _embedding_cache[word].clone()
    emb = get_text_embedding(word)
    _embedding_cache[word] = emb.cpu()
    return emb


def _run_segmentation_probs_chunk(image, cond_embeddings):
    processor, model = load_clipseg()
    device = get_device()
    w, h = image.size
    inputs = processor(images=[image] * len(cond_embeddings), return_tensors="pt").to(device)
    cond = torch.stack([c.squeeze() for c in cond_embeddings]).to(device)
    with torch.inference_mode():
        out = model(pixel_values=inputs["pixel_values"], conditional_embeddings=cond)
    logits = out.logits
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    probs = torch.sigmoid(logits)
    probs = torch.nn.functional.interpolate(
        probs.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
    )[:, 0]
    return probs.cpu().numpy()


def run_segmentation_batch(image, cond_embeddings, threshold=0.5, chunk_size=SEGMENTATION_CHUNK_SIZE):
    if len(cond_embeddings) <= chunk_size:
        probs = _run_segmentation_probs_chunk(image, cond_embeddings)
    else:
        chunks = [
            _run_segmentation_probs_chunk(image, cond_embeddings[i:i + chunk_size])
            for i in range(0, len(cond_embeddings), chunk_size)
        ]
        probs = np.concatenate(chunks, axis=0)
    return (probs > threshold).astype(np.uint8)


def compute_iou(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(inter / union) if union > 0 else 0.0


# =============================================================================
# WordNet disambiguation + blending (verbatim from expanded_benchmark_helpers.py
# / experiment_common.py: find_best_synset, build_word_sets_from_synset,
# top_k_neighbors, compute_centroid, compute_weighted_centroid, blend_embedding.
# No BabelNet fallback -- COCO's 80 canonical category words all resolve cleanly
# through WordNet alone; that fallback only exists for the other 3 datasets'
# rarer category words, which this script never touches.)
# =============================================================================

_nltk_lock = threading.Lock()


def to_wn_form(word: str) -> str:
    return word.strip().replace(" ", "_")


def to_display_form(word: str) -> str:
    return word.replace("_", " ")


def _ensure_wordnet():
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)


def get_all_synsets(word_wn: str):
    from nltk.corpus import wordnet as wn
    _ensure_wordnet()
    with _nltk_lock:
        synsets = wn.synsets(word_wn, pos=wn.NOUN)
        if not synsets:
            synsets = wn.synsets(word_wn)
        if not synsets:
            synsets = wn.synsets(word_wn.replace("_", " "))
    return synsets


def find_best_synset(word: str):
    """Pick the WordNet synset whose lemmas contain the COCO word -- automatic
    disambiguation against the canonical category name."""
    word_wn = to_wn_form(word)
    word_display = word.strip().lower()
    synsets = get_all_synsets(word_wn)
    if not synsets:
        return None

    matching = []
    for s in synsets:
        lemmas_lower = [l.lower() for l in s.lemma_names()]
        if word_wn.lower() in lemmas_lower or word_display in lemmas_lower:
            matching.append(s)

    if matching:
        def rank(s):
            lemmas_lower = [l.lower() for l in s.lemma_names()]
            try:
                return lemmas_lower.index(word_wn.lower())
            except ValueError:
                return lemmas_lower.index(word_display)
        matching.sort(key=rank)
        return matching[0]

    noun_synsets = [s for s in synsets if s.pos() == 'n']
    if noun_synsets:
        return noun_synsets[0]
    return synsets[0]


def build_word_sets_from_synset(synset):
    """{W_S, W_S_Hp, W_S_Hp_He}: direct synonyms / +depth-1 hyponyms / +depth-1 hypernyms."""
    w_s = list(synset.lemma_names())

    w_s_hp = list(w_s)
    for hypo in synset.hyponyms():
        for lemma in hypo.lemma_names():
            if lemma not in w_s_hp:
                w_s_hp.append(lemma)

    w_s_hp_he = list(w_s_hp)
    for hyper in synset.hypernyms():
        for lemma in hyper.lemma_names():
            if lemma not in w_s_hp_he:
                w_s_hp_he.append(lemma)

    return {"W_S": w_s, "W_S_Hp": w_s_hp, "W_S_Hp_He": w_s_hp_he}


def eligible_synset(word):
    s = find_best_synset(word)
    if s and len(s.lemma_names()) >= 2:
        return s
    return None


def cosine_sim(a, b):
    a, b = a.squeeze(), b.squeeze()
    return (a @ b).item() / (a.norm().item() * b.norm().item())


def top_k_neighbors(query_word: str, candidate_words: list, k=5):
    query_emb = get_text_embedding_cached(query_word)
    scored = []
    for w in candidate_words:
        if w.lower() == query_word.lower():
            continue
        emb = get_text_embedding_cached(w)
        scored.append((w, emb, cosine_sim(query_emb, emb)))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:k]


def compute_centroid(neighbors):
    return torch.stack([e.squeeze() for _, e, _ in neighbors]).mean(dim=0)


def compute_weighted_centroid(neighbors):
    sims = torch.tensor([max(s, 0.0) for _, _, s in neighbors])
    if sims.sum() <= 0:
        weights = torch.ones(len(neighbors)) / len(neighbors)
    else:
        weights = sims / sims.sum()
    embs = torch.stack([e.squeeze() for _, e, _ in neighbors])
    return (weights.unsqueeze(1) * embs).sum(dim=0)


def blend_embedding(query_emb, centroid, alpha):
    q = query_emb.squeeze()
    return (1 - alpha) * q + alpha * centroid.squeeze()


def get_variants(cat_name, synset):
    ws = build_word_sets_from_synset(synset)
    w_s = [to_display_form(w) for w in ws["W_S"]]
    w_s_hp = [to_display_form(w) for w in ws["W_S_Hp"]]
    w_s_hp_he = [to_display_form(w) for w in ws["W_S_Hp_He"]]

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


def blend_variants_for_class(cat_name, synset, k=5):
    variants, w_s = get_variants(cat_name, synset)
    info = {}
    for vname, word in variants.items():
        query_emb = get_text_embedding_cached(word)
        candidates = [w for w in w_s if w.lower() != word.lower()]
        neighbors = top_k_neighbors(word, candidates, k=k) if candidates else []
        info[vname] = {
            "word": word,
            "query_emb": query_emb,
            "centroid_uw": compute_centroid(neighbors) if neighbors else None,
            "centroid_w": compute_weighted_centroid(neighbors) if neighbors else None,
            "neighbors": [w for w, _, _ in neighbors],
        }
    return info


# =============================================================================
# COCO dataset access: local-first, download-and-cache-on-demand.
# =============================================================================

COCO_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"


def ensure_coco_annotations(coco_dir):
    """Downloads+extracts instances_val2017.json into <coco_dir>/annotations/ if it
    isn't already there -- lets --coco-dir point at either a fresh empty directory
    or an existing local COCO checkout interchangeably."""
    ann_path = os.path.join(coco_dir, "annotations", "instances_val2017.json")
    if os.path.exists(ann_path):
        return ann_path

    print(f"COCO annotations not found at {ann_path}. Downloading...")
    ann_dir = os.path.dirname(ann_path)
    os.makedirs(ann_dir, exist_ok=True)

    zip_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = os.path.join(coco_dir, "annotations_trainval2017.zip")
    r = requests.get(zip_url, stream=True)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete. Extracting instances_val2017.json...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        member = "annotations/instances_val2017.json"
        zf.extract(member, path=coco_dir)
    os.remove(zip_path)
    return ann_path


def get_local_image_path(coco_dir, file_name):
    """Local-first: returns the on-disk path, downloading+caching it there first if
    it isn't already present. Every later run (small local test or full-scale cluster
    run) that points at the same --coco-dir reuses whatever's already been fetched."""
    img_dir = os.path.join(coco_dir, "val2017")
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, file_name)
    if os.path.exists(path):
        return path

    img_id = int(os.path.splitext(file_name)[0])
    r = requests.get(COCO_URL.format(img_id), timeout=30)
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


def get_image_and_gt(coco, coco_dir, cat_name, img_id):
    cat_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}[cat_name]
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)
    if not anns:
        return None, None
    img_info = coco.loadImgs([img_id])[0]
    h, w = img_info["height"], img_info["width"]
    gt = np.zeros((h, w), dtype=np.uint8)
    for a in anns:
        gt = np.maximum(gt, coco.annToMask(a))

    path = get_local_image_path(coco_dir, img_info["file_name"])
    image = Image.open(path).convert("RGB")
    return image, gt


# =============================================================================
# Alpha sweep
# =============================================================================

SEED = 42
N_CATEGORIES = 10  # -1 = every eligible category (cluster / full-scale run)
N_IMAGES = 5        # -1 = every positive image for that category (cluster / full-scale run)
MIN_IMAGES = 5       # a category needs at least this many positive images to be eligible

ALPHA_GRID_COARSE = [round(x * 0.1, 2) for x in range(11)]          # 0.0, 0.1, ..., 1.0
ALPHA_GRID_FINE = [round(0.50 + 0.01 * i, 2) for i in range(31)]    # 0.50, 0.51, ..., 0.80


def pick_categories(positive_set, n=N_CATEGORIES, min_images=MIN_IMAGES, seed=SEED):
    eligible = [
        cat for cat in sorted(positive_set)
        if len(positive_set[cat]) >= min_images and eligible_synset(cat) is not None
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


def blended_embeddings_for_category(blend_info, alpha_grid):
    embeddings, tags = [], []
    for vname, info in blend_info.items():
        for weighted, centroid in [(False, info["centroid_uw"]), (True, info["centroid_w"])]:
            if centroid is None:
                continue
            for alpha in alpha_grid:
                embeddings.append(blend_embedding(info["query_emb"], centroid, alpha))
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


def run_grid(grid_name, alpha_grid, coco, coco_dir, positive_set, n_categories, n_images, out_path):
    fieldnames = ["grid", "category", "variant", "variant_word", "weighted", "img_ref", "alpha", "iou"]
    f, writer, seen = resumable_writer(out_path, fieldnames, ["grid", "category", "img_ref"])
    try:
        categories = pick_categories(positive_set, n=n_categories)
        print(f"[{grid_name}] {len(categories)} COCO categories selected: {categories}")

        for cat_name in categories:
            synset = eligible_synset(cat_name)
            blend_info = blend_variants_for_class(cat_name, synset)
            img_refs = pick_images(positive_set, cat_name, n=n_images)

            embeddings, tags = blended_embeddings_for_category(blend_info, alpha_grid)
            if not embeddings:
                print(f"  [skip class] '{cat_name}' -- no usable variants/neighbors")
                continue

            for img_ref in img_refs:
                key = (grid_name, cat_name, str(img_ref))
                if key in seen:
                    continue
                try:
                    image, gt = get_image_and_gt(coco, coco_dir, cat_name, img_ref)
                except Exception as e:
                    print(f"     [skip image] {img_ref}: fetch error {e}")
                    continue
                if image is None or gt is None or gt.sum() == 0:
                    continue

                masks = run_segmentation_batch(image, embeddings)
                for i, (vname, word, weighted, alpha) in enumerate(tags):
                    writer.writerow({
                        "grid": grid_name, "category": cat_name, "variant": vname,
                        "variant_word": word, "weighted": weighted, "img_ref": str(img_ref),
                        "alpha": alpha, "iou": compute_iou(masks[i], gt),
                    })
                seen.add(key)
                f.flush()
            print(f"  {cat_name}: done ({len(img_refs)} images)")
    finally:
        f.close()
    print(f"[{grid_name}] detail written to {out_path}")


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
        by_key[(r["grid"], r["weighted"], r["alpha"])].append(r["iou"])

    summary_rows = []
    for (grid, weighted, alpha), ious in sorted(by_key.items()):
        summary_rows.append({
            "grid": grid, "weighted": weighted, "alpha": alpha,
            "n": len(ious), "mean_iou": float(np.mean(ious)),
        })

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["grid", "weighted", "alpha", "n", "mean_iou"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"summary written to {summary_path}")

    print("\n=== mean IoU per (grid, weighted, alpha) ===")
    for grid in sorted({r["grid"] for r in summary_rows}):
        for weighted in [False, True]:
            subset = [r for r in summary_rows if r["grid"] == grid and r["weighted"] == weighted]
            if not subset:
                continue
            best = max(subset, key=lambda r: r["mean_iou"])
            label = "weighted-centroid" if weighted else "unweighted-centroid"
            print(f"  [{grid} / {label}] best alpha={best['alpha']:.2f}  mean_iou={best['mean_iou']:.4f}  (n={best['n']})")
            for r in subset:
                marker = "  <== best" if r["alpha"] == best["alpha"] else ""
                print(f"    alpha={r['alpha']:.2f}  n={r['n']:4d}  mean_iou={r['mean_iou']:.4f}{marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", default=os.environ.get("COCO_DIR", "./coco_data"),
                         help="Root dir with (or to populate with) annotations/instances_val2017.json "
                              "and val2017/*.jpg. Point this at an existing local COCO checkout to skip "
                              "downloading, or a fresh directory to download+cache into it.")
    parser.add_argument("--predownload-images", action="store_true",
                         help="Eagerly download every val2017 image up front instead of lazily on first "
                              "use (useful before a full-scale --n-categories -1 --n-images -1 run).")
    parser.add_argument("--grid", choices=["coarse", "fine", "both"], default="both")
    parser.add_argument("--n-categories", type=int, default=N_CATEGORIES,
                         help="-1 = every eligible COCO category (full-scale/cluster run)")
    parser.add_argument("--n-images", type=int, default=N_IMAGES,
                         help="-1 = every positive image per category (full-scale/cluster run)")
    parser.add_argument("--out-dir", default="./results")
    args = parser.parse_args()

    coco_dir = os.path.abspath(args.coco_dir)
    os.makedirs(coco_dir, exist_ok=True)
    ann_path = ensure_coco_annotations(coco_dir)
    coco = COCO(ann_path)

    if args.predownload_images:
        predownload_all_images(coco, coco_dir)

    positive_set = build_positive_set(coco)

    detail_out = os.path.join(args.out_dir, "alpha_value_experiment_coco_detail.csv")
    summary_out = os.path.join(args.out_dir, "alpha_value_experiment_coco_summary.csv")

    if args.grid in ("coarse", "both"):
        run_grid("coarse", ALPHA_GRID_COARSE, coco, coco_dir, positive_set,
                  args.n_categories, args.n_images, detail_out)
    if args.grid in ("fine", "both"):
        run_grid("fine", ALPHA_GRID_FINE, coco, coco_dir, positive_set,
                  args.n_categories, args.n_images, detail_out)

    summarize(detail_out, summary_out)
