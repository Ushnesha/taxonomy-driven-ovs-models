import csv
import os
import sys
from collections import defaultdict
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import expanded_benchmark_helpers as bm_hp
from expanded_benchmark_builder import build_positive_negative_sets, extract_gt_objects_and_masks

ALPHA = 0.6
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def eligible_synset(word):
    s = bm_hp.find_best_synset(word)
    if s and len(s.lemma_names()) >= 2:
        return s
    return None


def build_coco_ctx():
    coco = bm_hp.load_coco()
    positive_set, _, _ = build_positive_negative_sets(coco)
    return {"coco": coco}, positive_set


def build_lvis_ctx():
    lvis = bm_hp.load_lvis()
    cat_id_to_name = {c["id"]: c["name"].replace("_", " ") for c in lvis["categories"]}
    images_by_id = {im["id"]: im for im in lvis["images"]}
    positive_set = defaultdict(set)
    anns_by_image_cat = defaultdict(list)
    for a in lvis["annotations"]:
        cat_name = cat_id_to_name.get(a["category_id"])
        if cat_name is None:
            continue
        positive_set[cat_name].add(a["image_id"])
        anns_by_image_cat[(a["image_id"], cat_name)].append(a)
    positive_set = {cat: sorted(ids) for cat, ids in positive_set.items()}
    return {"lvis": lvis, "images_by_id": images_by_id, "anns_by_image_cat": anns_by_image_cat}, positive_set


def build_ade20k_ctx():
    ds = bm_hp.load_ade20k()
    ds_objs = ds.select_columns(["objects"])
    positive_set = defaultdict(list)
    for idx, ex in enumerate(ds_objs):
        for name in {o["raw_name"] for o in ex["objects"]}:
            positive_set[name].append(idx)
    return {"ade20k": ds}, dict(positive_set)


def build_pascalvoc_ctx():
    ds = bm_hp.load_pascalvoc()
    ds_masks = ds.select_columns(["mask"])
    positive_set = defaultdict(list)
    for idx in range(len(ds_masks)):
        gt_objects, _ = extract_gt_objects_and_masks({"mask": ds_masks[idx]["mask"]})
        for cls in gt_objects:
            positive_set[cls].append(idx)
    return {"pascalvoc": ds}, dict(positive_set)


def get_image_and_gt_coco(ctx, cat_name, img_ref):
    coco = ctx["coco"]
    cat_id = bm_hp.get_cat_name_to_id(coco)[cat_name]
    gt = bm_hp.get_gt_mask(coco, img_ref, cat_id)
    image = bm_hp.download_image(coco, img_ref)
    return image, gt


def get_image_and_gt_lvis(ctx, cat_name, img_ref):
    img_meta = ctx["images_by_id"][img_ref]
    h, w = img_meta["height"], img_meta["width"]
    anns = ctx["anns_by_image_cat"].get((img_ref, cat_name), [])
    gt = np.zeros((h, w), dtype=np.uint8)
    for a in anns:
        m = bm_hp.decode_lvis_ann_to_mask(a, h, w)
        if m is not None:
            gt = np.maximum(gt, m)
    url = img_meta.get("coco_url") or bm_hp.COCO_URL.format(img_ref)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        image = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        image = None
    return image, gt


def get_image_and_gt_ade20k(ctx, cat_name, img_ref):
    ex = ctx["ade20k"][img_ref]
    gt = bm_hp.get_gt_mask_for_ade(ex, cat_name)
    return ex["image"], gt


def get_image_and_gt_pascalvoc(ctx, cat_name, img_ref):
    item = ctx["pascalvoc"][img_ref]
    gt_objects, gt_bin_masks = extract_gt_objects_and_masks(item)
    if cat_name not in gt_objects:
        return item["image"], None
    import zlib
    i = gt_objects.index(cat_name)
    w, h = item["mask"].size
    gt = np.frombuffer(zlib.decompress(gt_bin_masks[i]), dtype=np.uint8).reshape((h, w))
    return item["image"], gt


FETCHERS = {
    "coco": get_image_and_gt_coco,
    "lvis": get_image_and_gt_lvis,
    "ade20k": get_image_and_gt_ade20k,
    "pascalvoc": get_image_and_gt_pascalvoc,
}
BUILDERS = {
    "coco": build_coco_ctx,
    "lvis": build_lvis_ctx,
    "ade20k": build_ade20k_ctx,
    "pascalvoc": build_pascalvoc_ctx,
}


SEGMENTATION_CHUNK_SIZE = 16  # cap peak memory: each item duplicates the full-res image in the batch


def _run_segmentation_probs_chunk(image, cond_embeddings):
    processor, model = bm_hp.load_model()
    device = bm_hp.get_device()
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


def run_segmentation_probs_batch(image, cond_embeddings, chunk_size=SEGMENTATION_CHUNK_SIZE):
    """Runs CLIPSeg over `cond_embeddings` in memory-bounded chunks (each chunk duplicates the
    full-res image `len(chunk)` times, so large K -- e.g. WaffleCLIP's 30+ descriptors times up
    to 4 linguistic variants -- can otherwise blow up peak memory). Chunking is purely a memory
    optimization: results are numerically identical to running everything in one batch, since
    each embedding's forward pass is independent."""
    if len(cond_embeddings) <= chunk_size:
        return _run_segmentation_probs_chunk(image, cond_embeddings)
    chunks = [
        _run_segmentation_probs_chunk(image, cond_embeddings[i:i + chunk_size])
        for i in range(0, len(cond_embeddings), chunk_size)
    ]
    return np.concatenate(chunks, axis=0)


def run_segmentation_batch(image, cond_embeddings, threshold=0.5):
    return (run_segmentation_probs_batch(image, cond_embeddings) > threshold).astype(np.uint8)


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


def compute_weighted_centroid(neighbors):
    sims = torch.tensor([max(s, 0.0) for _, _, s in neighbors])
    if sims.sum() <= 0:
        weights = torch.ones(len(neighbors)) / len(neighbors)
    else:
        weights = sims / sims.sum()
    embs = torch.stack([e.squeeze() for _, e, _ in neighbors])
    return (weights.unsqueeze(1) * embs).sum(dim=0)


def blend_variants_for_class(cat_name, synset, k=5):
    variants, w_s = get_variants(cat_name, synset)
    info = {}
    for vname, word in variants.items():
        query_emb = bm_hp.get_text_embedding_cached(word)
        candidates = [w for w in w_s if w.lower() != word.lower()]
        neighbors = bm_hp.top_k_neighbors(word, candidates, k=k) if candidates else []
        centroid_uw = bm_hp.compute_centroid(neighbors) if neighbors else None
        centroid_w = compute_weighted_centroid(neighbors) if neighbors else None
        info[vname] = {
            "word": word,
            "query_emb": query_emb,
            "centroid_uw": centroid_uw,
            "centroid_w": centroid_w,
            "neighbors": [w for w, _, _ in neighbors],
        }
    return info


def all_ids_for(dataset_name, ctx):
    if dataset_name == "coco":
        return ctx["coco"].getImgIds()
    if dataset_name == "lvis":
        return list(ctx["images_by_id"].keys())
    if dataset_name == "ade20k":
        return list(range(len(ctx["ade20k"])))
    if dataset_name == "pascalvoc":
        return list(range(len(ctx["pascalvoc"])))
    raise ValueError(dataset_name)


def build_negative_set(positive_set, all_ids):
    all_ids_set = set(all_ids)
    return {cat: sorted(all_ids_set - set(ids), key=str) for cat, ids in positive_set.items()}


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


def run_query_transform_experiment(out_path, method_name, transform_fn, limit_categories=None, limit_images=None):
    fieldnames = ["dataset", "category", "variant", "variant_word", "img_ref", "method", "iou"]
    f, writer, seen = resumable_writer(out_path, fieldnames, ["dataset", "category", "img_ref"])
    try:
        for dataset_name, builder in BUILDERS.items():
            ctx, positive_set = builder()
            categories = [c for c in sorted(positive_set) if eligible_synset(c) is not None]
            if limit_categories:
                categories = categories[:limit_categories]
            print(f"{dataset_name}: {len(categories)} categories")

            for cat_name in categories:
                synset = eligible_synset(cat_name)
                variants, _ = get_variants(cat_name, synset)
                query_info = {}
                for vname, word in variants.items():
                    baseline_emb = bm_hp.get_text_embedding_cached(word)
                    method_emb = transform_fn(word)
                    query_info[vname] = (word, baseline_emb, method_emb)

                img_refs = positive_set[cat_name]
                if limit_images:
                    img_refs = img_refs[:limit_images]

                for img_ref in img_refs:
                    key = (dataset_name, cat_name, str(img_ref))
                    if key in seen:
                        continue
                    try:
                        image, gt = FETCHERS[dataset_name](ctx, cat_name, img_ref)
                    except Exception as e:
                        print(f"  skip {dataset_name}/{cat_name}/{img_ref}: {e}")
                        continue
                    if image is None or gt is None or gt.sum() == 0:
                        continue

                    embeddings, tags = [], []
                    for vname, (word, baseline_emb, method_emb) in query_info.items():
                        embeddings.append(baseline_emb)
                        tags.append((vname, word, "baseline"))
                        embeddings.append(method_emb)
                        tags.append((vname, word, method_name))

                    masks = run_segmentation_batch(image, embeddings)
                    for i, (vname, word, method) in enumerate(tags):
                        writer.writerow({
                            "dataset": dataset_name, "category": cat_name, "variant": vname,
                            "variant_word": word, "img_ref": str(img_ref), "method": method,
                            "iou": bm_hp.compute_iou(masks[i], gt),
                        })
                    seen.add(key)
                    f.flush()
                print(f"  {cat_name}: done ({len(img_refs)} images)")
    finally:
        f.close()


def run_multi_descriptor_experiment(out_path, method_name, baseline_descriptors_fn, method_descriptors_fn,
                                     limit_categories=None, limit_images=None, threshold=0.5):
    """Score-level ensembling across independently-embedded descriptor sentences.

    Mirrors the *default* aggregation path in the official WaffleCLIP (`base_main.py`,
    `--merge_predictions` unset) and classify_by_description (`load.py: aggregate_similarity`,
    `aggregation_method='mean'`) repos: each descriptor is embedded and scored on its own, and
    the resulting *scores* are averaged -- not the text embeddings. CLIPSeg has no separate
    image-text cosine score to average (the conditional embedding drives the decoder directly),
    so the segmentation-domain analog used here is to run one CLIPSeg forward pass per descriptor
    and mean-average the resulting sigmoid probability maps before thresholding.
    """
    fieldnames = ["dataset", "category", "variant", "variant_word", "img_ref", "method", "iou"]
    f, writer, seen = resumable_writer(out_path, fieldnames, ["dataset", "category", "img_ref"])
    try:
        for dataset_name, builder in BUILDERS.items():
            ctx, positive_set = builder()
            categories = [c for c in sorted(positive_set) if eligible_synset(c) is not None]
            if limit_categories:
                categories = categories[:limit_categories]
            print(f"{dataset_name}: {len(categories)} categories")

            for cat_name in categories:
                synset = eligible_synset(cat_name)
                variants, _ = get_variants(cat_name, synset)
                query_info = {}
                for vname, word in variants.items():
                    baseline_sents = baseline_descriptors_fn(word)
                    method_sents = method_descriptors_fn(word)
                    baseline_embs = [bm_hp.get_text_embedding_cached(s) for s in baseline_sents]
                    method_embs = [bm_hp.get_text_embedding_cached(s) for s in method_sents]
                    query_info[vname] = (word, baseline_embs, method_embs)

                img_refs = positive_set[cat_name]
                if limit_images:
                    img_refs = img_refs[:limit_images]

                for img_ref in img_refs:
                    key = (dataset_name, cat_name, str(img_ref))
                    if key in seen:
                        continue
                    try:
                        image, gt = FETCHERS[dataset_name](ctx, cat_name, img_ref)
                    except Exception as e:
                        print(f"  skip {dataset_name}/{cat_name}/{img_ref}: {e}")
                        continue
                    if image is None or gt is None or gt.sum() == 0:
                        continue

                    all_embs, groups = [], []
                    for vname, (word, baseline_embs, method_embs) in query_info.items():
                        groups.append((vname, word, "baseline", len(all_embs), len(baseline_embs)))
                        all_embs.extend(baseline_embs)
                        groups.append((vname, word, method_name, len(all_embs), len(method_embs)))
                        all_embs.extend(method_embs)

                    probs = run_segmentation_probs_batch(image, all_embs)
                    for vname, word, method, start, count in groups:
                        avg_prob = probs[start:start + count].mean(axis=0)
                        mask = (avg_prob > threshold).astype(np.uint8)
                        writer.writerow({
                            "dataset": dataset_name, "category": cat_name, "variant": vname,
                            "variant_word": word, "img_ref": str(img_ref), "method": method,
                            "iou": bm_hp.compute_iou(mask, gt),
                        })
                    seen.add(key)
                    f.flush()
                print(f"  {cat_name}: done ({len(img_refs)} images)")
    finally:
        f.close()
