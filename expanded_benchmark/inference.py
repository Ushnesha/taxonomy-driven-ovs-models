"""
Inference Pipeline for Expanded COCO Benchmark
===============================================
Runtime inference functions:
  1. get_synset_groups() — fetch all meaning groups for a user-submitted word
  2. User selects the correct synset
  3. build_word_set_for_inference() — build word sets from selected synset
  4. run_inference_pipeline() — centroid blending + CLIPSeg segmentation

Underscore conversion:
  - Spaces in user input are converted to underscores for WordNet/BabelNet lookup
  - Underscores are converted back to spaces for display and model input
"""

import torch
import numpy as np
from PIL import Image
from collections import defaultdict

from .helpers import (
    to_wn_form, to_display_form,
    get_synset_groups_for_display,
    get_text_embedding, get_text_embedding_cached,
    build_word_sets_from_synset,
    supplement_word_sets_with_babelnet,
    compute_centroid, blend_embedding, top_k_neighbors,
    run_segmentation, compute_iou, cosine_sim,
    load_model, get_device,
    load_coco, download_image, get_gt_mask,
    COCO_ANN, COCO_80,
)


def fetch_synset_groups(word: str):
    """
    Step 1: Fetch all WordNet synset groups for a user-submitted word.
    Spaces are auto-converted to underscores internally.
    Returns a list of synset group dicts for user selection.
    """
    groups = get_synset_groups_for_display(word)

    # Add word count per synset
    result = []
    for g in groups:
        result.append({
            "synset_name": g["synset_name"],
            "definition": g["definition"],
            "pos": g["pos"],
            "lemma_count": len(g["lemma_names"]),
            "lemma_names_display": g["lemma_names_display"][:10],  # show first 10
            "total_lemmas": len(g["lemma_names"]),
        })

    return {
        "query_word": word,
        "query_word_wn": to_wn_form(word),
        "num_synsets": len(result),
        "synsets": result,
    }


def build_word_set_for_selected_synset(synset_name: str):
    """
    Step 2: Given a user-selected synset name (e.g., 'dog.n.01'),
    build the expanded word sets (W_S, W_S_Hp, W_S_Hp_He).
    Falls back to BabelNet if WordNet gives < 2 synonyms.
    """
    from nltk.corpus import wordnet as wn
    from .helpers import _ensure_wordnet
    _ensure_wordnet()

    try:
        synset = wn.synset(synset_name)
    except Exception:
        return {"error": f"Synset '{synset_name}' not found in WordNet."}

    ws = build_word_sets_from_synset(synset)

    # BabelNet fallback
    babelnet_used = False
    if len(ws["W_S"]) < 2:
        # Use the first lemma as the lookup word
        lookup_word = ws["W_S"][0] if ws["W_S"] else synset_name.split(".")[0]
        ws_enriched = supplement_word_sets_with_babelnet(lookup_word, ws)
        if len(ws_enriched["W_S"]) > len(ws["W_S"]):
            ws = ws_enriched
            babelnet_used = True

    return {
        "synset_name": synset_name,
        "definition": synset.definition(),
        "W_S": ws["W_S"],
        "W_S_Hp": ws["W_S_Hp"],
        "W_S_Hp_He": ws["W_S_Hp_He"],
        "W_S_display": [to_display_form(w) for w in ws["W_S"]],
        "W_S_Hp_display": [to_display_form(w) for w in ws["W_S_Hp"]],
        "W_S_Hp_He_display": [to_display_form(w) for w in ws["W_S_Hp_He"]],
        "babelnet_supplemented": babelnet_used,
        "counts": {
            "W_S": len(ws["W_S"]),
            "W_S_Hp": len(ws["W_S_Hp"]),
            "W_S_Hp_He": len(ws["W_S_Hp_He"]),
        },
    }


def run_inference_pipeline(
    query_word: str,
    synset_name: str,
    image: Image.Image,
    alpha: float = 0.7,
    n_neighbors: int = 5,
    threshold: float = 0.5,
    gt_mask: np.ndarray = None,
):
    """
    Step 3: Full inference pipeline.

    1. Build word sets from selected synset
    2. Find top-K CLIP-closest synonyms to the query word
    3. Compute unweighted centroid of top-K synonym embeddings
    4. Alpha-blend query embedding with centroid
    5. Run CLIPSeg segmentation with blended embedding
    6. Compute IoU if GT mask provided

    Returns dict with results for display.
    """
    from nltk.corpus import wordnet as wn
    from .helpers import _ensure_wordnet
    _ensure_wordnet()

    # 1. Build word sets
    try:
        synset = wn.synset(synset_name)
    except Exception:
        return {"error": f"Synset '{synset_name}' not found."}

    ws = build_word_sets_from_synset(synset)
    if len(ws["W_S"]) < 2:
        lookup_word = ws["W_S"][0] if ws["W_S"] else synset_name.split(".")[0]
        ws_enriched = supplement_word_sets_with_babelnet(lookup_word, ws)
        if len(ws_enriched["W_S"]) > len(ws["W_S"]):
            ws = ws_enriched

    # 2. Find top-K neighbors (CLIP-closest synonyms in the same synset)
    query_emb = get_text_embedding_cached(query_word)
    neighbors = top_k_neighbors(query_word, ws["W_S"], k=n_neighbors)

    neighbor_words = [w for w, _, _ in neighbors]
    neighbor_sims = [s for _, _, s in neighbors]

    if not neighbors:
        # No other synonyms available — use query embedding directly
        blended_emb = query_emb
        centroid = query_emb
        technique = "no_blending (insufficient synonyms)"
    else:
        # 3. Compute centroid
        centroid = compute_centroid(neighbors)

        # 4. Alpha-blend
        blended_emb = blend_embedding(query_emb, centroid, alpha)
        technique = f"alpha_blending (alpha={alpha}, k={len(neighbors)})"

    # 5. Run segmentation
    pred_mask = run_segmentation(image, blended_emb, threshold=threshold)

    # Also run with raw query for comparison
    raw_pred_mask = run_segmentation(image, query_emb, threshold=threshold)

    # 6. Compute IoU if GT provided
    iou_raw = None
    iou_blended = None
    if gt_mask is not None:
        iou_raw = compute_iou(raw_pred_mask, gt_mask)
        iou_blended = compute_iou(pred_mask, gt_mask)

    return {
        "query_word": query_word,
        "synset_name": synset_name,
        "synset_definition": synset.definition(),
        "alpha": alpha,
        "technique": technique,
        "word_sets": {
            "W_S": ws["W_S"],
            "W_S_Hp": ws["W_S_Hp"],
            "W_S_Hp_He": ws["W_S_Hp_He"],
        },
        "top_k_neighbors": [
            {"word": w, "similarity": round(s, 4)}
            for w, _, s in neighbors
        ],
        "centroid_computed_from": neighbor_words,
        "iou_raw_query": round(iou_raw, 4) if iou_raw is not None else None,
        "iou_blended": round(iou_blended, 4) if iou_blended is not None else None,
        "iou_delta": round(iou_blended - iou_raw, 4) if (iou_raw is not None and iou_blended is not None) else None,
        "pred_mask": pred_mask,          # numpy array
        "raw_pred_mask": raw_pred_mask,  # numpy array
        # Embedding stats
        "query_norm": round(query_emb.norm().item(), 2),
        "centroid_norm": round(centroid.norm().item(), 2) if neighbors else None,
        "blended_norm": round(blended_emb.norm().item(), 2),
        "query_centroid_cosine": round(cosine_sim(query_emb, centroid).item(), 4) if neighbors else None,
    }


def run_inference_on_coco_image(
    query_word: str,
    synset_name: str,
    coco_category: str = None,
    img_id: int = None,
    alpha: float = 0.7,
    n_neighbors: int = 5,
    threshold: float = 0.5,
):
    """
    Convenience wrapper: run inference on a COCO image with GT mask for IoU.

    If coco_category is provided, use it to find the GT mask and pick an image.
    If img_id is provided, use that specific image.
    """
    import random
    random.seed(42)

    coco = load_coco(COCO_ANN)

    # Determine category for GT
    if coco_category is None:
        # Try to auto-match: find if any COCO_80 word is in the synset lemmas
        from nltk.corpus import wordnet as wn
        synset = wn.synset(synset_name)
        synset_lemmas = set(l.lower() for l in synset.lemma_names())
        for cat in COCO_80:
            if cat.lower() in synset_lemmas or to_wn_form(cat).lower() in synset_lemmas:
                coco_category = cat
                break

    cat_id = None
    if coco_category:
        cat_name_to_id = {cat["name"]: cat["id"] for cat in coco.loadCats(coco.getCatIds())}
        cat_id = cat_name_to_id.get(coco_category)

    # Pick image
    if img_id is None:
        if cat_id:
            ann_ids = coco.getAnnIds(catIds=[cat_id])
            all_img_ids = list(set(
                coco.loadAnns(ann_ids)[i]["image_id"] for i in range(min(len(ann_ids), 100))
            ))
            if all_img_ids:
                img_id = random.choice(all_img_ids)
            else:
                img_id = coco.getImgIds()[0]
        else:
            img_id = coco.getImgIds()[0]

    # Download image
    image = download_image(coco, img_id)
    if image is None:
        return {"error": f"Failed to download image {img_id}"}

    # Get GT mask
    gt_mask = None
    if cat_id:
        gt_mask = get_gt_mask(coco, img_id, cat_id)

    result = run_inference_pipeline(
        query_word=query_word,
        synset_name=synset_name,
        image=image,
        alpha=alpha,
        n_neighbors=n_neighbors,
        threshold=threshold,
        gt_mask=gt_mask,
    )

    result["img_id"] = img_id
    result["coco_category"] = coco_category
    result["image_size"] = image.size

    return result
