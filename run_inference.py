#!/usr/bin/env python3
import argparse
import sys
import os
import pickle
import zlib
import numpy as np
import torch
from PIL import Image

# Import helpers from the repository
import expanded_benchmark_helpers as bm_hp
from ovs_eval.models import get_model, list_models
from ovs_eval.linguistics import get_linguistic_cats_v2
from ovs_eval.dataset import get_img

# Dummy COCO class to allow ADE20K images to work seamlessly with all models
class DummyCOCO:
    def __init__(self, cat_list):
        self.cat_to_id = {cat: i for i, cat in enumerate(cat_list)}
        self.id_to_cat = {i: cat for i, cat in enumerate(cat_list)}
        
    def getCatIds(self, catNms=None):
        if catNms is None:
            return list(self.cat_to_id.values())
        return [self.cat_to_id[name] for name in catNms if name in self.cat_to_id]
        
    def get(self, key, default=None):
        return self.cat_to_id.get(key, default)

def calculate_iou(pred_mask, gt_mask):
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    return intersection / union if union > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Run single/multi-image inference and verify cached ground truth.")
    parser.add_argument("--model", type=str, default="clipseg", choices=list_models(),
                        help=f"Model to evaluate. Choose from: {list_models()}")
    parser.add_argument("--index", type=int, default=None,
                        help="Index of the image in the metadata cache to test.")
    parser.add_argument("--img-id", type=str, default=None,
                        help="Image ID (e.g. 'coco_494869' or 'ade20k_366') to test.")
    parser.add_argument("--num-images", type=int, default=1,
                        help="Number of images to process in a batch (used if index/img-id is not specified).")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for prediction masks.")
    parser.add_argument("--desc", action="store_true",
                        help="Use descriptions instead of template prompts.")
    parser.add_argument("--metadata-path", type=str, default="data/img_metadata.pkl",
                        help="Path to the image metadata cache file.")
    parser.add_argument("--output", type=str, default="data/inference_results.json",
                        help="Path to save the computed IoUs as a JSON file.")
    args = parser.parse_args()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load cache
    if not os.path.exists(args.metadata_path):
        print(f"Error: Cache file '{args.metadata_path}' not found. Please run the benchmark builder first.")
        sys.exit(1)

    print(f"Loading cache from {args.metadata_path}...")
    with open(args.metadata_path, "rb") as f:
        img_bnch = pickle.load(f)
    print(f"Loaded {len(img_bnch)} cached images.")

    # Select the subset of image metadata entries
    img_data_list = []
    if args.img_id is not None:
        for img in img_bnch:
            if img["img_id"] == args.img_id:
                img_data_list.append(img)
                break
        if not img_data_list:
            print(f"Error: Image ID '{args.img_id}' not found in cache.")
            sys.exit(1)
    elif args.index is not None:
        if args.index < 0 or args.index >= len(img_bnch):
            print(f"Error: Index {args.index} out of range (0 to {len(img_bnch)-1}).")
            sys.exit(1)
        img_data_list.append(img_bnch[args.index])
    else:
        num_imgs = min(args.num_images, len(img_bnch))
        img_data_list = img_bnch[:num_imgs]
        print(f"No index or img-id specified. Running on first {num_imgs} images in cache.")

    # Load word sets database
    word_sets_path = "data/word_sets_v2.json"
    word_sets_db = {}
    if os.path.exists(word_sets_path):
        import json
        print(f"Loading taxonomy database from {word_sets_path}...")
        with open(word_sets_path, "r") as f:
            word_sets_db = json.load(f)
    else:
        print("Warning: data/word_sets_v2.json not found. Run the benchmark builder first.")

    # Load original images and ground truth masks
    print(f"\nLoading {len(img_data_list)} original images...")
    images_pil = []
    gt_masks_batch = []  # list of lists of ground-truth masks per image
    
    coco_obj = None
    ade_dataset = None

    for img_data in img_data_list:
        img_src = img_data["img_src"]
        img_src_id = img_data["img_src_id"]
        w, h = img_data["width"], img_data["height"]

        image_pil = None
        if img_src == "coco":
            if coco_obj is None:
                print("Loading COCO annotations helper...")
                from pycocotools.coco import COCO
                if not os.path.exists(bm_hp.COCO_ANN):
                    print(f"Error: COCO annotations not found at {bm_hp.COCO_ANN}")
                    sys.exit(1)
                coco_obj = COCO(bm_hp.COCO_ANN)
            image_pil = get_img(img_src_id, coco_obj)
        elif img_src == "ade20k":
            if ade_dataset is None:
                print("Loading ADE20K dataset helper...")
                ade_dataset = bm_hp.load_ade20k()
            ade_filenames = ade_dataset["filename"]
            filename = img_data["filename"]
            if filename in ade_filenames:
                ade_idx = ade_filenames.index(filename)
                image_pil = ade_dataset[ade_idx]["image"]
            else:
                print(f"Error: Could not find filename '{filename}' in the ADE20K dataset.")
                sys.exit(1)

        if image_pil is None:
            print(f"Error: Failed to load PIL Image for image ID {img_data['img_id']}")
            sys.exit(1)

        images_pil.append(image_pil)

        # Decompress cached ground truth masks
        gt_masks = []
        for compressed_mask in img_data["gt_bin_masks"]:
            mask = np.frombuffer(zlib.decompress(compressed_mask), dtype=np.uint8).reshape((h, w))
            gt_masks.append(mask)
        gt_masks_batch.append(gt_masks)

    # Initialize model
    print(f"\nInitializing model '{args.model}'...")
    model = get_model(name=args.model, device=device)

    # Initialize embedding cache if blending is possible
    bm_hp.load_embedding_cache()

    # Dictionary to hold final structured results
    structured_results = {}

    # Build parallel prompts batch
    image_prompts = []
    image_category_prompts = []  # Maps each category to its prompt variants per image

    for img_idx, img_data in enumerate(img_data_list):
        gt_objects = img_data["gt_objects"]
        prompts_for_img = []
        cat_prompts_map = {}

        for cat_name in gt_objects:
            ws_info = word_sets_db.get(cat_name, {})
            synonyms = ws_info.get("synonyms", [])
            hyponyms = ws_info.get("hyponyms", [])
            hypernyms = ws_info.get("hypernyms", [])

            orig = cat_name if args.desc else f"a photo of a {cat_name}"
            syns = [s if args.desc else f"a photo of a {s}" for s in synonyms]
            hypos = [h if args.desc else f"a photo of a {h}" for h in hyponyms]
            hypers = [hp if args.desc else f"a photo of a {hp}" for hp in hypernyms]

            cat_prompts_map[cat_name] = {
                "orig": orig,
                "syns": syns,
                "hypos": hypos,
                "hypers": hypers
            }

            # Add to the flat list of prompts for this specific image
            prompts_for_img.append(orig)
            prompts_for_img.extend(syns)
            prompts_for_img.extend(hypos)
            prompts_for_img.extend(hypers)

        image_prompts.append(prompts_for_img)
        image_category_prompts.append(cat_prompts_map)

    # Run predictions in a single GPU-batched call
    print(f"\nRunning GPU-batched model predictions for {len(img_data_list)} images...")
    pred_masks_batch = model.batch_inference_clipseg(images_pil, image_prompts, threshold=args.threshold)

    # Process results and calculate IoUs
    print(f"\nCalculating IoU vs cached ground truth:")
    print("=" * 105)
    print(f"{'Category (Original)':<20} | {'Variant Type':<12} | {'Specific Word':<20} | {'IoU':<6}")
    print("=" * 105)

    for img_idx, img_data in enumerate(img_data_list):
        img_id = img_data["img_id"]
        gt_objects = img_data["gt_objects"]
        gt_masks = gt_masks_batch[img_idx]
        image_pil = images_pil[img_idx]
        h, w = img_data["height"], img_data["width"]
        
        pred_masks = pred_masks_batch[img_idx]
        cat_prompts_map = image_category_prompts[img_idx]

        structured_results[img_id] = {}
        print(f"\nImage ID: {img_id}")
        print("-" * 105)

        for cat_idx, cat_name in enumerate(gt_objects):
            gt_mask = gt_masks[cat_idx]
            maps = cat_prompts_map[cat_name]
            ws_info = word_sets_db.get(cat_name, {})
            synonyms = ws_info.get("synonyms", [])
            hyponyms = ws_info.get("hyponyms", [])
            hypernyms = ws_info.get("hypernyms", [])

            # 1. Original IoU
            orig_mask = pred_masks.get(maps["orig"], np.zeros((h, w), dtype=np.uint8))
            iou_orig = calculate_iou(orig_mask, gt_mask)
            print(f"{cat_name:<20} | {'Original':<12} | {cat_name:<20} | {iou_orig:.4f}")

            # 2. Synonym IoUs
            syn_ious = {}
            for s, s_prompt in zip(synonyms, maps["syns"]):
                mask_s = pred_masks.get(s_prompt, np.zeros((h, w), dtype=np.uint8))
                syn_ious[s] = calculate_iou(mask_s, gt_mask)
                print(f"{'':<20} | {'Synonym':<12} | {s:<20} | {syn_ious[s]:.4f}")

            # 3. Hyponym IoUs
            hypo_ious = {}
            for h_w, h_prompt in zip(hyponyms, maps["hypos"]):
                mask_h = pred_masks.get(h_prompt, np.zeros((h, w), dtype=np.uint8))
                hypo_ious[h_w] = calculate_iou(mask_h, gt_mask)
                print(f"{'':<20} | {'Hyponym':<12} | {h_w:<20} | {hypo_ious[h_w]:.4f}")
            if not hyponyms:
                print(f"{'':<20} | {'Hyponym':<12} | {'(None)':<20} | N/A")

            # 4. Hypernym IoUs
            hyper_ious = {}
            for hp, hp_prompt in zip(hypernyms, maps["hypers"]):
                mask_hp = pred_masks.get(hp_prompt, np.zeros((h, w), dtype=np.uint8))
                hyper_ious[hp] = calculate_iou(mask_hp, gt_mask)
                print(f"{'':<20} | {'Hypernym':<12} | {hp:<20} | {hyper_ious[hp]:.4f}")
            if not hypernyms:
                print(f"{'':<20} | {'Hypernym':<12} | {'(None)':<20} | N/A")

            # 5. Blended Embedding prediction
            iou_blend = None
            if hasattr(model, "predict_with_embeddings"):
                # Compute blended embedding using synonyms, hyponyms, and hypernyms
                query_emb = bm_hp.get_text_embedding_cached(cat_name)
                neighbors = bm_hp.top_k_neighbors(cat_name, synonyms+hyponyms+hypernyms, k=5)
                
                if not neighbors:
                    blended_emb = query_emb
                    tech_info = "no blending"
                else:
                    centroid = bm_hp.compute_centroid(neighbors)
                    blended_emb = bm_hp.blend_embedding(query_emb, centroid, alpha=0.7)
                    tech_info = f"blend(k={len(neighbors)})"
                    
                pred_blend = model.predict_with_embeddings(image_pil, {cat_name: blended_emb}, threshold=args.threshold)
                mask_blend = pred_blend.get(cat_name, np.zeros((h, w), dtype=np.uint8))
                iou_blend = calculate_iou(mask_blend, gt_mask)
                print(f"{'':<20} | {'Blended':<12} | {tech_info:<20} | {iou_blend:.4f}")
            else:
                print(f"{'':<20} | {'Blended':<12} | {'(No support)':<20} | N/A")

            # Save to structured results
            structured_results[img_id][cat_name] = {
                "orig_iou": iou_orig,
                "syn_ious": syn_ious,
                "hypo_ious": hypo_ious,
                "hyper_ious": hyper_ious
            }
            if iou_blend is not None:
                structured_results[img_id][cat_name]["blended_iou"] = iou_blend

            print("-" * 105)

    # Save cache updates if we requested any new embeddings
    bm_hp.save_embedding_cache()
    print("=" * 105)

    # Save results to output path
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    if args.output.endswith(".json"):
        import json
        with open(args.output, "w") as f:
            json.dump(structured_results, f, indent=2)
        print(f"Saved results successfully in JSON format to: {args.output}")
    else:
        with open(args.output, "wb") as f:
            pickle.dump(structured_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved results successfully in pickle format to: {args.output}")

    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
