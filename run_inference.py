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
    parser = argparse.ArgumentParser(description="Run single-image inference and verify cached ground truth.")
    parser.add_argument("--model", type=str, default="clipseg", choices=list_models(),
                        help=f"Model to evaluate. Choose from: {list_models()}")
    parser.add_argument("--index", type=int, default=None,
                        help="Index of the image in the metadata cache to test.")
    parser.add_argument("--img-id", type=str, default=None,
                        help="Image ID (e.g. 'coco_494869' or 'ade20k_366') to test.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for prediction masks.")
    parser.add_argument("--desc", action="store_true",
                        help="Use descriptions instead of template prompts.")
    parser.add_argument("--metadata-path", type=str, default="data/img_metadata.pkl",
                        help="Path to the image metadata cache file.")
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

    # Find the image entry
    img_data = None
    if args.img_id is not None:
        for img in img_bnch:
            if img["img_id"] == args.img_id:
                img_data = img
                break
        if img_data is None:
            print(f"Error: Image ID '{args.img_id}' not found in cache.")
            sys.exit(1)
    elif args.index is not None:
        if args.index < 0 or args.index >= len(img_bnch):
            print(f"Error: Index {args.index} out of range (0 to {len(img_bnch)-1}).")
            sys.exit(1)
        img_data = img_bnch[args.index]
    else:
        # Default to first image
        img_data = img_bnch[0]
        print(f"No index or img-id specified. Using the first image in cache (index 0).")

    img_id = img_data["img_id"]
    img_src = img_data["img_src"]
    img_src_id = img_data["img_src_id"]
    w, h = img_data["width"], img_data["height"]
    gt_objects = img_data["gt_objects"]

    print(f"\nTarget Image Details:")
    print(f"  - Image ID: {img_id}")
    print(f"  - Source: {img_src} (original ID: {img_src_id})")
    print(f"  - Dimensions: {w}x{h}")
    print(f"  - Ground Truth Objects: {gt_objects}")

    # Load original image
    print(f"\nLoading original image...")
    image_pil = None

    if img_src == "coco":
        # Load COCO annotations for the image reader
        print("Loading COCO annotations helper...")
        from pycocotools.coco import COCO
        if not os.path.exists(bm_hp.COCO_ANN):
            print(f"Error: COCO annotations not found at {bm_hp.COCO_ANN}")
            sys.exit(1)
        coco_obj = COCO(bm_hp.COCO_ANN)
        image_pil = get_img(img_src_id, coco_obj)
    elif img_src == "ade20k":
        # Load ADE20K dataset to get the raw image
        print("Loading ADE20K dataset helper...")
        ade = bm_hp.load_ade20k()
        ade_filenames = ade["filename"]
        # Find matching filename
        filename = img_data["filename"]
        if filename in ade_filenames:
            ade_idx = ade_filenames.index(filename)
            image_pil = ade[ade_idx]["image"]
        else:
            print(f"Error: Could not find filename '{filename}' in the ADE20K dataset.")
            sys.exit(1)

    if image_pil is None:
        print("Error: Failed to load PIL Image.")
        sys.exit(1)

    # Initialize model
    print(f"\nInitializing model '{args.model}'...")
    model = get_model(name=args.model, device=device)

    # Decompress cached ground truth masks
    print("Decompressing cached ground truth masks...")
    gt_masks = []
    for compressed_mask in img_data["gt_bin_masks"]:
        mask = np.frombuffer(zlib.decompress(compressed_mask), dtype=np.uint8).reshape((h, w))
        gt_masks.append(mask)

    # Run predictions and compute IoUs
    print(f"\nRunning model prediction and calculating IoU vs cached ground truth:")
    print("=" * 80)
    print(f"{'Category Name':<20} | {'IoU':<6}")
    print("-" * 80)

    
    pred_masks = model.predict_v2(image_pil, gt_masks, gt_objects, threshold=args.threshold, desc=args.desc)

    for idx, cat in enumerate(gt_objects):

        gt_mask = gt_masks[idx]
        pred_mask = pred_masks.get(cat, np.zeros((h, w), dtype=np.uint8))

        iou = calculate_iou(pred_mask, gt_mask)
        print(f"{cat:<20} | {iou:.4f}")
    
    print("=" * 80)
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
