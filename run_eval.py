#!/usr/bin/env python3
import argparse
import sys
import os
import subprocess

# Bypass sitecustomize.py model logging crash due to missing 'logger' command
_original_popen = subprocess.Popen
def _patched_popen(args, *popenargs, **kwargs):
    if isinstance(args, list) and len(args) > 0 and args[0] == "logger":
        # Redirect to a dummy python call that does nothing
        args = [sys.executable, "-c", "pass"]
    return _original_popen(args, *popenargs, **kwargs)
subprocess.Popen = _patched_popen

import torch
import numpy as np
from pycocotools.coco import COCO

from ovs_eval.models import get_model, list_models
from ovs_eval.evaluation import run_evaluation, compute_lss_metrics
from ovs_eval.visualization import plot_taxonomy_evaluation, plot_lss_analysis

def main():
    parser = argparse.ArgumentParser(description="Taxonomy-Driven OVS Models Evaluation Pipeline")
    parser.add_argument("--model", type=str, required=True, choices=list_models(),
                        help=f"Model architecture to evaluate. Choose from: {list_models()}")
    parser.add_argument("--num-images", type=int, default=500,
                        help="Number of images from the COCO validation dataset to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for segmentation masks.")
    parser.add_argument("--desc", action="store_true",
                        help="Use descriptions instead of 'a photo of a <category>' prompt template.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a fast execution test with a single image.")
    parser.add_argument("--output", type=str, default="segmentation_results.json",
                        help="JSON file path to save computed segmentation masks and results.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to external model weights (for OpenSeg TF, OVSeg, SAN).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config (for OVSeg, SAN).")
    parser.add_argument("--use-actual", action="store_true",
                        help="If set, uses the actual complex wrapper from ovseg_runnable.py instead of the notebook baseline.")
    parser.add_argument("--plots-dir", type=str, default="visualizations",
                        help="Directory to save the generated visualization plots.")
    
    args = parser.parse_args()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device selected: {device}")

    # Set up annotations path
    ann_path = os.path.join("annotations", "instances_val2017.json")
    if not os.path.exists(ann_path):
        print(f"Error: COCO annotations not found at '{ann_path}'. Please check directory configuration.")
        sys.exit(1)

    print("Loading COCO validation annotations...")
    coco = COCO(ann_path)
    image_ids = coco.getImgIds()
    print(f"Loaded {len(image_ids)} total images from COCO, using {args.num_images}.")

    # Instantiate model
    print(f"Initializing model '{args.model}'...")
    try:
        model = get_model(
            name=args.model,
            device=device,
            weights=args.weights,
            config=args.config,
            baseline=not args.use_actual
        )
    except Exception as e:
        print(f"Failed to initialize model '{args.model}': {e}")
        sys.exit(1)

    # Determine subset of images to evaluate
    if args.smoke_test:
        # Use a single test image (e.g. 494869 as in the notebooks) or fallback to the first image ID
        test_img_id = 494869
        if test_img_id not in image_ids:
            test_img_id = image_ids[0]
        subset_ids = [test_img_id]
        print(f"Running smoke test on image ID: {test_img_id}")
    else:
        # Take the specified slice of images
        subset_ids = image_ids[:args.num_images]
        print(f"Running evaluation on first {args.num_images} images.")

    # Run evaluation
    results = run_evaluation(
        model=model,
        coco=coco,
        image_ids=subset_ids,
        threshold=args.threshold,
        desc=args.desc,
        output_file=args.output
    )

    # Compute and report LSS metrics
    print("\nComputing Linguistic Sensitivity Metrics...")
    LSS_M, lss_results = compute_lss_metrics(results)
    
    # Print console summary
    print("\n" + "=" * 80)
    print("SEGMENTATION PERFORMANCE SUMMARY".center(80))
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Total Images Evaluated: {len(results.get('Original', []))}")
    print("-" * 80)
    
    for cat_type in ['Original', 'Synonyms', 'Hypernyms', 'Hyponyms']:
        cat_results = results.get(cat_type, [])
        if cat_results:
            mious = [r['miou'] for r in cat_results]
            print(f"{cat_type:<12} | Mean mIoU: {np.mean(mious):.4f} | Std Dev: {np.std(mious):.4f}")
            
    print("-" * 80)
    print(f"Model-level LSS(M): {LSS_M:.4f}")
    print(f"Average spread over {len(lss_results)} evaluated classes.")
    print("=" * 80)

    # Plot results
    print("\nGenerating evaluation charts...")
    model_plots_dir = os.path.join(args.plots_dir, args.model)
    os.makedirs(model_plots_dir, exist_ok=True)
    plot_taxonomy_path = os.path.join(model_plots_dir, "taxonomy_ovs_evaluation.png")
    plot_lss_path = os.path.join(model_plots_dir, "lss_line_chart.png")
    plot_taxonomy_evaluation(results, save_path=plot_taxonomy_path)
    plot_lss_analysis(LSS_M, lss_results, save_path=plot_lss_path)
    
    print("\nEvaluation successfully completed!")
    print("Saved plots:")
    print(f"  - {plot_taxonomy_path}")
    print(f"  - {plot_lss_path}")

if __name__ == "__main__":
    main()
