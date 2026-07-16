"""
Expanded COCO Benchmark Module
===============================
Algorithmic dataset construction + WordNet/BabelNet inference pipeline.

Usage:
  # Build the expanded benchmark dataset
  from expanded_benchmark.dataset_builder import build_all
  positive_set, negative_set, word_sets, report = build_all()

  # Inference pipeline
  from expanded_benchmark.inference import fetch_synset_groups, run_inference_pipeline
  groups = fetch_synset_groups("bank")  # user sees all meaning groups
  result = run_inference_pipeline("bank", "bank.n.01", image, alpha=0.7)
"""

from .helpers import (
    COCO_80, COCO_ANN, DATA_DIR,
    to_wn_form, to_display_form,
    load_model, get_device,
    get_text_embedding, get_text_embedding_cached,
    get_synset_groups_for_display,
    find_best_synset,
    build_word_sets_from_synset,
    supplement_word_sets_with_babelnet,
    compute_centroid, blend_embedding, top_k_neighbors,
    run_segmentation, compute_iou,
    load_coco, download_image, get_gt_mask,
)

from .dataset_builder import (
    build_all,
    build_positive_negative_sets,
    build_word_sets,
)

from .inference import (
    fetch_synset_groups,
    build_word_set_for_selected_synset,
    run_inference_pipeline,
    run_inference_on_coco_image,
)
