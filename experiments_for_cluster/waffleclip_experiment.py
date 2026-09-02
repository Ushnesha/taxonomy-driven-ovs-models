"""WaffleCLIP baseline (Roth et al., ICCV 2023, "Waffling around for Performance: Visual
Classification with Random Words and Broad Concepts") convenience entry point -- thin wrapper
over positive_set_experiment.run() with approaches locked to baseline+waffleclip.

The real implementation lives in approaches.py's waffleclip_embedding() / waffle_descriptors()
(ported from the official https://github.com/ExplainableML/WaffleCLIP repo -- see that module's
docstring for the port notes: structured_descriptor_builder template, waffle_count=15 random
base-word + noise-word descriptors drawn once per run from waffleclip_word_list.pkl, mean-pooled
embedding). This file exists only so `python3 waffleclip_experiment.py --model X` matches the
one-script-per-baseline naming the team expects -- `python3 positive_set_experiment.py --model X
--approaches baseline,waffleclip` does exactly the same thing.

Usage:
    python3 waffleclip_experiment.py --model clipseg --limit-categories 5 --limit-images 3
    python3 waffleclip_experiment.py --model sclip   # full-scale, needs the sclip venv (see
                                                      # ../../models_reference/sclip/setup_env.sh)
"""
import argparse
import os

import approaches as ap
import benchmark_data as bd
import models as mdl
import positive_set_experiment as pse

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=mdl.SUPPORTED_MODELS)
    parser.add_argument("--waffle-count", type=int, default=ap.WAFFLE_COUNT)
    parser.add_argument("--benchmark-dir", default=None, help="defaults to ../../benchmark (or $BENCHMARK_DIR)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()
    ap.WAFFLE_COUNT = args.waffle_count

    out_path = os.path.join(args.out_dir, f"waffleclip_experiment_{args.model}.csv")
    summary_path = os.path.join(args.out_dir, f"waffleclip_experiment_{args.model}_summary.csv")

    pse.run(args.model, ["baseline", "waffleclip"], ap.DEFAULT_ALPHA, False, out_path, args.benchmark_dir,
            limit_categories=args.limit_categories, limit_images=args.limit_images, device=args.device)

    bd.summarize_csv(out_path, ["model", "approach", "variant"], "iou", summary_path)
    print(f"summary written to {summary_path}")
