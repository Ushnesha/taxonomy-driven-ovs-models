"""SHiNe baseline (Liu et al., CVPR 2024, "Semantic Hierarchy Nexus for Open-vocabulary
Object Detection") convenience entry point -- thin wrapper over
positive_set_experiment.run() with approaches locked to baseline+shine.

The real implementation lives in approaches.py's shine_embedding() / shine_candidate_sentences()
(ported from the official https://github.com/naver/shine repo -- see that module's docstring for
the port notes: SignatureComposer/Themer from shine/tools/composer.py + shine_cls/utils/themer.py,
hierarchy built from WordNet's synset.hypernym_paths() in place of SHiNe's curated iNat/FSOD/
ImageNet hierarchy trees). This file exists only so `python3 shine_experiment.py --model X`
matches the one-script-per-baseline naming the team expects -- `python3
positive_set_experiment.py --model X --approaches baseline,shine` does exactly the same thing.

Usage:
    python3 shine_experiment.py --model clipseg --limit-categories 5 --limit-images 3
    python3 shine_experiment.py --model sclip   # full-scale, needs the sclip venv (see
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
    parser.add_argument("--benchmark-dir", default=None, help="defaults to ../../benchmark (or $BENCHMARK_DIR)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    out_path = os.path.join(args.out_dir, f"shine_experiment_{args.model}.csv")
    summary_path = os.path.join(args.out_dir, f"shine_experiment_{args.model}_summary.csv")

    pse.run(args.model, ["baseline", "shine"], ap.DEFAULT_ALPHA, False, out_path, args.benchmark_dir,
            limit_categories=args.limit_categories, limit_images=args.limit_images, device=args.device)

    bd.summarize_csv(out_path, ["model", "approach", "variant"], "iou", summary_path)
    print(f"summary written to {summary_path}")
