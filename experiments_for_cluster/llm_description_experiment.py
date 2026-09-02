"""LLM-descriptor baseline (Menon & Vondrick, ICLR 2023, "Visual Classification via
Description from Large Language Models") convenience entry point -- thin wrapper over
positive_set_experiment.run() with approaches locked to baseline+llm_descriptor.

The real implementation lives in approaches.py's llm_descriptor_embedding() /
query_llm_descriptors() (ported from the official
https://github.com/sachit-menon/classify_by_description_release repo -- see that module's
docstring for the port notes: identical two-shot exemplar prompt, `stringtolist` bullet
parsing, one sentence per descriptor, mean-pooled embedding. Calls a local Ollama model instead
of the retired `text-davinci-003`; falls back to the WordNet gloss if no LLM is reachable). This
file exists only so `python3 llm_description_experiment.py --model X` matches the
one-script-per-baseline naming the team expects -- `python3 positive_set_experiment.py --model X
--approaches baseline,llm_descriptor` does exactly the same thing.

Usage:
    python3 llm_description_experiment.py --model clipseg --limit-categories 5 --limit-images 3
    python3 llm_description_experiment.py --model sclip   # full-scale, needs the sclip venv (see
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

    out_path = os.path.join(args.out_dir, f"llm_description_experiment_{args.model}.csv")
    summary_path = os.path.join(args.out_dir, f"llm_description_experiment_{args.model}_summary.csv")

    pse.run(args.model, ["baseline", "llm_descriptor"], ap.DEFAULT_ALPHA, False, out_path, args.benchmark_dir,
            limit_categories=args.limit_categories, limit_images=args.limit_images, device=args.device)

    bd.summarize_csv(out_path, ["model", "approach", "variant"], "iou", summary_path)
    print(f"summary written to {summary_path}")
