import argparse
import os

from experiment_common import RESULTS_DIR, run_query_transform_experiment
import expanded_benchmark_helpers as bm_hp

OUT_PATH = os.path.join(RESULTS_DIR, "prompt_template_experiment.csv")


def templated_embedding(word):
    return bm_hp.get_text_embedding_cached(f"Image of a {word}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()
    run_query_transform_experiment(
        args.out, "image_of_a", templated_embedding, args.limit_categories, args.limit_images
    )
