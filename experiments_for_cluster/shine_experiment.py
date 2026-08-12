import argparse
import os

from experiment_common import RESULTS_DIR, run_query_transform_experiment
import expanded_benchmark_helpers as bm_hp

OUT_PATH = os.path.join(RESULTS_DIR, "shine_experiment.csv")


def shine_embedding(word):
    synset = bm_hp.find_best_synset(word)
    if synset and synset.hypernyms():
        hyper = bm_hp.to_display_form(synset.hypernyms()[0].lemma_names()[0])
        prompt = f"a photo of a {word}, a type of {hyper}"
    else:
        prompt = f"a photo of a {word}"
    return bm_hp.get_text_embedding_cached(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()
    run_query_transform_experiment(args.out, "shine", shine_embedding, args.limit_categories, args.limit_images)
