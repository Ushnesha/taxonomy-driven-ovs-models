import argparse
import os

import requests

from experiment_common import RESULTS_DIR, run_query_transform_experiment
import expanded_benchmark_helpers as bm_hp

OUT_PATH = os.path.join(RESULTS_DIR, "llm_description_experiment.csv")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")


def llm_description(word):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": f"List 3 short visual features of a {word}, comma separated, no other text.",
            "stream": False,
        }, timeout=5)
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        if text:
            return text
    except Exception:
        pass
    synset = bm_hp.find_best_synset(word)
    if synset and synset.definition():
        return synset.definition()
    return word


def description_embedding(word):
    prompt = f"{word}, which is {llm_description(word)}"
    return bm_hp.get_text_embedding_cached(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--limit-categories", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()
    run_query_transform_experiment(
        args.out, "llm_description", description_embedding, args.limit_categories, args.limit_images
    )
