"""Audit the benchmark's syn/hypo/hyper words for wrong relations.

Spotted by eye in an SCLIP detail CSV:

    wheel, syn,  bicycle          <- a bicycle is not a synonym of a wheel
    wheel, hypo, mountain bike    <- nor a hyponym; these are PART-OF relations

Scoring "bicycle" against a wheel mask guarantees a near-zero IoU that has nothing to
do with a model's linguistic sensitivity. If that is widespread, part of the measured
orig->syn->hypo->hyper degradation is benchmark noise rather than model brittleness --
which is the first thing a reviewer will probe.

This reads word_sets_v2.json directly and flags the variant words that get_variants()
would actually pick (the FIRST entry at each level, which is what the experiments use),
against several heuristics:

  collides_with_other_category  the variant word is itself a separate benchmark
                                category. Strong smell: a real synonym should not be a
                                distinct class competing in the same vocabulary, and
                                part-whole confusions (wheel/bicycle) land here.
  same_as_orig                  variant identical to the base word -> measures nothing
  very_abstract_hypernym        entity/object/thing/artifact/whole/matter: technically
                                correct, visually useless
  proper_noun                   capitalised mid-list (Roger Bannister artifacts)
  multiword_gt4                 4+ words; usually a WordNet gloss fragment, not a word

None of these is proof of an error -- they are candidates for a human to read. The point
is to get a RATE: "N% of variant words are suspect" is a sentence you can put in a
limitations section, and a 2% rate means something very different from a 25% one.

stdlib only. No GPU, no model, no benchmark_data import (so no SentenceTransformer).

Usage:
    python3 check_word_sets.py
    python3 check_word_sets.py --benchmark-dir ~/workspace/OVS/benchmark
    python3 check_word_sets.py --show 40 --out-csv suspect_words.csv
"""
import argparse
import csv
import json
import os

LEVELS = [("syn", "synonyms"), ("hypo", "hyponyms"), ("hyper", "hypernyms")]

VERY_ABSTRACT = {
    "entity", "object", "physical object", "thing", "whole", "artifact", "artefact",
    "matter", "substance", "unit", "part", "device", "instrumentality", "instrument",
    "article", "structure", "material", "medium", "physical entity", "abstraction",
}


def to_display(w):
    """Mirror to_display_form: WordNet underscores become spaces."""
    return str(w).replace("_", " ").strip()


def first_variant(entry, key, cat):
    """What get_variants() would pick at this level: the first candidate, with
    synonyms filtered for being identical to the base word (matching that function)."""
    cands = [to_display(w) for w in entry.get(key, [])]
    if key == "synonyms":
        cands = [w for w in cands if w.lower() != cat.lower()]
    return cands[0] if cands else None


def flags_for(cat, level, word, all_categories):
    f = []
    if word.lower() == cat.lower():
        f.append("same_as_orig")
    if word.lower() in all_categories and word.lower() != cat.lower():
        f.append("collides_with_other_category")
    if level == "hyper" and word.lower() in VERY_ABSTRACT:
        f.append("very_abstract_hypernym")
    if word[:1].isupper() and not cat[:1].isupper():
        f.append("proper_noun")
    if len(word.split()) >= 4:
        f.append("multiword_gt4")
    return f


def main():
    ap = argparse.ArgumentParser()
    default = os.environ.get("BENCHMARK_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmark")
    ap.add_argument("--benchmark-dir", default=default)
    ap.add_argument("--show", type=int, default=25, help="examples to print per flag")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    path = os.path.join(os.path.expanduser(args.benchmark_dir), "word_sets_v2.json")
    if not os.path.isfile(path):
        raise SystemExit(f"ABORT: no word_sets_v2.json at {path}")
    with open(path) as f:
        word_sets = json.load(f)

    all_categories = {c.lower() for c in word_sets}
    print(f"{len(word_sets)} categories in {path}\n")

    rows, by_level, flagged_by_level = [], {}, {}
    for cat, entry in word_sets.items():
        for level, key in LEVELS:
            w = first_variant(entry, key, cat)
            if w is None:
                continue
            by_level[level] = by_level.get(level, 0) + 1
            fl = flags_for(cat, level, w, all_categories)
            if fl:
                flagged_by_level[level] = flagged_by_level.get(level, 0) + 1
            rows.append({"category": cat, "level": level, "word": w,
                         "flags": "|".join(fl)})

    print("=" * 74)
    print("RATE OF SUSPECT VARIANT WORDS (the word each experiment actually uses)")
    print("=" * 74)
    print(f"  {'level':<8}{'variants':>10}{'flagged':>10}{'rate':>9}")
    tot = totf = 0
    for level, _ in LEVELS:
        n, nf = by_level.get(level, 0), flagged_by_level.get(level, 0)
        tot += n
        totf += nf
        if n:
            print(f"  {level:<8}{n:>10}{nf:>10}{100.0*nf/n:>8.1f}%")
    if tot:
        print(f"  {'ALL':<8}{tot:>10}{totf:>10}{100.0*totf/tot:>8.1f}%")

    counts = {}
    for r in rows:
        for fl in (r["flags"].split("|") if r["flags"] else []):
            counts[fl] = counts.get(fl, 0) + 1
    print("\nBY FLAG")
    for fl, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {fl:<32}{n:>6}")

    for fl, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
        ex = [r for r in rows if fl in r["flags"]][:args.show]
        print(f"\n[{fl}] {counts[fl]} cases, showing {len(ex)}")
        for r in ex:
            print(f"    {r['category']:<24} --{r['level']:<6}-> {r['word']}")

    print("\n" + "-" * 74)
    print("collides_with_other_category is the one to read first: that is the")
    print("wheel->bicycle signature, and those rows are scored against the WRONG mask.")
    print("A high rate there means some of the measured degradation is benchmark error.")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["category", "level", "word", "flags"])
            w.writeheader()
            w.writerows(r for r in rows if r["flags"])
        print(f"\nflagged rows -> {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
