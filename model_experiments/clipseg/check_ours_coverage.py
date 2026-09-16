"""How often can "ours" actually produce an embedding?

"ours" is the only approach that can fail: it needs the query word to resolve to a
WordNet synset with at least one *other* lemma to build a synonym centroid from.
Words like "New Jersey" or "Frisbee" have no such synset, so ours_embedding()
returns None and that (category, variant) writes no row -- while baseline / shine /
waffleclip / llm_descriptor all still score. The resulting coverage gap is what
makes a naive 5-way mean-IoU comparison unpaired.

This script measures that gap over the WHOLE benchmark before any GPU time is
spent, so the policy decision (let ours be absent, vs. fall back to the raw query
embedding = alpha 0 = baseline) is made on real numbers rather than on a handful of
dry-run categories.

No CLIPSeg, no images, no segmentation -- pure WordNet lookups. (Importing
approaches.py does pull in expanded_benchmark_helpers, which instantiates a small
all-MiniLM SentenceTransformer at import time; this script never uses it. Seconds
to run, not hours.)

It reproduces exactly the two model-free gates inside approaches.ours_blend_info():

    1. eligible_synset(query_word) is None          -> no usable synset
    2. no candidate synonyms other than the word    -> nothing to average

The third gate (_top_k_neighbors returning empty) needs a model, but it can only
trigger if every candidate fails to embed, which does not happen for real strings.
So these numbers are exact in practice.

Usage:
    python3 check_ours_coverage.py
    python3 check_ours_coverage.py --benchmark-dir ~/workspace/OVS/benchmark
    python3 check_ours_coverage.py --show-failures 40 --out-csv coverage.csv
"""
import argparse
import csv
import os

import benchmark_data as bd
import approaches as ap
from expanded_benchmark_helpers import to_wn_form, get_all_synsets

VARIANTS = ["orig", "syn", "hypo", "hyper"]

# Why a word can fail, most-informative first. The distinction matters for the
# writeup: "WordNet has never heard of Frisbee" and "WordNet knows 'bed' perfectly
# well but English has no synonym for it" are completely different claims, and
# eligible_synset() collapses both into a bare None.
REASONS = [
    ("not_in_wordnet",
     "no synset at all (proper nouns, brand names: Frisbee, New Jersey)"),
    ("singleton_only_sense",
     "in WordNet, but every sense is a lone lemma -- no synonym exists"),
    ("singleton_chosen_other_sense_has_synonyms",
     "chosen sense is a lone lemma, but ANOTHER sense has synonyms "
     "(recoverable by sense re-selection -- at the risk of a sense error)"),
    ("no_synonym_candidates",
     "passed the gate but nothing left after dropping the word itself"),
]


def resolvable(word):
    """The model-free half of ours_blend_info(): can we build a centroid at all?

    Returns (ok, reason). `reason` is "" when ok.
    """
    synset = ap.eligible_synset(word)
    if synset is not None:
        w_s = [ap.to_display_form(w)
               for w in ap.build_word_sets_from_synset(synset)["W_S"]]
        candidates = [w for w in w_s if w.lower() != word.lower()]
        if not candidates:
            return False, "no_synonym_candidates"
        return True, ""

    # eligible_synset() returned None. Two very different causes -- separate them.
    senses = get_all_synsets(to_wn_form(word))
    if not senses:
        return False, "not_in_wordnet"
    # The word IS in WordNet. find_best_synset() ranks senses by how prominently
    # the word appears in each lemma list, never by whether the sense actually has
    # synonyms -- so a singleton sense can win over a sibling sense that has them
    # (apple.n.01 ['apple'] beats apple.n.02 ['apple', 'orchard_apple_tree']).
    if any(len(s.lemma_names()) >= 2 for s in senses):
        return False, "singleton_chosen_other_sense_has_synonyms"
    return False, "singleton_only_sense"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default=None,
                        help="defaults to ../../../benchmark (or $BENCHMARK_DIR)")
    parser.add_argument("--exclude-sources", default="",
                        help="comma-separated img_src values to drop, e.g. 'ade20k'")
    parser.add_argument("--show-failures", type=int, default=25,
                        help="how many unresolvable words to print per variant")
    parser.add_argument("--out-csv", default=None,
                        help="write the per-(category, variant) detail here")
    args = parser.parse_args()

    exclude = tuple(x.strip() for x in args.exclude_sources.split(",") if x.strip())
    bm = bd.load_benchmark(args.benchmark_dir, exclude_sources=exclude)

    rows = []
    for cat_name in bm.categories:
        for vname, word in bd.get_variants(cat_name, bm.word_sets).items():
            ok, reason = resolvable(word)
            rows.append({"category": cat_name, "variant": vname, "word": word,
                         "resolvable": int(ok), "reason": reason})

    total = len(rows)
    ok_total = sum(r["resolvable"] for r in rows)

    print()
    print("=" * 72)
    print(f"OURS COVERAGE -- {len(bm.categories)} categories, {total} (category, variant) cells")
    print("=" * 72)
    print(f"\nOVERALL: {ok_total}/{total} resolvable  ({100.0 * ok_total / total:.1f}%)")

    # ---- per variant: this is the number that decides the policy, because a
    # ---- method that only engages on half the hypernyms is a finding in itself.
    print("\nPER VARIANT")
    print(f"  {'variant':<8} {'resolvable':>12} {'total':>7} {'coverage':>10}")
    for v in VARIANTS:
        vr = [r for r in rows if r["variant"] == v]
        if not vr:
            print(f"  {v:<8} {'(none)':>12}")
            continue
        n_ok = sum(r["resolvable"] for r in vr)
        print(f"  {v:<8} {n_ok:>12} {len(vr):>7} {100.0 * n_ok / len(vr):>9.1f}%")

    # ---- why it failed, split by cause (see REASONS)
    print("\nFAILURE REASONS")
    counts = {}
    for r in rows:
        if not r["resolvable"]:
            counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    n_failed = total - ok_total
    if not counts:
        print("  (none -- ours resolves every word)")
    for reason, blurb in REASONS:
        n = counts.get(reason, 0)
        if not n:
            continue
        print(f"  {reason:<42} {n:>5}  ({100.0 * n / n_failed:>4.1f}% of failures)")
        print(f"  {'':<42}        {blurb}")

    # ---- the split that matters for the paper: a word English has no synonym for
    # ---- is a limitation of the METHOD; a word WordNet has never seen is a
    # ---- limitation of the RESOURCE. They need separate sentences.
    n_absent = counts.get("not_in_wordnet", 0)
    n_singleton = (counts.get("singleton_only_sense", 0)
                   + counts.get("singleton_chosen_other_sense_has_synonyms", 0))
    if n_failed:
        print("\nHEADLINE SPLIT")
        print(f"  in WordNet, but no usable synonym : {n_singleton:>5}"
              f"  ({100.0 * n_singleton / total:.1f}% of all cells)")
        print(f"  not in WordNet at all             : {n_absent:>5}"
              f"  ({100.0 * n_absent / total:.1f}% of all cells)")
        n_recover = counts.get("singleton_chosen_other_sense_has_synonyms", 0)
        if n_recover:
            print(f"  of which recoverable by picking a different sense: {n_recover}"
                  f"  (+{100.0 * n_recover / total:.1f} pts coverage, with sense-error risk)")

    # ---- which words, grouped by CAUSE so sense errors are spottable by eye
    if args.show_failures:
        print("\nUNRESOLVABLE WORDS, BY CAUSE (sample)")
        for reason, _ in REASONS:
            bad = [r for r in rows if r["reason"] == reason]
            if not bad:
                continue
            shown = ", ".join(r["word"] for r in bad[:args.show_failures])
            more = (f"  (+{len(bad) - args.show_failures} more)"
                    if len(bad) > args.show_failures else "")
            print(f"\n  [{reason}] {len(bad)}")
            print(f"    {shown}{more}")

    # ---- what it means for the comparison
    print("\n" + "-" * 72)
    pct = 100.0 * ok_total / total
    if pct >= 95:
        print("COVERAGE IS HIGH (>=95%). Falling back to the raw query embedding")
        print("(alpha=0, identical to baseline) for the unresolvable words costs almost")
        print("nothing and makes every approach's n match. Recommended.")
    elif pct >= 80:
        print("COVERAGE IS MODERATE (80-95%). A baseline fallback still makes the")
        print("comparison paired, but report the applicable-subset mean alongside the")
        print("headline so the dilution is visible.")
    else:
        print("COVERAGE IS LOW (<80%). A baseline fallback would dilute 'ours' toward")
        print("'baseline' enough to hide the real effect. Better to keep the rows")
        print("absent and report coverage as a first-class result -- a method that")
        print("only engages on part of the benchmark is a genuine limitation.")
    print("-" * 72)

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["category", "variant", "word",
                                              "resolvable", "reason"])
            w.writeheader()
            w.writerows(rows)
        print(f"\ndetail written to {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
