"""Merge sharded detail CSVs from a job array into one, then re-summarize.

`--category-shard I/N` makes each array task write its own
`<kind>_set_experiment_sclip_shardIofN.csv`, because two processes appending to one
CSV interleave rows and corrupt it. This stitches them back together.

The shards partition categories (categories[i::n]), so they are DISJOINT and a plain
concatenation is correct -- but this still deduplicates on (category, variant, img_id,
approach), because a shard resumed after a timeout, or an earlier sequential run passed
via --also-seen-from, can legitimately contribute the same row twice.

It also reports per-shard row counts and category coverage, so a shard that died early
is obvious rather than silently leaving a hole in the results.

stdlib only -- runs on the login node.

Usage:
    python3 merge_shards.py ~/workspace/OVS/results/positive_sclip_20260928_120000
    python3 merge_shards.py <run_dir> --value iou          # auto-detected otherwise
"""
import argparse
import csv
import glob
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--pattern", default="*_shard*of*.csv",
                    help="shard files to merge (default: *_shard*of*.csv)")
    ap.add_argument("--extra", default="",
                    help="comma-separated additional detail CSVs to fold in, e.g. the "
                         "sequential run's CSV from before you sharded")
    ap.add_argument("--value", default=None,
                    help="value column; auto-detected as iou or fpr")
    ap.add_argument("--out", default=None,
                    help="merged detail CSV (default: <run_dir>/merged_detail.csv)")
    args = ap.parse_args()

    run_dir = os.path.expanduser(args.run_dir)
    files = sorted(glob.glob(os.path.join(run_dir, args.pattern)))
    files += [x.strip() for x in args.extra.split(",") if x.strip()]
    if not files:
        raise SystemExit(f"ABORT: no files matching {args.pattern} in {run_dir}")

    print(f"{len(files)} source file(s):")
    fieldnames = None
    rows, seen = [], set()
    per_file = {}
    for path in files:
        if not os.path.isfile(path):
            print(f"  MISSING {path}")
            continue
        with open(path) as f:
            r = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(r.fieldnames or [])
            n_in = n_new = 0
            cats = set()
            for row in r:
                n_in += 1
                cats.add(row.get("category"))
                key = (row.get("category"), row.get("variant"),
                       row.get("img_id"), row.get("approach"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                n_new += 1
            per_file[path] = (n_in, n_new, len(cats))
            print(f"  {os.path.basename(path):<52} {n_in:>8,} rows "
                  f"({n_new:>8,} new, {len(cats):>4} categories)")

    if not rows:
        raise SystemExit("ABORT: no rows merged")

    value = args.value
    if value is None:
        value = "iou" if "iou" in (fieldnames or []) else "fpr"
    out = args.out or os.path.join(run_dir, "merged_detail.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    dupes = sum(n_in for n_in, _, _ in per_file.values()) - len(rows)
    all_cats = {r.get("category") for r in rows}
    print(f"\nmerged {len(rows):,} unique rows ({dupes:,} duplicates dropped)")
    print(f"{len(all_cats)} distinct categories -> {out}")

    # ---- summary, grouped like the experiments' own ----
    by = defaultdict(list)
    bad = 0
    for r in rows:
        try:
            by[(r["approach"], r["variant"])].append(float(r[value]))
        except (KeyError, ValueError):
            bad += 1
    if bad:
        print(f"  ({bad} rows skipped: missing/unparseable '{value}')")

    summary = os.path.join(run_dir, f"merged_summary_{value}.csv")
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["approach", "variant", "n", f"mean_{value}"])
        for (a, v), vals in sorted(by.items()):
            w.writerow([a, v, len(vals), sum(vals) / len(vals)])
    print(f"summary -> {summary}")

    print(f"\n  {'approach':<16}{'variant':<8}{'n':>9}{'mean':>10}")
    for (a, v), vals in sorted(by.items()):
        print(f"  {a:<16}{v:<8}{len(vals):>9,}{sum(vals)/len(vals):>10.4f}")

    print("\nNOTE: shard coverage above should total the full category count. A shard with")
    print("far fewer categories than its siblings died early -- resubmit just that index")
    print("with the same --out-dir before trusting this summary.")


if __name__ == "__main__":
    main()
