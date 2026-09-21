"""Read a positive/negative detail CSV and report what the summary CSV cannot.

The summary CSV written by the experiments averages every row an approach
produced. That is fine for the four approaches with full coverage, but "ours"
returns no embedding when a word has no WordNet synonyms (~50% of the
benchmark), so its mean comes from a DIFFERENT, systematically easier subset of
rows. Comparing those means directly conflates two things: the method, and the
fact that ours sat out the hard cases.

This script reports, side by side:

  1. ALL-AVAILABLE   -- every row, i.e. what the summary CSV already says.
  2. PAIRED          -- only (category, img_id, variant) cells where EVERY
                        approach produced a row. This is the fair comparison
                        and the one to quote.
  3. COVERAGE        -- what fraction of cells each approach could score, so the
                        gap is reported rather than hidden.
  4. LADDER          -- the orig->syn->hypo->hyper degradation, restricted to
                        categories that have all four variants (the per-variant
                        row counts differ otherwise, so the unrestricted ladder
                        compares different category subsets).
  5. PER-CATEGORY    -- per-category deltas vs baseline for one variant, to see
                        WHERE a win or loss comes from rather than just that it
                        happened.

Works on both experiments: the value column is detected automatically (`iou`
for positive, `fpr` for negative) and the sign convention flips accordingly --
for FPR, lower is better.

Usage:
    python3 analyze_benchmark.py positive_set_experiment_clipseg.csv
    python3 analyze_benchmark.py <detail.csv> --per-category hyper
    python3 analyze_benchmark.py <detail.csv> --out-dir analysis/
"""
import argparse
import csv
import os
from collections import defaultdict

VARIANTS = ["orig", "syn", "hypo", "hyper"]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def load(path):
    """rows + the detected value field ('iou' or 'fpr')."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no data rows")
    for field in ("iou", "fpr"):
        if field in rows[0]:
            break
    else:
        raise SystemExit(f"{path}: expected an 'iou' or 'fpr' column, got {list(rows[0])}")
    for r in rows:
        r[field] = float(r[field])
    return rows, field


def approaches_in(rows):
    """baseline first (it is the reference), then the rest alphabetically."""
    found = sorted({r["approach"] for r in rows})
    if "baseline" in found:
        found.remove("baseline")
        found.insert(0, "baseline")
    return found


def cells(rows):
    """{(category, img_id, variant): {approach: value}} -- one cell is one
    measurement opportunity that every approach had an equal shot at."""
    out = defaultdict(dict)
    for r in rows:
        out[(r["category"], r["img_id"], r["variant"])][r["approach"]] = r[field]
    return out


def _table(title, per_variant, approaches, value_field, note=""):
    better = "higher is better" if value_field == "iou" else "LOWER is better"
    print(f"\n{title}  ({value_field}, {better})")
    if note:
        print(f"  {note}")
    print(f"  {'approach':<16}" + "".join(f"{v:>12}" for v in VARIANTS))
    for a in approaches:
        line = f"  {a:<16}"
        for v in VARIANTS:
            vals = per_variant.get((a, v))
            line += f"{mean(vals):>12.4f}" if vals else f"{'-':>12}"
        print(line)

    print(f"\n  vs baseline (% change, sign-corrected so + always means BETTER)")
    print(f"  {'approach':<16}" + "".join(f"{v:>12}" for v in VARIANTS))
    for a in approaches:
        if a == "baseline":
            continue
        line = f"  {a:<16}"
        for v in VARIANTS:
            av, bv = per_variant.get((a, v)), per_variant.get(("baseline", v))
            if not av or not bv:
                line += f"{'-':>12}"
                continue
            am, bm = mean(av), mean(bv)
            pct = 100.0 * (am - bm) / bm if bm else float("nan")
            if value_field == "fpr":
                pct = -pct  # fewer false positives is an improvement
            line += f"{pct:>+11.1f}%"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("detail_csv")
    parser.add_argument("--per-category", default="hyper", choices=VARIANTS,
                        help="which variant to break down per category (default: hyper, "
                             "the variant whose benchmark result contradicts the COCO sweep)")
    parser.add_argument("--top", type=int, default=15,
                        help="how many categories to show at each end")
    parser.add_argument("--out-dir", default=None,
                        help="also write paired_summary.csv / coverage.csv / per_category.csv here")
    args = parser.parse_args()

    global field
    rows, field = load(args.detail_csv)
    approaches = approaches_in(rows)
    by_cell = cells(rows)
    n_app = len(approaches)

    print("=" * 78)
    print(f"{os.path.basename(args.detail_csv)}")
    print(f"{len(rows):,} rows | {len(by_cell):,} (category, img_id, variant) cells "
          f"| {len(approaches)} approaches: {', '.join(approaches)}")
    print("=" * 78)

    # ---------- 1. all available (what the summary CSV reports) ----------
    all_pv = defaultdict(list)
    for (cat, img, var), got in by_cell.items():
        for a, v in got.items():
            all_pv[(a, var)].append(v)
    _table("1. ALL AVAILABLE", all_pv, approaches, field,
           "every row -- ours is averaged over different cells than the others")

    # ---------- 2. paired ----------
    paired_pv = defaultdict(list)
    n_paired = 0
    for (cat, img, var), got in by_cell.items():
        if len(got) != n_app:
            continue
        n_paired += 1
        for a, v in got.items():
            paired_pv[(a, var)].append(v)
    _table("2. PAIRED  <-- QUOTE THIS ONE", paired_pv, approaches, field,
           f"only the {n_paired:,} cells ({100.0*n_paired/len(by_cell):.1f}%) "
           f"where all {n_app} approaches scored")

    # ---------- 3. coverage ----------
    print("\n3. COVERAGE  (cells this approach could score, per variant)")
    print(f"  {'approach':<16}" + "".join(f"{v:>12}" for v in VARIANTS))
    totals = defaultdict(int)
    for (cat, img, var) in by_cell:
        totals[var] += 1
    for a in approaches:
        line = f"  {a:<16}"
        for v in VARIANTS:
            got = len(all_pv.get((a, v), []))
            tot = totals.get(v, 0)
            line += f"{100.0*got/tot:>11.1f}%" if tot else f"{'-':>12}"
        print(line)

    # ---------- 4. degradation ladder on categories having all 4 variants ----------
    vars_by_cat = defaultdict(set)
    for (cat, img, var) in by_cell:
        vars_by_cat[cat].add(var)
    full_cats = {c for c, vs in vars_by_cat.items() if set(VARIANTS) <= vs}
    print(f"\n4. DEGRADATION LADDER  ({len(full_cats)} of {len(vars_by_cat)} categories "
          f"have all 4 variants; restricted to those so each step compares like with like)")
    ladder = defaultdict(list)
    for (cat, img, var), got in by_cell.items():
        if cat in full_cats and len(got) == n_app:
            for a, v in got.items():
                ladder[(a, var)].append(v)
    print(f"  {'approach':<16}" + "".join(f"{v:>12}" for v in VARIANTS) + f"{'orig->hyper':>14}")
    for a in approaches:
        line = f"  {a:<16}"
        for v in VARIANTS:
            vals = ladder.get((a, v))
            line += f"{mean(vals):>12.4f}" if vals else f"{'-':>12}"
        o, h = ladder.get((a, "orig")), ladder.get((a, "hyper"))
        line += f"{100.0*(mean(h)-mean(o))/mean(o):>+13.1f}%" if o and h else f"{'-':>14}"
        print(line)

    # ---------- 5. per-category, one variant, paired ----------
    var = args.per_category
    per_cat = defaultdict(lambda: defaultdict(list))
    for (cat, img, v), got in by_cell.items():
        if v == var and len(got) == n_app:
            for a, val in got.items():
                per_cat[cat][a].append(val)
    deltas = []
    for cat, got in per_cat.items():
        if "baseline" not in got or "ours" not in got:
            continue
        d = mean(got["ours"]) - mean(got["baseline"])
        if field == "fpr":
            d = -d
        deltas.append((d, cat, mean(got["baseline"]), mean(got["ours"]), len(got["baseline"])))
    deltas.sort(reverse=True)

    print(f"\n5. PER CATEGORY -- ours vs baseline on '{var}' (paired cells only, "
          f"{len(deltas)} categories)")
    if deltas:
        helped = sum(1 for d, *_ in deltas if d > 0)
        print(f"  ours helped {helped}/{len(deltas)} categories, hurt {len(deltas)-helped}")
        print(f"\n  {'BEST':<6}{'category':<28}{'baseline':>10}{'ours':>10}{'delta':>10}{'n':>6}")
        for d, cat, b, o, n in deltas[:args.top]:
            print(f"  {'':<6}{cat[:27]:<28}{b:>10.4f}{o:>10.4f}{d:>+10.4f}{n:>6}")
        print(f"\n  {'WORST':<6}{'category':<28}{'baseline':>10}{'ours':>10}{'delta':>10}{'n':>6}")
        for d, cat, b, o, n in deltas[-args.top:]:
            print(f"  {'':<6}{cat[:27]:<28}{b:>10.4f}{o:>10.4f}{d:>+10.4f}{n:>6}")

    # ---------- optional CSVs ----------
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        p = os.path.join(args.out_dir, "paired_summary.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["approach", "variant", "n", f"mean_{field}"])
            for a in approaches:
                for v in VARIANTS:
                    vals = paired_pv.get((a, v))
                    if vals:
                        w.writerow([a, v, len(vals), mean(vals)])
        p2 = os.path.join(args.out_dir, "coverage.csv")
        with open(p2, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["approach", "variant", "scored", "total", "coverage_pct"])
            for a in approaches:
                for v in VARIANTS:
                    got, tot = len(all_pv.get((a, v), [])), totals.get(v, 0)
                    if tot:
                        w.writerow([a, v, got, tot, 100.0 * got / tot])
        p3 = os.path.join(args.out_dir, f"per_category_{var}.csv")
        with open(p3, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["category", f"baseline_{field}", f"ours_{field}", "delta_better", "n"])
            for d, cat, b, o, n in deltas:
                w.writerow([cat, b, o, d, n])
        print(f"\nwrote {p}\n      {p2}\n      {p3}")


if __name__ == "__main__":
    main()
