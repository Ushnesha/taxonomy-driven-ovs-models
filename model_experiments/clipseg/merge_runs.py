"""
Merge alpha-sweep run directories into one, then re-summarize.

Use when the coarse and fine grids were run as SEPARATE parallel jobs (they must
write to separate --out-dir, since two processes appending to one detail CSV will
interleave rows and corrupt it).

    python3 merge_runs.py OUT_DIR RUN_DIR [RUN_DIR ...]

e.g.
    python3 merge_runs.py ~/.../results/alpha_sclip_merged \
                          ~/.../results/alpha_sclip_coarse \
                          ~/.../results/alpha_sclip_fine

Writes into OUT_DIR:
    <name>_detail.csv    all source detail rows, deduplicated
    <name>_summary.csv   mean IoU per (grid, weighted, alpha)
    run_meta_merged.json provenance of every source run

stdlib only -- no torch/mmcv, so it runs anywhere (login node is fine).
"""
import csv
import glob
import json
import os
import sys

KEY = ("grid", "category", "variant", "variant_word", "weighted", "img_ref", "alpha")
FIELDS = list(KEY) + ["iou"]

if len(sys.argv) < 3:
    sys.exit(__doc__)
out_dir, run_dirs = sys.argv[1], sys.argv[2:]
os.makedirs(out_dir, exist_ok=True)

rows, seen, sources, name = [], set(), [], None
for rd in run_dirs:
    matches = sorted(glob.glob(os.path.join(rd, "*_detail.csv")))
    if not matches:
        sys.exit(f"no *_detail.csv in {rd}")
    detail = matches[0]
    name = name or os.path.basename(detail).replace("_detail.csv", "")
    n_before = len(rows)
    with open(detail) as f:
        for r in csv.DictReader(f):
            k = tuple(r[c] for c in KEY)
            if k in seen:
                continue
            seen.add(k)
            rows.append({c: r[c] for c in FIELDS})
    grids = sorted({r["grid"] for r in rows[n_before:]})
    print(f"{detail}: +{len(rows)-n_before:,} rows  grids={grids}")

    meta_path = os.path.join(rd, "run_meta.json")
    entry = {"run_dir": rd, "detail_csv": detail, "rows_added": len(rows) - n_before}
    if os.path.exists(meta_path):
        try:
            entry["run_meta"] = json.load(open(meta_path))
        except Exception as e:
            entry["run_meta_error"] = repr(e)
    sources.append(entry)

detail_out = os.path.join(out_dir, f"{name}_detail.csv")
with open(detail_out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
print(f"\nmerged detail -> {detail_out}  ({len(rows):,} rows)")

# --- summarize: mean IoU per (grid, weighted, alpha) ---
buckets = {}
for r in rows:
    buckets.setdefault((r["grid"], r["weighted"], float(r["alpha"])), []).append(float(r["iou"]))

summary_out = os.path.join(out_dir, f"{name}_summary.csv")
with open(summary_out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["grid", "weighted", "alpha", "n", "mean_iou"])
    w.writeheader()
    for (grid, weighted, alpha), ious in sorted(buckets.items()):
        w.writerow({"grid": grid, "weighted": weighted, "alpha": alpha,
                    "n": len(ious), "mean_iou": sum(ious) / len(ious)})
print(f"merged summary -> {summary_out}")

json.dump({"merged_into": out_dir, "sources": sources},
          open(os.path.join(out_dir, "run_meta_merged.json"), "w"), indent=2, default=str)

# --- consistency checks ---
print("\n=== checks ===")
for grid in sorted({r["grid"] for r in rows}):
    ns = {len(v) for (g, _, _), v in buckets.items() if g == grid}
    print(f"  {grid:6}: {len([1 for (g,_,_) in buckets if g==grid])} (weighted,alpha) cells, "
          f"n per cell = {sorted(ns) if len(ns) <= 3 else f'{min(ns)}..{max(ns)} (VARIES)'}")
z = {(g, w): round(sum(v)/len(v), 10) for (g, w, a), v in buckets.items() if a == 0.0}
if len(set(z.values())) > 1:
    print(f"  WARNING: alpha=0 differs across weighted settings {z} -- expected identical")
elif z:
    print(f"  alpha=0 baseline identical for weighted/unweighted: {list(z.values())[0]:.6f}  OK")
else:
    print("  WARNING: no alpha=0 rows -- the coarse grid is MISSING, so there is no baseline")
