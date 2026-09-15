"""
Analyze the output of alpha_value_experiment.py (any model folder).

    python3 analyze_alpha.py /path/to/results/alpha_clipseg_YYYYMMDD_HHMMSS

Prints a text report (needs only pandas) and, if matplotlib is installed, writes
plots into the same run dir. Always writes recommended_alpha.csv.

What each section answers:
  1. HEADLINE     does blending beat the no-blend baseline, and by how much?
  2. PLATEAU      how sensitive is the result to the exact alpha?
  3. PER VARIANT  the actual thesis: does it rescue rare/abstract words (hypo/
                  hyper) at an acceptable cost on the canonical word (orig)?
  4. PER CATEGORY where does it help / hurt most?
"""
import glob
import os
import sys

import pandas as pd

run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
matches = sorted(glob.glob(os.path.join(run_dir, "*_detail.csv")))
if not matches:
    sys.exit(f"no *_detail.csv found in {run_dir}")
detail = matches[0]

df = pd.read_csv(detail)
df["weighted"] = df["weighted"].astype(str) == "True"
pd.set_option("display.width", 200)

print(f"file       : {detail}")
print(f"rows       : {len(df):,}")
print(f"grids      : {sorted(df.grid.unique())}")
print(f"categories : {df.category.nunique()}")
print(f"variants   : {sorted(df.variant.unique())}")
print(f"images     : {df.img_ref.nunique():,} unique")


def curve(sub):
    """mean IoU per alpha, ordered by alpha."""
    return sub.groupby("alpha").iou.mean().sort_index()


def at_nearest(series, target):
    """series value at the grid point closest to `target` (float keys never
    match exactly across a coarse/fine grid boundary)."""
    if series.empty:
        return float("nan")
    key = min(series.index, key=lambda a: abs(a - target))
    return series.loc[key]


# ---------------------------------------------------------------- 1. HEADLINE
print("\n" + "=" * 72)
print("1. HEADLINE  --  best alpha per (grid, centroid type)")
print("=" * 72)
baseline = curve(df[df.grid == "coarse"]).get(0.0)
rows = []
for grid in sorted(df.grid.unique()):
    for w in [False, True]:
        s = curve(df[(df.grid == grid) & (df.weighted == w)])
        if s.empty:
            continue
        rows.append({
            "grid": grid,
            "centroid": "weighted" if w else "unweighted",
            "best_alpha": s.idxmax(),
            "mean_iou_at_best": round(s.max(), 4),
            "baseline_alpha0": round(baseline, 4) if baseline is not None else None,
            "abs_gain": round(s.max() - baseline, 4) if baseline is not None else None,
            "rel_gain_%": round(100 * (s.max() - baseline) / baseline, 1) if baseline else None,
            "n_per_alpha": int(df[(df.grid == grid) & (df.weighted == w)]
                               .groupby("alpha").size().max()),
        })
head = pd.DataFrame(rows)
print(head.to_string(index=False))

best_row = head.loc[head.mean_iou_at_best.idxmax()]
print(f"\n  ==> USE alpha = {best_row.best_alpha:.2f} "
      f"({best_row.centroid} centroid, {best_row.grid} grid)")

# ----------------------------------------------------------------- 2. PLATEAU
print("\n" + "=" * 72)
print("2. PLATEAU  --  alphas within 1% of the peak (how sensitive is this?)")
print("=" * 72)
g = df[df.grid == "fine"] if (df.grid == "fine").any() else df[df.grid == "coarse"]
g = g[g.weighted == (best_row.centroid == "weighted")]
s = curve(g)
tol = s.max() * 0.99
flat = s[s >= tol]
print(f"  peak         : {s.max():.4f} at alpha={s.idxmax():.2f}")
print(f"  within 1%    : alpha {flat.index.min():.2f} .. {flat.index.max():.2f} "
      f"({len(flat)} of {len(s)} grid points)")
print(f"  spread there : {flat.max() - flat.min():.4f} IoU")
print("  -> a wide flat band means the method is NOT sensitive to the exact alpha.")

# -------------------------------------------------------------- 3. PER VARIANT
print("\n" + "=" * 72)
print("3. PER VARIANT  --  the core result for the paper")
print("=" * 72)
sub = df[(df.grid == "coarse") & (df.weighted == (best_row.centroid == "weighted"))]
piv = sub.pivot_table(index="alpha", columns="variant", values="iou", aggfunc="mean")
order = [c for c in ["orig", "syn", "hypo", "hyper"] if c in piv.columns]
piv = piv[order]
print("\nmean IoU vs alpha, per variant (coarse grid):")
print(piv.round(4).to_string())

print("\nsummary:")
vrows = []
for v in order:
    c = piv[v].dropna()
    n = int(sub[sub.variant == v].groupby("alpha").size().max())
    vrows.append({
        "variant": v,
        "example_word": (sub[sub.variant == v].variant_word.mode().tolist() or [""])[0],
        "iou_at_alpha0": round(c.get(0.0, float("nan")), 4),
        "best_alpha": c.idxmax(),
        "iou_at_best": round(c.max(), 4),
        "gain": round(c.max() - c.get(0.0, float("nan")), 4),
        "iou_at_chosen": round(at_nearest(c, best_row.best_alpha), 4),
        "n_per_alpha": n,
    })
vt = pd.DataFrame(vrows)
print(vt.to_string(index=False))
print("\n  Read this as: blending should LIFT hypo/hyper (rare, abstract words) a lot,")
print("  while costing a little on orig (the canonical word the model already knows).")

# ------------------------------------------------------------- 4. PER CATEGORY
print("\n" + "=" * 72)
print("4. PER CATEGORY  --  where blending helps / hurts")
print("=" * 72)
cw = df[(df.grid == "coarse") & (df.weighted == (best_row.centroid == "weighted"))]
cpiv = cw.pivot_table(index="category", columns="alpha", values="iou", aggfunc="mean")
a_chosen = min(cpiv.columns, key=lambda a: abs(a - best_row.best_alpha))
gain = (cpiv[a_chosen] - cpiv[0.0]).sort_values()
print(f"(IoU at alpha={a_chosen:.2f} minus IoU at alpha=0.00)\n")
print("  worst 8:")
for k, v in gain.head(8).items():
    print(f"    {k:<20} {v:+.4f}")
print("  best 8:")
for k, v in gain.tail(8)[::-1].items():
    print(f"    {k:<20} {v:+.4f}")
print(f"\n  helped: {(gain > 0).sum()}/{len(gain)} categories   "
      f"median gain: {gain.median():+.4f}")

# --------------------------------------------------------------- write outputs
out_csv = os.path.join(run_dir, "recommended_alpha.csv")
head.to_csv(out_csv, index=False)
vt.to_csv(os.path.join(run_dir, "per_variant_alpha.csv"), index=False)
gain.rename("gain_vs_alpha0").to_csv(os.path.join(run_dir, "per_category_gain.csv"))
print(f"\nwrote recommended_alpha.csv, per_variant_alpha.csv, per_category_gain.csv")

# ---------------------------------------------------------------------- plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("(matplotlib not installed -- skipping plots; pip install matplotlib to get them)")
    sys.exit(0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, grid in zip(axes, ["coarse", "fine"]):
    d = df[df.grid == grid]
    if d.empty:
        continue
    for w in [False, True]:
        s2 = curve(d[d.weighted == w])
        if s2.empty:
            continue
        ax.plot(s2.index, s2.values, marker="o", ms=3,
                label=("weighted" if w else "unweighted") + " centroid")
        ax.scatter([s2.idxmax()], [s2.max()], zorder=5)
        ax.annotate(f"  a={s2.idxmax():.2f}\n  {s2.max():.4f}",
                    (s2.idxmax(), s2.max()), fontsize=8)
    if grid == "coarse" and baseline is not None:
        ax.axhline(baseline, ls="--", c="gray", lw=1)
        ax.annotate(f"baseline a=0: {baseline:.4f}", (0.0, baseline), fontsize=8, va="bottom")
    ax.set_title(f"{grid} grid"); ax.set_xlabel("alpha"); ax.grid(alpha=.3); ax.legend()
axes[0].set_ylabel("mean IoU")
fig.tight_layout(); fig.savefig(os.path.join(run_dir, "plot_overall.png"), dpi=130)

fig, ax = plt.subplots(figsize=(8, 5))
for v in order:
    c = piv[v].dropna()
    ax.plot(c.index, c.values, marker="o", ms=3, label=f"{v}  (best a={c.idxmax():.1f})")
ax.axvline(best_row.best_alpha, ls="--", c="k", lw=1, alpha=.5)
ax.set_xlabel("alpha"); ax.set_ylabel("mean IoU"); ax.grid(alpha=.3)
ax.set_title(f"per-variant ({best_row.centroid} centroid, coarse grid)"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(run_dir, "plot_by_variant.png"), dpi=130)

fig, ax = plt.subplots(figsize=(8, max(4, .28 * len(gain))))
ax.barh(gain.index, gain.values,
        color=["#c0392b" if x < 0 else "#27ae60" for x in gain.values])
ax.axvline(0, c="k", lw=1)
ax.set_xlabel(f"IoU(alpha={a_chosen:.2f}) - IoU(alpha=0)")
ax.set_title("per-category effect of blending")
fig.tight_layout(); fig.savefig(os.path.join(run_dir, "plot_by_category.png"), dpi=130)

print("wrote plot_overall.png, plot_by_variant.png, plot_by_category.png")
