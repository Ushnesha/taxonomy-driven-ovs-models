"""
Per-category IoU for the `orig` variant at alpha=0 -- i.e. the model's plain,
unblended performance on the canonical class name. This is the number to sanity
check against the model's published mIoU before trusting a whole sweep.

    python3 check_orig_iou.py RUN_DIR [RUN_DIR_TO_COMPARE]

stdlib only; runs anywhere.
"""
import csv, glob, os, statistics, sys

def load(rd):
    f = sorted(glob.glob(os.path.join(rd, "*_detail.csv")))
    if not f: sys.exit(f"no *_detail.csv in {rd}")
    per = {}
    for r in csv.DictReader(open(f[0])):
        if r["variant"] == "orig" and float(r["alpha"]) == 0.0 and r["weighted"] == "True":
            per.setdefault(r["category"], []).append(float(r["iou"]))
    return {c: statistics.mean(v) for c, v in per.items()}, len(per)

if len(sys.argv) < 2: sys.exit(__doc__)
a, _ = load(sys.argv[1])
b = load(sys.argv[2])[0] if len(sys.argv) > 2 else None

print(f"{'category':<16}{'orig IoU':>10}" + (f"{'compare':>10}{'delta':>9}" if b else ""))
for c in sorted(a, key=a.get):
    line = f"{c:<16}{a[c]:10.4f}"
    if b and c in b:
        line += f"{b[c]:10.4f}{a[c]-b[c]:+9.4f}"
    print(line)

v = list(a.values())
print(f"\n{len(v)} categories | mean {statistics.mean(v):.4f} | median {statistics.median(v):.4f} "
      f"| max {max(v):.4f} | below 0.05: {sum(1 for x in v if x < 0.05)}")
print("\nSCLIP published COCO-Object mIoU is ~0.30. A mean far below that, or many\n"
      "categories pinned at 0.000, means the joint-softmax prob_thd is too high.")
