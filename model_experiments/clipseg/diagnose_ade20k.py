"""
Is the ADE20K misalignment a reordering (recoverable) or a different dataset?

    python3 diagnose_ade20k.py
"""
import collections
import benchmark_data as bd

bm = bd.load_benchmark()
by_src = collections.Counter(e["img_src"] for e in bm.img_by_id.values())
print("images per source:", dict(by_src))
tot = sum(by_src.values())
for s, n in by_src.items():
    print(f"  {s:11} {n:6}  ({100*n/tot:.1f}%)")

ade = [e for e in bm.img_by_id.values() if e["img_src"] == "ade20k"]
print(f"\n--- fields available on an ade20k entry (for re-alignment) ---")
e = ade[0]
for k, v in e.items():
    if k == "gt_bin_masks":
        print(f"  {k:14} <{len(v)} compressed masks>")
    else:
        print(f"  {k:14} {v!r}"[:130])

ds = bd._ade20k_dataset()
print(f"\ncurrent HF dataset: {len(ds)} rows | columns: {ds.column_names}")
print(f"benchmark expects max img_src_id: {max(x['img_src_id'] for x in ade)}")

# Is it a reordering? Do the recorded (h,w) pairs exist anywhere in the current dataset?
print("\n--- building (h,w) index over the current dataset (may take a minute) ---")
dims = collections.defaultdict(list)
for i in range(len(ds)):
    im = ds[i]["image"]
    dims[(im.size[1], im.size[0])].append(i)

hit = uniq = miss = 0
for x in ade[:200]:
    want = (x["height"], x["width"])
    c = dims.get(want, [])
    if not c:      miss += 1
    elif len(c)==1: uniq += 1; hit += 1
    else:           hit += 1
print(f"of 200 benchmark ade20k entries:")
print(f"  {hit:3} have their recorded (h,w) present in the current dataset")
print(f"  {uniq:3} of those match a UNIQUE row (re-alignable by dims alone)")
print(f"  {miss:3} have no row with those dims at all")
print("\nInterpretation:")
print("  miss ~0  -> same images, reordered. Re-alignable.")
print("  miss high-> different dataset snapshot. Not re-alignable; drop ADE20K or rebuild.")
