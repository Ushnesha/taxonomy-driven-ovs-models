"""
Does the fetched image match the dimensions recorded in img_metadata.pkl?

    python3 check_image_dims.py [N_PER_SOURCE]

The GT masks are stored as flat buffers reshaped to (entry["height"], entry["width"]).
The predicted mask is produced at the fetched PIL image's size. If those ever disagree,
compute_iou() raises on the broadcast. This samples each source and reports mismatches.
"""
import sys, collections
import benchmark_data as bd

n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
bm = bd.load_benchmark()
by_src = collections.defaultdict(list)
for e in bm.img_by_id.values():
    by_src[e["img_src"]].append(e)

print(f"{len(bm.categories)} categories, {len(bm.img_by_id)} images")
print({k: len(v) for k, v in by_src.items()}, "\n")

bad = 0
for src, entries in by_src.items():
    print(f"--- {src} ---")
    for e in entries[:n]:
        try:
            img = bd.fetch_image(e)
        except Exception as ex:
            print(f"  {e['img_id']}: FETCH ERROR {type(ex).__name__}: {ex}")
            continue
        got = (img.size[1], img.size[0])          # PIL .size is (w, h)
        want = (e["height"], e["width"])
        if got != want:
            bad += 1
            print(f"  {e['img_id']}: MISMATCH fetched={got} recorded={want}")
    print(f"  checked {min(n, len(entries))}")

print(f"\n{bad} mismatches found.")
print("Any mismatch means compute_iou() will raise on those images.")
