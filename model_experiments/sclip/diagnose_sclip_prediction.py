"""What class DOES win the pixels, when the right one doesn't?

`sky` covers 38.9% of an ADE20K image and SCLIP predicts it on 0 pixels even at
prob_thd=0, where every pixel is assigned to its argmax class. So those pixels went
somewhere. Which class took them is the single most informative thing left, and it
splits the remaining possibilities cleanly:

  * a plausible confusion ("ceiling", "wall", "water") -> the model and the index
    mapping are fine, and the 326-class vocabulary is simply too confusable. That is a
    result to report, not a bug to fix.
  * one class taking nearly everything, or names unrelated to the image -> the class
    index mapping is broken: _class_idx_for() and the model's own row order disagree,
    and every number this pipeline has produced is meaningless.
  * the GT class absent from the vocabulary entirely -> a name-matching failure.

For each probed image this prints the top predicted classes BY NAME with pixel counts,
what the GT region was labelled instead, and where the GT class ranked.

No assumptions about thresholds: prob_thd is forced to 0 so nothing is suppressed.

Usage:
    python3 diagnose_sclip_prediction.py
    python3 diagnose_sclip_prediction.py --categories sky,wall,floor --n-images 1
"""
import argparse
import os
from collections import Counter

import numpy as np
import torch

import benchmark_data as bd
from sclip import SClipModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--categories", default="sky,wall,floor,person",
                   help="comma-separated benchmark categories to probe")
    p.add_argument("--n-images", type=int, default=1)
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    bm = bd.load_benchmark(args.benchmark_dir)
    name_path = os.path.join("/tmp", f"sclip_diag_{os.getpid()}.txt")
    bd.build_sclip_name_path(bm, name_path)

    # The class-index -> name mapping AS THIS WRAPPER BELIEVES IT. Line 0 of the name
    # file is background, so foreground classes start at 1 (see sclip.py).
    with open(name_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    idx_to_name = {0: f"<background> [{lines[0][:40]}...]"}
    for i, ln in enumerate(lines[1:]):
        idx_to_name[i + 1] = ln.split(", ")[0]
    print(f"name file: {len(lines)} lines -> 1 background + {len(lines)-1} classes")

    print("loading SCLIP ...", flush=True)
    model = SClipModel(name_path=name_path, prob_thd=0.0)

    # Cross-check the wrapper's mapping against the model's OWN query_idx. If these
    # disagree, that alone is the bug.
    n_cls_model = int(model.model.query_idx.max()) + 1
    print(f"model query_idx: {len(model.model.query_idx)} queries -> "
          f"{n_cls_model} classes;  wrapper knows {len(idx_to_name)} names")
    if n_cls_model != len(idx_to_name):
        print("  !! MISMATCH between the model's class count and the wrapper's name map.")
        print("     Every extracted mask would be for the wrong class. This is the bug.")

    for cat in [c.strip() for c in args.categories.split(",") if c.strip()]:
        if cat not in bm.positive_set:
            print(f"\n[{cat}] not a benchmark category with positive images -- skipped")
            continue
        try:
            want_idx = model._class_idx_for(cat)
        except KeyError as e:
            print(f"\n[{cat}] NOT IN THE MODEL VOCABULARY: {e}")
            continue

        for img_id in bm.positive_set[cat][:args.n_images]:
            entry = bm.img_by_id.get(img_id)
            if entry is None:
                continue
            try:
                image = bd.fetch_image(entry)
            except Exception as e:
                print(f"\n[{cat}/{img_id}] fetch error {e}")
                continue
            gt = bd.decode_gt_mask(entry, cat)
            if gt is None or gt.sum() == 0:
                continue

            img_tensor, data_sample, _ = model._load_image(image)
            with model._lock:
                orig = model.model.prob_thd
                try:
                    model.model.prob_thd = 0.0
                    batch = dict(inputs=[img_tensor], data_samples=[data_sample])
                    processed = model.model.data_preprocessor(batch, False)
                    with torch.no_grad():
                        out = model.model.predict(processed["inputs"],
                                                  processed["data_samples"])
                    pred = out[0].pred_sem_seg.data.squeeze(0).cpu().numpy()
                finally:
                    model.model.prob_thd = orig

            print(f"\n{'='*72}")
            print(f"[{cat}]  img={img_id}  image={image.size[0]}x{image.size[1]}  "
                  f"pred_map={pred.shape}  gt={gt.shape}")
            if pred.shape != gt.shape:
                print("  !! PREDICTION AND GT SHAPES DIFFER -- IoU is meaningless. "
                      "This is the bug.")
                continue
            print(f"  GT covers {100.0*gt.mean():.1f}% of pixels; "
                  f"expected class index for '{cat}' = {want_idx}")

            counts = Counter(pred.ravel().tolist())
            total = pred.size
            print(f"\n  TOP {args.top} PREDICTED CLASSES OVER THE WHOLE IMAGE")
            for cidx, n in counts.most_common(args.top):
                mark = "  <-- the class we wanted" if cidx == want_idx else ""
                print(f"    {cidx:>4} {idx_to_name.get(cidx, '???')[:38]:<40} "
                      f"{n:>10,} px ({100.0*n/total:5.1f}%){mark}")

            got = counts.get(want_idx, 0)
            rank = [c for c, _ in counts.most_common()].index(want_idx) + 1 \
                if want_idx in counts else None
            print(f"\n  '{cat}' (idx {want_idx}) won {got:,} px "
                  f"({100.0*got/total:.2f}%), rank {rank}")

            # What the GT REGION was labelled instead -- the sharpest signal.
            inside = Counter(pred[gt.astype(bool)].ravel().tolist())
            n_inside = sum(inside.values())   # not Counter.total(): that is py3.10+ only
            print(f"  INSIDE the GT region, the model said:")
            for cidx, n in inside.most_common(5):
                mark = "  <-- correct" if cidx == want_idx else ""
                print(f"    {cidx:>4} {idx_to_name.get(cidx, '???')[:38]:<40} "
                      f"{n:>10,} px ({100.0*n/max(n_inside,1):5.1f}%){mark}")

    print(f"\n{'='*72}")
    print("READ THE 'INSIDE the GT region' BLOCK. Plausible confusions mean the model")
    print("works and the vocabulary is too confusable -- a finding. Unrelated names, or")
    print("one class taking everything, means the index mapping is broken -- a bug.")


if __name__ == "__main__":
    main()
