"""Prove the SCLIP prompt-convention fix before re-running a 3-day job.

The positive-set run came back as walls of 0.0 IoU while the alpha sweep on the same
model reported 0.2028. Diagnosis: SCLIP warms its whole vocabulary with the
openai_imagenet_template ensemble (SCLIP/clip_segmentor.py:38), but approaches.embed()
was injecting BARE, template-free words into one row of it. In a joint softmax the
bare-word class then loses the argmax to its templated competitors -- "wheel" loses to
"car"/"bicycle", "garment" loses to "wet suit" -- so its mask is empty.

This measures three things on the same images so the diagnosis is proven, not assumed:

  1. NATIVE    -- no injection at all. SCLIP's own query_features row for the class,
                  i.e. ordinary SCLIP inference. This is the ceiling.
  2. TEMPLATED -- get_text_embedding(word, desc=False): the 80-template ensemble,
                  matching how the vocabulary was built. Should reproduce NATIVE.
  3. BARE      -- get_text_embedding(word, desc=True): template-free, the OLD
                  approaches.embed() behaviour. Should be far worse.

Reading it:
  * TEMPLATED ~= NATIVE and BARE << NATIVE  -> diagnosis confirmed, fix is correct.
  * all three ~= NATIVE                     -> the convention was NOT the problem;
                                               do not re-run, come back and re-diagnose.
  * NATIVE itself near 0                     -> something else is broken in the
                                               inference path; the fix is irrelevant.

It also scores a SHiNe-style long sentence both ways, because templating SENTENCES (as
opposed to bare class names) is a judgement call, not something native parity can
settle: wrapping an 80-token hypernym chain in 80 templates forces heavy truncation.

Pure benchmark data + the real model, no COCO needed. Needs a GPU.

Usage:
    python3 check_native_iou.py                       # 8 categories x 2 images
    python3 check_native_iou.py --n-categories 12 --n-images 3
"""
import argparse
import os
import sys

import numpy as np
import torch

import benchmark_data as bd
import approaches as ap
from sclip import SClipModel

THRESHOLD = 0.1  # SCLIP's published COCO-Object prob_thd. 0.5 zeroes most classes.


def _free():
    """Release cached GPU blocks between forward passes."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = (pred | gt).sum()
    return float((pred & gt).sum()) / float(union) if union else float("nan")


def native_scores(model, image, cat_name):
    """One forward pass -> (winning class index, winning score) per pixel.

    prob_thd is forced to 0 so NOTHING is suppressed, and the patched
    postprocess_result stores the winning score in `seg_logits`. That lets a whole
    range of thresholds be evaluated from a SINGLE forward pass instead of one pass
    per threshold -- the threshold is only ever a post-hoc comparison against this
    score, never something the network recomputes.
    """
    img_tensor, data_sample, _ = model._load_image(image)
    with model._lock:
        original = model.model.prob_thd
        try:
            model.model.prob_thd = 0.0
            batch = dict(inputs=[img_tensor], data_samples=[data_sample])
            processed = model.model.data_preprocessor(batch, False)
            with torch.no_grad():
                out = model.model.predict(processed["inputs"], processed["data_samples"])
            idx = out[0].pred_sem_seg.data.squeeze(0).cpu().numpy()
            val = out[0].seg_logits.data.squeeze(0).float().cpu().numpy()
        finally:
            model.model.prob_thd = original
    return idx, val


def native_mask(model, image, cat_name, threshold=THRESHOLD):
    """SCLIP's own prediction for `cat_name` with NO row swap.

    predict_with_embeddings() with an empty-ish swap is not the same thing, so this
    re-runs the model's real joint forward pass and extracts the class's mask the same
    way _predict_joint does, leaving query_features untouched.
    """
    img_tensor, data_sample, _ = model._load_image(image)
    idx = model._class_idx_for(cat_name)
    with model._lock:
        original_prob_thd = model.model.prob_thd
        try:
            model.model.prob_thd = threshold
            batch = dict(inputs=[img_tensor], data_samples=[data_sample])
            processed = model.model.data_preprocessor(batch, False)
            with torch.no_grad():
                out = model.model.predict(processed["inputs"], processed["data_samples"])
            # squeeze(0), matching _predict_joint exactly -- a bare squeeze() would
            # also collapse a height or width of 1 and mis-shape the mask.
            pred = out[0].pred_sem_seg.data.squeeze(0).cpu().numpy()
        finally:
            model.model.prob_thd = original_prob_thd
    return (pred == idx).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-dir", default=None)
    p.add_argument("--n-categories", type=int, default=8)
    p.add_argument("--n-images", type=int, default=2)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--thresholds", default="0,0.005,0.01,0.02,0.05,0.1",
                   help="comma-separated prob_thd values to evaluate from ONE forward "
                        "pass per image. SCLIP's published 0.1 is calibrated for ~81 "
                        "classes; with a 326-class vocabulary the winning softmax "
                        "probability is ~0.03, so 0.1 blanks every mask.")
    args = p.parse_args()
    # sentinel: was --threshold given, or is it just the module default?
    args.threshold_explicit = any(a.startswith("--threshold=") or a == "--threshold"
                                  for a in sys.argv[1:])

    bm = bd.load_benchmark(args.benchmark_dir)
    name_path = os.path.join("/tmp", f"sclip_native_{os.getpid()}.txt")
    bd.build_sclip_name_path(bm, name_path)

    # Prefer categories the model will actually predict: most-annotated first, and
    # lowercase before capitalised (capitals are WordNet proper-noun artifacts like
    # "Bannister", which legitimately win no pixels and make every comparison vacuous).
    cats = sorted((c for c in bm.categories if bm.positive_set.get(c)),
                  key=lambda c: (c[:1].isupper(), -len(bm.positive_set.get(c, []))))
    cats = cats[:args.n_categories]
    print(f"categories: {cats}")

    print("\nloading SCLIP ...", flush=True)
    model = SClipModel(name_path=name_path, prob_thd=args.threshold)

    # ---- threshold sweep FIRST: if every mask is empty at prob_thd=0.1 but healthy
    # ---- at 0.0, the threshold is the bug and the 3-way prompt comparison below is
    # ---- measuring nothing at all.
    thds = [float(x) for x in args.thresholds.split(",") if x.strip() != ""]
    print(f"\n=== THRESHOLD SWEEP (native prediction, one forward pass per image) ===")
    print(f"  {'category':<22}" + "".join(f"{t:>9}" for t in thds))
    sweep = {t: [] for t in thds}
    for cat in cats:
        for img_id in bm.positive_set[cat][:args.n_images]:
            entry = bm.img_by_id.get(img_id)
            if entry is None:
                continue
            try:
                image = bd.fetch_image(entry)
            except Exception:
                continue
            gt = bd.decode_gt_mask(entry, cat)
            if gt is None or gt.sum() == 0:
                continue
            idx_map, val_map = native_scores(model, image, cat)
            cls = model._class_idx_for(cat)
            line = f"  {cat:<22}"
            for t in thds:
                m = ((idx_map == cls) & (val_map > t)).astype(np.uint8)
                v = iou(m, gt)
                sweep[t].append(v)
                line += f"{v:>9.4f}"
            print(line, flush=True)
            _free()
    print(f"  {'MEAN':<22}" + "".join(f"{float(np.nanmean(sweep[t])):>9.4f}" for t in thds))
    best_t = max(thds, key=lambda t: float(np.nanmean(sweep[t])))
    print(f"\n  best prob_thd = {best_t}  (mean IoU {float(np.nanmean(sweep[best_t])):.4f})")
    hi, lo = float(np.nanmean(sweep[max(thds)])), float(np.nanmean(sweep[best_t]))
    if lo < 0.02:
        print(f"  -> STILL BROKEN: even the best threshold gives {lo:.4f}. Something other")
        print(f"     than the threshold is wrong -- run diagnose_sclip_prediction.py.")
    elif hi < 0.5 * lo:
        print(f"  -> prob_thd={max(thds)} costs most of the signal ({hi:.4f} vs {lo:.4f}).")
        print(f"     Use prob_thd={best_t}. This says nothing about the prompt convention;")
        print(f"     the 3-way comparison below tests that, at this same threshold.")
    else:
        print(f"  -> the threshold barely matters here ({hi:.4f} vs {lo:.4f}).")

    # The prompt comparison must run at a threshold where masks are NOT empty, or all
    # three arms are trivially 0.0 and the comparison tests nothing. Use whatever the
    # sweep just found, unless --threshold was given explicitly.
    cmp_thd = args.threshold if args.threshold_explicit else best_t
    print(f"\nprompt comparison will use prob_thd={cmp_thd}"
          f"{' (from the sweep)' if not args.threshold_explicit else ' (from --threshold)'}")

    rows = []
    for cat in cats:
        for img_id in bm.positive_set[cat][:args.n_images]:
            entry = bm.img_by_id.get(img_id)
            if entry is None:
                continue
            try:
                image = bd.fetch_image(entry)
            except Exception as e:
                print(f"  skip {cat}/{img_id}: fetch error {e}")
                continue
            gt = bd.decode_gt_mask(entry, cat)
            if gt is None or gt.sum() == 0:
                continue

            # Three full joint forward passes per image, each holding a
            # [num_queries, H, W] map at ORIGINAL resolution (5.4 GiB on a 2200x1650
            # ADE20K image with the 326-class vocabulary). Without freeing between
            # them the allocator accumulates all three and OOMs on anything under
            # ~30 GiB. empty_cache() after each keeps peak to one pass.
            m_nat = native_mask(model, image, cat, cmp_thd)
            _free()
            m_tpl = model.predict_with_embeddings(
                image, {cat: model.get_text_embedding(cat, desc=False)},
                threshold=cmp_thd)[cat]
            _free()
            m_bare = model.predict_with_embeddings(
                image, {cat: model.get_text_embedding(cat, desc=True)},
                threshold=cmp_thd)[cat]
            _free()

            rows.append((cat, img_id, iou(m_nat, gt), iou(m_tpl, gt), iou(m_bare, gt)))
            print(f"  {cat:<22} {str(img_id):<16} gt={100*gt.mean():5.1f}%  "
                  f"native={rows[-1][2]:.4f}  templated={rows[-1][3]:.4f}  "
                  f"bare={rows[-1][4]:.4f}", flush=True)

    if not rows:
        sys.exit("ABORT: no usable (category, image) pairs.")

    mean = lambda i: float(np.nanmean([r[i] for r in rows]))
    mn, mt, mb = mean(2), mean(3), mean(4)
    if torch.cuda.is_available():
        print(f"\npeak GPU memory: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
              f"of {torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GiB")
    print("\n" + "=" * 68)
    print(f"  {len(rows)} (category, image) pairs, prob_thd={cmp_thd}")
    print(f"  1. NATIVE    (SCLIP's own row, no injection)   : {mn:.4f}")
    print(f"  2. TEMPLATED (desc=False, the FIX)             : {mt:.4f}")
    print(f"  3. BARE      (desc=True, the OLD behaviour)    : {mb:.4f}")
    print("=" * 68)
    if mn < 0.02:
        print("\n-> NATIVE is near zero too. The prompt convention is NOT the problem;")
        print("   something else in the inference path is. Do NOT re-run yet.")
    elif mt > 0.8 * mn and mb < 0.5 * mn:
        print("\n-> CONFIRMED. Templated injection reproduces native; bare does not.")
        print("   The approaches.py desc=False fix is correct -- re-run positive+negative.")
    elif abs(mt - mb) < 0.05 * max(mn, 1e-9):
        print("\n-> Templated and bare score the SAME, so the convention was not the")
        print("   cause of the zeros. Re-diagnose before spending GPU time.")
    else:
        print("\n-> Mixed signal. Send this table over before re-running.")

    # ---- the judgement call native parity cannot settle: descriptor SENTENCES ----
    print("\nSENTENCE handling (shine/waffleclip/llm_descriptor inject full sentences)")
    print("  Templating a sentence wraps it in all 80 imagenet templates and forces")
    print("  heavy 77-token truncation. Native parity cannot arbitrate this -- the model")
    print("  has no 'own row' for a sentence. Compare the two on one category:")
    cat = rows[0][0]
    sent = (f"a {cat}, which is a physical object, which is an entity, "
            f"which is a thing that can be seen and touched in a photograph")
    for img_id in bm.positive_set[cat][:args.n_images]:
        entry = bm.img_by_id.get(img_id)
        if entry is None:
            continue
        try:
            image = bd.fetch_image(entry)
        except Exception:
            continue
        gt = bd.decode_gt_mask(entry, cat)
        if gt is None or gt.sum() == 0:
            continue
        t = model.predict_with_embeddings(
            image, {cat: model.get_text_embedding(sent, desc=False)},
            threshold=cmp_thd)[cat]
        b = model.predict_with_embeddings(
            image, {cat: model.get_text_embedding(sent, desc=True)},
            threshold=cmp_thd)[cat]
        print(f"  {cat:<22} {str(img_id):<16} sentence templated={iou(t, gt):.4f}  "
              f"verbatim={iou(b, gt):.4f}")
    print("\n  If verbatim beats templated here, split the convention: desc=False for")
    print("  bare class names, desc=True for full sentences. Otherwise leave it as is.")


if __name__ == "__main__":
    main()
