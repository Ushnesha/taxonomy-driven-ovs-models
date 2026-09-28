"""Is CAT-Seg itself underperforming, or are OUR injected embeddings weak?

The alpha dry run returned mean IoU ~0.02 on COCO, an order of magnitude below
CLIPSeg (0.296) and SCLIP (0.203) at alpha=0. Two very different causes produce
that, and no amount of staring at the sweep output distinguishes them:

  A. the inference path is wrong (preprocessing, colour order, resize, mask
     extraction) -- then even CAT-Seg's OWN un-injected prediction scores ~0.02
  B. the injected embedding is weaker than the model's own -- then CAT-Seg's
     native prediction is fine and only our swapped-in row is bad

So measure three things on the same images and compare:

  1. NATIVE    -- no injection at all. The model's own cached row for the class,
                  exactly as normal CAT-Seg inference. This is the ceiling.
  2. BARE      -- our pipeline's embedding: approaches.embed(), i.e. the bare word,
                  which is what baseline_embedding and ours(alpha=0) both use.
  3. TEMPLATED -- get_text_embedding(desc=False), i.e. wrapped in
                  DEFAULT_PROMPT_TEMPLATE, matching CAT-Seg's own convention.

Reading the result:
  * NATIVE ~= 0.02  -> cause A. The wrapper's inference path is broken; fix that
                      before any sweep. Compare against CAT-Seg's published numbers.
  * NATIVE high, BARE low, TEMPLATED high
                    -> cause B, and specifically the PROMPT convention: injecting a
                       bare word into a joint softmax whose other 79 classes were
                       warmed with a template puts the probed class at a systematic
                       disadvantage. Fix by making embed() use desc=False for catseg.
  * NATIVE ~= BARE ~= TEMPLATED, all low
                    -> the model genuinely scores this low on these categories;
                       check whether the sample is just small/hard objects.

Usage:
    python3 check_native_iou.py --config ./CAT-Seg/configs/vitl_336.yaml \\
        --weights /scratch/$USER/catseg_ckpt/catseg_model_large.pth \\
        --coco-dir ~/workspace/OVS/coco_data --n-categories 8 --n-images 3
"""
import argparse
import json
import os

import numpy as np
import torch

import approaches as ap
from expanded_benchmark_helpers import COCO_80
from catseg import CATSegModel


def iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = (pred | gt).sum()
    return float((pred & gt).sum()) / float(union) if union else float("nan")


def native_mask(model, image, cat_name):
    """CAT-Seg's own prediction for `cat_name` -- NO row swap at all.

    Mirrors predict_with_embeddings' mask extraction exactly (same argmax, same
    _as_probabilities), so any difference against the injected variants is due to the
    embedding and nothing else.
    """
    import numpy as _np
    from PIL import Image as _Image
    if isinstance(image, _np.ndarray):
        image = _Image.fromarray(image)
    img_bgr = _np.array(image.convert("RGB"))[:, :, ::-1]
    idx = model._class_idx_for(cat_name)
    model._warm_cache()
    with torch.no_grad():
        predictions = model.demo.predictor(img_bgr)
    sem_seg = predictions["sem_seg"]
    if not torch.is_tensor(sem_seg):
        sem_seg = torch.as_tensor(_np.asarray(sem_seg))
    # Mirror predict_with_embeddings exactly: argmax on the raw tensor, no
    # normalisation assumption (threshold 0.0 needs none).
    label = sem_seg.argmax(0)
    return (label.cpu().numpy() == idx).astype(_np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--coco-dir", default=os.environ.get("COCO_DIR", "./coco_data"))
    p.add_argument("--catseg-path", default=None)
    p.add_argument("--categories", default=None,
                   help="comma-separated COCO category names to test, instead of the "
                        "first --n-categories of COCO_80. Use this to compare against "
                        "the EXACT categories the alpha sweep picked -- the sweep filters "
                        "to WordNet-eligible categories, so its set is different (and "
                        "harder) than the first-N default, which on its own explains a "
                        "large apparent gap.")
    p.add_argument("--n-categories", type=int, default=8)
    p.add_argument("--n-images", type=int, default=3)
    args = p.parse_args()

    from pycocotools.coco import COCO
    # Reuse the sweep's own COCO layout helpers rather than guessing paths: it keeps
    # the annotations FLAT at <coco_dir>/instances_val2017.json (not under
    # annotations/), downloads them on demand, and caches images in <coco_dir>/val2017/.
    from alpha_value_experiment import ensure_coco_annotations, get_local_image_path
    coco_dir = os.path.abspath(os.path.expanduser(args.coco_dir))
    ann = ensure_coco_annotations(coco_dir)
    coco = COCO(ann)

    class_json = os.path.join("/tmp", f"catseg_native_{os.getpid()}.json")
    with open(class_json, "w") as f:
        json.dump(COCO_80, f)

    print("building CAT-Seg ...", flush=True)
    model = CATSegModel(config=args.config, weights=args.weights,
                        catseg_path=args.catseg_path, class_name_path=class_json)
    if model.baseline:
        raise SystemExit("ABORT: fell back to BASELINE (CLIPSeg) mode.")

    name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    if args.categories:
        wanted = [c.strip() for c in args.categories.split(",") if c.strip()]
        missing = [c for c in wanted if c not in name_to_id]
        if missing:
            raise SystemExit(f"ABORT: not COCO categories: {missing}")
        cats = wanted
    else:
        cats = [c for c in COCO_80 if c in name_to_id][:args.n_categories]
    print(f"categories: {cats}")

    rows = []
    for cat in cats:
        cid = name_to_id[cat]
        img_ids = coco.getImgIds(catIds=[cid])[:args.n_images]
        for img_id in img_ids:
            meta = coco.loadImgs([img_id])[0]
            try:
                path = get_local_image_path(coco_dir, meta["file_name"])
            except Exception as e:
                print(f"  skip {cat}/{img_id}: {e}")
                continue
            from PIL import Image
            image = Image.open(path).convert("RGB")

            anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], catIds=[cid], iscrowd=None))
            gt = np.zeros((meta["height"], meta["width"]), dtype=np.uint8)
            for a in anns:
                gt |= coco.annToMask(a).astype(np.uint8)
            if gt.sum() == 0:
                continue

            m_native = native_mask(model, image, cat)
            m_bare = model.predict_with_embeddings(image, {cat: ap.embed(model, cat)})[cat]
            m_tmpl = model.predict_with_embeddings(
                image, {cat: model.get_text_embedding(cat, desc=False)})[cat]

            rows.append((cat, img_id, gt.mean(),
                         iou(m_native, gt), iou(m_bare, gt), iou(m_tmpl, gt)))
            print(f"  {cat:<16} {img_id:<12} gt={gt.mean()*100:5.1f}%  "
                  f"native={rows[-1][3]:.4f}  bare={rows[-1][4]:.4f}  tmpl={rows[-1][5]:.4f}",
                  flush=True)

    if not rows:
        raise SystemExit("ABORT: no usable (category, image) pairs found.")

    mean = lambda i: float(np.nanmean([r[i] for r in rows]))
    n, mn, mb, mt = len(rows), mean(3), mean(4), mean(5)
    print("\n" + "=" * 64)
    print(f"  {n} (category, image) pairs")
    print(f"  1. NATIVE    (CAT-Seg's own row, no injection) : {mn:.4f}")
    print(f"  2. BARE      (approaches.embed, bare word)     : {mb:.4f}")
    print(f"  3. TEMPLATED (DEFAULT_PROMPT_TEMPLATE)         : {mt:.4f}")
    print("=" * 64)
    if mn < 0.05:
        print("\n-> NATIVE is as low as the sweep. The INFERENCE PATH is the problem, not")
        print("   the injection. Suspect preprocessing (BGR/RGB), resize, or mask")
        print("   extraction. Do not run the sweep.")
    elif mb < 0.5 * mn and mt > 0.8 * mn:
        print("\n-> The PROMPT CONVENTION is the problem: a bare word injected into a joint")
        print("   softmax whose other classes were warmed with a template is at a")
        print("   systematic disadvantage. Make approaches.embed() use desc=False here.")
    elif mb > 0.8 * mn:
        print("\n-> Injection is faithful (bare ~= native). The low sweep number is the")
        print("   model's genuine score on this sample -- check category difficulty.")
    else:
        print("\n-> Mixed signal; send this table over and we'll read it together.")


if __name__ == "__main__":
    main()
