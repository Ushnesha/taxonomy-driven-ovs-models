# catseg/ (standalone) -- extra caveats beyond ../README.md

Same layout/conventions as `../clipseg/`, `../groupvit/`, `../sclip/` -- see the parent
`model_experiments/README.md` first. This file only covers what's different about CAT-Seg.

## Why this folder needs its own README

Every other model folder's wrapper (`clipseg.py`/`groupvit.py`/`sclip.py`) has been run
end-to-end and verified. `catseg.py`'s production mode has **not** -- there is no local
CUDA/detectron2 to test it against (per `../../CLAUDE.md`, CAT-Seg integration is
Ushnesha's, on ASU's cluster). Every fact `catseg.py`'s docstrings state about CAT-Seg's
internals (attribute names, the `class_embeddings()`/`get_text_embeds()`/`self.cache`
mechanism, the default prompt template, its L2-normalize convention) was confirmed by
reading the real source at https://github.com/KU-CVLAB/CAT-Seg, not guessed -- but reading
source isn't the same as running it. **Run the verification below before trusting any
catseg `positive_set`/`negative_set` numbers.**

## Setup

```bash
cd catseg
bash setup_env.sh   # builds ./venv (detectron2 from source, several minutes) and
                     # vendors CAT-Seg into ./CAT-Seg -- see that script's own comments
                     # for adjusting the torch/CUDA pin to your cluster.
source venv/bin/activate
```
Then download a checkpoint from CAT-Seg's model zoo (linked at the end of `setup_env.sh`'s
output) -- CAT-Seg (L) / `configs/vitl_336.yaml` is the paper's stronger model; CAT-Seg (B)
/ `configs/vitb_384.yaml` is lighter.

## Verifying the injection hook before a real run

`catseg.py`'s `predict_with_embeddings` swaps one row of the model's cached text-embedding
tensor per call (see that file's module docstring for exactly why and how). Before trusting
a real sweep, confirm the swap actually reaches the output:

```bash
python3 - << 'EOF'
from PIL import Image
from catseg import CATSegModel

model = CATSegModel(
    config="./CAT-Seg/configs/vitl_336.yaml", weights="/path/to/model_large.pth",
    class_name_path="./CAT-Seg/datasets/coco.json",  # or any real class-list json
)
img = Image.open("/path/to/any/test.jpg")

# 1. Real embedding for a real class present in the image -- should give a plausible mask.
emb = model.get_text_embedding("dog", desc=False)
m1 = model.predict_with_embeddings(img, {"dog": emb})["dog"]
print("real-embedding mask pixels:", m1.sum())

# 2. Nonsense embedding (a class it has never seen) swapped into the SAME row -- if the
#    injection hook is doing nothing, this mask will be identical to m1. If it works, this
#    should differ substantially (typically much smaller / near-empty).
noise_emb = model.get_text_embedding("a chaotic explosion of static and noise", desc=True)
m2 = model.predict_with_embeddings(img, {"dog": noise_emb})["dog"]
print("noise-embedding mask pixels:", m2.sum())

assert (m1 != m2).any(), "predict_with_embeddings' row-swap is NOT reaching the output -- do not trust results yet"
print("OK: swapped embedding measurably changed the output.")
EOF
```

If the assertion fails, `_locate_predictor_module`'s attribute path
(`self.demo.predictor.model.sem_seg_head.predictor`) or the `self.cache` row-swap mechanism
has drifted from what was read in `catseg.py`'s module docstring -- inspect
`model._predictor_module` directly in a debugger against your actual installed CAT-Seg
version before proceeding.

## Full-scale run safety gate

Unlike the other three folders, `alpha_value_experiment.py` / `positive_set_experiment.py`
/ `negative_set_experiment.py` here **refuse to run at full scale without `--config` and
`--weights`** -- without a real checkpoint they'd silently fall back to CLIPSeg (baseline
mode) and report those numbers under CAT-Seg's name. Pass `--limit-categories`/
`--limit-images` for a small local pipeline check (baseline fallback allowed there), or
`--allow-baseline` to force it at full scale (not recommended).

```bash
# 1. derive CAT-Seg's alpha on COCO, full scale
python3 alpha_value_experiment.py --coco-dir ./coco_data \
    --config ./CAT-Seg/configs/vitl_336.yaml --weights /path/to/model_large.pth

# 2. run the full approaches comparison on the real benchmark, using that alpha
python3 positive_set_experiment.py --alpha <BEST_ALPHA> \
    --config ./CAT-Seg/configs/vitl_336.yaml --weights /path/to/model_large.pth
python3 negative_set_experiment.py --alpha <BEST_ALPHA> --limit-images -1 \
    --config ./CAT-Seg/configs/vitl_336.yaml --weights /path/to/model_large.pth
```
