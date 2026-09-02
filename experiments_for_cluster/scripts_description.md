# Scripts Description

**benchmark_data.py**
Shared library for the baseline/ours/shine/waffleclip/llm_descriptor suite: loads the
precomputed `../../benchmark/` (346 categories, 22,357 images across LVIS/ADE20K/PascalVOC --
built separately, no COCO), resolves (dataset, img_ref) -> (PIL image, GT mask), builds each
category's {orig, syn, hypo, hyper} variant words, and groups categories that share a hypernym
(`hyper_siblings`) so a shared-hypernym query (e.g. "seat" for both "chair" and "sofa") is scored
against the union of every present sibling's mask, not just the one the query was derived from
(`decode_gt_mask_for_variant` for IoU on the positive set, `hyper_variant_contaminated` to skip a
now-invalid false-positive row on the negative set). Also provides the resumable/checkpointed CSV
writer and summary aggregation both experiment scripts below share.

**approaches.py**
The 5 query-transform methods compared in the paper (baseline / ours / shine / waffleclip /
llm_descriptor), each computing a text embedding for a linguistic variant word against a given
pluggable model. `ours_blend_info`/`ours_embedding` do the WordNet-neighbor centroid blending;
`shine_embedding`, `waffleclip_embedding`, `llm_descriptor_embedding` are ports of the respective
papers' official repos (see the module's own docstring for exact provenance per method).
`approach_embedding()` is the single dispatch entrypoint every script below calls.

**models.py**
Factory over the 3 pluggable models this suite runs (clipseg, groupvit, sclip -- clip_vit_large
was dropped: its patch-similarity mask extraction is self-implemented by this project, not native
to or ported from an official CLIP-ViT-L segmentation pipeline). `predict_masks_for_tags()` papers
over SCLIP's one-embedding-per-class-per-call restriction so callers don't need model-specific
branching.

**positive_set_experiment.py**
For one model (`--model`) and every eligible category, across all 4 linguistic variants and any
subset of the 5 approaches (`--approaches`), measures IoU against ground truth on the category's
positive images. Checkpointed -- safe to interrupt and resume. Run once per model to fill the
paper's model x approach matrix.

**negative_set_experiment.py**
Same (model, variant, approach) grid as positive_set_experiment.py, run on each category's
negative images instead. Reports false-positive rate (fraction of image pixels predicted
positive) rather than IoU, since IoU against an empty ground truth is 0 by construction.

**shine_experiment.py / waffleclip_experiment.py / llm_description_experiment.py**
Thin convenience wrappers (~30 lines each) over `positive_set_experiment.run()` with
`--approaches` locked to `baseline,<method>` -- they exist only so
`python3 shine_experiment.py --model X` matches the one-script-per-baseline naming, and produce
identical results to calling `positive_set_experiment.py --model X --approaches baseline,shine`
directly. The actual method implementations live in `approaches.py`.

**prompt_template_experiment.py**
Compares the bare-word baseline against the "Image of a {word}" prompt template across all 4
linguistic variants and every model, isolating the effect of prompt phrasing (paper's E2,
extended) -- not one of approaches.py's 5 dispatched methods, so it's its own small script
against benchmark_data.py/models.py rather than a positive_set_experiment.py wrapper.

**alpha_value_experiment.py**
A separate experiment from the suite above: derives each model's best alpha value by sweeping on
COCO specifically (not the `../../benchmark/` the other scripts use), since COCO is the
calibration dataset -- once a model's alpha is found here at full scale, that fixed value is what
gets passed as `--alpha` to positive_set_experiment.py / negative_set_experiment.py when running
"ours" on the other 3 datasets. Coarse grid (0.0-1.0, step 0.1) then fine grid (0.50-0.80, step
0.01), both unweighted- and cosine-weighted-centroid. Run once per model
(`--model clipseg|groupvit|sclip`); defaults to full scale (`--n-categories -1 --n-images -1`).

**alpha_value_ablation/**
Not a script in this directory -- see `../../alpha_value_ablation/`. Self-contained per-model
copies of this same COCO alpha sweep (own venv, own vendored model code where needed, e.g.
SCLIP's official repo), meant to run standalone on the cluster with nothing else from this repo
checked out.
