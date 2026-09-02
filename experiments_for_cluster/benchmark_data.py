"""
Loads and serves the precomputed benchmark at ../../benchmark/ (sibling of
this repo, built by expanding LVIS + ADE20K + Pascal VOC -- no COCO): 346
canonical categories, 22,357 images, positive/negative image sets and
one-level WordNet synonym/hyponym/hypernym sets per category.

This replaces experiment_common.py's BUILDERS, which instead re-derived
positive/negative sets and word variants live from the raw datasets on every
run. Every orchestration script (positive_set_experiment.py,
negative_set_experiment.py, and later ablations) shares this module for:
loading the 4 benchmark files, resolving (dataset, img_ref) -> (PIL image,
GT mask), and building each category's {orig, syn, hypo, hyper} variant words.

Known caveat (deliberately out of scope here, see conversation decision log):
word_sets_v2.json has real WordNet-sense disambiguation errors for some
categories (e.g. "dog" resolves to "someone morally reprehensible", not the
animal). Fixing that is a separate benchmark-rebuild task; this module serves
the data as-is.
"""
import csv
import json
import os
import pickle
import sys
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
import requests
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import expanded_benchmark_helpers as bm_hp  # noqa: E402
from expanded_benchmark_helpers import to_display_form  # noqa: E402

DEFAULT_BENCHMARK_DIR = os.environ.get(
    "BENCHMARK_DIR", os.path.normpath(os.path.join(REPO_ROOT, "..", "benchmark"))
)


@dataclass
class Benchmark:
    word_sets: dict          # {category: {canonical_name, synset_id, definition, synonyms, hyponyms, hypernyms}}
    positive_set: dict       # {category: [img_id, ...]}
    negative_set: dict       # {category: [img_id, ...]}
    img_by_id: dict          # {img_id: {img_src, img_src_id, img_id, width, height, filename, img_url, gt_objects, gt_bin_masks}}
    categories: list = field(default_factory=list)  # sorted category names with >=1 usable linguistic variant
    hyper_siblings: dict = field(default_factory=dict)  # {category: [category, ...siblings sharing its hyper word]}


def load_benchmark(benchmark_dir=None):
    benchmark_dir = benchmark_dir or DEFAULT_BENCHMARK_DIR
    with open(os.path.join(benchmark_dir, "word_sets_v2.json")) as f:
        word_sets = json.load(f)
    with open(os.path.join(benchmark_dir, "positive_set_v2.json")) as f:
        positive_set = json.load(f)
    with open(os.path.join(benchmark_dir, "negative_set_v2.json")) as f:
        negative_set = json.load(f)
    with open(os.path.join(benchmark_dir, "img_metadata.pkl"), "rb") as f:
        img_metadata = pickle.load(f)
    img_by_id = {entry["img_id"]: entry for entry in img_metadata}

    categories = sorted(
        cat for cat in word_sets
        if len(get_variants(cat, word_sets)) > 1  # has >=1 variant beyond "orig"
    )
    hyper_siblings = build_hyper_siblings(categories, word_sets)
    return Benchmark(word_sets, positive_set, negative_set, img_by_id, categories, hyper_siblings)


def build_hyper_siblings(categories, word_sets):
    """{category: [category, ...siblings]} -- categories whose "hyper" variant word
    (get_variants()'s first-hypernym-candidate pick) matches this category's own,
    including itself. Categories with no shared hypernym map to [category] alone.

    Two canonical categories can resolve to the same hypernym word (e.g. "chair" and
    "sofa" both -> "seat"); when scoring the "hyper" variant, an image containing both
    should count either being correctly segmented, not just the one "chair"/"sofa" the
    query happened to be derived from -- see decode_gt_mask_for_variant below."""
    by_word = defaultdict(list)
    cat_hyper_word = {}
    for cat in categories:
        hyper = get_variants(cat, word_sets).get("hyper")
        word = hyper.lower() if hyper else None
        cat_hyper_word[cat] = word
        if word:
            by_word[word].append(cat)
    return {cat: (by_word[word] if word else [cat]) for cat, word in cat_hyper_word.items()}


def get_variants(cat_name, word_sets):
    """
    {"orig": cat_name, "syn": ..., "hypo": ..., "hyper": ...} (levels missing
    from word_sets_v2.json's arrays for this category are simply absent from
    the dict) -- one representative word per linguistic-distance level,
    sourced from the precomputed word_sets_v2.json fields rather than a live
    WordNet call. Mirrors the old experiment_common.get_variants() picking
    convention (first candidate at each level).
    """
    entry = word_sets.get(cat_name)
    variants = {"orig": cat_name}
    if entry is None:
        return variants

    syn_candidates = [to_display_form(w) for w in entry.get("synonyms", [])
                       if to_display_form(w).lower() != cat_name.lower()]
    hypo_candidates = [to_display_form(w) for w in entry.get("hyponyms", [])]
    hyper_candidates = [to_display_form(w) for w in entry.get("hypernyms", [])]

    if syn_candidates:
        variants["syn"] = syn_candidates[0]
    if hypo_candidates:
        variants["hypo"] = hypo_candidates[0]
    if hyper_candidates:
        variants["hyper"] = hyper_candidates[0]
    return variants


# =============================================================================
# Image + GT mask retrieval.
# =============================================================================

_ADE20K_DS = None
_PASCALVOC_DS = None


def _ade20k_dataset():
    global _ADE20K_DS
    if _ADE20K_DS is None:
        _ADE20K_DS = bm_hp.load_ade20k()
    return _ADE20K_DS


def _pascalvoc_dataset():
    global _PASCALVOC_DS
    if _PASCALVOC_DS is None:
        _PASCALVOC_DS = bm_hp.load_pascalvoc()
    return _PASCALVOC_DS


def fetch_image(entry):
    """PIL image (RGB) for one img_metadata.pkl entry. lvis downloads by URL
    (img_url is a real COCO CDN url, since LVIS images ARE COCO images);
    ade20k/pascalvoc index into the same HF datasets the benchmark was built
    from (img_src_id is the dataset row index, filename/img_url are empty
    for these two sources)."""
    src = entry["img_src"]
    if src == "lvis":
        r = requests.get(entry["img_url"], timeout=30)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    if src == "ade20k":
        return _ade20k_dataset()[entry["img_src_id"]]["image"].convert("RGB")
    if src == "pascalvoc":
        return _pascalvoc_dataset()[entry["img_src_id"]]["image"].convert("RGB")
    raise ValueError(f"Unknown img_src '{src}'")


def decode_gt_mask(entry, cat_name):
    """np.uint8[h, w] binary mask for `cat_name` in this image, or None if
    the category isn't present (gt_bin_masks are zlib-compressed flat
    uint8 0/1 buffers, one per entry in gt_objects, in gt_objects order)."""
    gt_objects = entry["gt_objects"]
    if cat_name not in gt_objects:
        return None
    idx = gt_objects.index(cat_name)
    raw = zlib.decompress(entry["gt_bin_masks"][idx])
    h, w = entry["height"], entry["width"]
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w).copy()


def decode_gt_mask_for_variant(bm, entry, cat_name, variant_name):
    """decode_gt_mask(entry, cat_name), except for variant_name == "hyper": unions in
    every sibling category's mask (bm.hyper_siblings[cat_name]) that's also present in
    this image. Querying a shared hypernym (e.g. "seat" for both "chair" and "sofa") and
    correctly segmenting either one is a correct answer to that query -- scoring it
    against only the one canonical category's mask would spuriously penalize IoU on
    images that contain both. No-op (identical to decode_gt_mask) for categories with no
    hypernym siblings, and for every other variant level."""
    mask = decode_gt_mask(entry, cat_name)
    if variant_name != "hyper":
        return mask
    for sib in bm.hyper_siblings.get(cat_name, [cat_name]):
        if sib == cat_name:
            continue
        sib_mask = decode_gt_mask(entry, sib)
        if sib_mask is not None:
            mask = sib_mask if mask is None else np.maximum(mask, sib_mask)
    return mask


def hyper_variant_contaminated(bm, entry, cat_name):
    """True if this image (already known to lack cat_name, i.e. it's in cat_name's
    negative set) contains a sibling category sharing cat_name's hyper word. Querying
    that shared hypernym on such an image has a legitimate target (the sibling), so a
    model correctly segmenting it is not a false positive for cat_name's negative-set
    FPR -- callers should skip the "hyper" variant row for this image rather than score
    it. Cheap membership check against gt_objects, no mask decode needed."""
    gt_objects = set(entry.get("gt_objects", []))
    return any(sib in gt_objects for sib in bm.hyper_siblings.get(cat_name, [cat_name]) if sib != cat_name)


def false_positive_rate(pred_mask):
    """Fraction of pixels predicted positive -- used on the negative set,
    where IoU against an empty GT mask is 0 by construction."""
    return float(pred_mask.sum()) / pred_mask.size


compute_iou = bm_hp.compute_iou


# =============================================================================
# SCLIP needs a class-list config file naming every category it might be
# asked to embed (see models.py's module docstring). Its default
# configs/cls_coco_object.txt only lists COCO's 80 classes.
# =============================================================================

# configs/cls_coco_object.txt's own line 0 (generic scene/stuff background
# vocabulary) -- reused as-is since it's dataset-agnostic, not COCO-specific.
SCLIP_BACKGROUND_LINE = (
    "sky, wall, tree, wood, grass, road, sea, river, mountain, sands, desk, bed, "
    "building, cloud, lamp, door, window, wardrobe, ceiling, shelf, curtain, "
    "stair, floor, hill, rail, fence"
)


def build_sclip_name_path(bm, out_path):
    """Writes a SCLIP-format class-list file (one line per class, first
    comma-separated token is the class name predict_with_embeddings() keys
    must match) covering every category in `bm`, so SCLIP can be queried
    against our benchmark instead of just COCO's 80 classes."""
    lines = [SCLIP_BACKGROUND_LINE]
    for cat in bm.categories:
        entry = bm.word_sets[cat]
        syns = [to_display_form(w) for w in entry.get("synonyms", [])]
        syns = [w for w in syns if w.lower() != cat.lower()]
        lines.append(", ".join([cat] + syns))
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


# =============================================================================
# Shared checkpointed CSV writer + summary aggregation, used identically by
# every orchestration script.
# =============================================================================

def resumable_csv_writer(path, fieldnames, key_fields):
    """Opens `path` for append, returns (file, DictWriter, seen) where `seen`
    is the set of key_fields tuples already written -- callers skip any row
    whose key is already in `seen`, so an interrupted run resumes instead of
    recomputing/duplicating rows."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    exists = os.path.exists(path)
    seen = set()
    if exists:
        with open(path) as f:
            for row in csv.DictReader(f):
                seen.add(tuple(row[k] for k in key_fields))
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
        f.flush()
    return f, writer, seen


def summarize_csv(detail_path, group_fields, value_field, summary_path):
    """Aggregates mean(value_field) grouped by group_fields -- the "mean IoU
    per (model, approach, variant)" / "mean FPR per (model, approach)" step
    every orchestration script's __main__ runs after the detail CSV is done."""
    rows = []
    with open(detail_path) as f:
        for row in csv.DictReader(f):
            row[value_field] = float(row[value_field])
            rows.append(row)

    from collections import defaultdict
    by_key = defaultdict(list)
    for r in rows:
        by_key[tuple(r[g] for g in group_fields)].append(r[value_field])

    summary_fieldnames = list(group_fields) + ["n", f"mean_{value_field}"]
    summary_rows = []
    for key, values in sorted(by_key.items()):
        row = dict(zip(group_fields, key))
        row["n"] = len(values)
        row[f"mean_{value_field}"] = float(np.mean(values))
        summary_rows.append(row)

    out_dir = os.path.dirname(summary_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows
