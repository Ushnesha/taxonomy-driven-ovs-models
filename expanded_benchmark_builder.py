"""
Expanded COCO Benchmark — Dataset Builder
==========================================
Algorithmically constructs the expanded benchmark:
  1. positive_set.json  — {cat_name: [img_ids with >=1 instance]}
  2. negative_set.json  — {cat_name: [img_ids with 0 instances]}
  3. word_sets.json     — {cat_name: {synset_name, definition, W_S, W_S_Hp, W_S_Hp_He}}

Synset disambiguation is AUTOMATIC: for each COCO category, we find the
WordNet synset whose lemmas contain the original COCO word. No manual
curation file is needed.

WordNet-first: if WordNet gives < 2 synonyms, BabelNet supplements.
All internal words use underscore form; display form uses spaces.
"""

from PIL import TiffImagePlugin
import json, os
from collections import defaultdict
import numpy as np
import pickle
# from expanded_benchmark_helpers import (
#     COCO_80, DATA_DIR, COCO_ANN,
#     load_coco, get_cat_name_to_id, download_image, get_gt_mask,
#     to_wn_form, to_display_form,
#     find_best_synset, build_word_sets_from_synset,
#     supplement_word_sets_with_babelnet,
# )

from expanded_benchmark_helpers import *

# ── Output paths ──
POS_PATH = os.path.join(DATA_DIR, "positive_set.json")
NEG_PATH = os.path.join(DATA_DIR, "negative_set.json")
WS_PATH  = os.path.join(DATA_DIR, "word_sets.json")
METADATA_PATH  = os.path.join(DATA_DIR, "img_metadata.pkl")
WS2_PATH  = os.path.join(DATA_DIR, "word_sets_v2.json")
POS2_PATH = os.path.join(DATA_DIR, "positive_set_v2.json")
NEG2_PATH = os.path.join(DATA_DIR, "negative_set_v2.json")


def build_positive_negative_sets(coco=None, force_rebuild=False):
    """
    Build (or load cached) positive and negative image sets.

    positive_set[cat_name] = [img_ids containing >=1 instance of that category]
    negative_set[cat_name] = [all_other_img_ids]

    Returns (positive_set, negative_set, all_cat_names).
    """
    if not force_rebuild and os.path.exists(POS_PATH) and os.path.exists(NEG_PATH):
        print("Loading cached positive/negative sets...")
        with open(POS_PATH) as f:
            positive_set = json.load(f)
        with open(NEG_PATH) as f:
            negative_set = json.load(f)
        all_cat_names = sorted(positive_set.keys())
        print(f"  {len(all_cat_names)} categories loaded from disk")
        return positive_set, negative_set, all_cat_names

    if coco is None:
        coco = load_coco(COCO_ANN)

    all_img_ids = coco.getImgIds()
    cat_name_to_id = get_cat_name_to_id(coco)
    cat_id_to_name = {v: k for k, v in cat_name_to_id.items()}
    all_cat_names = sorted(cat_name_to_id.keys())
    print(f"  {len(all_img_ids)} images, {len(all_cat_names)} categories")

    # Build positive set
    print("Building positive set...")
    positive_set = defaultdict(set)
    for ann in coco.dataset["annotations"]:
        cat_name = cat_id_to_name.get(ann["category_id"])
        if cat_name:
            positive_set[cat_name].add(ann["image_id"])
    positive_set = {cat: sorted(ids) for cat, ids in positive_set.items()}

    with open(POS_PATH, "w") as f:
        json.dump(positive_set, f)
    print(f"  Saved -> {POS_PATH}")

    # Build negative set
    print("Building negative set...")
    all_ids_set = set(all_img_ids)
    negative_set = {}
    for cat_name in all_cat_names:
        pos_ids = set(positive_set.get(cat_name, []))
        negative_set[cat_name] = sorted(all_ids_set - pos_ids)

    with open(NEG_PATH, "w") as f:
        json.dump(negative_set, f)
    print(f"  Saved -> {NEG_PATH}")

    return positive_set, negative_set, all_cat_names

from collections import defaultdict
def build_pos_neg_sets_from_bnchmrk(img_bnch):
    """
    Builds positive and negative image sets for categories in the benchmark.
    
    Returns:
      positive_set: {category: [img_ids where category exists]}
      negative_set: {category: [img_ids where category does NOT exist]}
    """
    # 1. Collect all unique image IDs in this benchmark
    all_img_ids = [img["img_id"] for img in img_bnch]
    all_img_ids_set = set(all_img_ids)
    
    # 2. Build the positive set
    positive_set = defaultdict(list)
    for img in img_bnch:
        img_id = img["img_id"]
        for cat in img["gt_objects"]:
            if img_id not in positive_set[cat]:
                positive_set[cat].append(img_id)
                
    # Sort positive lists for consistency
    positive_set = {cat: sorted(ids) for cat, ids in positive_set.items()}
    
    # 3. Build the negative set (All IDs minus positive IDs)
    negative_set = {}
    for cat, pos_ids in positive_set.items():
        pos_set = set(pos_ids)
        negative_set[cat] = sorted(list(all_img_ids_set - pos_set))
        
    return positive_set, negative_set


def build_word_sets(force_rebuild=False, use_babelnet=True):
    """
    Algorithmically build word sets for all 80 COCO categories.

    For each category:
      1. Convert spaces -> underscores for WordNet lookup
      2. Find the synset whose lemmas contain the original COCO word (auto-disambiguation)
      3. Extract W_S (synonyms), W_S_Hp (+ hyponyms), W_S_Hp_He (+ hypernyms)
      4. If W_S has < 2 synonyms and use_babelnet=True, supplement with BabelNet

    Returns (word_sets, level_counts, disambiguation_report).
    """
    if not force_rebuild and os.path.exists(WS_PATH):
        print(f"Loading cached word sets from {WS_PATH}...")
        with open(WS_PATH) as f:
            ws_data = json.load(f)
        return (
            ws_data["categories"],
            ws_data["metadata"]["word_counts"],
            ws_data["metadata"].get("disambiguation_report", {})
        )

    print("Building word sets algorithmically (WordNet-first, BabelNet-fallback)...")
    print(f"  Processing {len(COCO_80)} COCO categories...\n")

    word_sets = {}
    level_counts = {"W_S": 0, "W_S_Hp": 0, "W_S_Hp_He": 0}
    report = {
        "total": len(COCO_80),
        "wordnet_found": 0,
        "wordnet_missing": 0,
        "babelnet_supplemented": 0,
        "babelnet_failed": 0,
        "missing_synsets": [],
        "babelnet_details": [],
    }

    for cat_name in COCO_80:
        word_wn = to_wn_form(cat_name)
        print(f"  {cat_name:20s} -> {word_wn:30s} ...", end=" ", flush=True)

        # Step 1-2: Find best synset via auto-disambiguation
        synset = find_best_synset(cat_name)

        if synset is None:
            report["wordnet_missing"] += 1
            report["missing_synsets"].append(cat_name)
            print("NO SYNSET FOUND")
            continue

        report["wordnet_found"] += 1

        # Step 3: Build word sets from WordNet
        ws = build_word_sets_from_synset(synset)
        synset_name = synset.name()
        definition = synset.definition()
        babelnet_used = False

        # Step 4: Supplement with BabelNet if needed
        if use_babelnet and len(ws["W_S"]) < 2:
            print(f"[WN:{len(ws['W_S'])}syn]", end=" ", flush=True)
            ws_enriched = supplement_word_sets_with_babelnet(cat_name, ws)
            if len(ws_enriched["W_S"]) > len(ws["W_S"]):
                report["babelnet_supplemented"] += 1
                delta_syn = len(ws_enriched["W_S"]) - len(ws["W_S"])
                delta_hypo = len(ws_enriched["W_S_Hp"]) - len(ws["W_S_Hp"])
                delta_hyper = len(ws_enriched["W_S_Hp_He"]) - len(ws["W_S_Hp_He"])
                report["babelnet_details"].append({
                    "category": cat_name,
                    "wn_synonyms": len(ws["W_S"]),
                    "bn_added_synonyms": delta_syn,
                    "bn_added_hyponyms": delta_hypo,
                    "bn_added_hypernyms": delta_hyper,
                })
                ws = ws_enriched
                babelnet_used = True
                print(f"-> [BN:{len(ws['W_S'])}syn]", end=" ", flush=True)
            else:
                report["babelnet_failed"] += 1

        # Store
        word_sets[cat_name] = {
            "synset_name": synset_name,
            "definition": definition,
            "W_S": ws["W_S"],
            "W_S_Hp": ws["W_S_Hp"],
            "W_S_Hp_He": ws["W_S_Hp_He"],
            "babelnet_supplemented": babelnet_used,
        }

        for level in ["W_S", "W_S_Hp", "W_S_Hp_He"]:
            level_counts[level] += len(word_sets[cat_name][level])

        print(f"| W_S:{len(ws['W_S'])} Hp:{len(ws['W_S_Hp'])} He:{len(ws['W_S_Hp_He'])}")

    # Save
    ws_data = {
        "metadata": {
            "total_categories": len(word_sets),
            "source": "Algorithmic WordNet disambiguation + BabelNet fallback",
            "word_counts": level_counts,
            "disambiguation_report": report,
        },
        "categories": word_sets,
    }
    with open(WS_PATH, "w") as f:
        json.dump(ws_data, f, indent=2)
    print(f"\n  Saved -> {WS_PATH}")

    return word_sets, level_counts, report

def create_benchmark_from_coco(coco_dataset, limit=5000, img_bnch=[], cat_bnch={}):
    img_ids = coco_dataset.getImgIds()
    print(len(img_ids))
    for img_id in img_ids[:limit]:
        img_meta = coco_dataset.loadImgs(img_id)[0]
        h, w = img_meta["height"], img_meta["width"]

        gt_objects = []
        gt_bin_masks = []
        anns = get_anns_for_img_id(img_id, coco_dataset)
        
        for ann in anns:
            # start_time = time()
            category = coco_dataset.loadCats(ann["category_id"])[0]
            # print(category)
            cat_name = category["name"]
            if cat_name not in gt_objects:
                gt_objects.append(cat_name)
                gt_bin_masks.append(get_gt_mask(coco_dataset, img_id, category['id']))
                # gt_bin_masks.append(get_bin_masks(ann["segmentation"], h, w))
            if cat_name not in cat_bnch:
                # 1. Query local WordNet first (extremely fast)
                definition_wn, w_s_wn, w_s_hp_wn, w_s_he_wn = build_word_sets_from_synset_v2(
                    word=cat_name, supporting_words=category["supercategory"]
                )
                definition = definition_wn
                synonyms = w_s_wn
                hyponyms = w_s_hp_wn
                hypernyms = w_s_he_wn
                
                # 2. Fallback to BabelNet ONLY if WordNet fails or has fewer than 2 synonyms
                if not w_s_wn:
                    print(f"WordNet failed/insufficient for '{cat_name}'. Querying BabelNet API...")
                    definition_bn, w_s_bn, w_s_hp_bn, w_s_he_bn = supplement_word_sets_with_babelnet_v2(
                        word=cat_name, supporting_words=category["supercategory"]
                    )
                
                    definition = definition_bn if definition_wn == "" else definition_wn
                    synonyms.extend(w_s_bn)
                    hyponyms.extend(w_s_hp_bn)
                    hypernyms.extend(w_s_he_wn)
                
                
                # 3. Save to cat_bnch, limiting lists to at most 5 unique elements
                cat_bnch[cat_name] = {
                    "cat_src_id" : category["id"],
                    "cat_src" : "coco",
                    "cat_id" : f"coco_{category['id']}",
                    "definition" : definition,
                    "synonyms": clean_taxonomy_list(synonyms[:5]),
                    "hyponyms": clean_taxonomy_list(hyponyms[:5]),
                    "hypernyms": clean_taxonomy_list(hypernyms[:5])
                }
            # print(f"time taken: {(time()-start_time)*1000:.1f} ms")
        
        img_bnch.append({
            "img_src" : "coco",
            "img_src_id": img_id,
            "img_id" : f"coco_{img_id}",
            "width": w,
            "height": h,
            "filename": img_meta["file_name"],
            "img_url": img_meta.get("coco_url", ""),
            "gt_objects": gt_objects,
            "gt_bin_masks": gt_bin_masks
        })

    return img_bnch, cat_bnch

def create_benchmark_from_ade20K(ade_dataset, limit=2000, img_bnch=[], cat_bnch={}):
    for idx in range(limit):
        data = ade_dataset[idx]
        img_pil = data["image"]
        img_id = int(os.path.splitext(data['filename'])[0].split('_')[-1])
        w, h = img_pil.size

        gt_objects = []
        gt_bin_masks = []
        
        for obj_id,obj in enumerate(data["objects"]):
            # start_time = time()
            # print(category)
            cat_name = obj["raw_name"]
            if cat_name not in gt_objects:
                gt_objects.append(cat_name)
                gt_bin_masks.append(get_gt_mask_for_ade(data, cat_name))
            if cat_name not in cat_bnch:
                # 1. Query local WordNet first (extremely fast)
                hypernyms = obj["hypernym"] if len(obj["hypernym"]) > 0 else []
                definition_wn, w_s_wn, w_s_hp_wn, w_s_he_wn = build_word_sets_from_synset_v2(
                    word=cat_name, supporting_words=obj["hypernym"][:2]
                )
                definition = definition_wn
                synonyms = w_s_wn
                hyponyms = w_s_hp_wn
                hypernyms.extend(w_s_he_wn)
                
                # 2. Fallback to BabelNet ONLY if WordNet fails or has fewer than 2 synonyms
                if not w_s_wn:
                    print(f"WordNet failed/insufficient for '{cat_name}'. Querying BabelNet API...")
                    definition_bn, w_s_bn, w_s_hp_bn, w_s_he_bn = supplement_word_sets_with_babelnet_v2(
                        word=cat_name, supporting_words=obj["hypernym"][:2]
                    )
                
                    definition = definition_bn if definition_wn == "" else definition_wn
                    synonyms.extend(w_s_bn)
                    hyponyms.extend(w_s_hp_bn)
                    hypernyms.extend(w_s_he_wn)
                
                
                # 3. Save to cat_bnch, limiting lists to at most 5 unique elements
                cat_bnch[cat_name] = {
                    "cat_src_id" : obj['name_ndx'],
                    "cat_src" : "ade20k",
                    "cat_id" : f"ade20k_{obj['name_ndx']}",
                    "definition" : definition,
                    "synonyms": clean_taxonomy_list(synonyms)[:5],
                    "hyponyms": clean_taxonomy_list(hyponyms)[:5],
                    "hypernyms": clean_taxonomy_list(hypernyms)[:5]
                }
            # print(f"time taken: {(time()-start_time)*1000:.1f} ms")
        
        img_bnch.append({
            "img_src" : "ade20k",
            "img_src_id": img_id,
            "img_id" : f"ade20k_{img_id}",
            "width": w,
            "height": h,
            "filename": data["filename"],
            "img_url": "",
            "gt_objects": gt_objects,
            "gt_bin_masks": gt_bin_masks
        })

    return img_bnch, cat_bnch

def build_benchmark(coco_dataset, ade_dataset, limit_coco=5000, limit_ade=2000):
    img_bnch, cat_bnch = [], defaultdict(list)
    
    # 1. COCO
    print("Processing COCO...")
    img_bnch, cat_bnch = create_benchmark_from_coco(coco_dataset, limit=limit_coco, img_bnch=img_bnch, cat_bnch=cat_bnch)
    # 2. ADE
    print("Processing ADE...")
    img_bnch, cat_bnch = create_benchmark_from_ade20K(ade_dataset, limit=limit_ade, img_bnch=img_bnch, cat_bnch=cat_bnch)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(img_bnch, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  Saved -> {METADATA_PATH}")
    with open(WS2_PATH, "w") as f:
        json.dump(cat_bnch, f, indent=2)
    print(f"\n  Saved -> {WS2_PATH}")
    
    # 3. Combined pos/neg sets
    pos_set, neg_set = build_pos_neg_sets_from_bnchmrk(img_bnch)
    with open(POS2_PATH, "w") as f:
        json.dump(pos_set, f, indent=2)
    print(f"\n  Saved -> {POS2_PATH}")
    with open(NEG2_PATH, "w") as f:
        json.dump(neg_set, f, indent=2)
    print(f"\n  Saved -> {NEG2_PATH}")

    return img_bnch, cat_bnch


    


def build_all(force_rebuild=False, use_babelnet=True):
    """
    Run the full dataset construction pipeline.
    Returns (positive_set, negative_set, word_sets, report).
    """
    print("=" * 60)
    print("EXPANDED COCO BENCHMARK — Dataset Builder")
    print("=" * 60)

    # 1. Load COCO
    print("\n[1/4] Loading COCO and ADE20K datasets...")
    coco = load_coco()
    ade = load_ade20k()


    # 2. Positive & negative sets
    print("\n[2/4] Building positive/negative image sets...")
    positive_set, negative_set, all_cat_names = build_positive_negative_sets(
        coco=coco, force_rebuild=force_rebuild
    )

    # Stats
    pos_counts = [len(v) for v in positive_set.values()]
    neg_counts = [len(v) for v in negative_set.values()]
    print(f"  Positive: min={min(pos_counts)}, max={max(pos_counts)}, mean={np.mean(pos_counts):.0f}")
    print(f"  Negative: min={min(neg_counts)}, max={max(neg_counts)}, mean={np.mean(neg_counts):.0f}")

    # 3. Word sets
    print("\n[3/4] Building word sets...")
    word_sets, level_counts, report = build_word_sets(
        force_rebuild=force_rebuild, use_babelnet=use_babelnet
    )

     # 4. Benchmark Datasets    
    print("\n[4/4] Building benchmark datasets...")
    

    

    # Summary
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"  Categories with word sets: {len(word_sets)}/{len(COCO_80)}")
    print(f"  Total words across all categories:")
    for level, count in level_counts.items():
        print(f"    {level}: {count}")

    blendable = [c for c, w in word_sets.items() if len(w["W_S"]) >= 2]
    limited  = [c for c, w in word_sets.items() if len(w["W_S"]) < 2]
    print(f"\n  Blendable (>=2 synonyms): {len(blendable)}")
    print(f"  Limited (<2 synonyms):   {len(limited)}")

    if report.get("babelnet_supplemented", 0) > 0:
        print(f"\n  BabelNet supplemented: {report['babelnet_supplemented']} categories")
        for detail in report.get("babelnet_details", []):
            print(f"    {detail['category']}: +{detail['bn_added_synonyms']}syn "
                  f"+{detail['bn_added_hyponyms']}hypo +{detail['bn_added_hypernyms']}hyper")

    if report.get("missing_synsets"):
        print(f"\n  Categories with no synset: {report['missing_synsets']}")

    return positive_set, negative_set, word_sets, report


# ── Standalone execution ──
if __name__ == "__main__":
    # build_all(force_rebuild=False, use_babelnet=True)
    coco_dataset = load_coco()
    ade_dataset = load_ade20k()
    build_benchmark(coco_dataset, ade_dataset, limit_coco=10, limit_ade=10)
