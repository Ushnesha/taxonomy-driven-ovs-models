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


VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
    "train", "tvmonitor"
]
VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
]
VOC_CLS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

def extract_gt_objects_and_masks(dataset_item):
    """
    Parses a single dataset item dictionary with keys 'image' and 'mask'.
    Returns:
        gt_objects (list): List of class names present in the image.
        gt_bin_masks (list): List of zlib-compressed binary masks.
    """
    import zlib
    mask_img = dataset_item["mask"]
    
    # Convert mask PIL image to numpy array of shape (H, W, 3)
    mask_np = np.array(mask_img.convert("RGB"))
    
    gt_objects = []
    gt_bin_masks = []
    
    # Loop through foreground classes (1 to 20)
    for class_id in range(1, len(VOC_CLASSES)):
        class_name = VOC_CLASSES[class_id]
        color = VOC_COLORMAP[class_id]
        
        # Check where the RGB channels match this class color
        binary_mask = (mask_np == color).all(axis=-1)
        
        # If the class is present in the image, keep it
        if binary_mask.any():
            gt_objects.append(class_name)
            # Store as compressed binary mask of uint8 (0 or 1)
            compressed = zlib.compress(binary_mask.astype(np.uint8).tobytes())
            gt_bin_masks.append(compressed)
            
    return gt_objects, gt_bin_masks


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

def process_single_coco_image(img_id, coco_dataset):
    """Worker function to process a single COCO image."""
    img_meta = coco_dataset.loadImgs(img_id)[0]
    h, w = img_meta["height"], img_meta["width"]

    gt_objects = []
    gt_bin_masks = []
    anns = get_anns_for_img_id(img_id, coco_dataset)
    
    for ann in anns:
        category = coco_dataset.loadCats(ann["category_id"])[0]
        cat_name = category["name"]
        if cat_name not in gt_objects:
            gt_objects.append(cat_name)
            mask = get_gt_mask(coco_dataset, img_id, category['id'])
            if mask is not None:
                import zlib
                gt_bin_masks.append(zlib.compress(mask.tobytes()))
            else:
                gt_bin_masks.append(b"")
            
    return {
        "img_src" : "coco",
        "img_src_id": img_id,
        "img_id" : f"coco_{img_id}",
        "width": w,
        "height": h,
        "filename": img_meta["file_name"],
        "img_url": img_meta.get("coco_url", ""),
        "gt_objects": gt_objects,
        "gt_bin_masks": gt_bin_masks
    }

def create_benchmark_from_coco(coco_dataset, img_ids_to_process, img_bnch=[], cat_bnch={}, num_workers=8):
    img_ids = img_ids_to_process
    if not img_ids:
        return img_bnch, cat_bnch
        
    # 1. Build taxonomy sequentially first (handles API/NLTK locks safely)
    print("Building taxonomy for COCO categories...")
    for img_id in img_ids:
        anns = get_anns_for_img_id(img_id, coco_dataset)
        for ann in anns:
            category = coco_dataset.loadCats(ann["category_id"])[0]
            cat_name = category["name"]
            if cat_name not in cat_bnch:
                definition_wn, w_s_wn, w_s_hp_wn, w_s_he_wn = build_word_sets_from_synset_v2(
                    word=cat_name, supporting_words=category["supercategory"]
                )
                definition = definition_wn
                synonyms = list(w_s_wn)
                hyponyms = list(w_s_hp_wn)
                hypernyms = list(w_s_he_wn)
                
                if not w_s_wn:
                    definition_bn, w_s_bn, w_s_hp_bn, w_s_he_bn = supplement_word_sets_with_babelnet_v2(
                        word=cat_name, supporting_words=category["supercategory"]
                    )
                    definition = definition_bn if definition_wn == "" else definition_wn
                    synonyms.extend(w_s_bn)
                    hyponyms.extend(w_s_hp_bn)
                    hypernyms.extend(w_s_he_bn)
                
                cat_bnch[cat_name] = {
                    "cat_src_id" : category["id"],
                    "cat_src" : "coco",
                    "cat_id" : f"coco_{category['id']}",
                    "definition" : definition,
                    "synonyms": clean_taxonomy_list(synonyms)[:5],
                    "hyponyms": clean_taxonomy_list(hyponyms)[:5],
                    "hypernyms": clean_taxonomy_list(hypernyms)[:5]
                }
                
    # 2. Process image masks sequentially
    from tqdm import tqdm
    
    print(f"Processing {len(img_ids)} COCO images...")
    checkpoint_interval = 100
    for i, img_id in enumerate(tqdm(img_ids, desc="COCO Images")):
        result = process_single_coco_image(img_id, coco_dataset)
        img_bnch.append(result)
        
        # Periodic crash-safe checkpointing
        if (i + 1) % checkpoint_interval == 0 or (i + 1) == len(img_ids):
            tmp_path = METADATA_PATH + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(img_bnch, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, METADATA_PATH)
            
            tmp_ws_path = WS2_PATH + ".tmp"
            with open(tmp_ws_path, "w") as f:
                json.dump(cat_bnch, f, indent=2)
            os.replace(tmp_ws_path, WS2_PATH)
            
    return img_bnch, cat_bnch


def create_benchmark_from_pascalvoc(pascal_dataset, indices_to_process, img_bnch=[], cat_bnch={}, num_workers=8):
    if not indices_to_process:
        return img_bnch, cat_bnch

    # 1. Build taxonomy sequentially first
    print("Building taxonomy for Pascal VOC categories...")
    for class_id in range(1, len(VOC_CLASSES)):
        cat_name = VOC_CLASSES[class_id]
        if cat_name not in cat_bnch:
            definition_wn, w_s_wn, w_s_hp_wn, w_s_he_wn = build_word_sets_from_synset_v2(
                word=cat_name, supporting_words=[cat_name]
            )
            definition = definition_wn
            synonyms = list(w_s_wn)
            hyponyms = list(w_s_hp_wn)
            hypernyms = list(w_s_he_wn)
            
            if not w_s_wn:
                print(f"WordNet failed/insufficient for '{cat_name}'. Querying BabelNet API...")
                definition_bn, w_s_bn, w_s_hp_bn, w_s_he_bn = supplement_word_sets_with_babelnet_v2(
                    word=cat_name, supporting_words=[cat_name]
                )
                definition = definition_bn if definition_wn == "" else definition_wn
                synonyms.extend(w_s_bn)
                hyponyms.extend(w_s_hp_bn)
                hypernyms.extend(w_s_he_bn)
            
            cat_bnch[cat_name] = {
                "cat_src_id": class_id,
                "cat_src": "pascalvoc",
                "cat_id": f"pascalvoc_{class_id}",
                "definition": definition,
                "synonyms": clean_taxonomy_list(synonyms)[:5],
                "hyponyms": clean_taxonomy_list(hyponyms)[:5],
                "hypernyms": clean_taxonomy_list(hypernyms)[:5]
            }

    # 2. Process Pascal VOC images
    from tqdm import tqdm
    print(f"Processing {len(indices_to_process)} Pascal VOC images...")
    checkpoint_interval = 100
    for i, idx in enumerate(tqdm(indices_to_process, desc="Pascal VOC Images")):
        data = pascal_dataset[idx]
        img_pil = data["image"]
        w, h = img_pil.size
        
        gt_objects, gt_bin_masks = extract_gt_objects_and_masks(data)
        
        img_bnch.append({
            "img_src": "pascalvoc",
            "img_src_id": idx,
            "img_id": f"pascalvoc_{idx}",
            "width": w,
            "height": h,
            "filename": "",
            "img_url": "",
            "gt_objects": gt_objects,
            "gt_bin_masks": gt_bin_masks
        })
        
        # Periodic crash-safe checkpointing
        if (i + 1) % checkpoint_interval == 0 or (i + 1) == len(indices_to_process):
            tmp_path = METADATA_PATH + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(img_bnch, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, METADATA_PATH)
            
            tmp_ws_path = WS2_PATH + ".tmp"
            with open(tmp_ws_path, "w") as f:
                json.dump(cat_bnch, f, indent=2)
            os.replace(tmp_ws_path, WS2_PATH)
            
    return img_bnch, cat_bnch


def create_benchmark_from_lvis(lvis_data, indices_to_process, img_bnch=[], cat_bnch={}):
    if not indices_to_process:
        return img_bnch, cat_bnch

    # 1. Build maps for fast lookup
    cat_id_to_meta = {cat["id"]: cat for cat in lvis_data["categories"]}
    
    # We want indices_to_process to index into the lvis_data["images"] list
    target_images = [lvis_data["images"][idx] for idx in indices_to_process]
    target_image_ids = {img["id"] for img in target_images}
    
    # Group annotations by image_id
    img_id_to_anns = defaultdict(list)
    for ann in lvis_data["annotations"]:
        if ann["image_id"] in target_image_ids:
            img_id_to_anns[ann["image_id"]].append(ann)
            
    # 2. Build taxonomy sequentially first (handles API/NLTK locks safely)
    print("Building taxonomy for LVIS categories...")
    present_cat_ids = set()
    for img_id in target_image_ids:
        for ann in img_id_to_anns[img_id]:
            present_cat_ids.add(ann["category_id"])
            
    for cat_id in present_cat_ids:
        cat_meta = cat_id_to_meta[cat_id]
        cat_name = cat_meta["name"].replace("_", " ")
        
        if cat_name not in cat_bnch:
            definition = cat_meta.get("def", "")
            syns = cat_meta.get("synonyms", [])
            synset = cat_meta.get("synset", None)
            definition_wn, w_s_wn, w_s_hp_wn, w_s_he_wn = build_word_sets_from_synset_v2(
                word=cat_name, supporting_words=syns, synset=synset
            )
            definition = definition_wn if definition == "" else definition
            synonyms = list(w_s_wn)
            hyponyms = list(w_s_hp_wn)
            hypernyms = list(w_s_he_wn)
            
            if not w_s_wn:
                print(f"WordNet failed/insufficient for '{cat_name}'. Querying BabelNet API...")
                definition_bn, w_s_bn, w_s_hp_bn, w_s_he_bn = supplement_word_sets_with_babelnet_v2(
                    word=cat_name, supporting_words=syns
                )
                definition = definition_bn if definition == "" else definition
                synonyms.extend(w_s_bn)
                hyponyms.extend(w_s_hp_bn)
                hypernyms.extend(w_s_he_bn)
                
            cat_bnch[cat_name] = {
                "cat_src_id": cat_id,
                "cat_src": "lvis",
                "cat_id": f"lvis_{cat_id}",
                "definition": definition,
                "synonyms": clean_taxonomy_list(synonyms)[:5],
                "hyponyms": clean_taxonomy_list(hyponyms)[:5],
                "hypernyms": clean_taxonomy_list(hypernyms)[:5]
            }
            
    # 3. Process image masks
    from tqdm import tqdm
    import zlib
    
    print(f"Processing {len(indices_to_process)} LVIS images...")
    checkpoint_interval = 100
    
    for i, idx in enumerate(tqdm(indices_to_process, desc="LVIS Images")):
        img_meta = lvis_data["images"][idx]
        img_id = img_meta["id"]
        w, h = img_meta["width"], img_meta["height"]
        
        # Group annotations by category name to merge masks
        cat_to_anns = defaultdict(list)
        for ann in img_id_to_anns[img_id]:
            cat_name = cat_id_to_meta[ann["category_id"]]["name"].replace("_", " ")
            cat_to_anns[cat_name].append(ann)
            
        gt_objects = []
        gt_bin_masks = []
        
        for cat_name, anns in cat_to_anns.items():
            merged_mask = np.zeros((h, w), dtype=np.uint8)
            for ann in anns:
                mask = decode_lvis_ann_to_mask(ann, h, w)
                if mask is not None:
                    merged_mask = np.maximum(merged_mask, mask)
            
            gt_objects.append(cat_name)
            gt_bin_masks.append(zlib.compress(merged_mask.tobytes()))
            
        img_bnch.append({
            "img_src": "lvis",
            "img_src_id": img_id,
            "img_id": f"lvis_{img_id}",
            "width": w,
            "height": h,
            "filename": img_meta.get("file_name", ""),
            "img_url": img_meta.get("coco_url", ""),
            "gt_objects": gt_objects,
            "gt_bin_masks": gt_bin_masks
        })
        
        # Periodic crash-safe checkpointing
        if (i + 1) % checkpoint_interval == 0 or (i + 1) == len(indices_to_process):
            tmp_path = METADATA_PATH + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(img_bnch, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, METADATA_PATH)
            
            tmp_ws_path = WS2_PATH + ".tmp"
            with open(tmp_ws_path, "w") as f:
                json.dump(cat_bnch, f, indent=2)
            os.replace(tmp_ws_path, WS2_PATH)
            
    return img_bnch, cat_bnch


def process_single_ade_image_data(idx, data, ade_dataset_full=None):
    """Worker function to process a single ADE20K image data dict."""
    # Try to get width and height from instances (which is extremely fast and avoids decoding raw image)
    instances = data.get("instances", [])
    fallback_img = None
    if instances:
        w, h = instances[0].size
    elif ade_dataset_full is not None:
        full_data = ade_dataset_full[idx]
        fallback_img = full_data["image"]
        w, h = fallback_img.size
    else:
        # Fallback default
        w, h = 256, 256
        
    img_id = int(os.path.splitext(data['filename'])[0].split('_')[-1])

    # Group object instances and masks by category name
    objects_by_cat = {}
    for obj_id, obj in enumerate(data["objects"]):
        cat_name = obj["raw_name"]
        if cat_name not in objects_by_cat:
            objects_by_cat[cat_name] = []
        objects_by_cat[cat_name].append(data["instances"][obj_id])

    gt_objects = []
    gt_bin_masks = []
    
    for cat_name, masks in objects_by_cat.items():
        # Combine all masks for this category using np.maximum
        merged_mask = np.zeros((h, w), dtype=np.uint8)
        for mask_img in masks:
            merged_mask = np.maximum(merged_mask, (np.array(mask_img) > 0).astype(np.uint8))
            
        gt_objects.append(cat_name)
        # Compress the mask to save massive memory
        import zlib
        gt_bin_masks.append(zlib.compress(merged_mask.tobytes()))
        
    # Explicitly close PIL images to prevent memory leaks
    for mask_img in instances:
        mask_img.close()
    if fallback_img is not None:
        fallback_img.close()
        
    return {
        "img_src" : "ade20k",
        "img_src_id": img_id,
        "img_id" : f"ade20k_{img_id}",
        "width": w,
        "height": h,
        "filename": data["filename"],
        "img_url": "",
        "gt_objects": gt_objects,
        "gt_bin_masks": gt_bin_masks
    }

def create_benchmark_from_ade20K(ade_dataset, indices_to_process, img_bnch=[], cat_bnch={}, num_workers=8):
    if not indices_to_process:
        return img_bnch, cat_bnch
        
    # Remove 'image' and 'segmentations' columns to avoid decoding/caching them during processing
    ade_dataset_light = ade_dataset.remove_columns(["image", "segmentations"])

    # 1. Build taxonomy sequentially first (handles API/NLTK locks safely)
    print("Building taxonomy for ADE20K categories...")
    # Select only the 'objects' column to avoid loading and decoding heavy image/mask data
    objects_column = ade_dataset_light.select(indices_to_process)["objects"]
    for objects in objects_column:
        for obj in objects:
            cat_name = obj["raw_name"]
            if cat_name not in cat_bnch:
                hypernyms = obj["hypernym"] if len(obj["hypernym"]) > 0 else []
                definition_wn, w_s_wn, w_s_hp_wn, w_s_he_wn = build_word_sets_from_synset_v2(
                    word=cat_name, supporting_words=obj["hypernym"][:2]
                )
                definition = definition_wn
                synonyms = list(w_s_wn)
                hyponyms = list(w_s_hp_wn)
                hypernyms = list(w_s_he_wn)
                
                if not w_s_wn:
                    print(f"WordNet failed/insufficient for '{cat_name}'. Querying BabelNet API...")
                    definition_bn, w_s_bn, w_s_hp_bn, w_s_he_bn = supplement_word_sets_with_babelnet_v2(
                        word=cat_name, supporting_words=obj["hypernym"][:2]
                    )
                    definition = definition_bn if definition_wn == "" else definition_wn
                    synonyms.extend(w_s_bn)
                    hyponyms.extend(w_s_hp_bn)
                    hypernyms.extend(w_s_he_wn)
                
                cat_bnch[cat_name] = {
                    "cat_src_id" : obj['name_ndx'],
                    "cat_src" : "ade20k",
                    "cat_id" : f"ade20k_{obj['name_ndx']}",
                    "definition" : definition,
                    "synonyms": clean_taxonomy_list(synonyms)[:5],
                    "hyponyms": clean_taxonomy_list(hyponyms)[:5],
                    "hypernyms": clean_taxonomy_list(hypernyms)[:5]
                }
                
    # 2. Process image masks in parallel (safe with ThreadPoolExecutor since memory is shared and GIL is released during Pillow C-decodes)
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    import pyarrow as pa
    import gc
    
    print(f"Processing {len(indices_to_process)} ADE20K images with {num_workers} threads...")
    checkpoint_interval = 100
    
    # Select the subset of rows to iterate sequentially (prevents PyArrow index-caching overhead)
    sub_dataset = ade_dataset_light.select(indices_to_process)
    
    def worker(item):
        i, data = item
        idx = indices_to_process[i]
        return process_single_ade_image_data(idx, data, ade_dataset)
        
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to monitor the progress of the parallel map
        for i, result in enumerate(tqdm(executor.map(worker, enumerate(sub_dataset)), total=len(sub_dataset), desc="ADE20K Images")):
            img_bnch.append(result)
            
            # Free memory frequently in the main thread
            if (i + 1) % 20 == 0:
                gc.collect()
                pa.default_memory_pool().release_unused()
                
            # Periodic crash-safe checkpointing
            if (i + 1) % checkpoint_interval == 0 or (i + 1) == len(indices_to_process):
                tmp_path = METADATA_PATH + ".tmp"
                with open(tmp_path, "wb") as f:
                    pickle.dump(img_bnch, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, METADATA_PATH)
                
                tmp_ws_path = WS2_PATH + ".tmp"
                with open(tmp_ws_path, "w") as f:
                    json.dump(cat_bnch, f, indent=2)
                os.replace(tmp_ws_path, WS2_PATH)
                
                gc.collect()
                pa.default_memory_pool().release_unused()
            
    return img_bnch, cat_bnch

def build_benchmark(coco_dataset, ade_dataset, pascal_dataset, lvis_dataset, limit_coco=5000, limit_ade=2000, limit_pascal=1500, limit_lvis=5000):
    import pickle
    
    # 1. Load existing cache if present
    img_bnch = []
    cat_bnch = {}
    
    if os.path.exists(METADATA_PATH):
        print(f"Loading existing benchmark images from {METADATA_PATH}...")
        try:
            with open(METADATA_PATH, "rb") as f:
                img_bnch = pickle.load(f)
            print(f"  Loaded {len(img_bnch)} previously processed images.")
        except Exception as e:
            print(f"  Warning: Failed to load existing metadata, starting fresh: {e}")
            img_bnch = []
            
    if os.path.exists(WS2_PATH):
        print(f"Loading existing category taxonomy from {WS2_PATH}...")
        try:
            with open(WS2_PATH, "r") as f:
                cat_bnch = json.load(f)
            print(f"  Loaded {len(cat_bnch)} category taxonomy entries.")
        except Exception as e:
            print(f"  Warning: Failed to load taxonomy cache: {e}")
            cat_bnch = {}

    # 2. Determine what has already been processed
    processed_coco_ids = {img["img_src_id"] for img in img_bnch if img["img_src"] == "coco"}
    processed_ade_ids = {img["img_src_id"] for img in img_bnch if img["img_src"] == "ade20k"}
    processed_pascal_ids = {img["img_src_id"] for img in img_bnch if img["img_src"] == "pascalvoc"}
    processed_lvis_ids = {img["img_src_id"] for img in img_bnch if img["img_src"] == "lvis"}

    # 3. Filter COCO images
    print("Checking COCO cache...")
    all_coco_img_ids = coco_dataset.getImgIds()
    coco_target_ids = all_coco_img_ids[:limit_coco]
    coco_ids_to_process = [img_id for img_id in coco_target_ids if img_id not in processed_coco_ids]

    # 4. Filter ADE20K images
    print("Checking ADE20K cache...")
    ade_filenames = ade_dataset["filename"]
    ade_img_ids = [int(os.path.splitext(f)[0].split('_')[-1]) for f in ade_filenames]
    ade_target_indices = list(range(min(limit_ade, len(ade_dataset))))
    ade_indices_to_process = [idx for idx in ade_target_indices if ade_img_ids[idx] not in processed_ade_ids]

    # 5. Filter Pascal VOC images
    print("Checking Pascal VOC cache...")
    pascal_target_indices = list(range(min(limit_pascal, len(pascal_dataset))))
    pascal_indices_to_process = [idx for idx in pascal_target_indices if idx not in processed_pascal_ids]

    # 6. Filter LVIS images
    print("Checking LVIS cache...")
    lvis_target_indices = list(range(min(limit_lvis, len(lvis_dataset["images"]))))
    lvis_indices_to_process = [idx for idx in lvis_target_indices if lvis_dataset["images"][idx]["id"] not in processed_lvis_ids]

    # 7. Process COCO if needed
    if coco_ids_to_process:
        print(f"Processing COCO ({len(coco_ids_to_process)} new images out of target limit {limit_coco})...")
        img_bnch, cat_bnch = create_benchmark_from_coco(
            coco_dataset, coco_ids_to_process, img_bnch=img_bnch, cat_bnch=cat_bnch
        )
    else:
        print(f"All {limit_coco} target COCO images are already processed.")

    # 8. Process ADE20K if needed
    if ade_indices_to_process:
        print(f"Processing ADE ({len(ade_indices_to_process)} new images out of target limit {limit_ade})...")
        img_bnch, cat_bnch = create_benchmark_from_ade20K(
            ade_dataset, ade_indices_to_process, img_bnch=img_bnch, cat_bnch=cat_bnch
        )
    else:
        print(f"All {limit_ade} target ADE20K images are already processed.")

    # 9. Process Pascal VOC if needed
    if pascal_indices_to_process:
        print(f"Processing Pascal VOC ({len(pascal_indices_to_process)} new images out of target limit {limit_pascal})...")
        img_bnch, cat_bnch = create_benchmark_from_pascalvoc(
            pascal_dataset, pascal_indices_to_process, img_bnch=img_bnch, cat_bnch=cat_bnch
        )
    else:
        print(f"All {limit_pascal} target Pascal VOC images are already processed.")

    # 10. Process LVIS if needed
    if lvis_indices_to_process:
        print(f"Processing LVIS ({len(lvis_indices_to_process)} new images out of target limit {limit_lvis})...")
        img_bnch, cat_bnch = create_benchmark_from_lvis(
            lvis_dataset, lvis_indices_to_process, img_bnch=img_bnch, cat_bnch=cat_bnch
        )
    else:
        print(f"All {limit_lvis} target LVIS images are already processed.")

    # 11. Save updated sets to disk
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(img_bnch, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n  Saved -> {METADATA_PATH} (total images: {len(img_bnch)})")
    
    with open(WS2_PATH, "w") as f:
        json.dump(cat_bnch, f, indent=2)
    print(f"  Saved -> {WS2_PATH} (total categories: {len(cat_bnch)})")
    
    # 12. Rebuild positive/negative sets from combined metadata and save
    print("Rebuilding positive/negative sets...")
    pos_set, neg_set = build_pos_neg_sets_from_bnchmrk(img_bnch)
    with open(POS2_PATH, "w") as f:
        json.dump(pos_set, f, indent=2)
    print(f"  Saved -> {POS2_PATH}")
    with open(NEG2_PATH, "w") as f:
        json.dump(neg_set, f, indent=2)
    print(f"  Saved -> {NEG2_PATH}")

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
    pascal_dataset = load_pascalvoc()
    lvis_dataset = load_lvis()
    len_coco = len(coco_dataset.getImgIds())
    len_ade = len(ade_dataset)
    len_pascal = len(pascal_dataset)
    len_lvis = len(lvis_dataset["images"])
    build_benchmark(
        coco_dataset, ade_dataset, pascal_dataset, lvis_dataset,
        limit_coco=len_coco, limit_ade=len_ade, limit_pascal=len_pascal, limit_lvis=len_lvis
    )
