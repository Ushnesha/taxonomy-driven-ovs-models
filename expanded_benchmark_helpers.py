"""
Shared helpers for the Expanded COCO Benchmark.
WordNet-first, BabelNet-fallback approach.
All internal word lookups use underscores (spaces converted).
"""

from IPython.core import async_helpers
import sys, os, json, threading
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from collections import defaultdict
from datasets import load_dataset
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer, util
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='mps') 

# ── Paths ──
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
COCO_ANN  = os.path.join(REPO_ROOT, "datasets", "coco", "instances_val2017.json")
ADE20K_PATH = os.path.join(REPO_ROOT, "datasets", "ade20k")
DATA_DIR  = os.path.join(REPO_ROOT, "data")
COCO_URL  = "http://images.cocodataset.org/val2017/{:012d}.jpg"
os.makedirs(DATA_DIR, exist_ok=True)

# ── BabelNet config ──
BABELNET_URL = "https://babelnet.io/v9/"
BABELNET_API_KEY = os.environ.get("BABELNET_API_KEY", "")

# ═══════════════════════════════════════════════
# Underscore / space conversion
# ═══════════════════════════════════════════════

def to_wn_form(word: str) -> str:
    """Convert a word/phrase to WordNet-compatible form: spaces → underscores."""
    return word.strip().replace(" ", "_")

def to_display_form(word: str) -> str:
    """Convert a WordNet-compatible word to display form: underscores → spaces."""
    return word.replace("_", " ")

# ═══════════════════════════════════════════════
# COCO-80 categories (canonical list)
# ═══════════════════════════════════════════════

COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# ── Globals (lazy loaded) ──
_processor = None
_model = None
_device = None
_embedding_cache = {}
_nltk_lock = threading.Lock()

# ═══════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════

def get_device():
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = "cuda"
        elif torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
    return _device

def load_model():
    """Load CLIPSeg once and cache globally."""
    global _processor, _model
    if _processor is None:
        from transformers import AutoProcessor, CLIPSegForImageSegmentation
        print("Loading CLIPSeg (CIDAS/clipseg-rd64-refined)...", flush=True)
        _processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        _model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        _model.to(get_device())
        _model.eval()
    return _processor, _model

# ═══════════════════════════════════════════════
# CLIP text embeddings (raw norm — NO L2)
# ═══════════════════════════════════════════════

def get_text_embedding(word: str):
    """Bare-word CLIP embedding. Raw norm preserved (~8-10)."""
    processor, model = load_model()
    inputs = processor(text=word, return_tensors="pt", padding=True, truncation=True).to(get_device())
    with torch.inference_mode():
        emb = model.clip.get_text_features(**inputs).pooler_output  # [1, 512]
    return emb.cpu()

def cosine_sim(a, b):
    """Cosine similarity between two 1-D tensors."""
    a, b = a.squeeze(), b.squeeze()
    return (a @ b).item() / (a.norm().item() * b.norm().item())

# ═══════════════════════════════════════════════
# Embedding caching
# ═══════════════════════════════════════════════

def _cache_path():
    return os.path.join(DATA_DIR, "embedding_cache.pt")

def load_embedding_cache():
    global _embedding_cache
    path = _cache_path()
    if os.path.exists(path):
        _embedding_cache = torch.load(path, map_location="cpu", weights_only=False)
        return True
    return False

def save_embedding_cache():
    path = _cache_path()
    torch.save(dict(_embedding_cache), path)

def get_text_embedding_cached(word: str):
    """CLIP embedding with disk cache."""
    global _embedding_cache
    if word in _embedding_cache:
        return _embedding_cache[word].clone()
    emb = get_text_embedding(word)
    _embedding_cache[word] = emb.cpu()
    return emb

# ═══════════════════════════════════════════════
# CLIPSeg inference
# ═══════════════════════════════════════════════

def run_segmentation(image: Image.Image, cond_embedding: torch.Tensor, threshold=0.5):
    """Run CLIPSeg with a pre-computed conditional embedding. Returns binary mask."""
    processor, model = load_model()
    w, h = image.size
    inputs = processor(images=image, return_tensors="pt").to(get_device())
    cond = cond_embedding.to(get_device())
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    with torch.inference_mode():
        out = model(pixel_values=inputs["pixel_values"], conditional_embeddings=cond)
    logits = out.logits
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    probs = torch.sigmoid(logits)
    probs = torch.nn.functional.interpolate(probs.unsqueeze(0), size=(h, w),
                                            mode="bilinear", align_corners=False).squeeze()
    return (probs.cpu().numpy() > threshold).astype(np.uint8)

def compute_iou(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(inter / union) if union > 0 else 0.0

# ═══════════════════════════════════════════════
# COCO utilities
# ═══════════════════════════════════════════════

def load_coco(ann_path=COCO_ANN):
    import zipfile
    from pycocotools.coco import COCO
    
    path = ann_path
    
    if not os.path.exists(path):
        print(f"COCO annotations not found at {path}. Downloading...")
        parent_dir = os.path.dirname(path)
        os.makedirs(parent_dir, exist_ok=True)
        
        # Download the annotations zip file
        zip_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        zip_path = os.path.join(parent_dir, "annotations_trainval2017.zip")
        
        try:
            r = requests.get(zip_url, stream=True)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete. Extracting instances_val2017.json...")
            
            # Extract only the instances_val2017.json file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                val_ann_member = "annotations/instances_val2017.json"
                if val_ann_member in zip_ref.namelist():
                    zip_ref.extract(val_ann_member, path=parent_dir)
                    extracted_path = os.path.join(parent_dir, val_ann_member)
                    os.rename(extracted_path, path)
                    
                    # Remove the empty annotations folder created by extraction
                    extracted_dir = os.path.join(parent_dir, "annotations")
                    if os.path.exists(extracted_dir):
                        os.rmdir(extracted_dir)
                else:
                    zip_ref.extractall(path=parent_dir)
            
            # Clean up the zip file
            os.remove(zip_path)
            print("Extraction complete and temporary files cleaned up.")
            
        except Exception as e:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise RuntimeError(f"Failed to download/extract COCO annotations: {e}")
            
    return COCO(path)

def get_anns_for_img_id(image_id, coco):
    ann_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(ann_ids)
    return annotations


def get_cat_name_to_id(coco):
    return {cat["name"]: cat["id"] for cat in coco.loadCats(coco.getCatIds())}

def download_image(coco, img_id):
    try:
        r = requests.get(COCO_URL.format(img_id), timeout=30)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

def get_gt_mask(coco, img_id, cat_id):
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)
    if not anns:
        return None
    img_info = coco.loadImgs([img_id])[0]
    h, w = img_info["height"], img_info["width"]
    gt = np.zeros((h, w), dtype=np.uint8)
    for a in anns:
        gt = np.maximum(gt, coco.annToMask(a))
    return gt

def get_images_for_category(coco, cat_id, n=5, seed=42):
    ann_ids = coco.getAnnIds(catIds=[cat_id])
    img_ids = list(set(coco.loadAnns(ann_ids)[i]["image_id"] for i in range(len(ann_ids))))
    rng = np.random.RandomState(seed)
    rng.shuffle(img_ids)
    return img_ids[:n]

# ═══════════════════════════════════════════════
# ADE20K utilities
# ═══════════════════════════════════════════════

def load_ade20k(local_path=ADE20K_PATH, split="validation"):
    """Loads the ADE20K dataset from local cache if available, otherwise downloads it."""
    import os
    from datasets import load_dataset
    
    # Save the original environment state
    orig_offline = os.environ.get("HF_DATASETS_OFFLINE")
    
    try:
        # Enforce offline mode to avoid remote hub latency
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        dataset = load_dataset("uva-cv-lab/ade20k-150", cache_dir=local_path)[split]
        print(f"Loaded ADE20K {split} dataset from local cache.")
        return dataset
    except Exception:
        print("ADE20K dataset not found in local cache. Downloading from Hugging Face...")
        # Disable offline mode to allow downloading
        os.environ["HF_DATASETS_OFFLINE"] = "0"
        try:
            dataset = load_dataset("uva-cv-lab/ade20k-150", cache_dir=local_path)[split]
            print("Download and load complete.")
            return dataset
        finally:
            # Restore the original environment state
            if orig_offline is not None:
                os.environ["HF_DATASETS_OFFLINE"] = orig_offline
            else:
                os.environ.pop("HF_DATASETS_OFFLINE", None)
    finally:
        # Restore the original environment state if loading succeeded
        if orig_offline is not None:
            os.environ["HF_DATASETS_OFFLINE"] = orig_offline
        else:
            os.environ.pop("HF_DATASETS_OFFLINE", None)

def get_gt_mask_for_ade(data, cat_name):
    """
    Given an ADE20K data item and a category name, finds all instances 
    matching that category and merges their masks using np.maximum.
    """
    import numpy as np
    
    # Get image dimensions (width, height)
    w, h = data["image"].size
    gt = np.zeros((h, w), dtype=np.uint8)
    
    found = False
    for idx, obj in enumerate(data["objects"]):
        if obj["raw_name"] == cat_name:
            # Get the binary mask for this instance
            instance_mask = (np.array(data["instances"][idx]) > 0).astype(np.uint8)
            # Perform logical OR (merge)
            gt = np.maximum(gt, instance_mask)
            found = True
            
    return gt if found else None

# ═══════════════════════════════════════════════
# Data Cleaning utilities
# ═══════════════════════════════════════════════
def clean_taxonomy_list(raw_list):
    _ensure_wordnet()
    from nltk.corpus import wordnet as wn
    """
    Cleans a list of taxnomy words by:
      1. Splitting comma-separated strings into individual elements.
      2. Normalizing spaces to underscores and converting to lowercase.
      3. Filtering out misspelled words/typos using WordNet.
      4. Deduplicating while preserving order.
    """
    cleaned = []
    for item in raw_list:
        # Split by comma (handles "altar, communion table, Lord's table" -> ['altar', 'communion table', "Lord's table"])
        parts = [p.strip().lower() for p in item.split(",")]
        
        for part in parts:
            # Replace spaces with underscores
            part_normalized = part.replace(" ", "_")
            if not part_normalized:
                continue
                
            # Check if it's already in the cleaned list
            if part_normalized not in cleaned:
                # Use WordNet to filter out typos/misspellings
                # (e.g. "fa\u00e7ade" or "botle" will return [] and be skipped)
                if wn.synsets(part_normalized) or wn.synsets(part.replace("_", " ")):
                    cleaned.append(part_normalized)
                    
    return cleaned


# ═══════════════════════════════════════════════
# WordNet utilities
# ═══════════════════════════════════════════════

def _ensure_wordnet():
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

def get_all_synsets(word_wn: str):
    """
    Return ALL WordNet synsets for a word (underscore form).
    Noun priority, then any POS, then underscore form of original.
    Returns list of nltk Synset objects.
    """
    from nltk.corpus import wordnet as wn
    _ensure_wordnet()

    with _nltk_lock:
        synsets = wn.synsets(word_wn, pos=wn.NOUN)
        if not synsets:
            synsets = wn.synsets(word_wn)
        if not synsets:
            # Try with spaces instead of underscores
            synsets = wn.synsets(word_wn.replace("_", " "))
    return synsets

def get_synset_groups_for_display(word: str):
    """
    Given a user-submitted word (may contain spaces), return all synset groups
    formatted for user display. Used by the inference pipeline.
    Returns list of dicts: [{synset_name, definition, pos, lemma_names, lemma_names_display}]
    """
    word_wn = to_wn_form(word)
    synsets = get_all_synsets(word_wn)

    results = []
    for s in synsets:
        lemmas = s.lemma_names()
        results.append({
            "synset_name": s.name(),
            "definition": s.definition(),
            "pos": s.pos(),
            "lemma_names": lemmas,                          # underscore form
            "lemma_names_display": [to_display_form(l) for l in lemmas],  # space form
        })
    return results

def find_best_synset(word: str):
    """
    For dataset construction: find the synset whose lemmas contain the
    original COCO word (or its underscored form). This is automatic
    disambiguation — pick the synset that matches the COCO meaning.

    Returns an nltk Synset object, or None if no synset found.
    """
    from nltk.corpus import wordnet as wn
    _ensure_wordnet()

    word_wn = to_wn_form(word)
    word_display = word.strip().lower()

    synsets = get_all_synsets(word_wn)

    if not synsets:
        return None

    # Step 1: Find synsets whose lemma_names contain the COCO word
    matching = []
    for s in synsets:
        lemmas_lower = [l.lower() for l in s.lemma_names()]
        if word_wn.lower() in lemmas_lower or word_display in lemmas_lower:
            matching.append(s)

    if matching:
        # Pick the one where the word appears earliest in the lemma list
        # (most prominent meaning)
        def rank(s):
            lemmas_lower = [l.lower() for l in s.lemma_names()]
            try:
                return lemmas_lower.index(word_wn.lower())
            except ValueError:
                return lemmas_lower.index(word_display)
        matching.sort(key=rank)
        return matching[0]

    # Step 2: Fallback — first noun synset, or first synset overall
    noun_synsets = [s for s in synsets if s.pos() == 'n']
    if noun_synsets:
        return noun_synsets[0]
    return synsets[0]

def find_best_synset_v2(word: str, supporting_words):
    from nltk.corpus import wordnet as wn
    _ensure_wordnet()

    # Try exact match (underscore form and display form)
    word_wn = word.replace(" ", "_")
    word_display = word.strip().lower()
    
    synsets = wn.synsets(word_wn, pos=wn.NOUN)
    if not synsets:
        synsets = wn.synsets(word_display, pos=wn.NOUN)
        
    # Try lemmatization with morphy for plural forms (e.g. "boxes" -> "box")
    lemma = None
    if not synsets:
        lemma = wn.morphy(word_wn, pos=wn.NOUN)
        if lemma:
            synsets = wn.synsets(lemma, pos=wn.NOUN)
        if not synsets:
            lemma_display = wn.morphy(word_display, pos=wn.NOUN)
            if lemma_display:
                synsets = wn.synsets(lemma_display, pos=wn.NOUN)
                lemma = lemma_display
                
    # Fallback to general synsets (any POS)
    if not synsets:
        synsets = wn.synsets(word_wn)
        if not synsets and lemma:
            synsets = wn.synsets(lemma)
            
    if not synsets:
        return None, ""
    if len(synsets) == 1:
        return synsets[0], synsets[0].definition()



    # 1. Create a combined context string
    if isinstance(supporting_words, (list, tuple, set)):
        context_text = ", ".join(supporting_words)
    else:
        context_text = str(supporting_words)

    context_emb = embedding_model.encode(context_text, convert_to_tensor=True)
    
    best_synset = None
    best_definition = ""
    max_score = -1.0
    
    # 2. Score each candidate
    for s in synsets:
        definition = ""
        # Build candidate features: definition + examples + immediate hypernyms
        if s.definition() and s.definition() != "":
            definition = s.definition()
        hypernym_names = [h.name().split('.')[0].replace('_', ' ') for h in s.hypernyms()]
        candidate_text = f"{definition} {' '.join(s.examples())} {' '.join(hypernym_names)}"
        
        candidate_emb = embedding_model.encode(candidate_text, convert_to_tensor=True)
        score = util.cos_sim(context_emb, candidate_emb).item()
        
        if score > max_score:
            max_score = score
            best_synset = s
            best_definition = definition

    return best_synset, best_definition

def build_word_sets_from_synset(synset):
    """
    Given an nltk Synset, return {W_S, W_S_Hp, W_S_Hp_He}.
    All lemmas in underscore form.
    """
    w_s = list(synset.lemma_names())

    # + depth-1 hyponyms
    w_s_hp = list(w_s)
    for hypo in synset.hyponyms():
        for lemma in hypo.lemma_names():
            if lemma not in w_s_hp:
                w_s_hp.append(lemma)

    # + depth-1 hypernyms
    w_s_hp_he = list(w_s_hp)
    for hyper in synset.hypernyms():
        for lemma in hyper.lemma_names():
            if lemma not in w_s_hp_he:
                w_s_hp_he.append(lemma)

    return {
        "W_S": w_s,
        "W_S_Hp": w_s_hp,
        "W_S_Hp_He": w_s_hp_he,
    }

def build_word_sets_from_synset_v2(word: str, supporting_words, limit=5):
    w_s_hp = []
    w_s_he = []
    w_s = []
    synset, definition = find_best_synset_v2(word, supporting_words)
    if synset:
        w_s = list(synset.lemma_names())[:5]
        i = 0
        for hypo in synset.hyponyms():
            for lemma in hypo.lemma_names():
                if lemma not in w_s_hp:
                    w_s_hp.append(lemma)
                if i >= limit: break
        i = 0
        for hyper in synset.hypernyms():
            for lemma in hyper.lemma_names():
                if lemma not in w_s_he:
                    w_s_he.append(lemma)
                if i >= limit: break

    return definition, w_s, w_s_hp, w_s_he

def build_synset_group(synset):
    w_s_hp = []
    w_s_he = []
    w_s = []
    if synset:
        w_s = list(synset.lemma_names())
        print(w_s)
        for hypo in synset.hyponyms():
            for lemma in hypo.lemma_names():
                if lemma not in w_s_hp:
                    w_s_hp.append(lemma)
        for hyper in synset.hypernyms():
            for lemma in hyper.lemma_names():
                if lemma not in w_s_he:
                    w_s_he.append(lemma)

    return synset.definition(), w_s, w_s_hp, w_s_he

# ═══════════════════════════════════════════════
# BabelNet fallback
# ═══════════════════════════════════════════════

BN_CACHE_PATH = os.path.join(DATA_DIR, "babelnet_cache.json")
_babelnet_cache = {}

def _load_babelnet_cache():
    global _babelnet_cache
    if os.path.exists(BN_CACHE_PATH):
        try:
            with open(BN_CACHE_PATH, "r") as f:
                _babelnet_cache = json.load(f)
            print(f"Loaded {len(_babelnet_cache)} cached BabelNet queries from {BN_CACHE_PATH}")
        except Exception as e:
            print(f"Warning: Failed to load BabelNet cache: {e}")
            _babelnet_cache = {}
    else:
        _babelnet_cache = {}

_load_babelnet_cache()

def _babelnet_get(path: str, params: dict, timeout=10):
    """Cached GET request to BabelNet API."""
    import hashlib
    cache_key = hashlib.md5((path + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    if cache_key in _babelnet_cache:
        return _babelnet_cache[cache_key]
    try:
        params["key"] = BABELNET_API_KEY
        r = requests.get(f"{BABELNET_URL}{path}", params=params, timeout=timeout)
        if r.status_code != 200:
            result = None
        else:
            result = r.json()
    except Exception:
        result = None
        
    if result is not None:
        _babelnet_cache[cache_key] = result
        try:
            with open(BN_CACHE_PATH, "w") as f:
                json.dump(_babelnet_cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save BabelNet cache: {e}")
    return result

def _babelnet_get_synset_ids(word: str, pos="NOUN"):
    """Get BabelNet synset IDs for a word. Prefer WordNet-sourced IDs."""
    params = {"lemma": word, "searchLang": "EN"}
    if pos:
        params["pos"] = pos
    data = _babelnet_get("getSynsetIds", params)
    if not data:
        return []
    # Sort: WordNet-sourced first
    wn_ids = [d["id"] for d in data if d.get("source") == "WN"]
    other_ids = [d["id"] for d in data if d.get("source") != "WN"]
    return wn_ids + other_ids


def _babelnet_get_synset_data(synset_id: str):
    """Get full synset data including lemmas and glosses."""
    return _babelnet_get("getSynset", {"id": synset_id, "targetLang": "EN"}) or {}

def _babelnet_get_outgoing_edges(synset_id: str):
    """Get outgoing semantic edges (hyponyms, hypernyms)."""
    return _babelnet_get("getOutgoingEdges", {"id": synset_id}) or []

def _babelnet_extract_english_lemmas(synset_data: dict):
    """Extract all English lemmas from a BabelNet synset data dict."""
    lemmas = []
    for sense in synset_data.get("senses", []):
        props = sense.get("properties", {})
        lemma = props.get("simpleLemma", "")
        lang = props.get("language", "")
        if lang == "EN" and lemma and lemma.lower().replace(" ", "_") not in lemmas:
            # Convert to underscore form for consistency
            lemmas.append(lemma.lower().replace(" ", "_"))
    return lemmas

def _babelnet_get_related_lemmas(synset_id: str, edge_types: tuple):
    """
    Get lemmas from related synsets via BabelNet edges.
    edge_types: tuple of edge shortNames to follow (e.g., ("+~", "has-kind") for hyponyms).
    """
    edges = _babelnet_get_outgoing_edges(synset_id)
    related_lemmas = []
    for edge in edges:
        pointer = edge.get("pointer", {}).get("shortName", "")
        if pointer in edge_types:
            target_id = edge.get("target")
            if target_id:
                target_data = _babelnet_get_synset_data(target_id)
                related_lemmas.extend(_babelnet_extract_english_lemmas(target_data))
    return related_lemmas

def get_best_synset_id_frm_bn(word: str, supporting_words) -> str:
    """
    Finds the best BabelNet Synset ID matching the supporting_words context,
    robustly handling word formatting (spaces vs. underscores vs. hyphens).
    """
    # 1. Normalize the input word
    word_clean = word.replace("-", " ").replace("_", " ").strip()
    word_spaced = word_clean
    word_underscored = word_clean.replace(" ", "_")
    # 2. Query BabelNet using both formats
    synset_ids = _babelnet_get_synset_ids(word_spaced)
    if not synset_ids:
        synset_ids = _babelnet_get_synset_ids(word_underscored)
    if not synset_ids:
        # Final fallback to raw input
        synset_ids = _babelnet_get_synset_ids(word)
    if not synset_ids:
        return None, None, ""
    if len(synset_ids) == 1:
        sid = synset_ids[0]
        data = _babelnet_get_synset_data(sid)
        definition = ""
        for gloss in data.get("glosses", []):
            if gloss.get("language") == "EN":
                definition = gloss.get("gloss", "")
                break
        return sid, data, definition

    # 3. Format the context text dynamically
    if isinstance(supporting_words, (list, tuple, set)):
        context_text = ", ".join(supporting_words)
    else:
        context_text = str(supporting_words)
    context_emb = embedding_model.encode(context_text, convert_to_tensor=True)
    best_synset_id = None
    best_synset_data = None
    best_synset_def = ""
    max_score = -1.0
    # 4. Score candidates
    for sid in synset_ids[:5]:  # Check first 5 candidates
        data = _babelnet_get_synset_data(sid)
        
        # Get definition
        definition = ""
        for gloss in data.get("glosses", []):
            if gloss.get("language") == "EN":
                definition = gloss.get("gloss", "")
                break
                
        # Calculate similarity score
        candidate_emb = embedding_model.encode(definition, convert_to_tensor=True)
        score = util.cos_sim(context_emb, candidate_emb).item()
        if score > max_score and definition != "":
            max_score = score
            best_synset_id = sid
            best_synset_data = data
            best_synset_def = definition
    return best_synset_id, best_synset_data, best_synset_def

def supplement_word_sets_with_babelnet(word: str, existing_word_sets: dict):
    """
    If WordNet gives insufficient synonyms (< 2 in W_S), try BabelNet
    to supplement the word sets. Returns enriched word_sets dict.

    Strategy:
    1. Look up the word in BabelNet, get all synset IDs
    2. Find the synset whose lemmas contain the original word (disambiguation)
    3. Extract all English lemmas from that synset
    4. Get hyponyms and hypernyms via edges
    5. Merge with existing WordNet results (deduplicate)
    """
    word_wn = to_wn_form(word)
    word_display = word_wn.replace("_", " ")

    # Try underscored form first, then space form
    synset_ids = _babelnet_get_synset_ids(word_wn)
    if not synset_ids:
        synset_ids = _babelnet_get_synset_ids(word_display)
    if not synset_ids:
        return existing_word_sets  # BabelNet couldn't find it

    # Find the synset whose lemmas contain our word
    best_synset_id = None
    best_lemmas = []
    for sid in synset_ids[:5]:  # check first 5
        data = _babelnet_get_synset_data(sid)
        lemmas = _babelnet_extract_english_lemmas(data)
        lemmas_lower = [l.lower() for l in lemmas]
        if word_wn.lower() in lemmas_lower or word_display.lower() in lemmas_lower:
            best_synset_id = sid
            best_lemmas = lemmas
            break

    if best_synset_id is None:
        return existing_word_sets  # no matching synset

    # Get hyponyms and hypernyms from BabelNet edges
    bn_hyponyms = _babelnet_get_related_lemmas(best_synset_id, ("+~", "has-kind"))
    bn_hypernyms = _babelnet_get_related_lemmas(best_synset_id, ("+@", "is-a"))

    # Merge with existing WordNet results (BabelNet supplements, doesn't replace)
    w_s = list(existing_word_sets["W_S"])
    for lemma in best_lemmas:
        if lemma not in w_s:
            w_s.append(lemma)

    w_s_hp = list(existing_word_sets["W_S_Hp"])
    # BabelNet synonyms must also go into W_S_Hp
    for lemma in best_lemmas:
        if lemma not in w_s_hp:
            w_s_hp.append(lemma)
    for lemma in bn_hyponyms:
        if lemma not in w_s_hp:
            w_s_hp.append(lemma)

    # W_S_Hp_He = enriched W_S_Hp + BabelNet hypernyms (superset of w_s_hp)
    w_s_hp_he = list(w_s_hp)
    for lemma in bn_hypernyms:
        if lemma not in w_s_hp_he:
            w_s_hp_he.append(lemma)

    return {"W_S": w_s, "W_S_Hp": w_s_hp, "W_S_Hp_He": w_s_hp_he}




def supplement_word_sets_with_babelnet_v2(word: str, supporting_words, limit=5):
    """
    If WordNet gives insufficient synonyms (< 2 in W_S), try BabelNet
    to supplement the word sets. Returns enriched word_sets dict using
    supporting_words context to disambiguate.
    """
    definition = ""
    w_s, w_s_hp, w_s_he = [], [], []
    # 1. Find the best synset ID using context
    best_synset_id, data, definition = get_best_synset_id_frm_bn(word, supporting_words)
    if not best_synset_id:
        return definition, w_s, w_s_hp, w_s_he
    
    # 2. Extract Synonyms (lemmas from that synset)
    w_s = _babelnet_extract_english_lemmas(data)
    word_normalized = word.lower().replace(" ", "_")
    if word_normalized not in w_s:
        w_s.insert(0, word_normalized)

    # 3. Extract Hyponyms and Hypernyms via outgoing edges
    bn_hyponyms = _babelnet_get_related_lemmas(best_synset_id, ("+~", "has-kind"))
    bn_hypernyms = _babelnet_get_related_lemmas(best_synset_id, ("+@", "is-a"))

    # Format output lists
    for lemma in bn_hyponyms[:limit]:
        if lemma not in w_s_hp:
            w_s_hp.append(lemma)

    for lemma in bn_hypernyms[:limit]:
        if lemma not in w_s_he:
            w_s_he.append(lemma)

    return definition, w_s[:limit], w_s_hp, w_s_he

# ═══════════════════════════════════════════════
# Blending
# ═══════════════════════════════════════════════

def compute_centroid(embeddings: list):
    """Unweighted mean of a list of (word, tensor) tuples."""
    return torch.stack([e for _, e in embeddings]).mean(dim=0)

def blend_embedding(query_emb, centroid, alpha):
    """(1-α) * query + α * centroid. Preserves raw norm."""
    q = query_emb.squeeze()
    return (1 - alpha) * q + alpha * centroid.squeeze()

def top_k_neighbors(query_word: str, candidate_words: list, k=5):
    """
    Given a query word and a list of candidate words, return the top-K
    by CLIP cosine similarity (excluding the query itself).
    Returns list of (word, embedding, similarity_score).
    """
    query_emb = get_text_embedding(query_word)
    scored = []
    for w in candidate_words:
        if w.lower() == query_word.lower():
            continue
        emb = get_text_embedding(w)
        scored.append((w, emb, cosine_sim(query_emb, emb)))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:k]
