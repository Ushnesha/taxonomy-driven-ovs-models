# Taxonomy-Driven Open-Vocabulary Segmentation Models — Explanation

## Overview

This repository explores **taxonomy-driven open-vocabulary segmentation (OVS)** — the idea that a segmentation model's performance varies depending on where in a semantic taxonomy (hierarchy) the text prompt sits. It uses **CLIPSeg**, a CLIP-based zero-shot segmentation model, to segment objects in COCO 2017 validation images by prompting it with terms at different levels of a taxonomy: exact category names, synonyms, hypernyms (broader categories), and functional descriptions.

The key research question: _How does the taxonomic specificity of a text prompt affect segmentation quality?_

## Repository structure

```
taxonomy-driven-ovs-models/
├── dataset.ipynb            # COCO 2017 data download and exploration
├── testing_clipSeg.ipynb    # Core CLIPSeg inference and taxonomy experiments
└── explanation.md           # This file
```

There are exactly two notebooks, no auxiliary scripts, no config files, and no local data. Everything is downloaded at runtime from the COCO dataset and Hugging Face Hub.

---

## 1. Environment

Both notebooks run in a **conda environment named `ml`** with Python 3.10.19.

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pycocotools` | 2.0.11 | COCO annotation parsing and mask utilities |
| `transformers` | 5.5.0 | Loading CLIPSeg model and processor from Hugging Face |
| `opencv-python` | 4.13.0.92 | Image processing and mask resizing |
| `nltk` | 3.9.4 | WordNet lookups for synonym/ hypernym expansion |
| `torch` | — | Model inference backend |
| `matplotlib` | — | 3-panel visualizations |
| `Pillow` | — | Image loading |
| `numpy` | 2.2.6 | Array operations |

---

## 2. Dataset: COCO 2017 Validation

**Source:** [COCO 2017](http://images.cocodataset.org/)  
**Split:** Validation (`val2017`) — 5,000 images  
**Annotations:** `annotations/instances_val2017.json`  
**Categories:** 80 object classes (IDs 1–90, with 10 gaps)

### Data flow

1. The annotations archive (`annotations_trainval2017.zip`) is downloaded from `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` and extracted locally.
2. `pycocotools.coco.COCO` loads `annotations/instances_val2017.json`.
3. All annotations are enumerated and indexed by `category_id` → `[image_ids]`.
4. Individual images are downloaded on-the-fly from `http://images.cocodataset.org/val2017/<file_name>.jpg` when needed (not pre-downloaded).

### Annotation schema (per the COCO format)

```
annotation {
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": list[list[float]] | RLE,  # polygon or RLE encoding
    "area": float,
    "bbox": [x, y, width, height],
    "iscrowd": 0 | 1
}
```

### All 80 COCO categories

`person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush`

---

## 3. Model: CLIPSeg

**Model ID:** `CIDAS/clipseg-rd64-refined`  
**Architecture:** `CLIPSegForImageSegmentation` (from Hugging Face `transformers`)  
**Type:** Zero-shot open-vocabulary image segmentation, built on top of CLIP  
**Internal resolution:** 352 × 352  

### Model loading

```python
from transformers import AutoProcessor, CLIPSegForImageSegmentation

processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
```

Two unexpected weight keys (`position_ids` for vision and text embeddings) appear during loading — these are harmless architecture artifacts and do not affect inference.

### Inference pipeline

1. **Preprocessing:** The processor tokenizes the text prompt and transforms the input image.
2. **Forward pass:** The model takes `(pixel_values, input_ids, attention_mask)` and outputs logits.
3. **Sigmoid:** Logits → probability map via `torch.sigmoid()`.
4. **Threshold:** Probabilities ≥ 0.5 become the binary segmentation mask.
5. **Resize:** The 352×352 mask is upsampled to the original image dimensions via bilinear interpolation.

---

## 4. Core functions (`testing_clipSeg.ipynb`, cell 5)

### `get_class_name(coco, category_id)`
Resolves a numeric COCO category ID to its human-readable class name string.

### `get_annotations_for_image(image_ids, coco)`
Returns all COCO annotation objects for the given image IDs.

### `get_objects_from_annotations(annotations, coco)`
Extracts ground-truth class names and builds a mapping of `category_id → [image_ids]` indexing which images contain which object categories.

### `get_img(image_id, coco)`
Downloads a single COCO image from its `coco_url` using `load_image()` from the transformers library.

### `visualize_segmentation_with_labels(image, annotations, text_cats, coco, model, processor, threshold=0.5, desc=False)`
The central visualization function. Produces a **3-panel figure**:

| Panel | Content |
|-------|---------|
| **Left** | Original image |
| **Middle** | Ground-truth segmentation masks overlaid with class labels at centroids |
| **Right** | CLIPSeg predicted masks overlaid with text prompt labels at centroids |

For each text prompt in `text_cats`:
- If `desc=True`, the raw prompt is used directly (e.g., `"an object to sit on"`).
- If `desc=False` (default), the prompt is templated as `"a photo of a {text_cat}"`.
- The processor runs text + image through the model.
- Sigmoid + threshold → binary mask.
- Mask is resized to the original image dimensions.
- Overlaid with a color from the `tab20` colormap.
- Per-class probability statistics (shape, mean, max, min) are printed.

### `calculate_miou(gt_masks, pred_masks)`
Computes **mean Intersection-over-Union (mIoU)** and per-class IoU between ground-truth and predicted masks. Implemented but **not yet executed** in the notebook (the call is commented out).

### `get_syn(word)`
Looks up a WordNet synset for a given word and returns the first lemma name. Intended for automated synonym/hypernym expansion from WordNet.

---

## 5. Taxonomy-driven experiments

The core experiments test CLIPSeg's response to prompts at different levels of a semantic hierarchy:

### 5a. Exact category name (baseline)
Prompt: `"a photo of a couch"`  
Result: mean probability 0.32, max 0.84 — strong activation.

### 5b. Synonyms
Prompts: `"a photo of a couch"`, `"a photo of a sofa"`, `"a photo of a divan"`, `"a photo of a daybed"`  

| Prompt | Mean prob | Max prob |
|--------|-----------|----------|
| couch | 0.32 | 0.84 |
| sofa | 0.32 | 0.85 |
| divan | — | <0.50 (no mask) |
| daybed | 0.28 | 0.70 |

**Finding:** Direct synonyms ("couch" / "sofa") produce nearly identical segmentation quality. Less common synonyms ("divan") fail to activate.

### 5c. Hypernyms (broader categories)
Prompt: `"a photo of furniture"`  
Result: mean probability 0.31, max 0.73 — still activates on couch regions but with lower confidence.

### 5d. Functional descriptions
Prompt: `"an object to sit on"` (with `desc=True` to skip the "a photo of a" template)  
Result: mean probability 0.11, max 0.56 — much weaker activation. The model can relate the functional description to the object class, but segmentation quality degrades significantly compared to naming the category directly.

### 5e. Commented-out dog taxonomy experiment
A planned but not-yet-executed experiment listed dog terms at multiple taxonomy levels: `['dog', 'canine', 'golden retriever', 'labrador retriever', 'bulldog', 'poodle', 'hound', 'pup', 'german shepherd', 'husky', 'mammal', 'animal', 'pet']` — from specific breed → species → family → class → kingdom.

---

## 6. How to run

1. **Activate the conda environment:**
   ```bash
   conda activate ml
   ```

2. **Run `dataset.ipynb` first** to download the COCO annotations and verify the data.

3. **Run `testing_clipSeg.ipynb`** cells sequentially:
   - Cell 0–2: Install dependencies and load CLIPSeg.
   - Cell 3–5: Load COCO annotations and define utility functions.
   - Cell 6 onward: Run inference experiments on selected images.

4. **To test a different category**, modify the `text_cats` list and select image IDs from the corresponding COCO category.

---

## 7. Key observations

- CLIPSeg outputs segmentation masks at **352×352** resolution, which is bilinearly upsampled to match the original image dimensions. This limits fine-grained boundary accuracy.
- The **"a photo of a ..."** prompt template works well for standard objects but may hurt performance for abstract or functional descriptions — hence the `desc` flag to bypass it.
- Taxonomy level matters: **naming the object directly gives the best results**, synonyms at the same specificity level perform similarly, and broader/functional terms degrade gracefully.
- mIoU computation is implemented but **not yet run** — quantitative evaluation remains future work.

---

## 8. Limitations and future directions

- **No quantitative evaluation:** The `calculate_miou` function exists but hasn't been executed across a representative sample.
- **Small prompt set:** Only 10 categories in `text_cats` and a handful of synonym experiments.
- **No automated taxonomy expansion:** WordNet (`get_syn`) is available but not yet integrated into the experiment loop.
- **Single model:** Only CLIPSeg is tested. Comparisons with other OVS models (e.g., SAM, OVSeg, ODISE) would strengthen the findings.
- **No dataset-level benchmarking:** Currently tests individual images manually rather than running across the full COCO validation set.
