import os
import numpy as np
from PIL import Image
from collections import defaultdict
from transformers.image_utils import load_image
from pycocotools import mask as maskUtils

def get_class_name(coco, category_id):
    cat_info = coco.loadCats(category_id)[0]
    return cat_info['name']

def get_annotations_for_image(image_id, coco):
    ann_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(ann_ids)
    return annotations

def get_objects_from_annotations(annotations, coco):
    ground_truth_objects = []
    img_ids_per_cat = defaultdict(list)
    for ann in annotations:
        category_id = ann['category_id']
        class_name = get_class_name(coco, category_id)
        ground_truth_objects.append(class_name)
        img_ids_per_cat[category_id].append(ann['image_id'])
    return ground_truth_objects, img_ids_per_cat

def get_img(image_id, coco):
    """
    Get image by image ID. First checks local 'val2017' directory.
    Falls back to COCO URL if local file is not found.
    """
    # COCO image filenames are 12 digits zero-padded
    local_path = os.path.join("val2017", f"{image_id:012d}.jpg")
    if os.path.exists(local_path):
        try:
            return Image.open(local_path).convert("RGB")
        except Exception:
            pass
            
    img_meta = coco.loadImgs(image_id)[0]
    url = img_meta['coco_url']
    # load_image handles both local paths and URLs
    return load_image(url)

def decode_annotations_to_masks(annotations, h, w):
    """
    Decode COCO segmentations into binary masks.
    Returns:
        dict: {category_id: np.ndarray [H, W] binary mask}
    """
    gt_masks = {}
    for ann in annotations:
        cat_id = ann["category_id"]
        seg = ann["segmentation"]
        
        # Decode the segment using pycocotools.mask
        if isinstance(seg, list):
            if len(seg) == 0:
                mask = np.zeros((h, w), np.uint8)
            elif isinstance(seg[0], dict):
                mask = maskUtils.decode(seg)
            else:
                from pycocotools.mask import frPyObjects
                mask = maskUtils.decode(frPyObjects(seg, h, w))
        elif isinstance(seg, dict):
            if isinstance(seg["counts"], list):
                from pycocotools.mask import frPyObjects
                mask = maskUtils.decode(frPyObjects([seg], h, w))
            else:
                mask = maskUtils.decode(seg)
        else:
            mask = np.zeros((h, w), np.uint8)
            
        if mask.ndim == 3:
            mask = np.any(mask, axis=2).astype(np.uint8)
            
        if cat_id not in gt_masks:
            gt_masks[cat_id] = mask
        else:
            gt_masks[cat_id] = np.logical_or(gt_masks[cat_id], mask).astype(np.uint8)
            
    return gt_masks
