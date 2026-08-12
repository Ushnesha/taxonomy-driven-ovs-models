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

def get_img(image_id, coco=None, img_url=None, filename=None):
    """
    Get image by image ID, filename, or URL. First checks local image directories.
    Falls back to provided img_url, COCO metadata, or standard COCO URL format.
    """
    if isinstance(image_id, int):
        fn = f"{image_id:012d}.jpg"
    elif isinstance(image_id, str) and image_id.isdigit():
        fn = f"{int(image_id):012d}.jpg"
    else:
        fn = filename or (str(image_id) if isinstance(image_id, str) else None)

    local_dirs = [
        "val2017",
        "train2017",
        "images",
        os.path.join("datasets", "coco", "val2017"),
        os.path.join("datasets", "coco", "train2017"),
    ]

    if fn:
        for d in local_dirs:
            local_path = os.path.join(d, fn)
            if os.path.exists(local_path):
                try:
                    return Image.open(local_path).convert("RGB")
                except Exception:
                    pass

    if img_url:
        try:
            return load_image(img_url)
        except Exception:
            pass

    if coco is not None and hasattr(coco, "loadImgs"):
        try:
            imgs = coco.loadImgs(image_id)
            if imgs and len(imgs) > 0 and "coco_url" in imgs[0]:
                url = imgs[0]["coco_url"]
                if url:
                    return load_image(url)
        except (KeyError, IndexError, Exception):
            pass

    if isinstance(image_id, (int, str)) and str(image_id).isdigit():
        val_url = f"http://images.cocodataset.org/val2017/{int(image_id):012d}.jpg"
        try:
            return load_image(val_url)
        except Exception:
            train_url = f"http://images.cocodataset.org/train2017/{int(image_id):012d}.jpg"
            return load_image(train_url)

    raise ValueError(f"Could not load image for image_id: {image_id}")

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
