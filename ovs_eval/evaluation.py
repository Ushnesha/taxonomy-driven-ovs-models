import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from ovs_eval.dataset import get_img, get_annotations_for_image, get_objects_from_annotations, decode_annotations_to_masks
from ovs_eval.linguistics import get_linguistic_cats_v2

def calculate_miou(gt_masks, pred_masks):
    """
    Calculate mean Intersection over Union (mIoU).
    """
    ious = {}
    
    # Use union of keys to ensure we score all predicted/ground truth classes
    all_class_ids = set(gt_masks.keys()).union(pred_masks.keys())
    
    # Find active shape for default zero mask
    default_shape = None
    for mask in gt_masks.values():
        default_shape = mask.shape
        break
    if default_shape is None:
        for mask in pred_masks.values():
            default_shape = mask.shape
            break
            
    if default_shape is None:
        return 0.0, {}
        
    for class_id in all_class_ids:
        gt_mask = gt_masks.get(class_id, np.zeros(default_shape, dtype=np.uint8))
        pred_mask = pred_masks.get(class_id, np.zeros(default_shape, dtype=np.uint8))
        
        gt_binary = (gt_mask > 0).astype(np.uint8)
        pred_binary = (pred_mask > 0).astype(np.uint8)
        
        intersection = np.logical_and(gt_binary, pred_binary).sum()
        union = np.logical_or(gt_binary, pred_binary).sum()
        
        if union == 0:
            iou = 0.0
        else:
            iou = intersection / union
            
        ious[class_id] = iou
        
    miou = np.mean(list(ious.values())) if ious else 0.0
    return miou, ious

def lss_per_class(variant_iou_dict, variants=('Original', 'Synonyms', 'Hypernyms', 'Hyponyms')):
    """
    LSS(M, c) = std-dev of mean-IoU values across variants.
    """
    available = {v: variant_iou_dict[v] for v in variants
                 if v in variant_iou_dict and len(variant_iou_dict[v]) > 0}
    
    if len(available) < 2:
        return None, {}, None
        
    mu_per_variant = {v: np.mean(ious) for v, ious in available.items()}
    mu_values = np.array(list(mu_per_variant.values()))
    mu_M_c = np.mean(mu_values)
    lss_c = np.sqrt(np.mean((mu_values - mu_M_c) ** 2))
    
    return lss_c, mu_per_variant, mu_M_c

def prepare_image_data(img_id, coco):
    """
    CPU and I/O bound preprocessing task.
    """
    image = get_img(img_id, coco)
    annotations = get_annotations_for_image(img_id, coco)
    
    # Get original image shape for default masks
    if hasattr(image, "size"):
        w, h = image.size
    else:
        h, w = image.shape[:2]
        
    gt_masks = decode_annotations_to_masks(annotations, h, w)
    gt_objects, _ = get_objects_from_annotations(annotations, coco)
    cats = list(set(gt_objects))
    
    if not cats:
        return None
        
    orig_cats, syn_cats, hyper_cats, hypo_cats = get_linguistic_cats_v2(cats)
    
    category_types = {
        'Original': orig_cats,
        'Synonyms': syn_cats,
        'Hypernyms': hyper_cats,
        'Hyponyms': hypo_cats
    }
    
    return {
        'img_id': img_id,
        'image': image,
        'gt_masks': gt_masks,
        'category_types': category_types
    }

def run_evaluation(model, coco, image_ids, threshold=0.5, desc=False, output_file="segmentation_results.json"):
    """
    Run evaluation loop over a list of image IDs using a parallelized preprocessing pipeline.
    """
    # Configure file logging with naming convention: evaluation_<model_name>.log
    model_name = model.__class__.__name__.lower()
    if model_name.endswith("model"):
        model_name = model_name[:-5]
    log_filename = f"evaluation_{model_name}.log"
    
    logger = logging.getLogger("ovs_eval.evaluation")
    logger.setLevel(logging.ERROR)
    
    # Avoid duplicate handlers if function called multiple times
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    results = defaultdict(list)
    
    # Step 1: Preprocess images/masks/annotations in parallel
    preprocessed_data = []
    
    # We use a ThreadPoolExecutor since this is primarily network/disk IO-bound
    # (e.g. downloading/reading images) and releases the GIL during file operations.
    max_workers = min(16, len(image_ids))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(prepare_image_data, img_id, coco) for img_id in image_ids]
        
        # Gathering results in submitted order to preserve determinism
        for f, img_id in tqdm(zip(futures, image_ids), total=len(image_ids), desc="Preprocessing images"):
            try:
                data = f.result()
                if data is not None:
                    preprocessed_data.append(data)
            except Exception as e:
                msg = f"Error preprocessing image {img_id}: {e}"
                print(msg)
                logger.error(msg, exc_info=True)
                
    # Step 2: Run sequential model inference on the GPU (avoids CUDA concurrency/memory issues)
    for data in tqdm(preprocessed_data, desc="Running model inference"):
        img_id = data['img_id']
        image = data['image']
        gt_masks = data['gt_masks']
        category_types = data['category_types']
        
        for cat_type, cat_dict in category_types.items():
            try:
                pred_masks = model.predict(image, cat_dict, coco, threshold=threshold, desc=desc)
                miou, ious = calculate_miou(gt_masks, pred_masks)
                
                results[cat_type].append({
                    'img_id': img_id,
                    'miou': miou,
                    'ious': ious,
                    'categories': list(cat_dict.values())
                })
            except Exception as e:
                msg = f"Error running model inference on image {img_id}: {e}"
                print(msg)
                logger.error(msg, exc_info=True)
                continue
            
    # Convert results to JSON-serializable format and save
    results_json = {}
    for cat_type, results_list in results.items():
        results_json[cat_type] = []
        for r in results_list:
            results_json[cat_type].append({
                'img_id': r['img_id'],
                'miou': float(r['miou']),
                'categories': r['categories'],
                'ious': {str(k): float(v) for k, v in r['ious'].items()}
            })
            
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
        
    print(f"Results saved to {output_file}")
    return results

def compute_lss_metrics(results, variants=('Original', 'Synonyms', 'Hypernyms', 'Hyponyms')):
    """
    Compute Linguistic Sensitivity Score metrics from evaluation results.
    """
    class_variant_ious = defaultdict(lambda: defaultdict(list))
    
    for variant in variants:
        if variant not in results:
            continue
        for entry in results[variant]:
            img_id = entry['img_id']
            
            # Find the matching Original entry to get canonical class names
            orig_entry = next(
                (e for e in results['Original'] if e['img_id'] == img_id), None
            )
            if orig_entry is None:
                continue
                
            orig_classes = orig_entry['categories']
            ious = entry['ious']
            
            for cls_name, (cat_id, iou_val) in zip(orig_classes, ious.items()):
                class_variant_ious[cls_name][variant].append(iou_val)
                
    lss_results = {}
    for cls_name, variant_dict in class_variant_ious.items():
        # Only score classes that have data in ALL variants
        if all(len(variant_dict[v]) > 0 for v in variants):
            lss_c, mu_per_v, mu_c = lss_per_class(variant_dict, variants)
            lss_results[cls_name] = {
                'LSS': lss_c,
                'mu_M_c': mu_c,
                'mu_per_variant': mu_per_v,
                'n_images': len(variant_dict['Original']),
            }
            
    all_lss_values = [v['LSS'] for v in lss_results.values() if v['LSS'] is not None]
    LSS_M = np.mean(all_lss_values) if all_lss_values else 0.0
    
    return LSS_M, lss_results
