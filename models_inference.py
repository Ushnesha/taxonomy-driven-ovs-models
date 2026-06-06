import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO

device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# Refactored host function: GT decoding once, prediction delegated.
# ======================================================================
def get_segmentation_masks(image, annotations, text_cats, coco,
                           predict_fn, threshold=0.5):
    from pycocotools import mask as maskUtils

    img_array = np.array(image.convert("RGB")) if hasattr(image, "convert") else image
    h, w = img_array.shape[:2]
    image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image

    gt_masks = {}
    for ann in annotations:
        cat_id = ann["category_id"]
        seg = ann["segmentation"]
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
        gt_masks[cat_id] = np.logical_or(gt_masks.get(cat_id, 0), mask).astype(np.uint8)

    # The only model-specific part:
    pred_masks = predict_fn(image_pil, text_cats, coco, threshold)
    return gt_masks, pred_masks

def make_clipseg_predict(model, processor):
    def predict(image_pil, text_cats, coco, threshold):
        cat_ids = _ordered_cat_ids(text_cats, coco)
        prompts = [f"a photo of a {v}" for v in text_cats.values()]
        W, H = image_pil.size
        inputs = processor(text=prompts, images=[image_pil] * len(prompts),
                           padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits)  # [C, h, w] in [0,1]
        probs = F.interpolate(probs.unsqueeze(1), size=(H, W),
                              mode="bilinear", align_corners=False)[:, 0]
        probs = probs.cpu().numpy()
        return {cid: (probs[i] > threshold).astype(np.uint8)
                for i, cid in enumerate(cat_ids)}
    return predict