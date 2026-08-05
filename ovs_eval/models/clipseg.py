import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from ovs_eval.models.base import BaseOVSModel

class CLIPSegModel(BaseOVSModel):
    def __init__(self, device=None, model_id="CIDAS/clipseg-rd64-refined"):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(self.device).eval()
        
        # Adjust processor size
        self.processor.image_processor.size = {
            "height": self.model.config.vision_config.image_size,
            "width": self.model.config.vision_config.image_size
        }

    def predict(self, image, text_cats, coco, threshold=0.5, desc=False):
        if isinstance(image, np.ndarray):
            from PIL import Image
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            
        img_array = np.array(image_pil.convert("RGB"))
        original_height, original_width = img_array.shape[:2]
        
        pred_masks = {}
        
        prompts = [p_var if desc else f"a photo of a {p_var}" for p_var in text_cats.values()]
        
        # Batch all prompts and duplicate the image accordingly
        inputs = self.processor(
            text=prompts, images=[image_pil] * len(prompts), return_tensors="pt", padding=True
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
            
        probs = torch.sigmoid(logits)
        probs = F.interpolate(probs.unsqueeze(1), size=(original_height, original_width), mode='bilinear', align_corners=False)[:, 0]
        probs_np = probs.cpu().numpy()
        
        for idx, (p_base, p_var) in enumerate(text_cats.items()):
            if hasattr(coco, 'getCatIds'):
                cat_ids = coco.getCatIds(catNms=[p_base])
                cat_id = cat_ids[0] if cat_ids else idx
            else:
                cat_id = coco.get(p_base, idx)
                
            pred_masks[cat_id] = (probs_np[idx] > threshold).astype(np.uint8)
            
        return pred_masks

    def predict_v2(self, image_pil, gt_masks, gt_objects, threshold=0.5, desc = False):
        """
        Predict binary masks for each text prompt.
        
        Args:
            image_pil: PIL.Image or np.ndarray
            gt_masks: dict/list of ground-truth binary masks
            text_prompts: dict (cat_key -> prompt) or list of prompt strings
            threshold: float, confidence threshold
            
        Returns:
            dict: predicted binary masks mapped by category keys (or indices if list was passed)
        """
        if isinstance(image_pil, np.ndarray):
            from PIL import Image
            image_pil = Image.fromarray(image_pil)
            
        img_array = np.array(image_pil.convert("RGB"))
        original_height, original_width = img_array.shape[:2]
        
        pred_masks = {}
        prompts = [p_var if desc else f"a photo of a {p_var}" for p_var in gt_objects]
        
        if not prompts:
            return {}
            
        # Batch all prompts and duplicate the image accordingly
        inputs = self.processor(
            text=prompts, images=[image_pil] * len(prompts), return_tensors="pt", padding=True
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
            
        probs = torch.sigmoid(logits)
        probs = F.interpolate(probs.unsqueeze(1), size=(original_height, original_width), mode='bilinear', align_corners=False)[:, 0]
        probs_np = probs.cpu().numpy()
        
        for idx, key in enumerate(gt_objects):
            pred_masks[key] = (probs_np[idx] > threshold).astype(np.uint8)
            
        return pred_masks
