import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor as AP2, AutoModel
from ovs_eval.models.base import BaseOVSModel

class SiglipModel(BaseOVSModel):
    def __init__(self, device=None, model_id="google/siglip-base-patch16-224"):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        self.processor = AP2.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

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
        
        # Preprocess text and image separately and efficiently
        image_inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
        text_inputs = self.processor(text=prompts, padding="max_length", return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            # Get vision outputs (batch_size=1)
            vision_outputs = self.model.vision_model(**image_inputs)
            vision_features = vision_outputs.last_hidden_state # [1, num_patches, hidden_dim]
            
            if hasattr(self.model, "visual_projection"):
                vision_features = self.model.visual_projection(vision_features)
                
            b, num_patches, hidden_dim = vision_features.shape
            patch_size = int(np.sqrt(num_patches))
            vision_features = vision_features.reshape(patch_size, patch_size, hidden_dim)
            vision_features = F.normalize(vision_features, dim=-1)
            
            # Get text features (C, hidden_dim)
            text_features = self.model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute similarity maps [C, patch_size, patch_size]
            similarity_map = torch.einsum(
                'pqd,cd->cpq',
                vision_features,
                text_features
            )
            
            # Apply SigLIP learnable temperature scale and bias for calibrated alignment
            logit_scale = self.model.logit_scale.exp() if hasattr(self.model, "logit_scale") else 1.0
            logit_bias = self.model.logit_bias if hasattr(self.model, "logit_bias") else 0.0
            similarity_map = similarity_map * logit_scale + logit_bias
            
            # Compute class probabilities
            probs = torch.sigmoid(similarity_map) # [C, patch_size, patch_size]
            probs_interpolated = F.interpolate(
                probs.unsqueeze(1), 
                size=(original_height, original_width), 
                mode='bilinear', 
                align_corners=False
            )[:, 0] # [C, H, W]
            
            probs_np = probs_interpolated.cpu().numpy()
            
        for idx, (p_base, p_var) in enumerate(text_cats.items()):
            if hasattr(coco, 'getCatIds'):
                cat_ids = coco.getCatIds(catNms=[p_base])
                cat_id = cat_ids[0] if cat_ids else idx
            else:
                cat_id = coco.get(p_base, idx)
                
            prob_map = probs_np[idx]
            max_val = prob_map.max()
            min_val = prob_map.min()
            
            if max_val < threshold:
                pred_masks[cat_id] = np.zeros_like(prob_map, dtype=np.uint8)
            else:
                dynamic_thresh = min_val + 0.5 * (max_val - min_val)
                threshold_to_use = max(dynamic_thresh, threshold)
                pred_masks[cat_id] = (prob_map > threshold_to_use).astype(np.uint8)
            
        return pred_masks

