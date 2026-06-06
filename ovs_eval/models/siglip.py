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
        
        inputs = self.processor(
            text=prompts, images=[image_pil] * len(prompts), return_tensors="pt", padding=True
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
            
        text_features = outputs.text_model_output.last_hidden_state[:, 0, :]
        text_features = F.normalize(text_features, dim=-1)

        vision_features = outputs.vision_model_output.last_hidden_state
        batch_size, num_patches, hidden_dim = vision_features.shape
        patch_size = int(np.sqrt(num_patches))
        vision_features = vision_features.reshape(batch_size, patch_size, patch_size, hidden_dim)
        vision_features = F.normalize(vision_features, dim=-1)

        # Compute similarity between each patch and text in parallel
        with torch.inference_mode():
            similarity_map = torch.einsum(
                'bpqd,bd->bpq',
                vision_features,
                text_features
            ) # [C, patch_size, patch_size]
            
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
                
            pred_masks[cat_id] = (probs_np[idx] > threshold).astype(np.uint8)
            
        return pred_masks
