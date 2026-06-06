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
        
        for idx, (p_base, p_var) in enumerate(text_cats.items()):
            text_prompt = p_var if desc else f"a photo of a {p_var}"
            
            inputs = self.processor(
                text=text_prompt, images=image_pil, return_tensors="pt", padding=True
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model(**inputs)
                
            if hasattr(coco, 'getCatIds'):
                cat_ids = coco.getCatIds(catNms=[p_base])
                cat_id = cat_ids[0] if cat_ids else idx
            else:
                cat_id = coco.get(p_base, idx)
                
            text_features = outputs.text_model_output.last_hidden_state[:, 0, :]
            text_features = F.normalize(text_features, dim=-1)

            vision_features = outputs.vision_model_output.last_hidden_state
            batch_size, num_patches, hidden_dim = vision_features.shape
            patch_size = int(np.sqrt(num_patches))
            vision_features = vision_features.reshape(batch_size, patch_size, patch_size, hidden_dim)
            vision_features = F.normalize(vision_features, dim=-1)

            # Compute similarity between each patch and text
            with torch.inference_mode():
                similarity_map = torch.einsum(
                    'bpqd,bd->bpq',
                    vision_features,
                    text_features
                )
                
                probs = torch.sigmoid(similarity_map)[0]
                mask = (probs > threshold).float().unsqueeze(0).unsqueeze(0)
                mask = F.interpolate(mask, size=(original_height, original_width), mode='bilinear', align_corners=False)
                mask = mask.squeeze(0).squeeze(0)
                
            pred_masks[cat_id] = mask.cpu().numpy().astype(np.uint8)
            
        return pred_masks
