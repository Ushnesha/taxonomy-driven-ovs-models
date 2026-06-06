import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from ovs_eval.models.base import BaseOVSModel

class CLIPViTLargeModel(BaseOVSModel):
    def __init__(self, device=None, model_id="openai/clip-vit-large-patch14"):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        
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
            _, text_hidden_dim = text_features.shape

            vision_features = outputs.vision_model_output.last_hidden_state[:, 1:, :]
            batch_size, num_patches, vision_hidden_dim = vision_features.shape
            patch_size = int(np.sqrt(num_patches))
            vision_features = vision_features.reshape(batch_size, patch_size, patch_size, vision_hidden_dim)
            vision_features = F.normalize(vision_features, dim=-1)

            # Randomly initialized projection layer (defined in loop as in original notebook)
            # Setting seed to 42 for reproducibility of the random baseline
            torch.manual_seed(42)
            text_proj = torch.nn.Linear(text_hidden_dim, vision_hidden_dim).to(self.device)
            
            with torch.inference_mode():
                text_features = text_proj(text_features)
                text_features = F.normalize(text_features, dim=-1)

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
