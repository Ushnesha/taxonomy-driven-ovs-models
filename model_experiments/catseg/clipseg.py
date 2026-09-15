"""CLIPSeg model wrapper (CIDAS/clipseg-rd64-refined via HF Transformers). Copied from
taxonomy-driven-ovs-models/ovs_eval/models/clipseg.py -- trimmed to the two methods
alpha_value_experiment.py needs (get_text_embedding, predict_with_embeddings); predict()/
predict_v2()/batch_inference() (text-prompt-driven inference, not embedding-driven) are
dropped since this folder's sweep never calls them. `from base import BaseOVSModel`
replaces the original `from ovs_eval.models.base import BaseOVSModel` since base.py is a
local sibling file here, not part of a package.
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from base import BaseOVSModel
from PIL import Image


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

    def get_text_embedding(self, word, desc=False):
        """
        Raw (NOT L2-normalized) conditional text embedding -- the blending
        hook. Shape [1, 512]. CLIPSeg's FiLM decoder was trained on these raw-
        norm vectors (norm ~9-10); unit-normalizing changes decoder behavior,
        so this must stay unnormalized all the way through any blending.
        """
        prompt = word if desc else f"a photo of a {word}"
        inputs = self.processor(text=[prompt], return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            text_features = self.model.clip.get_text_features(**inputs)
            if hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
        return text_features.cpu()

    def predict_with_embeddings(self, image_pil, embeddings_dict, threshold=0.5):
        """
        Predict binary masks using pre-computed conditional embeddings.

        Args:
            image_pil: PIL.Image or np.ndarray
            embeddings_dict: dict mapping category keys to torch.Tensor of shape [1, 512] or [512]
            threshold: float, confidence threshold

        Returns:
            dict: predicted binary masks mapped by category keys
        """
        if isinstance(image_pil, np.ndarray):
            image_pil = Image.fromarray(image_pil)

        w, h = image_pil.size
        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)

        pred_masks = {}
        for key, cond_emb in embeddings_dict.items():
            cond = cond_emb.to(self.device)
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)

            with torch.inference_mode():
                out = self.model(pixel_values=inputs["pixel_values"], conditional_embeddings=cond)

            logits = out.logits
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)
            probs = torch.sigmoid(logits)
            probs = F.interpolate(probs.unsqueeze(0), size=(h, w),
                                  mode="bilinear", align_corners=False).squeeze()

            pred_masks[key] = (probs.cpu().numpy() > threshold).astype(np.uint8)

        return pred_masks
