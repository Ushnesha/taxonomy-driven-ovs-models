import os
import numpy as np
import torch
import torch.nn.functional as F
from ovs_eval.models.base import BaseOVSModel

class OpenSegModel(BaseOVSModel):
    def __init__(self, device=None, weights=None, baseline=True):
        """
        If baseline=True, uses the baseline from testing_openSeg.ipynb (CLIP-base image-level baseline).
        If baseline=False and weights is provided, attempts to load the TensorFlow OpenSeg model.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        self.baseline = baseline or (weights is None)
        
        if self.baseline:
            from transformers import CLIPModel, CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
            
            # Adjust processor size
            self.processor.image_processor.size = {
                "height": self.model.config.vision_config.image_size,
                "width": self.model.config.vision_config.image_size
            }
        else:
            # Load TensorFlow OpenSeg model
            import tensorflow.compat.v2 as tf
            import open_clip
            self.openseg_model = tf.saved_model.load(weights)
            
            # Load CLIP ViT-L-14 text encoder
            self.clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", "openai")
            self.clip_model = self.clip_model.to(self.device).eval()
            self.tokenizer = open_clip.get_tokenizer("ViT-L-14")

    def predict(self, image, text_cats, coco, threshold=0.5, desc=False):
        if isinstance(image, np.ndarray):
            from PIL import Image
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            
        if self.baseline:
            return self._predict_baseline(image_pil, text_cats, coco, threshold, desc)
        else:
            return self._predict_openseg(image_pil, text_cats, coco, threshold, desc)

    def _predict_baseline(self, image_pil, text_cats, coco, threshold, desc):
        img_array = np.array(image_pil.convert("RGB"))
        original_height, original_width = img_array.shape[:2]
        pred_masks = {}
        
        for idx, (p_base, p_var) in enumerate(text_cats.items()):
            text_prompt = p_var if desc else f"a photo of a {p_var}"
            
            text_inputs = self.processor(
                text=text_prompt, images=None, return_tensors="pt", padding=True
            )['input_ids'].to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(text_inputs)
                text_features = F.normalize(text_features, dim=-1)
                
            image_inputs = self.processor(
                text=None, images=image_pil, return_tensors="pt"
            )['pixel_values'].to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(image_inputs)
                image_features = F.normalize(image_features, dim=-1)
                
            similarity = (text_features @ image_features.T).cpu().numpy()[0, 0]
            
            if hasattr(coco, 'getCatIds'):
                cat_ids = coco.getCatIds(catNms=[p_base])
                cat_id = cat_ids[0] if cat_ids else idx
            else:
                cat_id = coco.get(p_base, idx)
                
            # Mapped to the whole image
            mask = np.ones((original_height, original_width), dtype=np.float32) * similarity
            mask = (mask > threshold).astype(np.uint8)
            pred_masks[cat_id] = mask
            
        return pred_masks

    def _predict_openseg(self, image_pil, text_cats, coco, threshold, desc):
        import io
        import tensorflow.compat.v2 as tf
        
        # Build category lists
        names = list(text_cats.values())
        if hasattr(coco, 'getCatIds'):
            cat_ids = [coco.getCatIds(catNms=[name])[0] for name in text_cats.keys()]
        else:
            cat_ids = [coco[name] for name in text_cats.keys()]
            
        W, H = image_pil.size
        buf = io.BytesIO()
        image_pil.convert("RGB").save(buf, format="JPEG")
        
        out = self.openseg_model.signatures["serving_default"](
            inp_image_bytes=tf.convert_to_tensor(buf.getvalue()),
            inp_text_emb=tf.zeros([1, 1, 768], tf.float32)
        )
        
        info = out["image_info"]
        ch, cw = int(info[0, 0] * info[2, 0]), int(info[0, 1] * info[2, 1])
        feat = out["ppixel_ave_feat"][:, :ch, :cw]
        feat = tf.image.resize(feat, (H, W), method="nearest")[0].numpy()
        feat /= (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8)
        
        # Encode text features using open_clip
        toks = self.tokenizer([f"a photo of a {n}" for n in names]).to(self.device)
        with torch.no_grad():
            t = self.clip_model.encode_text(toks)
            t = F.normalize(t, dim=-1)
        class_emb = t.cpu().numpy().astype(np.float32)
        
        sims = torch.from_numpy(feat @ class_emb.T).permute(2, 0, 1) # [C, H, W]
        
        # Convert logits to masks
        conf, label = sims.max(0)
        label, conf = label.cpu().numpy(), conf.cpu().numpy()
        
        pred_masks = {}
        for i, cid in enumerate(cat_ids):
            pred_masks[cid] = ((label == i) & (conf > threshold)).astype(np.uint8)
            
        return pred_masks
