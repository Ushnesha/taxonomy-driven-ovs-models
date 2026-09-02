from expanded_benchmark_helpers import device
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from ovs_eval.models.base import BaseOVSModel
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

    def predict(self, image, text_cats, coco, threshold=0.5, desc=False):
        if isinstance(image, np.ndarray):
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
            from PIL import Image
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

    def batch_inference(self, images, image_prompts, threshold=0.5):
        """
        Runs batched inference on multiple images with different prompts.
        
        Args:
            images: list of PIL Images (length B)
            image_prompts: list of lists of strings (length B), where image_prompts[i]
                        contains the prompts for images[i].
            
        Returns:
            List of dicts: list of length B containing predicted masks for each prompt
        """
        flat_images = []
        flat_prompts = []
        
        # Store indices to split the flat output back into per-image results
        split_indices = []
        current_idx = 0
        
        for img, prompts in zip(images, image_prompts):
            num_prompts = len(prompts)
            flat_images.extend([img] * num_prompts)
            flat_prompts.extend(prompts)
            
            split_indices.append((current_idx, current_idx + num_prompts))
            current_idx += num_prompts
        # Process the flat pairs in smaller sub-batches (e.g. 64 pairs at a time) to prevent OOM
        sub_batch_size = 64
        all_logits = []
        
        for i in range(0, len(flat_prompts), sub_batch_size):
            sub_prompts = flat_prompts[i : i + sub_batch_size]
            sub_images = flat_images[i : i + sub_batch_size]
            
            inputs = self.processor(
                text=sub_prompts, 
                images=sub_images, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model(**inputs)
                
            logits = outputs.logits  # Shape: [sub_batch_size, H_model, W_model]
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)
            all_logits.append(logits.cpu()) # Move to CPU immediately to free VRAM/RAM
            
        # Concatenate all logits along batch dimension
        logits = torch.cat(all_logits, dim=0)
        
        probs = torch.sigmoid(logits)
        probs_np = probs.numpy()
        
        # Split the flat batch outputs back into individual image results
        batched_results = []
        for img_idx, (start, end) in enumerate(split_indices):
            img_results = {}
            prompts_for_img = image_prompts[img_idx]
            w, h = images[img_idx].size  # Get original size of this specific image
            
            for p_idx, prompt in enumerate(prompts_for_img):
                global_idx = start + p_idx
                prob_slice = probs_np[global_idx]
                
                # Resize specifically to this image's height and width (h, w)
                prob_tensor = torch.tensor(prob_slice).unsqueeze(0).unsqueeze(0)  # [1, 1, H_model, W_model]
                prob_resized = F.interpolate(
                    prob_tensor, 
                    size=(h, w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
                
                binary_mask = (prob_resized.numpy() > threshold).astype(np.uint8)
                img_results[prompt] = binary_mask
                
            batched_results.append(img_results)
            
        return batched_results


    
