import numpy as np
import torch
import torch.nn.functional as F
from ovs_eval.models.base import BaseOVSModel
from transformers import CLIPModel, CLIPProcessor

class SAMCLIPModel(BaseOVSModel):
    def __init__(self, device=None, weights=None, model_type="vit_h"):
        """
        weights: Path to the SAM checkpoint (e.g. 'sam_vit_h_4b8939.pth').
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        if weights is None:
            raise ValueError(
                "SAM+CLIP model requires the SAM weights checkpoint. "
                "Please download 'sam_vit_h_4b8939.pth' from the Segment Anything repository "
                "and pass it via --weights."
            )
            
        # Load SAM Automatic Mask Generator
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry[model_type](checkpoint=weights).to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100
        )
        
        # Load CLIP
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device).eval()

    def predict(self, image, text_cats, coco, threshold=0.5, desc=False):
        if isinstance(image, np.ndarray):
            from PIL import Image
            image_pil = Image.fromarray(image)
            img_np = image
        else:
            image_pil = image
            img_np = np.array(image_pil.convert("RGB"))
            
        W, H = image_pil.size
        
        # Format names and categories
        names = list(text_cats.values())
        prompts = [p if desc else f"a photo of a {p}" for p in names]
        
        if hasattr(coco, 'getCatIds'):
            cat_ids = [coco.getCatIds(catNms=[name])[0] for name in text_cats.keys()]
        else:
            cat_ids = [coco[name] for name in text_cats.keys()]
            
        # Encode text features once using CLIP
        toks = self.clip_processor(
            text=prompts, images=None, return_tensors="pt", padding=True
        ).to(self.device)
        
        with torch.inference_mode():
            txt_feats = self.clip_model.get_text_features(**toks)
            if hasattr(txt_feats, "pooler_output"):
                txt_feats = txt_feats.pooler_output
            txt = F.normalize(txt_feats, dim=-1)
            
        pred = {cid: np.zeros((H, W), np.uint8) for cid in cat_ids}
        
        # Generate class-agnostic mask proposals using SAM
        masks = self.mask_generator.generate(img_np)
        
        crops = []
        valid_masks = []
        for m in masks:
            if m["area"] < 100:
                continue
            seg = m["segmentation"]
            ys, xs = np.where(seg)
            if len(xs) == 0:
                continue
            crop = image_pil.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            crops.append(crop)
            valid_masks.append(m)
            
        if len(crops) == 0:
            return pred
            
        # Process and extract CLIP features in batches to utilize GPU parallelism
        batch_size = 32
        im_features_list = []
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            ci = self.clip_processor(images=batch_crops, return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                im_feats = self.clip_model.get_image_features(**ci)
                if hasattr(im_feats, "pooler_output"):
                    im_feats = im_feats.pooler_output
                im_feats = F.normalize(im_feats, dim=-1)
                im_features_list.append(im_feats)
                
        im_features = torch.cat(im_features_list, dim=0) # [N, D]
        
        # Parallel similarity and class selection
        scores = im_features @ txt.T # [N, C]
        conf, idx = scores.max(dim=1)
        
        conf = conf.cpu().numpy()
        idx = idx.cpu().numpy()
        
        for i, m in enumerate(valid_masks):
            c_score = conf[i]
            if c_score > threshold:
                target_cat_id = cat_ids[idx[i]]
                pred[target_cat_id][m["segmentation"]] = 1
                
        return pred
