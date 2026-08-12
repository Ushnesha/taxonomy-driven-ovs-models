import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ovs_eval.models.base import BaseOVSModel
from ovs_eval.models.clipseg import CLIPSegModel

class CATSegModel(BaseOVSModel):
    """
    CAT-Seg (Cost Aggregation Transformer for Open-Vocabulary Segmentation) wrapper for ovs_eval.
    """
    def __init__(self, device=None, config=None, weights=None, baseline=True, catseg_path="CAT-Seg"):
        """
        If baseline=True, uses CLIPSeg as fallback/baseline.
        If baseline=False, initializes the actual CAT-Seg predictor using Detectron2 config and weights.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        self.baseline = baseline or (config is None and weights is None)
        
        if self.baseline:
            self.clipseg = CLIPSegModel(device=self.device)
        else:
            # Ensure CAT-Seg repository directory is on sys.path
            abs_catseg_path = os.path.abspath(catseg_path)
            if abs_catseg_path not in sys.path:
                sys.path.insert(0, abs_catseg_path)
                
            from detectron2.config import get_cfg
            from detectron2.projects.deeplab import add_deeplab_config
            
            try:
                from cat_seg import add_cat_seg_config
            except ImportError:
                from cat_seg.config import add_cat_seg_config
                
            try:
                from cat_seg.predictor import VisualizationDemo
            except ImportError:
                try:
                    from demo.predictor import VisualizationDemo
                except ImportError:
                    from cat_seg.demo.predictor import VisualizationDemo

            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_cat_seg_config(cfg)
            cfg.merge_from_file(config)
            cfg.merge_from_list(["MODEL.WEIGHTS", weights])
            cfg.freeze()
            
            self.demo = VisualizationDemo(cfg)

    def predict(self, image, text_cats, coco, threshold=0.5, desc=False):
        if self.baseline:
            return self.clipseg.predict(image, text_cats, coco, threshold, desc)
            
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            
        # Detectron2 models expect BGR format
        img_bgr = np.array(image_pil.convert("RGB"))[:, :, ::-1]
        
        # Build prompt vocabulary
        class_names = [p_var if desc else f"a photo of a {p_var}" for p_var in text_cats.values()]
        
        # Run CAT-Seg demo predictor
        predictions, _ = self.demo.run_on_image(img_bgr, class_names)
        sem_seg = predictions["sem_seg"]  # Tensor shape [C, H, W] containing logits
        
        if not torch.is_tensor(sem_seg):
            sem_seg = torch.as_tensor(np.asarray(sem_seg))
            
        # Resolve category IDs
        if hasattr(coco, 'getCatIds'):
            cat_ids = [coco.getCatIds(catNms=[name])[0] if coco.getCatIds(catNms=[name]) else idx 
                       for idx, name in enumerate(text_cats.keys())]
        else:
            cat_ids = [coco.get(name, idx) for idx, name in enumerate(text_cats.keys())]
            
        # Convert logits to probabilities and generate binary masks
        probs = sem_seg.softmax(0)
        probs_np = probs.cpu().numpy()
        
        pred_masks = {}
        for idx, cid in enumerate(cat_ids):
            pred_masks[cid] = (probs_np[idx] > threshold).astype(np.uint8)
            
        return pred_masks
