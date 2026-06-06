import numpy as np
import torch
from ovs_eval.models.base import BaseOVSModel
from ovs_eval.models.clipseg import CLIPSegModel

class SANModel(BaseOVSModel):
    def __init__(self, device=None, config=None, weights=None, baseline=True):
        """
        If baseline=True, uses the baseline from testing_san.ipynb (which is CLIPSeg).
        If baseline=False, uses the actual MendelXu/SAN Predictor.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        self.baseline = baseline or (config is None and weights is None)
        
        if self.baseline:
            self.clipseg = CLIPSegModel(device=self.device)
        else:
            from predict import Predictor # SAN repo root has predict.py
            self.predictor = Predictor(config_file=config, model_path=weights)

    def predict(self, image, text_cats, coco, threshold=0.5, desc=False):
        if self.baseline:
            return self.clipseg.predict(image, text_cats, coco, threshold, desc)
        else:
            if isinstance(image, np.ndarray):
                from PIL import Image
                image_pil = Image.fromarray(image)
                image_np = image
            else:
                image_pil = image
                image_np = np.array(image_pil)
                
            vocab = list(text_cats.values())
            
            # Predict
            result = self.predictor.predict(image_np, vocabulary=vocab, augment_vocabulary=False)
            sem_seg = result["sem_seg"] # [C, H, W]
            
            if not torch.is_tensor(sem_seg):
                sem_seg = torch.as_tensor(np.asarray(sem_seg))
                
            # Process outputs
            if hasattr(coco, 'getCatIds'):
                cat_ids = [coco.getCatIds(catNms=[name])[0] for name in text_cats.keys()]
            else:
                cat_ids = [coco[name] for name in text_cats.keys()]
                
            probs = sem_seg.softmax(0)
            conf, label = probs.max(0)
            label, conf = label.cpu().numpy(), conf.cpu().numpy()
            
            return {cid: ((label == i) & (conf > threshold)).astype(np.uint8)
                    for i, cid in enumerate(cat_ids)}
