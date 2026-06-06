import numpy as np
import torch
from ovs_eval.models.base import BaseOVSModel
from ovs_eval.models.clipseg import CLIPSegModel

class OVSegModel(BaseOVSModel):
    def __init__(self, device=None, config=None, weights=None, baseline=True):
        """
        If baseline=True, uses the baseline from testing_ovseg.ipynb (which is CLIPSeg).
        If baseline=False, uses the actual open-vocab-seg wrapper.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        self.baseline = baseline or (config is None and weights is None)
        
        if self.baseline:
            self.clipseg = CLIPSegModel(device=self.device)
        else:
            from detectron2.config import get_cfg
            from detectron2.projects.deeplab import add_deeplab_config
            from open_vocab_seg import add_ovseg_config
            from open_vocab_seg.utils import VisualizationDemo
            
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_ovseg_config(cfg)
            cfg.merge_from_file(config)
            cfg.merge_from_list(["MODEL.WEIGHTS", weights])
            cfg.freeze()
            
            self.demo = VisualizationDemo(cfg)

    def predict(self, image, text_cats, coco, threshold=0.5, desc=False):
        if self.baseline:
            return self.clipseg.predict(image, text_cats, coco, threshold, desc)
        else:
            if isinstance(image, np.ndarray):
                from PIL import Image
                image_pil = Image.fromarray(image)
            else:
                image_pil = image
                
            img_bgr = np.array(image_pil)[:, :, ::-1] # RGB -> BGR
            class_names = list(text_cats.values())
            
            # Run inference
            predictions, _ = self.demo.run_on_image(img_bgr, class_names)
            sem_seg = predictions["sem_seg"] # [C, H, W] logits
            
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
