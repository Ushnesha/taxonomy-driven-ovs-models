import numpy as np
import torch
import torch.nn.functional as F
from ovs_eval.models.base import BaseOVSModel
from transformers import AutoProcessor, GroundingDinoForObjectDetection

class GroundedSAMModel(BaseOVSModel):
    def __init__(self, device=None, weights=None, model_type="vit_h", gd_model_id="IDEA-Research/grounding-dino-tiny"):
        """
        Grounded SAM Model combining Grounding DINO for zero-shot object detection
        and SAM for high-quality mask generation.
        
        weights: Path to the SAM checkpoint (e.g. 'sam_vit_h_4b8939.pth').
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        if weights is None:
            raise ValueError(
                "Grounded SAM model requires the SAM weights checkpoint. "
                "Please download 'sam_vit_h_4b8939.pth' from the Segment Anything repository "
                "and pass it via --weights."
            )
            
        # Load SAM model
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[model_type](checkpoint=weights).to(self.device)
        self.sam_predictor = SamPredictor(sam)
        
        # Load Grounding DINO model & processor
        self.gd_processor = AutoProcessor.from_pretrained(gd_model_id)
        self.gd_model = GroundingDinoForObjectDetection.from_pretrained(gd_model_id).to(self.device).eval()

    def predict(self, image, text_cats, coco, threshold=0.3, desc=False):
        """
        Runs Grounding DINO to predict bounding boxes for each text category,
        then runs SAM to convert the bounding boxes into segmentations.
        """
        if isinstance(image, np.ndarray):
            from PIL import Image
            image_pil = Image.fromarray(image)
            img_np = image
        else:
            image_pil = image
            img_np = np.array(image_pil.convert("RGB"))
            
        W, H = image_pil.size
        
        # Determine category mapping
        names = list(text_cats.values())
        prompts = [p if desc else f"a photo of a {p}" for p in names]
        
        # Format text prompt for Grounding DINO (each query should be terminated with a dot)
        text_prompt = ". ".join(prompts) + "."
        
        if hasattr(coco, 'getCatIds'):
            cat_ids = [coco.getCatIds(catNms=[name])[0] for name in text_cats.keys()]
        else:
            cat_ids = [coco[name] for name in text_cats.keys()]
            
        # Helper to normalize strings for comparison
        def normalize_str(s):
            return s.strip().lower().replace(".", "").replace("a photo of a ", "").replace("a photo of an ", "").replace("a photo of ", "")
            
        prompt_to_cat_id = {}
        for (canon_name, p_text), cid in zip(text_cats.items(), cat_ids):
            prompt_to_cat_id[normalize_str(p_text)] = cid
            
        # Run Grounding DINO
        inputs = self.gd_processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gd_model(**inputs)
            
        # Post-process detections
        # Note: box_threshold controls object detection confidence. We set text_threshold similarly.
        # Bounding boxes scores behave differently from pixel probabilities, so we set a sensible limit.
        box_thresh = threshold if threshold < 0.5 else 0.25
        
        results = self.gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_thresh,
            text_threshold=box_thresh,
            target_sizes=[(H, W)]
        )[0]
        
        pred = {cid: np.zeros((H, W), np.uint8) for cid in cat_ids}
        
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        text_labels = results.get("text_labels", results.get("labels", []))
        
        if len(boxes) == 0:
            return pred
            
        # Run SAM predictor using box prompts
        self.sam_predictor.set_image(img_np)
        
        for i in range(len(boxes)):
            box = boxes[i].cpu().numpy() # [xmin, ymin, xmax, ymax]
            label_str = text_labels[i]
            
            # Match detected label to original category
            label_norm = normalize_str(label_str)
            matched_cat_id = None
            
            # Try exact match first
            if label_norm in prompt_to_cat_id:
                matched_cat_id = prompt_to_cat_id[label_norm]
            else:
                # Substring match
                for p_norm, cid in prompt_to_cat_id.items():
                    if label_norm in p_norm or p_norm in label_norm:
                        matched_cat_id = cid
                        break
                        
            if matched_cat_id is not None:
                # Run SAM for this box
                masks, _, _ = self.sam_predictor.predict(
                    box=box,
                    multimask_output=False
                )
                # masks is [1, H, W]
                pred[matched_cat_id] = np.logical_or(pred[matched_cat_id], masks[0]).astype(np.uint8)
                
        return pred
