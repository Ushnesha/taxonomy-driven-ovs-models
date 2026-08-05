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
        
        try:
            results = self.gd_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_thresh,
                text_threshold=box_thresh,
                target_sizes=[(H, W)]
            )[0]
        except TypeError:
            # Fallback for older transformers versions where the parameter is named 'threshold'
            results = self.gd_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_thresh,
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

    def predict_v2(self, image_pil, gt_masks, text_prompts, threshold=0.3):
        """
        Predicts binary masks for multiple prompts on a single PIL image.
        text_prompts can be a list of prompts or a dictionary of {key: prompt_string}.
        """
        if isinstance(image_pil, np.ndarray):
            from PIL import Image
            image_pil = Image.fromarray(image_pil)
            img_np = image_pil
        else:
            img_np = np.array(image_pil.convert("RGB"))
            
        W, H = image_pil.size
        
        # Handle list vs dict
        if isinstance(text_prompts, list):
            prompts_dict = {p: p for p in text_prompts}
        else:
            prompts_dict = text_prompts
            
        if not prompts_dict:
            return {}

        # Normalize helper for string matching
        def normalize_str(s):
            if not isinstance(s, str):
                return ""
            return s.strip().lower().replace(".", "").replace("a photo of a ", "").replace("a photo of an ", "").replace("a photo of ", "").replace("_", " ").replace("-", " ")

        # Grounding DINO has a 256 token limit. To safely prevent exceeding this limit,
        # we chunk the prompts into groups of at most 25.
        chunk_size = 25
        prompts_keys = list(prompts_dict.keys())
        pred_masks = {key: np.zeros((H, W), dtype=np.uint8) for key in prompts_dict.keys()}

        # Set SAM image once outside the loop
        self.sam_predictor.set_image(img_np)

        for k_idx in range(0, len(prompts_keys), chunk_size):
            chunk_keys = prompts_keys[k_idx : k_idx + chunk_size]
            sub_prompts_dict = {k: prompts_dict[k] for k in chunk_keys}

            prompt_to_key = {}
            text_prompts_list = []
            for key, p_text in sub_prompts_dict.items():
                norm_p = normalize_str(p_text)
                prompt_to_key[norm_p] = key
                text_prompts_list.append(p_text if p_text.endswith(".") else f"{p_text}.")

            # Join prompts for Grounding DINO call
            text_prompt = " ".join(text_prompts_list)

            # Run Grounding DINO
            inputs = self.gd_processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.gd_model(**inputs)
                
            box_thresh = threshold if threshold < 0.5 else 0.25
            try:
                results = self.gd_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=box_thresh,
                    text_threshold=box_thresh,
                    target_sizes=[(H, W)]
                )[0]
            except TypeError:
                results = self.gd_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=box_thresh,
                    text_threshold=box_thresh,
                    target_sizes=[(H, W)]
                )[0]

            boxes = results.get("boxes", [])
            text_labels = results.get("text_labels")
            if text_labels is None:
                text_labels = results.get("labels", [])

            # Diagnostic print for debugging Grounding DINO matching
            print(f"DEBUG DINO (chunk {k_idx//chunk_size}): prompts={len(chunk_keys)} | boxes={len(boxes)} | labels={text_labels}")

            if len(boxes) > 0:
                for i in range(len(boxes)):
                    box = boxes[i].cpu().numpy() # [xmin, ymin, xmax, ymax]
                    label_str = text_labels[i]
                    # Ensure label_str is string (might be integer ID in some transformers versions)
                    if not isinstance(label_str, str):
                        try:
                            # Try to decode input token index to string if possible, or fallback
                            token_id = int(label_str)
                            label_str = self.gd_processor.tokenizer.decode([token_id])
                        except Exception:
                            label_str = str(label_str)
                            
                    label_norm = normalize_str(label_str)
                    
                    matched_key = None
                    if label_norm in prompt_to_key:
                        matched_key = prompt_to_key[label_norm]
                    else:
                        for p_norm, key in prompt_to_key.items():
                            if label_norm in p_norm or p_norm in label_norm:
                                matched_key = key
                                break
                                
                    if matched_key is not None:
                        masks, _, _ = self.sam_predictor.predict(
                            box=box,
                            multimask_output=False
                        )
                        pred_masks[matched_key] = np.logical_or(pred_masks[matched_key], masks[0]).astype(np.uint8)
                        
        return pred_masks

    def batch_inference(self, images, image_prompts, threshold=0.3):
        """
        Runs batch prediction sequentially over multiple images.
        """
        batched_results = []
        for img, prompts in zip(images, image_prompts):
            prompts_dict = {p: p for p in prompts}
            pred_dict = self.predict_v2(img, None, prompts_dict, threshold=threshold)
            batched_results.append(pred_dict)
        return batched_results
