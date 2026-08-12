class BaseOVSModel:
    """
    Abstract base class/interface for OVS models.
    """
    def predict(self, image, text_cats, coco, threshold=0.5, desc=False):
        """
        Run inference on the image for the specified text categories.
        
        Args:
            image (PIL.Image or np.ndarray): Input image.
            text_cats (dict): Mapping of canonical category name (e.g. 'car') to linguistic query variant (e.g. 'automobile').
            coco (COCO or dict): The COCO dataset object or category name-to-ID mapping.
            threshold (float): Inference threshold.
            desc (bool): Whether query variant is description (True) or standard prompt (False).
            
        Returns:
            dict: {category_id: np.ndarray [H, W] binary mask}
        """
        raise NotImplementedError("Each OVS Model must implement the predict method.")

    def batch_inference(self, images, image_prompts, threshold=0.5):
        """
        Default fallback implementation of batch_inference iterating over images.
        
        Args:
            images (list): List of PIL.Image or np.ndarray images.
            image_prompts (list of lists): List where each element is a list of prompt strings for the corresponding image.
            threshold (float): Confidence threshold.
            
        Returns:
            list of dicts: List of dicts mapping each prompt string to its binary mask [H, W].
        """
        results = []
        for img, prompts in zip(images, image_prompts):
            if not prompts:
                results.append({})
                continue
            text_cats = {prompt: prompt for prompt in prompts}
            coco = {prompt: prompt for prompt in prompts}
            res = self.predict(img, text_cats=text_cats, coco=coco, threshold=threshold, desc=True)
            results.append(res)
        return results
