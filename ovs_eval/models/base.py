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
