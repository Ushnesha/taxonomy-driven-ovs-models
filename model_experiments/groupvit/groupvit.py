"""GroupViT model wrapper (nvidia/groupvit-gcc-yfcc via HF Transformers). Copied from
taxonomy-driven-ovs-models/ovs_eval/models/groupvit.py -- trimmed to the two methods
alpha_value_experiment.py needs (get_text_embedding, predict_with_embeddings); predict()
(text-prompt-driven inference) is dropped since this folder's sweep never calls it.
`from base import BaseOVSModel` replaces the original `from ovs_eval.models.base import
BaseOVSModel` since base.py is a local sibling file here, not part of a package.
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor
from transformers import GroupViTModel as HFGroupViTModel
from transformers.models.groupvit.modeling_groupvit import get_grouping_from_attentions
from base import BaseOVSModel
from PIL import Image


class GroupViTOVSModel(BaseOVSModel):
    """
    nvidia/groupvit-gcc-yfcc via transformers.GroupViTModel.

    Unlike CLIPSeg (FiLM decoder conditioned on a raw-norm text embedding),
    GroupViT segments by cosine similarity between projected+L2-normalized
    text embeddings and per-group image embeddings produced by its
    hierarchical grouping blocks (see get_grouping_from_attentions). The
    blending hook here is `text_embeds` -- the output of `text_projection`
    on the text encoder's pooled output, shape [1, projection_dim=256] --
    analogous to CLIPSeg's `conditional_embeddings`.
    """

    def __init__(self, device=None, model_id="nvidia/groupvit-gcc-yfcc"):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = HFGroupViTModel.from_pretrained(model_id).to(self.device).eval()

    @staticmethod
    def _threshold(prob_map, threshold):
        """
        GroupViT's per-pixel similarity is only meaningfully separable in a
        *relative* sense (it was designed for argmax over a full label set,
        not an independent per-class decision boundary) -- the whole image
        tends to score positively for any single word. Fall back to a
        per-image dynamic threshold: normalize within this map's own
        [min, max] range so the mask isn't degenerate.
        """
        max_val = prob_map.max()
        min_val = prob_map.min()
        if max_val < threshold:
            return np.zeros_like(prob_map, dtype=np.uint8)
        dynamic_thresh = min_val + 0.5 * (max_val - min_val)
        threshold_to_use = max(dynamic_thresh, threshold)
        return (prob_map > threshold_to_use).astype(np.uint8)

    def get_text_embedding(self, word, desc=False):
        """
        Raw projected text embedding -- NOT L2-normalized -- the blending hook.
        Shape [1, 256]. GroupViT normalizes this internally right before use,
        so unlike CLIPSeg, downstream behavior only depends on direction, not
        the norm you hand it.
        """
        prompt = word if desc else f"a photo of a {word}"
        inputs = self.processor(text=[prompt], return_tensors="pt", padding=True).to(self.device)

        with torch.inference_mode():
            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            text_embeds = self.model.text_projection(text_outputs.pooler_output)

        return text_embeds.cpu()

    def predict_with_embeddings(self, image_pil, embeddings_dict, threshold=0.5):
        """
        Predict binary masks from pre-computed (e.g. alpha-blended) text
        embeddings, bypassing the text encoder entirely. Mirrors CLIPSeg's
        predict_with_embeddings hook.

        embeddings_dict: {key: torch.Tensor [1, 256] or [256]}, i.e. the
        (possibly blended) output of get_text_embedding().
        """
        if isinstance(image_pil, np.ndarray):
            image_pil = Image.fromarray(image_pil)

        w, h = image_pil.size
        image_inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            vision_outputs = self.model.vision_model(
                pixel_values=image_inputs["pixel_values"],
                output_attentions=True,
                return_dict=True,
            )

            image_group_embeds = vision_outputs.last_hidden_state  # [1, num_group, hidden]
            image_group_embeds = self.model.visual_projection(
                image_group_embeds.reshape(-1, image_group_embeds.shape[-1])
            )
            image_group_embeds = F.normalize(image_group_embeds, dim=-1)

            grouping = get_grouping_from_attentions(
                vision_outputs.attentions, image_inputs["pixel_values"].shape[2:]
            )  # [1, num_group, H, W]
            flatten_grouping = grouping.reshape(grouping.shape[0], grouping.shape[1], -1)

            logit_scale = self.model.logit_scale.exp()

            pred_masks = {}
            for key, emb in embeddings_dict.items():
                emb = emb.to(self.device)
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                emb = F.normalize(emb, dim=-1)

                logits_per_group = torch.matmul(image_group_embeds, emb.t()) * logit_scale  # [num_group, 1]
                logits_per_group = logits_per_group.reshape(1, 1, -1)  # [1, 1, num_group]

                seg_logits = torch.matmul(logits_per_group, flatten_grouping)
                seg_logits = seg_logits.reshape(1, grouping.shape[2], grouping.shape[3])

                probs = torch.sigmoid(seg_logits)
                probs = F.interpolate(
                    probs.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
                )[0, 0]

                pred_masks[key] = self._threshold(probs.cpu().numpy(), threshold)

        return pred_masks
