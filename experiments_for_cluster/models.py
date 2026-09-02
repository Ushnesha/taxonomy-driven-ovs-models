"""
Thin factory over the 3 pluggable OVS models this experiment suite targets:
CLIPSeg, GroupViT, SCLIP. The wrapper classes themselves
(get_text_embedding / predict_with_embeddings / predict) are not redefined
here -- source of truth stays ovs_eval/models/*.py (browsable copies also
live in ../../models_reference/ for reference). This module exists to:

  1. restrict get_model() to just the 3 models this suite runs (the other
     registered architectures -- openseg/ovseg/san/sam_clip/grounded_sam/
     sam_siglip -- are out of scope here, see helper_functions.py docstring;
     clip_vit_large was dropped -- its patch-similarity mask extraction is
     self-implemented by this project, not native to or ported from any
     official CLIP-ViT-L segmentation pipeline, so it doesn't meet this
     suite's "native/published mask prediction only" bar),
  2. give SCLIP the custom benchmark-wide name_path it needs (its default
     config only lists COCO's 80 classes as valid embeddings_dict keys; our
     346-category benchmark needs its own class list -- see
     benchmark_data.build_sclip_name_path), and
  3. paper over the one real interface difference between SCLIP and the
     other two: predict_masks_for_tags() below.

SCLIP quirk (see models_reference/sclip/sclip.py's module docstring):
predict_with_embeddings() dict keys must be real registered class names
(one embedding "swapped in" per class, inside one shared joint softmax
forward pass), not arbitrary tags -- so passing e.g. both a "baseline" and
an "ours" embedding for the same category in one call would just overwrite
each other under the same class-index key. CLIPSeg/GroupViT have no such
restriction: their embeddings_dict keys are arbitrary labels, so all tags
for one image can be batched into a single predict_with_embeddings call.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ovs_eval.models.registry import get_model as _get_model  # noqa: E402

SUPPORTED_MODELS = ["clipseg", "groupvit", "sclip"]

# Models whose predict_with_embeddings() can only carry one embedding per
# canonical category per call -- see module docstring.
ONE_EMBEDDING_PER_CLASS_PER_CALL = {"sclip"}


def get_model(name, device=None, sclip_name_path=None):
    """
    Instantiates one of the 4 supported models. `sclip_name_path` (only used
    when name == "sclip") should point at a class-list file in SCLIP's
    config format (see benchmark_data.build_sclip_name_path) covering our
    benchmark's categories -- without it, SCLIP falls back to its default
    COCO-80 class list and predict_with_embeddings() will raise KeyError for
    any category outside that list.
    """
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown/unsupported model '{name}'. Supported: {SUPPORTED_MODELS}")

    if name == "sclip":
        from ovs_eval.models.sclip import SClipModel
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if sclip_name_path is not None:
            kwargs["name_path"] = sclip_name_path
        return SClipModel(**kwargs)

    return _get_model(name, device=device)


def predict_masks_for_tags(model, model_name, image, canonical_name, tag_to_embedding, threshold=0.5):
    """
    Runs `model.predict_with_embeddings` over every (tag -> embedding) pair
    against `image`, returning {tag: mask}.

    For models that accept an arbitrary embeddings_dict key (clipseg,
    groupvit), all tags are batched into a single call.

    For models in ONE_EMBEDDING_PER_CLASS_PER_CALL (sclip), the embeddings_dict
    key must resolve to a real class and only carries one embedding per call,
    so this issues one predict_with_embeddings call per tag instead, each
    keyed by `canonical_name` (the actual class every tag's embedding is a
    variant/method computed for), and re-keys the single-entry result back
    onto the caller's tag.
    """
    if not tag_to_embedding:
        return {}

    if model_name in ONE_EMBEDDING_PER_CLASS_PER_CALL:
        masks = {}
        for tag, emb in tag_to_embedding.items():
            result = model.predict_with_embeddings(image, {canonical_name: emb}, threshold=threshold)
            masks[tag] = result[canonical_name]
        return masks

    return model.predict_with_embeddings(image, tag_to_embedding, threshold=threshold)
