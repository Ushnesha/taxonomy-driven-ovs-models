from ovs_eval.models.clipseg import CLIPSegModel
from ovs_eval.models.clip_vit_large import CLIPViTLargeModel
from ovs_eval.models.siglip import SiglipModel
from ovs_eval.models.openseg import OpenSegModel
from ovs_eval.models.ovseg import OVSegModel
from ovs_eval.models.san import SANModel
from ovs_eval.models.sam_clip import SAMCLIPModel
from ovs_eval.models.grounded_sam import GroundedSAMModel
from ovs_eval.models.sam_siglip import SAMSiglipModel
from ovs_eval.models.groupvit import GroupViTOVSModel

_REGISTRY = {
    "clipseg": CLIPSegModel,
    "clip_vit_large": CLIPViTLargeModel,
    "siglip": SiglipModel,
    "openseg": OpenSegModel,
    "ovseg": OVSegModel,
    "san": SANModel,
    "sam_clip": SAMCLIPModel,
    "grounded_sam": GroundedSAMModel,
    "sam_siglip": SAMSiglipModel,
    "groupvit": GroupViTOVSModel,
}

# SClip needs mmengine/mmcv/mmsegmentation, which live in their own venv
# (models_sandbox/sclip_venv) separate from this project's base
# requirements.txt -- import it lazily so importing this registry doesn't
# fail in a venv that never installed those deps.
try:
    from ovs_eval.models.sclip import SClipModel
    _REGISTRY["sclip"] = SClipModel
except ImportError as _sclip_import_error:
    SClipModel = None

def list_models():
    return list(_REGISTRY.keys())

def get_model(name, device=None, weights=None, config=None, baseline=True):
    """
    Factory function to retrieve model instance by name.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model architecture: '{name}'. Available: {list_models()}")
        
    model_cls = _REGISTRY[name]
    
    # Pass config and weights if available
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    if name in ["openseg", "sam_clip", "grounded_sam", "sam_siglip"]:
        kwargs["weights"] = weights
        if name == "openseg":
            kwargs["baseline"] = baseline
    elif name in ["ovseg", "san"]:
        kwargs["config"] = config
        kwargs["weights"] = weights
        kwargs["baseline"] = baseline
        
    return model_cls(**kwargs)

