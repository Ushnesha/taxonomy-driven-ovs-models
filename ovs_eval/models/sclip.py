"""
Real SCLIP (Wang et al., ECCV 2024, "SCLIP: Rethinking Self-Attention for
Dense Vision-Language Inference") wrapped in the BaseOVSModel interface.

Runs the actual wangf3014/SCLIP repo code via its CLIPForSegmentation class --
the correlative self-attention modification to CLIP's last attention block
(csa=True in encode_image, inside the vendored `clip/` package) is untouched,
no architecture shortcuts. Ported from the validated prototype at
models_sandbox/sclip_blend_experiment.py (confirmed run end-to-end against
real COCO images; results cached in models_sandbox/results/).

SCLIP needs mmengine/mmcv/mmsegmentation, which are NOT in this project's
base requirements.txt (heavier, separately-pinned deps -- see
models_sandbox/setup_env.sh / sclip_venv). This module is imported lazily by
registry.py specifically so the base venv can still import everything else
when mmcv isn't installed.

Unlike CLIPSeg's FiLM decoder (raw-norm conditional_embeddings) or GroupViT's
internally-renormalized text_embeds, SCLIP scores L2-normalized per-patch
image features against L2-normalized text embeddings via dot product +
softmax -- CLAUDE.md's "never L2-normalize" rule is specific to CLIPSeg's
raw-norm convention and does not apply here. get_text_embedding() below
L2-normalizes by construction (twice, matching the repo's own per-query
embedding computation), and any blended embedding must be re-normalized
before use for the same reason -- predict_with_embeddings() does this.

SCLIP is also architecturally different from CLIPSeg/GroupViT in one more
way: it classifies every pixel against a single shared, fixed
`model.query_features` tensor (one joint softmax across all COCO classes at
once), not an independent per-prompt call. So predict_with_embeddings() here
must know which COCO class each embedding belongs to -- its `embeddings_dict`
keys must be canonical COCO category names (e.g. "dog"), not arbitrary
labels, so the right query_features row(s) can be swapped in before the
single joint forward pass and restored after.
"""
import os
import sys
import tempfile
import threading
import types
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from ovs_eval.models.base import BaseOVSModel


def _patch_mmcv_ops():
    """
    mmcv's compiled `mmcv.ops` extension is not needed by CLIPForSegmentation
    (it never calls into custom conv/attention ops), but mmseg.models eagerly
    imports unrelated decode heads/losses at package-import time that do
    `from mmcv.ops import ...`. On platforms where the compiled extension
    fails to load (observed: mmcv built from source on macOS arm64, missing
    MPS symbols at dlopen time) this stub lets those unused imports succeed
    instead of crashing the process. Real prebuilt mmcv wheels (e.g.
    Linux+CUDA cluster installs) import fine on their own, so this only
    activates as a fallback.
    """
    try:
        import mmcv.ops  # noqa: F401
        return
    except Exception:
        pass

    class _DummyOpsModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            class _Dummy:
                def __init__(self, *a, **k): pass
                def __call__(self, *a, **k):
                    raise NotImplementedError(f"mmcv.ops.{name} stubbed (unused by SCLIP)")
            _Dummy.__name__ = name
            return _Dummy

    sys.modules["mmcv.ops"] = _DummyOpsModule("mmcv.ops")


_SCLIP_DIR_DEFAULT = os.environ.get(
    "SCLIP_DIR",
    # taxonomy-driven-ovs-models/SCLIP -- vendored (code only, no .git) from
    # https://github.com/wangf3014/SCLIP so it travels with this repo onto
    # the cluster, rather than depending on a sibling folder that only
    # exists in this local dev checkout.
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "SCLIP")),
)


class SClipModel(BaseOVSModel):
    def __init__(self, device=None, sclip_dir=None, clip_path="ViT-B/16",
                 name_path=None, logit_scale=50, prob_thd=0.1):
        self.sclip_dir = os.path.abspath(sclip_dir or _SCLIP_DIR_DEFAULT)
        if self.sclip_dir not in sys.path:
            # Must come first on sys.path so `import clip` resolves to SCLIP's
            # vendored clip/ package, not any pip-installed `clip` package.
            sys.path.insert(0, self.sclip_dir)

        _patch_mmcv_ops()
        try:
            import clip_segmentor
            import clip as sclip_clip
            from prompts.imagenet_template import openai_imagenet_template
        except ImportError as e:
            raise ImportError(
                "SClipModel requires mmengine/mmcv/mmsegmentation and SCLIP's own "
                "`clip` package, which live in a separate venv from this project's base "
                "requirements.txt. Set up and activate models_sandbox/sclip_venv (see "
                "models_sandbox/setup_env.sh), or point SCLIP_DIR / sclip_dir at a "
                "working checkout of https://github.com/wangf3014/SCLIP."
            ) from e

        self._clip_segmentor = clip_segmentor
        self._clip_module = sclip_clip
        self._template_list = openai_imagenet_template

        if device is None:
            # MPS deliberately excluded: postprocess_result's one-hot class-merge
            # builds a (num_classes x num_queries x H x W) tensor that exceeds
            # MPS's per-buffer allocation cap at full COCO image resolution.
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._name_path = name_path or os.path.join(self.sclip_dir, "configs", "cls_coco_object.txt")
        self.model = clip_segmentor.CLIPForSegmentation(
            clip_path=clip_path, name_path=self._name_path,
            device=torch.device(self.device), logit_scale=logit_scale, prob_thd=prob_thd,
        )
        self.model.eval()
        self.model.to(self.device)  # __init__ moves self.net but not data_preprocessor

        # configs/cls_coco_object.txt line 0 is a background/"stuff" class whose
        # comma-separated words can collide with a real COCO class name (e.g.
        # "bed" is both a stuff-word and its own foreground class), so class
        # names are resolved from foreground lines only (line 1 onward).
        with open(self._name_path) as f:
            fg_class_names = [line.strip().split(", ")[0] for line in f.readlines()[1:]]
        self._name_to_class_idx = {name.lower(): i + 1 for i, name in enumerate(fg_class_names)}

        rows_by_class = defaultdict(list)
        for row, class_idx in enumerate(self.model.query_idx.tolist()):
            rows_by_class[class_idx].append(row)
        self._rows_by_class = dict(rows_by_class)

        self._lock = threading.Lock()  # model.query_features/prob_thd are shared mutable state

    def _class_idx_for(self, canonical_name):
        idx = self._name_to_class_idx.get(canonical_name.lower())
        if idx is None:
            raise KeyError(
                f"'{canonical_name}' is not a foreground class in {os.path.basename(self._name_path)}"
            )
        return idx

    def get_text_embedding(self, word, desc=False):
        """
        Raw text embedding for `word`: mean of per-template CLIP text features,
        L2-normalized twice (per-template, then after mean) -- verbatim
        reproduction of CLIPForSegmentation.__init__'s per-query embedding
        computation, factored out so it can be called on arbitrary WordNet
        variant words, not just the fixed COCO class list.

        `desc` is accepted for interface parity with other models: SCLIP's own
        prompt ensembling already covers template variation for a bare word,
        so a caller-supplied description string is instead embedded verbatim,
        template-free, as the single query.
        """
        with torch.no_grad():
            if desc:
                query = self._clip_module.tokenize([word]).to(self.device)
            else:
                query = self._clip_module.tokenize(
                    [t(word) for t in self._template_list]
                ).to(self.device)
            feature = self.model.net.encode_text(query)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            feature = feature.mean(dim=0)
            feature = feature / feature.norm()
        return feature.cpu()

    def _load_image(self, image):
        """
        Runs the real SCLIP test_pipeline (configs/cfg_coco_object.py):
        LoadImageFromFile -> Resize(scale=(2048,336), keep_ratio=True) ->
        PackSegInputs. LoadImageFromFile reads from disk, so an in-memory
        image is written to a temp file first (matches how the validated
        prototype fed real COCO images through this same pipeline).
        """
        from mmcv.transforms import LoadImageFromFile, Resize
        from mmseg.datasets.transforms import PackSegInputs

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as f:
            image.convert("RGB").save(f.name)
            results = dict(img_path=f.name)
            results = LoadImageFromFile()(results)
            results = Resize(scale=(2048, 336), keep_ratio=True)(results)
            packed = PackSegInputs()(results)
        return packed["inputs"], packed["data_samples"], results["ori_shape"]

    def _predict_joint(self, image, class_to_embedding, threshold):
        """
        Swap in `class_to_embedding[class_idx]` for every query row mapped to
        that class, then run ONE real joint forward pass -- every class,
        including ones not being swapped, competes in the same softmax/argmax
        exactly as in normal SCLIP inference -- extract each swapped class's
        binary mask, then restore the model's original state. Locked because
        model.query_features/prob_thd are mutable state shared across calls.
        """
        img_tensor, data_sample, _ = self._load_image(image)

        with self._lock:
            original_prob_thd = self.model.prob_thd
            originals = {}
            try:
                self.model.prob_thd = threshold
                for class_idx, emb in class_to_embedding.items():
                    emb = emb.to(self.device)
                    if emb.dim() == 2:
                        emb = emb.squeeze(0)
                    emb = emb / emb.norm()
                    for r in self._rows_by_class.get(class_idx, []):
                        if r not in originals:
                            originals[r] = self.model.query_features[r].clone()
                        self.model.query_features[r] = emb

                batch = dict(inputs=[img_tensor], data_samples=[data_sample])
                processed = self.model.data_preprocessor(batch, False)
                with torch.no_grad():
                    out = self.model.predict(processed["inputs"], processed["data_samples"])
                pred = out[0].pred_sem_seg.data.squeeze(0).cpu().numpy()
            finally:
                for r, orig in originals.items():
                    self.model.query_features[r] = orig
                self.model.prob_thd = original_prob_thd

        return {class_idx: (pred == class_idx).astype(np.uint8) for class_idx in class_to_embedding}

    def predict(self, image, text_cats, coco, threshold=0.5, desc=False):
        class_to_embedding = {}
        class_idx_to_cat_id = {}
        for p_base, p_var in text_cats.items():
            class_idx = self._class_idx_for(p_base)
            class_to_embedding[class_idx] = self.get_text_embedding(p_var, desc=desc)

            if hasattr(coco, "getCatIds"):
                cat_ids = coco.getCatIds(catNms=[p_base])
                cat_id = cat_ids[0] if cat_ids else class_idx
            else:
                cat_id = coco.get(p_base, class_idx)
            class_idx_to_cat_id[class_idx] = cat_id

        masks_by_class = self._predict_joint(image, class_to_embedding, threshold)
        return {class_idx_to_cat_id[c]: m for c, m in masks_by_class.items()}

    def predict_with_embeddings(self, image_pil, embeddings_dict, threshold=0.5):
        """
        embeddings_dict: {canonical_coco_category_name: torch.Tensor}, i.e.
        the (possibly alpha-blended) output of get_text_embedding(), keyed by
        the COCO class name it should replace (see module docstring for why
        the key must resolve to a class, unlike CLIPSeg/GroupViT).
        """
        class_to_embedding = {}
        class_idx_to_key = {}
        for key, emb in embeddings_dict.items():
            class_idx = self._class_idx_for(key)
            class_to_embedding[class_idx] = emb
            class_idx_to_key[class_idx] = key

        masks_by_class = self._predict_joint(image_pil, class_to_embedding, threshold)
        return {class_idx_to_key[c]: m for c, m in masks_by_class.items()}
