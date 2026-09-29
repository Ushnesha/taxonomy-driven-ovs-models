"""
Real SCLIP (Wang et al., ECCV 2024, "SCLIP: Rethinking Self-Attention for
Dense Vision-Language Inference") wrapped in the BaseOVSModel interface. Copied from
taxonomy-driven-ovs-models/ovs_eval/models/sclip.py, trimmed to the two methods
alpha_value_experiment.py needs (get_text_embedding, predict_with_embeddings); predict()
(text-prompt-driven inference) is dropped since this folder's sweep never calls it.
`from base import BaseOVSModel` replaces the original package-relative import, and
`_SCLIP_DIR_DEFAULT` now points at the sibling `SCLIP/` folder copied into this same
directory (see ../README.md), rather than a path two levels up a package tree.

Runs the actual wangf3014/SCLIP repo code via its CLIPForSegmentation class -- the
correlative self-attention modification to CLIP's last attention block (csa=True in
encode_image, inside the vendored `clip/` package) is untouched, no architecture
shortcuts. Attribution: https://github.com/wangf3014/SCLIP (code vendored under SCLIP/,
license/citation to be added -- see this folder's README).

SCLIP needs mmengine/mmcv/mmsegmentation -- see requirements.txt / setup_env.sh in this
same folder.

Unlike CLIPSeg's FiLM decoder (raw-norm conditional_embeddings) or GroupViT's
internally-renormalized text_embeds, SCLIP scores L2-normalized per-patch image features
against L2-normalized text embeddings via dot product + softmax -- CLAUDE.md's "never
L2-normalize" rule is specific to CLIPSeg's raw-norm convention and does not apply here.
get_text_embedding() below L2-normalizes by construction (twice, matching the repo's own
per-query embedding computation), and any blended embedding must be re-normalized before
use for the same reason -- predict_with_embeddings() does this.

SCLIP is also architecturally different from CLIPSeg/GroupViT in one more way: it
classifies every pixel against a single shared, fixed `model.query_features` tensor (one
joint softmax across all classes at once), not an independent per-prompt call. So
predict_with_embeddings() here must know which class each embedding belongs to -- its
`embeddings_dict` keys must be canonical category names present in this folder's class-list
config (COCO's 80 classes by default -- see name_path below), not arbitrary labels, so the
right query_features row(s) can be swapped in before the single joint forward pass and
restored after.
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

from base import BaseOVSModel


def _patch_postprocess_class_merge(clip_segmentor):
    """
    Replace CLIPForSegmentation.postprocess_result's class-merge step with a
    mathematically identical but vastly smaller one.

    SCLIP maps `num_queries` prompts onto `num_cls` classes (113 -> 81 for
    cls_coco_object.txt: several classes have multiple names). Upstream merges
    them by broadcasting a one-hot matrix over the full logit map:

        cls_index  = one_hot(query_idx).T.view(num_cls, num_queries, 1, 1)
        seg_logits = (seg_logits * cls_index).max(1)[0]

    That materialises a (num_cls x num_queries x H x W) tensor -- 9,153 copies
    of the image -- before immediately reducing it away along dim 1. seg_logits
    is at the ORIGINAL image resolution (predict() upsamples to ori_shape), so
    the cost scales with the input image, not with the 336px network input:
    fine on 640x480 COCO, fatal on the benchmark's large LVIS/ADE20K images
    (observed: a 2962 GiB allocation request on an 80 GiB A100).

    The reduction is a segmented max -- for each class, the max over the queries
    belonging to it -- so it can be done straight into a (num_cls x H x W)
    buffer, using ~num_queries times less memory. Equivalence: post-softmax
    logits are non-negative, so the zeros the one-hot mask introduces for
    non-member queries never beat a real member value; classes with no queries
    stay 0 under both forms.
    """
    cls = clip_segmentor.CLIPForSegmentation
    if getattr(cls, "_ovs_lowmem_merge", False):
        return

    def _merge(logits, query_idx, num_cls):
        """logits [num_queries, H, W] -> [num_cls, H, W], per-class max."""
        idx = query_idx.to(logits.device)
        out = logits.new_zeros((num_cls,) + tuple(logits.shape[1:]))
        try:
            return out.index_reduce_(0, idx, logits, "amax", include_self=False)
        except (RuntimeError, AttributeError):
            # index_reduce_ is missing on old torch and can lack a kernel for
            # some dtypes; the per-class loop is slower but always available and
            # still never allocates more than one class's worth of rows.
            for c in range(num_cls):
                rows = (idx == c).nonzero(as_tuple=True)[0]
                if rows.numel():
                    out[c] = logits.index_select(0, rows).amax(0)
            return out

    def _merge_argmax_streaming(logits, query_idx, num_cls):
        """Per-class max + argmax WITHOUT ever materialising [num_cls, H, W].

        The merged tensor is the single largest allocation in this path. At benchmark
        scale it is ruinous: 326 classes on a 2200x1650 ADE20K image is 4.4 GiB, on top
        of the 5.4 GiB the [num_queries, H, W] logits already occupy. On a 19.5 GiB GPU
        that OOMs (observed: "Tried to allocate 9.06 GiB").

        Only the per-pixel winner and its score are actually needed, so keep a running
        best over classes: peak extra memory is a couple of [H, W] planes instead of
        num_cls of them.
        """
        idx = query_idx.to(logits.device)
        hw = logits.shape[1:]
        best_val = logits.new_full(hw, float("-inf"))
        best_idx = torch.zeros(hw, dtype=torch.long, device=logits.device)
        for c in range(num_cls):
            rows = (idx == c).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            v = logits.index_select(0, rows).amax(0) if rows.numel() > 1 else logits[rows[0]]
            better = v > best_val
            best_val = torch.where(better, v, best_val)
            best_idx = torch.where(better, torch.full_like(best_idx, c), best_idx)
        return best_val, best_idx

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            logits = seg_logits[i] * self.logit_scale
            logits = logits.softmax(0)  # num_queries * H * W

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)

            if self.area_thd is not None:
                # area_thd needs the full merged tensor; SCLIP's default is None, so this
                # branch is not normally taken. Memory-hungry by necessity.
                if num_cls != num_queries:
                    logits = _merge(logits, self.query_idx, num_cls)
                predictions = torch.nn.functional.one_hot(
                    logits.argmax(0), num_cls).to(logits.dtype)
                area_pred = predictions[:, :, 1:].sum((0, 1), keepdim=True)
                area_pred = (area_pred > self.area_thd * area_pred.sum()).to(logits.dtype)
                logits[1:] *= area_pred.transpose(0, -1)
                best_val, best_idx = logits.max(0)
            elif num_cls != num_queries:
                best_val, best_idx = _merge_argmax_streaming(logits, self.query_idx, num_cls)
            else:
                best_val, best_idx = logits.max(0)

            seg_pred = best_idx.unsqueeze(0).clone()
            seg_pred[best_val.unsqueeze(0) < self.prob_thd] = 0

            # 'seg_logits' stores only the winning score, not the full per-class map.
            # Nothing in this project reads it (_predict_joint uses pred_sem_seg only),
            # and keeping the full map is what blew the GPU budget.
            data_samples[i].set_data({
                "seg_logits": clip_segmentor.PixelData(**{"data": best_val.unsqueeze(0)}),
                "pred_sem_seg": clip_segmentor.PixelData(**{"data": seg_pred}),
            })
            del logits, best_val, best_idx
        return data_samples

    cls.postprocess_result = postprocess_result
    cls._ovs_lowmem_merge = True


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
    # Sibling ./SCLIP -- vendored (code only, no .git) from
    # https://github.com/wangf3014/SCLIP so this folder is self-contained.
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "SCLIP"),
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
                "`clip` package -- run setup_env.sh in this folder first, or point "
                "SCLIP_DIR / sclip_dir at a working checkout of "
                "https://github.com/wangf3014/SCLIP."
            ) from e

        _patch_postprocess_class_merge(clip_segmentor)

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
        variant words, not just the fixed class list.

        `desc` is accepted for interface parity with other models: SCLIP's own
        prompt ensembling already covers template variation for a bare word,
        so a caller-supplied description string is instead embedded verbatim,
        template-free, as the single query.
        """
        with torch.no_grad():
            # truncate=True: OpenAI CLIP's tokenize() defaults to truncate=False, which
            # RAISES on anything over the 77-token context length. SHiNe's
            # "a X, which is a Y, which is a Z, ..." sentences, built from WordNet
            # hypernym paths, routinely exceed it. The class name comes first, so
            # truncation drops only the most abstract tail ancestors.
            if desc:
                query = self._clip_module.tokenize([word], truncate=True).to(self.device)
            else:
                query = self._clip_module.tokenize(
                    [t(word) for t in self._template_list], truncate=True
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
        image is written to a temp file first.
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

    def predict_with_embeddings(self, image_pil, embeddings_dict, threshold=0.5):
        """
        embeddings_dict: {canonical_category_name: torch.Tensor}, i.e. the
        (possibly alpha-blended) output of get_text_embedding(), keyed by the
        class name it should replace (see module docstring for why the key
        must resolve to a class, unlike CLIPSeg/GroupViT).
        """
        class_to_embedding = {}
        class_idx_to_key = {}
        for key, emb in embeddings_dict.items():
            class_idx = self._class_idx_for(key)
            class_to_embedding[class_idx] = emb
            class_idx_to_key[class_idx] = key

        masks_by_class = self._predict_joint(image_pil, class_to_embedding, threshold)
        return {class_idx_to_key[c]: m for c, m in masks_by_class.items()}
