"""
CAT-Seg (Cheng et al., CVPR 2024, "CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic
Segmentation") wrapped in the BaseOVSModel interface. Extends the draft Ushnesha shared
(openVocabSegmentation/catseg.py) into this folder's standalone layout, matching
clipseg.py/groupvit.py/sclip.py's structure and conventions.

Runs the real cvlab-kaist/CAT-Seg repo (org now published as KU-CVLAB/CAT-Seg) via
Detectron2 -- real pretrained CLIP (ViT-L/14@336px by default) + the paper's trained
cost-aggregation transformer on top, classifying every pixel via a joint softmax over the
model's text-embedding vocabulary. No architecture shortcuts: get_text_embedding() and
predict_with_embeddings() drive the model's own class_embeddings()/get_text_embeds() path
and its own CATSegHead forward pass, not a reimplementation.

=====================================================================================
IMPORTANT -- unlike clipseg.py/groupvit.py/sclip.py, PRODUCTION mode below has NOT been
run end-to-end against a real checkpoint: this project has no local CUDA/detectron2
(CLAUDE.md: CAT-Seg integration is Ushnesha's, on ASU's cluster). Every fact this file's
docstrings state about CAT-Seg's real internals (attribute names, tensor shapes, the
class_embeddings()/get_text_embeds()/self.cache mechanism, the default prompt template,
its L2-normalize convention) was confirmed by reading the actual source at
https://github.com/KU-CVLAB/CAT-Seg (cat_seg/cat_seg_model.py,
cat_seg/modeling/transformer/cat_seg_predictor.py, cat_seg/config.py, configs/vitl_336.yaml,
requirements.txt, INSTALL.md) as of this file's writing -- not guessed. What is NOT
verified is that this actually runs correctly end-to-end on a real GPU box. Before trusting
any catseg positive_set/negative_set numbers, a teammate with cluster access should run the
verification in this folder's README.md ("Verifying the injection hook before a real run")
and confirm predict_with_embeddings' swapped-in embedding actually changes the output mask
for a probe class vs. its un-swapped baseline.
=====================================================================================

BASELINE mode (config/weights omitted, or baseline=True): falls back to CLIPSegModel
(clipseg.py, sibling file) so this folder's scripts are exercisable end-to-end locally with
no detectron2/GPU -- useful for testing the pipeline plumbing (variant building, CSV
resume, hyper-sibling merge, etc.) ONLY. It does NOT produce CAT-Seg numbers; every call
prints a loud warning, and this folder's experiment scripts refuse to run at full scale in
baseline mode (see positive_set_experiment.py / negative_set_experiment.py / and
alpha_value_experiment.py's argparse --config/--weights requirement) so a real run can never
silently end up reporting CLIPSeg numbers under CAT-Seg's name.

PRODUCTION mode (config + weights given): real CAT-Seg via Detectron2. Needs the vendored
CAT-Seg repo checked out at ./CAT-Seg (see setup_env.sh) plus a checkpoint (CAT-Seg (B),
ViT-B/16, configs/vitb_384.yaml, or CAT-Seg (L), ViT-L/14@336px, configs/vitl_336.yaml --
official checkpoints linked from the CAT-Seg README's model zoo).

Text-embedding convention: CAT-Seg's own class_embeddings()/get_text_embeds() L2-normalize
every embedding right after CLIP's encode_text() (`emb / emb.norm(dim=-1, keepdim=True)`) --
like SCLIP, NOT like CLIPSeg's raw-norm FiLM convention. get_text_embedding() below
reproduces this exactly, and predict_with_embeddings() re-normalizes any blended embedding
before injecting it, for the same reason SCLIP's wrapper does (CLAUDE.md's "never
L2-normalize" rule is CLIPSeg-specific and does not apply here).

Architecture note (why predict_with_embeddings looks like SCLIP's, not CLIPSeg's): CAT-Seg
classifies every pixel via ONE joint softmax across its whole text-embedding vocabulary in
a single forward pass (cat_seg_predictor.py's forward() computes one cost-aggregation pass
over ALL classes at once) -- there is no per-prompt independent sigmoid the way CLIPSeg/
GroupViT have. So, exactly like sclip.py's `_predict_joint` (see that file's module
docstring), embeddings_dict's keys here must be real canonical category names present in
this model's FIXED vocabulary (the class_name_path JSON given at construction), not
arbitrary tags -- predict_with_embeddings swaps in ONE embedding at that class's row,
leaving every other real class's row (and thus the rest of the joint softmax) untouched,
runs ONE forward pass, extracts that class's mask, and restores the original row. This
folder's experiment scripts therefore call it once per (variant, approach) tag, all keyed
by the real cat_name, exactly like sclip/positive_set_experiment.py's
predict_masks_for_tags -- see that file's docstring for why (injecting all 5 approach tags
into one forward call at once would make them compete against only each other, not a real
background/vocabulary, silently inflating IoU).

The injection mechanism itself: CATSegPredictor.get_text_embeds() computes real embeddings
from self.test_class_texts via CLIP's encode_text() only on its FIRST eval-mode call, then
caches the result in `self.cache` and returns that cache UNCHANGED on every later call
(`if self.cache is not None and not self.training: return self.cache`) regardless of the
classnames passed in -- confirmed from source. So: this wrapper (a) warms that cache once
per instance with the real fixed vocabulary's real embeddings (CAT-Seg's own text tower,
no injection), then (b) overwrites one row of `self.cache` per predict_with_embeddings call,
exactly like SCLIP's `model.query_features[r]` row-swap. Detectron2's DefaultPredictor
(which VisualizationDemo wraps) calls `model.eval()` in its own __init__, so `self.training`
is False by construction -- this part is standard Detectron2 behavior, not CAT-Seg-specific
guesswork. The one unverified detail is the exact attribute path down to that predictor
instance (`self.demo.predictor.model.sem_seg_head.predictor` -- inferred from
cat_seg_model.py's `self.sem_seg_head.predictor.clip_model.encode_image(...)` call site);
if it's wrong the constructor's `_locate_predictor_module` will raise AttributeError with
the path it tried, immediately and loudly, rather than silently mis-injecting.
"""
import json
import os
import sys
import threading

import numpy as np
import torch
from PIL import Image

from base import BaseOVSModel
from clipseg import CLIPSegModel

# CATSegPredictor's own default (configs/*.yaml's PROMPT_ENSEMBLE_TYPE: "single") --
# confirmed from cat_seg/modeling/transformer/cat_seg_predictor.py. If --config sets a
# different PROMPT_ENSEMBLE_TYPE ("imagenet" / "imagenet_select"), this single template is
# no longer what the real class_embeddings() path used to warm the cache would use --
# get_text_embedding()'s single-template behavior would then diverge from the model's own
# convention for that config, and this constant should be updated to match.
DEFAULT_PROMPT_TEMPLATE = "A photo of a {} in the scene"

_CATSEG_DIR_DEFAULT = os.environ.get(
    "CATSEG_DIR",
    # Sibling ./CAT-Seg -- vendored (code only, no .git) from
    # https://github.com/KU-CVLAB/CAT-Seg so this folder is self-contained. See
    # setup_env.sh.
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "CAT-Seg"),
)


class CATSegModel(BaseOVSModel):
    def __init__(self, device=None, config=None, weights=None, baseline=None,
                 catseg_path=None, class_name_path=None):
        """
        If baseline=True (or config/weights missing), falls back to CLIPSeg -- see module
        docstring. Otherwise builds the real CAT-Seg predictor via Detectron2.

        class_name_path: path to a JSON list of class names (see
        benchmark_data.build_catseg_class_json) -- this model's FIXED vocabulary, i.e. the
        full set of real classes predict_with_embeddings can swap a row for. Required in
        production mode. Every category the experiment scripts will ever pass as an
        embeddings_dict key must appear in this list.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device

        self.baseline = baseline if baseline is not None else (config is None or weights is None)

        if self.baseline:
            print("=" * 78)
            print("[CATSegModel] BASELINE MODE -- using CLIPSeg as a stand-in. This is NOT")
            print("  CAT-Seg. Only for exercising this folder's pipeline without a real")
            print("  checkpoint/detectron2 -- never trust these numbers as CAT-Seg results.")
            print("=" * 78)
            self.clipseg = CLIPSegModel(device=self.device)
            self._logged_cache_shape = True   # nothing to log: no CAT-Seg cache in baseline mode
            return

        if not class_name_path:
            raise ValueError(
                "CATSegModel production mode requires class_name_path (a JSON list of "
                "every real category this model will be asked to predict -- see "
                "benchmark_data.build_catseg_class_json)."
            )

        catseg_path = os.path.abspath(catseg_path or _CATSEG_DIR_DEFAULT)
        if catseg_path not in sys.path:
            sys.path.insert(0, catseg_path)

        try:
            from detectron2.config import get_cfg
            from detectron2.projects.deeplab import add_deeplab_config
            from cat_seg import add_cat_seg_config
            from cat_seg.third_party import clip as cat_seg_clip
            from demo.predictor import VisualizationDemo  # CAT-Seg repo's demo/predictor.py
        except ImportError as e:
            raise ImportError(
                "CATSegModel production mode requires detectron2 + a checkout of "
                "https://github.com/KU-CVLAB/CAT-Seg at catseg_path (default ./CAT-Seg) -- "
                "run setup_env.sh in this folder first."
            ) from e
        self._clip_module = cat_seg_clip

        with open(class_name_path) as f:
            self._vocab_names = json.load(f)
        self._name_to_idx = {name.lower(): i for i, name in enumerate(self._vocab_names)}

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_cat_seg_config(cfg)
        cfg.merge_from_file(config)
        cfg.merge_from_list([
            "MODEL.WEIGHTS", weights,
            "MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON", class_name_path,
            "MODEL.DEVICE", self.device,
        ])
        cfg.freeze()

        print(f"[CATSegModel] Building real CAT-Seg (config={config}, weights={weights}, "
              f"{len(self._vocab_names)}-class vocabulary)...", flush=True)
        self.demo = VisualizationDemo(cfg)
        self._predictor_module = self._locate_predictor_module()
        self._lock = threading.Lock()  # self._predictor_module.cache is shared mutable state
        self._logged_cache_shape = False  # set on the first _warm_cache()

    def _locate_predictor_module(self):
        """The CATSegPredictor instance holding .cache/.clip_model/.tokenizer/
        .prompt_templates/.test_class_texts -- see module docstring for how this path was
        inferred and why it's the one part of this file that needs on-cluster confirmation.
        Fails loudly (not silently) if the real object graph doesn't match."""
        obj = self.demo.predictor.model
        path = "self.demo.predictor.model"
        for attr in ("sem_seg_head", "predictor"):
            if not hasattr(obj, attr):
                raise AttributeError(
                    f"CATSegModel._locate_predictor_module: expected {path}.{attr} to exist "
                    f"but it doesn't -- CAT-Seg's internal module structure may have "
                    f"changed since this wrapper was written against "
                    f"https://github.com/KU-CVLAB/CAT-Seg. Inspect {path} directly."
                )
            obj = getattr(obj, attr)
            path += f".{attr}"
        return obj

    def _class_idx_for(self, canonical_name):
        idx = self._name_to_idx.get(canonical_name.lower())
        if idx is None:
            raise KeyError(
                f"'{canonical_name}' is not in this CATSegModel's vocabulary "
                f"({len(self._vocab_names)} classes from class_name_path)."
            )
        return idx

    def get_text_embedding(self, word, desc=False):
        """
        L2-normalized text embedding for `word` -- CAT-Seg's own convention (see module
        docstring). `desc=True` embeds `word` verbatim (already a full prompt/sentence);
        `desc=False` wraps it in DEFAULT_PROMPT_TEMPLATE, matching
        CATSegPredictor.class_embeddings()'s single-template default. Shape [1, D].
        """
        if self.baseline:
            return self.clipseg.get_text_embedding(word, desc=desc)

        pred = self._predictor_module
        text = word if desc else DEFAULT_PROMPT_TEMPLATE.format(word)
        tok_fn = pred.tokenizer if pred.tokenizer is not None else self._clip_module.tokenize
        # truncate=True: CLIP's text encoder has a hard 77-token limit. SHiNe's
        # "a X, which is a Y, which is a Z, ..." sentences, built from WordNet hypernym
        # paths, routinely exceed it, and OpenAI CLIP's tokenize() defaults to
        # truncate=False -- which RAISES rather than truncating. This killed the CLIPSeg
        # and SCLIP runs partway through their first real dry run; do not remove it.
        # Guarded because a HF-style tokenizer (pred.tokenizer) takes different kwargs.
        try:
            tokens = tok_fn([text], truncate=True)
        except TypeError:
            tokens = tok_fn([text], truncation=True, max_length=77)
        if hasattr(tokens, "to"):
            tokens = tokens.to(self.device)
        with torch.no_grad():
            emb = pred.clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float().cpu()

    def _warm_cache(self):
        """CATSegPredictor.get_text_embeds() only computes real embeddings from
        self.test_class_texts on its first eval-mode call, then caches them in self.cache
        and returns that cache UNCHANGED (ignoring its arguments) on every later call --
        see module docstring. This forces that first real call once per model instance, so
        predict_with_embeddings' row-swaps below always start from real per-class
        embeddings for every OTHER class in the vocabulary."""
        pred = self._predictor_module
        if pred.cache is None:
            with torch.no_grad():
                pred.get_text_embeds(pred.test_class_texts, pred.prompt_templates, pred.clip_model)
        if pred.cache is None:
            raise RuntimeError(
                "CATSegModel._warm_cache: predictor_module.cache is still None after "
                "calling get_text_embeds() -- CAT-Seg's caching behavior may have changed; "
                "see this file's module docstring before trusting predict_with_embeddings."
            )

        # The cache is [num_classes, num_templates, dim]. num_templates depends on
        # cfg.MODEL.PROMPT_ENSEMBLE_TYPE: 1 for "single", but ~80 for "imagenet" /
        # "imagenet_select". That matters enormously -- see _swap_row below -- so assert
        # the rank once, loudly, instead of letting a shape assumption fail silently.
        if pred.cache.dim() != 3:
            raise RuntimeError(
                f"CATSegModel: expected predictor cache of rank 3 "
                f"[num_classes, num_templates, dim], got shape {tuple(pred.cache.shape)}. "
                f"predict_with_embeddings' row-swap indexing assumes rank 3 -- inspect "
                f"CAT-Seg's CATSegPredictor.get_text_embeds() before trusting any numbers."
            )
        if not self._logged_cache_shape:
            n_cls, n_tpl, dim = pred.cache.shape
            print(f"[CATSegModel] text-embedding cache: {n_cls} classes x {n_tpl} "
                  f"prompt template(s) x {dim} dims", flush=True)
            if n_tpl > 1:
                print(f"[CATSegModel] NOTE: {n_tpl} prompt templates are ensembled. "
                      f"predict_with_embeddings overwrites ALL {n_tpl} template rows for "
                      f"the target class with the single injected embedding, so the "
                      f"injection is not diluted. get_text_embedding() correspondingly "
                      f"uses one template (DEFAULT_PROMPT_TEMPLATE), which means an "
                      f"injected 'baseline' embedding is NOT bit-identical to the model's "
                      f"own ensembled row for that class -- expected, and consistent "
                      f"across all 5 approaches, so comparisons remain fair.", flush=True)
            self._logged_cache_shape = True

    def predict_with_embeddings(self, image_pil, embeddings_dict, threshold=0.5):
        """
        embeddings_dict: {canonical_category_name: torch.Tensor}, keyed by the real class
        name it should replace in this model's fixed vocabulary -- see module docstring for
        why (CAT-Seg's joint softmax, like SCLIP's). Each key is looked up via
        _class_idx_for, its cache row swapped in, ONE real forward pass run, that class's
        mask extracted (argmax == this class AND softmax confidence > threshold, matching
        sem_seg_to_pred_masks' convention elsewhere in this project), then the original row
        restored -- every other class's row (and thus the rest of the softmax) is untouched
        during that pass, exactly as in normal CAT-Seg inference.
        """
        if self.baseline:
            return self.clipseg.predict_with_embeddings(image_pil, embeddings_dict, threshold=threshold)

        if isinstance(image_pil, np.ndarray):
            image_pil = Image.fromarray(image_pil)
        img_bgr = np.array(image_pil.convert("RGB"))[:, :, ::-1]

        self._warm_cache()
        pred = self._predictor_module

        pred_masks = {}
        with self._lock:
            for key, emb in embeddings_dict.items():
                idx = self._class_idx_for(key)
                e = emb.to(self.device).float().squeeze()
                if e.dim() != 1:
                    raise ValueError(f"expected a 1-D embedding for '{key}', got shape {tuple(e.shape)}")
                e = e / e.norm()

                # Overwrite EVERY prompt-template row for this class, not just row 0.
                # CAT-Seg's cost aggregation consumes all num_templates rows; with
                # PROMPT_ENSEMBLE_TYPE="imagenet" that is ~80 rows, so writing only
                # [idx, 0, :] would leave ~79/80 of the class's signal as the model's
                # ORIGINAL embedding. The alpha sweep would then produce a nearly flat
                # IoU-vs-alpha curve and look like "blending does nothing" -- a silent
                # false negative, not a crash. This is the CLIPSeg "person" bug
                # (8 sub-prompts, one embedding) in a different guise.
                original_rows = pred.cache[idx].clone()          # [num_templates, dim]
                try:
                    pred.cache[idx] = e.unsqueeze(0).expand_as(original_rows)
                    # predictor(), not demo.run_on_image(): the latter also renders a
                    # visualization we immediately discard, on every one of the
                    # ~20 forward passes per image.
                    predictions = self.demo.predictor(img_bgr)
                    sem_seg = predictions["sem_seg"]
                    if not torch.is_tensor(sem_seg):
                        sem_seg = torch.as_tensor(np.asarray(sem_seg))
                    probs = sem_seg.softmax(0)
                    conf, label = probs.max(0)
                    label_np, conf_np = label.cpu().numpy(), conf.cpu().numpy()
                    pred_masks[key] = ((label_np == idx) & (conf_np > threshold)).astype(np.uint8)
                finally:
                    pred.cache[idx] = original_rows

        return pred_masks
