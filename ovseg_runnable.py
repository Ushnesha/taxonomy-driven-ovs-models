"""
COMPLETE, RUNNABLE builders + predict_fns for open-vocab segmentation on COCO-80.

HOW TO USE
----------
Each model lives in its own repo and (usually) its own conda env. You cannot
import all of them in one process. Pick a model, set up its env (commands in
each section), then:

    python this_file.py --model ovseg \
        --weights /path/ovseg_swinbase_vitL14_ft_mpt.pth \
        --config  configs/ovseg_swinB_vitL_demo.yaml \
        --image   sample.jpg

Every predict_fn returns the SAME thing:  {cat_id: np.uint8 [H, W] mask}.
Plug it into your get_segmentation_masks(...) host, then into the shared mIoU
evaluator so all models are scored identically.

VERIFY: Detectron2 repo APIs drift between commits. The 1-2 spots most worth a
quick check are flagged with `# VERIFY` comments.
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# The 80 COCO "thing" classes, in COCO category order (ids 1..90 with gaps).
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


# =====================================================================
# Shared helpers
# =====================================================================
def _ordered_cat_ids(text_cats, coco):
    """Returns cat_ids aligned with the order of text_cats. `coco` is a pycocotools
    COCO object OR a {name: id} dict (handy for standalone testing)."""
    if isinstance(coco, dict):
        return [coco[name] for name in text_cats]
    return [coco.getCatIds(catNms=[name])[0] for name in text_cats]


def sem_seg_to_pred_masks(sem_seg, cat_ids, threshold, is_logits=True):
    """sem_seg: torch [C, H, W] aligned with cat_ids. Argmax over classes, keep a
    pixel only if winning confidence > threshold. Returns {cat_id: uint8 [H,W]}."""
    probs = sem_seg.softmax(0) if is_logits else sem_seg
    conf, label = probs.max(0)
    label, conf = label.cpu().numpy(), conf.cpu().numpy()
    return {cid: ((label == i) & (conf > threshold)).astype(np.uint8)
            for i, cid in enumerate(cat_ids)}


# =====================================================================
# Detectron2 vocabulary registration (used by CAT-Seg and FC-CLIP)
# =====================================================================
def register_coco80_metadata(name="coco80_ov", class_names=COCO80):
    """FC-CLIP (and other Detectron2 OV models) build the text classifier from the
    class names stored in dataset metadata at model-build time. Register them here
    BEFORE constructing the model/predictor."""
    from detectron2.data import DatasetCatalog, MetadataCatalog
    if name in MetadataCatalog.list():
        MetadataCatalog.remove(name)
    DatasetCatalog.register(name, lambda: [])          # empty; inference-only
    md = MetadataCatalog.get(name)
    md.set(
        stuff_classes=list(class_names),               # FC-CLIP semantic path
        thing_classes=list(class_names),
        stuff_dataset_id_to_contiguous_id={i: i for i in range(len(class_names))},
        thing_dataset_id_to_contiguous_id={i: i for i in range(len(class_names))},
        ignore_label=255,
    )
    return md


def write_catseg_vocab_json(path, class_names=COCO80):
    """CAT-Seg reads its test vocabulary from a JSON list of class names, pointed to
    by MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON. Write your 80 names there."""
    with open(path, "w") as f:
        json.dump(list(class_names), f)
    return path


# =====================================================================
# 1. CLIPSeg  (HuggingFace)  -- multi-label sigmoid
#    pip install transformers torch pillow
# =====================================================================
def build_clipseg():
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = (CLIPSegForImageSegmentation
             .from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval())
    return model, proc


def make_clipseg_predict(model, processor):
    def predict(image_pil, text_cats, coco, threshold):
        cat_ids = _ordered_cat_ids(text_cats, coco)
        prompts = [f"a photo of a {v}" for v in text_cats.values()]
        W, H = image_pil.size
        inputs = processor(text=prompts, images=[image_pil] * len(prompts),
                           padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits)
        probs = F.interpolate(probs.unsqueeze(1), size=(H, W),
                              mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
        return {cid: (probs[i] > threshold).astype(np.uint8)
                for i, cid in enumerate(cat_ids)}
    return predict


# =====================================================================
# 2. OVSeg  (facebookresearch/ov-seg)  -- mask-adapted CLIP
#    git clone https://github.com/facebookresearch/ov-seg
#    install detectron2; pip install -r requirements.txt
#    checkpoint: ovseg_swinbase_vitL14_ft_mpt.pth   (Swin-B + CLIP ViT-L/14)
#    config:     configs/ovseg_swinB_vitL_demo.yaml
# =====================================================================
def build_ovseg(config_file, weights):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from open_vocab_seg import add_ovseg_config
    from open_vocab_seg.utils import VisualizationDemo
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHTS", weights])
    cfg.freeze()
    return VisualizationDemo(cfg)


def make_ovseg_predict(demo):
    def predict(image_pil, text_cats, coco, threshold):
        cat_ids = _ordered_cat_ids(text_cats, coco)
        class_names = list(text_cats.values())
        img_bgr = np.array(image_pil)[:, :, ::-1]                  # RGB -> BGR
        predictions, _ = demo.run_on_image(img_bgr, class_names)   # VERIFY signature
        sem_seg = predictions["sem_seg"]                           # [C, H, W] logits
        return sem_seg_to_pred_masks(sem_seg, cat_ids, threshold)
    return predict


# =====================================================================
# 3. OpenSeg  (Google Research, TF SavedModel)  -- dense feats + CLIP text emb
#    Get the exported SavedModel from the openseg/ README in google-research.
#    pip install tensorflow open_clip_torch
# =====================================================================
def build_openseg(saved_model_dir):
    import tensorflow.compat.v2 as tf
    return tf.saved_model.load(saved_model_dir)


def build_clip_text_fn(model_name="ViT-L-14", pretrained="openai"):
    """Returns fn(list_of_names) -> np.float32 [C, 768] normalized text embeddings."""
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained)
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    def text_fn(names):
        toks = tokenizer([f"a photo of a {n}" for n in names]).to(device)
        with torch.no_grad():
            t = clip_model.encode_text(toks)
            t = F.normalize(t, dim=-1)
        return t.cpu().numpy().astype(np.float32)
    return text_fn


def make_openseg_predict(openseg_model, clip_text_fn):
    import io
    import tensorflow.compat.v2 as tf

    def predict(image_pil, text_cats, coco, threshold):
        cat_ids = _ordered_cat_ids(text_cats, coco)
        W, H = image_pil.size
        buf = io.BytesIO(); image_pil.convert("RGB").save(buf, format="JPEG")
        out = openseg_model.signatures["serving_default"](
            inp_image_bytes=tf.convert_to_tensor(buf.getvalue()),
            inp_text_emb=tf.zeros([1, 1, 768], tf.float32))
        info = out["image_info"]
        ch, cw = int(info[0, 0] * info[2, 0]), int(info[0, 1] * info[2, 1])
        feat = out["ppixel_ave_feat"][:, :ch, :cw]
        feat = tf.image.resize(feat, (H, W), method="nearest")[0].numpy()   # [H,W,D]
        feat /= (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8)
        class_emb = clip_text_fn(list(text_cats.values()))                  # [C,D]
        sims = torch.from_numpy(feat @ class_emb.T).permute(2, 0, 1)        # [C,H,W]
        return sem_seg_to_pred_masks(sims, cat_ids, threshold, is_logits=False)
    return predict


# =====================================================================
# 4. SAN  (MendelXu/SAN)  -- side adapter network
#    git clone https://github.com/MendelXu/SAN ; bash install.sh
#    checkpoint: huggingface.co/Mendel192/san  ->  san_vit_large_14.pth
#    config:     configs/san_clip_vit_large_res4_coco.yaml
# =====================================================================
def build_san(config_file, model_path):
    from predict import Predictor          # SAN repo root has predict.py
    return Predictor(config_file=config_file, model_path=model_path)


def make_san_predict(predictor):
    def predict(image_pil, text_cats, coco, threshold):
        cat_ids = _ordered_cat_ids(text_cats, coco)
        vocab = list(text_cats.values())
        result = predictor.predict(np.array(image_pil), vocabulary=vocab,
                                   augment_vocabulary=False)     # VERIFY kwargs
        sem_seg = result["sem_seg"]                              # VERIFY key -> [C,H,W]
        if not torch.is_tensor(sem_seg):
            sem_seg = torch.as_tensor(np.asarray(sem_seg))
        return sem_seg_to_pred_masks(sem_seg, cat_ids, threshold)
    return predict


# =====================================================================
# 5. CAT-Seg  (cvlab-kaist/CAT-Seg)  -- cost aggregation (slide-mode inference)
#    git clone https://github.com/cvlab-kaist/CAT-Seg ; install detectron2
#    checkpoint: model_final.pth from their model zoo (ViT-L: vitl_swinb_384)
#    config:     configs/vitl_swinb_384.yaml
# =====================================================================
def build_catseg(config_file, weights, vocab_json="coco80_catseg.json"):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from cat_seg import add_cat_seg_config
    from demo.predictor import VisualizationDemo     # cat-seg/demo/predictor.py
    write_catseg_vocab_json(vocab_json, COCO80)
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list([
        "MODEL.WEIGHTS", weights,
        "MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON", vocab_json,        # VERIFY key name
    ])
    cfg.freeze()
    return VisualizationDemo(cfg)


def make_catseg_predict(demo):
    def predict(image_pil, text_cats, coco, threshold):
        cat_ids = _ordered_cat_ids(text_cats, coco)             # order must match json
        img_bgr = np.array(image_pil)[:, :, ::-1]
        predictions, _ = demo.run_on_image(img_bgr)
        return sem_seg_to_pred_masks(predictions["sem_seg"], cat_ids, threshold)
    return predict


# =====================================================================
# 6. FC-CLIP  (bytedance/fc-clip)  -- frozen ConvNeXt-L CLIP, panoptic-capable
#    git clone https://github.com/bytedance/fc-clip ; install detectron2
#    checkpoint: fcclip_cocopan.pth
#    config:     configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml
# =====================================================================
def build_fcclip(config_file, weights, dataset_name="coco80_ov"):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.engine import DefaultPredictor
    from fcclip import add_maskformer2_config, add_fcclip_config
    register_coco80_metadata(dataset_name, COCO80)               # MUST precede build
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fcclip_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list([
        "MODEL.WEIGHTS", weights,
        "DATASETS.TEST", (dataset_name,),                        # vocab source
    ])
    cfg.freeze()
    return DefaultPredictor(cfg)


def make_fcclip_predict(predictor):
    def predict(image_pil, text_cats, coco, threshold):
        cat_ids = _ordered_cat_ids(text_cats, coco)
        img_bgr = np.array(image_pil)[:, :, ::-1]
        out = predictor(img_bgr)
        sem_seg = out["sem_seg"]                                 # [C, H, W] over vocab
        return sem_seg_to_pred_masks(sem_seg, cat_ids, threshold)
    return predict


# =====================================================================
# 7. SAM + CLIP  -- class-agnostic masks classified by CLIP
#    pip install git+https://github.com/facebookresearch/segment-anything
#    checkpoint: sam_vit_h_4b8939.pth
#      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# =====================================================================
def build_sam(checkpoint, model_type="vit_h"):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return SamAutomaticMaskGenerator(sam)


def build_clip():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, proc


def make_sam_clip_predict(mask_generator, clip_model, clip_processor):
    def predict(image_pil, text_cats, coco, threshold):
        cat_ids = _ordered_cat_ids(text_cats, coco)
        names = list(text_cats.values())
        W, H = image_pil.size
        img_np = np.array(image_pil.convert("RGB"))
        toks = clip_processor(text=[f"a photo of a {n}" for n in names], images=None,
                              return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            txt_feats = clip_model.get_text_features(**toks)
            if hasattr(txt_feats, "pooler_output"):
                txt_feats = txt_feats.pooler_output
            txt = F.normalize(txt_feats, dim=-1)
        pred = {cid: np.zeros((H, W), np.uint8) for cid in cat_ids}
        for m in mask_generator.generate(img_np):
            seg = m["segmentation"]
            ys, xs = np.where(seg)
            if len(xs) == 0:
                continue
            crop = image_pil.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            ci = clip_processor(images=crop, return_tensors="pt").to(device)
            with torch.no_grad():
                im_feats = clip_model.get_image_features(**ci)
                if hasattr(im_feats, "pooler_output"):
                    im_feats = im_feats.pooler_output
                im = F.normalize(im_feats, dim=-1)
            conf, idx = (im @ txt.T)[0].max(0)
            if conf.item() > threshold:
                pred[cat_ids[idx.item()]][seg] = 1
        return pred
    return predict


# =====================================================================
# Runnable dispatcher: builds ONE model and runs its predict_fn on an image.
# =====================================================================
def build_predict_fn(args):
    if args.model == "clipseg":
        return make_clipseg_predict(*build_clipseg())
    if args.model == "ovseg":
        return make_ovseg_predict(build_ovseg(args.config, args.weights))
    if args.model == "openseg":
        return make_openseg_predict(build_openseg(args.weights), build_clip_text_fn())
    if args.model == "san":
        return make_san_predict(build_san(args.config, args.weights))
    if args.model == "catseg":
        return make_catseg_predict(build_catseg(args.config, args.weights))
    if args.model == "fcclip":
        return make_fcclip_predict(build_fcclip(args.config, args.weights))
    if args.model == "sam":
        return make_sam_clip_predict(build_sam(args.weights), *build_clip())
    raise ValueError(args.model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["clipseg", "ovseg", "openseg", "san", "catseg",
                             "fcclip", "sam"])
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", default=None, help="config file (Detectron2 models)")
    ap.add_argument("--weights", default=None, help="checkpoint / SavedModel dir")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    predict_fn = build_predict_fn(args)

    image = Image.open(args.image).convert("RGB")
    # text_cats: {coco_base_name: prompt_word}. Use a few classes for a quick test.
    text_cats = {n: n for n in ["person", "dog", "car", "chair", "bottle"]}
    # Standalone id map (replace with your real pycocotools `coco` object for eval):
    coco = {n: i for i, n in enumerate(COCO80)}

    pred_masks = predict_fn(image, text_cats, coco, args.threshold)
    for name in text_cats:
        cid = coco[name]
        print(f"{name:12s} (cat_id={cid}): {int(pred_masks[cid].sum())} px")
