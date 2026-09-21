"""Is this .pth actually the CAT-Seg checkpoint the config expects?

Downloading the wrong variant (model_base.pth for a vitl_336 config, or some other
repo's .pth entirely) does NOT fail loudly: Detectron2's checkpointer skips keys it
cannot match and carries on with RANDOM weights for those layers. The run then
produces plausible-looking-but-meaningless masks. This script inspects the file
before any of that, with no GPU and no detectron2.

It reports:
  * file size and the top-level structure of the checkpoint
  * the CLIP visual backbone width inferred from the weights (768 => ViT-B/16,
    1024 => ViT-L/14), so you can confirm it matches the config you intend to use
  * whether CAT-Seg's own head tensors (sem_seg_head / cost aggregation) are present,
    which is what distinguishes a real CAT-Seg checkpoint from a bare CLIP dump

Usage:
    python3 check_checkpoint.py ~/workspace/OVS/checkpoints/catseg_model_large.pth
    python3 check_checkpoint.py <path> --config ./CAT-Seg/configs/vitl_336.yaml
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--config", default=None,
                    help="optional: the CAT-Seg yaml you plan to pair it with, so the "
                         "backbone width can be cross-checked against the config name")
    ap.add_argument("--show", type=int, default=8, help="sample keys to print")
    args = ap.parse_args()

    path = os.path.expanduser(args.weights)
    if not os.path.isfile(path):
        sys.exit(f"ABORT: no such file: {path}")

    size_gb = os.path.getsize(path) / 1024**3
    print(f"file      : {path}")
    print(f"size      : {size_gb:.2f} GB")
    # A truncated/failed wget is the single most common cause of a "corrupt checkpoint".
    if size_gb < 0.1:
        print("  !! far too small for a CAT-Seg checkpoint -- almost certainly a failed")
        print("     download (an HTML error page saved as .pth). Re-download.")

    import torch
    print("loading (map_location='cpu') ...", flush=True)
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        sys.exit(f"ABORT: torch.load failed -- file is corrupt or not a checkpoint: {e!r}")

    if isinstance(ckpt, dict):
        print(f"top-level : dict with {len(ckpt)} keys: {list(ckpt)[:6]}")
    sd = ckpt
    for wrapper in ("model", "state_dict"):
        if isinstance(sd, dict) and wrapper in sd and isinstance(sd[wrapper], dict):
            print(f"  unwrapping ['{wrapper}']")
            sd = sd[wrapper]
    if not isinstance(sd, dict):
        sys.exit(f"ABORT: expected a state dict, got {type(sd)}")

    keys = list(sd)
    print(f"tensors   : {len(keys)}")
    print(f"sample    : {keys[:args.show]}")

    # ---- CLIP backbone width: 768 = ViT-B/16, 1024 = ViT-L/14@336 ----
    width = None
    for k in keys:
        if k.endswith("positional_embedding") and hasattr(sd[k], "shape") and sd[k].dim() == 2:
            width = sd[k].shape[-1]
            print(f"\nCLIP text width inferred from '{k}': {width}")
            break
    variant = {512: "ViT-B/32", 768: "ViT-L/14 (text 768)", 1024: "ViT-L/14@336 (text 768/vis 1024)"}
    if width:
        print(f"  => looks like {variant.get(width, f'unknown (width {width})')}")

    # ---- is CAT-Seg's own trained head here, or just a CLIP dump? ----
    head = [k for k in keys if "sem_seg_head" in k]
    agg = [k for k in keys if "aggregator" in k or "cost" in k.lower()]
    print(f"\nsem_seg_head tensors     : {len(head)}")
    print(f"cost-aggregation tensors : {len(agg)}")
    if not head:
        print("  !! NO sem_seg_head weights. This is NOT a CAT-Seg checkpoint -- probably")
        print("     a plain CLIP release. Detectron2 would randomly initialise the whole")
        print("     segmentation head and your masks would be noise.")
    else:
        print("  OK: CAT-Seg's trained head is present.")

    if args.config:
        cfg = os.path.basename(args.config)
        print(f"\nconfig    : {cfg}")
        if "vitl" in cfg and width and width < 768:
            print("  !! MISMATCH: config is vit-L but the checkpoint looks like vit-B.")
        elif "vitb" in cfg and width and width >= 1024:
            print("  !! MISMATCH: config is vit-B but the checkpoint looks like vit-L.")
        else:
            print("  no obvious size mismatch with the config name.")

    print("\nNOTE: the definitive check is the Detectron2 load log at model build time --")
    print("watch for 'Some model parameters or buffers are not found' / 'not used'. A")
    print("handful of lines is normal; dozens means the wrong checkpoint.")


if __name__ == "__main__":
    main()
