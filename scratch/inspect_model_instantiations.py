import re
import os

files = [
    "scratch/testing_clip_vit_large_extracted.py",
    "scratch/testing_clipSeg_extracted.py",
    "scratch/testing_openSeg_extracted.py",
    "scratch/testing_ovseg_extracted.py",
    "scratch/testing_sigLip_extracted.py",
    "scratch/testing_san_extracted.py",
    "scratch/testing_cocodataset_extracted.py"
]

for fp in files:
    if not os.path.exists(fp):
        continue
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()
    
    print(f"FILE: {fp} (lines: {len(content.splitlines())})")
    # find lines with model creation
    model_lines = []
    lines = content.splitlines()
    for idx, l in enumerate(lines):
        if any(keyword in l for keyword in ["AutoProcessor", "from_pretrained", "clip", "siglip", "ovseg", "open_clip", "san"]):
            if "=" in l and not l.strip().startswith("#"):
                model_lines.append((idx + 1, l.strip()))
    print("Model Instantiations:")
    for line_num, l in model_lines[:10]:
        print(f"  [{line_num}] {l}")
    print("-" * 40)
