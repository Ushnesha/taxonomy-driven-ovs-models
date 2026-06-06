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
    print("=" * 80)
    print(f"FILE: {fp}")
    print("=" * 80)
    
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Find functions: def get_segmentation_masks or any function containing predict or forward or inference
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if "def get_segmentation_masks" in line or "def visualize_segmentation_with_labels" in line:
            print(f"Line {idx+1}: {line}")
            # Print the next 50 lines of this function to see its implementation
            for j in range(idx + 1, min(idx + 60, len(lines))):
                # if we encounter another def or class, stop printing
                if lines[j].strip().startswith("def ") or lines[j].strip().startswith("class "):
                    break
                print(f"  {j+1}: {lines[j]}")
            print("-" * 50)
            
    # Also find if there is another prediction function or script cells that run inference
    # Let's search for "with torch.no_grad():" or "with torch.inference_mode():"
    print("Inference / Prediction lines:")
    for idx, line in enumerate(lines):
        if "with torch.inference_mode():" in line or "with torch.no_grad():" in line:
            print(f"Line {idx+1}: {line}")
            for j in range(max(0, idx - 2), min(len(lines), idx + 8)):
                print(f"  {j+1}: {lines[j]}")
            print()
