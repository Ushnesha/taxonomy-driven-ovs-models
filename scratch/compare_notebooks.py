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

for filepath in files:
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        continue
    
    print("=" * 80)
    print(f"FILE: {filepath}")
    print("=" * 80)
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    lines = content.splitlines()
    
    # Let's print the first 30 lines (imports & setup)
    print("--- SETUP / IMPORTS / MODEL LOADING ---")
    for i, line in enumerate(lines[:35]):
        print(f"{i+1}: {line}")
        
    # Find functions defined
    print("--- DEFINED FUNCTIONS ---")
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            print(f"Line {i+1}: {line.strip()}")
            
    # Find any model instantiation
    print("--- PRETRAINED / MODEL INSTANTIATION ---")
    for i, line in enumerate(lines):
        if "from_pretrained" in line or "CLIPSeg" in line or "open_clip" in line or "CLIP" in line or "Siglip" in line or "OV-Seg" in line or "SAN" in line or "seg" in line.lower():
            if "=" in line and ("model" in line or "processor" in line or "tokenizer" in line):
                print(f"Line {i+1}: {line.strip()}")
                
    # Let's print the end of the file (usually the evaluation loop / main block)
    print("--- END OF FILE (LAST 20 LINES) ---")
    for i, line in enumerate(lines[-20:]):
        print(f"{len(lines) - 20 + i + 1}: {line}")
    print("\n\n")
