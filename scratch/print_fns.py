import os

files_to_print = [
    ("scratch/testing_clip_vit_large_extracted.py", "get_segmentation_masks"),
    ("scratch/testing_openSeg_extracted.py", "get_segmentation_masks"),
    ("scratch/testing_sigLip_extracted.py", "get_segmentation_masks")
]

for filepath, fn_name in files_to_print:
    if not os.path.exists(filepath):
        continue
    print("=" * 80)
    print(f"FILE: {filepath} -> FUNCTION: {fn_name}")
    print("=" * 80)
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    found = False
    for idx, line in enumerate(lines):
        if f"def {fn_name}" in line:
            found = True
            # print up to 100 lines of the function
            for j in range(idx, min(idx + 150, len(lines))):
                # if we hit another def that is at same indentation level (except the one we started with), stop
                if j > idx and lines[j].startswith("def "):
                    break
                print(f"{j+1}: {lines[j]}", end="")
            break
    if not found:
        print(f"Function {fn_name} not found.")
    print("\n" + "="*80)
