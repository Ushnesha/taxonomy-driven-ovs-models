import difflib
import os

files = [
    "scratch/testing_clipSeg_extracted.py",
    "scratch/testing_clip_vit_large_extracted.py",
    "scratch/testing_openSeg_extracted.py",
    "scratch/testing_ovseg_extracted.py",
    "scratch/testing_sigLip_extracted.py",
    "scratch/testing_san_extracted.py",
    "scratch/testing_cocodataset_extracted.py"
]

def show_diff(file1, file2):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        l1 = f1.readlines()
        l2 = f2.readlines()
    
    diff = list(difflib.unified_diff(l1, l2, fromfile=file1, tofile=file2, n=1))
    print(f"Diff between {file1} and {file2}: {len(diff)} lines of diff.")
    # Show first 20 lines of diff
    for line in diff[:30]:
        print(line, end="")
    print("\n" + "="*50)

# compare clipSeg with others
for f in files[1:]:
    show_diff("scratch/testing_clipSeg_extracted.py", f)
