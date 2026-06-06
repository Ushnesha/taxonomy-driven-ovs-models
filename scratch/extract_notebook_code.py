import json
import os
import glob

notebooks = [
    "testing_clip_vit_large.ipynb",
    "testing_clipSeg.ipynb",
    "testing_openSeg.ipynb",
    "testing_cocodataset.ipynb",
    "testing_ovseg.ipynb",
    "testing_sigLip.ipynb",
    "testing_san.ipynb"
]

os.makedirs("scratch", exist_ok=True)

for nb in notebooks:
    if not os.path.exists(nb):
        print(f"Skipping {nb} as it does not exist.")
        continue
    
    with open(nb, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading {nb}: {e}")
            continue
            
    code_lines = []
    cell_idx = 1
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            # source can be a list of lines or a single string
            if isinstance(source, list):
                source_code = "".join(source)
            else:
                source_code = str(source)
            
            # Skip pip installs or empty lines to keep it clean
            clean_lines = []
            for line in source_code.splitlines():
                if line.strip().startswith("%pip") or line.strip().startswith("!pip"):
                    continue
                clean_lines.append(line)
            
            if clean_lines:
                code_lines.append(f"# --- Cell {cell_idx} ---")
                code_lines.extend(clean_lines)
                code_lines.append("")
                cell_idx += 1
                
    out_name = os.path.join("scratch", nb.replace(".ipynb", "_extracted.py"))
    with open(out_name, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(code_lines))
    print(f"Extracted {len(code_lines)} lines from {nb} to {out_name}")
