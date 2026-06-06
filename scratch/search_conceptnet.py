import json

with open("testing_cocodataset.ipynb", "r") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "conceptnet" in source:
        print(f"Cell {idx}:")
        print(source[:500])
        print("="*40)
