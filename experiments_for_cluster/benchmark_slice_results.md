# Benchmark Slice Results

All numbers below come from small random slices of the benchmark (~4 categories, 1-2 images each, out of a dataset with tens of thousands of category/image pairs). **Do not treat any of this as final — it is only a sanity check that the scripts work end to end and point in the right direction before the full-dataset run.**

**positive_set_experiment.py** — blending clearly rescues degraded linguistic-variant queries, e.g. COCO "airplane" queried as hypernym "heavier-than-air craft": baseline IoU 0.0 to blended 0.86.

**negative_set_experiment.py** — false-positive rate stayed ~0.0 for baseline and blended alike; the one nonzero case (ADE20K "adding machine") dropped to 0.0 after blending. No sign of blending causing hallucinated masks.

**shine_experiment.py / waffleclip_experiment.py / llm_description_experiment.py / prompt_template_experiment.py** — each ran without errors across all 4 datasets, producing plausible IoU values for both the baseline and the alternative-method embedding. Sample too small to compare methods yet.
