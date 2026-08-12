# Scripts Description

**experiment_common.py**
Shared library for all experiment scripts: builds the positive set for each of the 4 datasets (COCO, LVIS, ADE20K, Pascal VOC) via the exact benchmark methodology, fetches (image, ground-truth mask) pairs, builds all 4 linguistic variants (orig/syn/hyper/hypo) per category, computes unweighted and cosine-weighted centroid blends, builds negative sets from positive sets, runs batched CLIPSeg inference, and provides a resumable/checkpointed CSV writer plus a generic runner for comparing a baseline embedding against any alternative query-transform method.

**positive_set_experiment.py**
For every eligible category and every one of its positive images (all 4 datasets, full scale, no sampling caps), across all 4 linguistic variants, measures IoU for the plain query, the alpha-blended embedding, and the weighted-centroid-blended embedding. Checkpoints incrementally so it can resume on interruption.

**negative_set_experiment.py**
Same variants/methods as positive_set_experiment.py but run on the full negative set (every image without the category, full scale). Reports false-positive rate (fraction of image pixels activated) instead of IoU, since IoU against an empty ground truth is always 0 by construction.

**shine_experiment.py**
Simplified reimplementation of SHiNe (Liu et al., CVPR 2024): builds a hierarchy-anchored prompt ("a photo of a {word}, a type of {hypernym}") instead of blending embeddings, compared against the bare-word baseline on the full positive set.

**waffleclip_experiment.py**
Reimplementation of WaffleCLIP (Roth et al., ICCV 2023): averages the embeddings of the word combined with 5 deterministic random-character suffixes, compared against the bare-word baseline on the full positive set.

**llm_description_experiment.py**
Reimplementation of Menon & Vondrick (ICLR 2023): embeds the word plus a short visual description instead of the bare word. Tries a local Ollama server first; falls back to the WordNet gloss when no LLM is reachable.

**prompt_template_experiment.py**
Compares the bare-word baseline against the "Image of a {word}" prompt template across all 4 linguistic variants, isolating the effect of prompt phrasing.
