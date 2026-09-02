# Per-model experiments (standalone)

Each subfolder is a complete, standalone copy of everything needed to run this project's
experiments for one model: COCO alpha derivation, and the full baseline/ours/shine/
waffleclip/llm_descriptor comparison on the real benchmark (positive-set IoU and
negative-set false-positive rate). Each folder can be `cd`-ed into and run with its own
venv, with nothing else from this repo checked out -- deliberately, so running one model's
experiments never requires setting up (or risks breaking) another's environment, and each
folder can be zipped/rsynced to the cluster independently. SCLIP in particular needs heavy,
version-pinned deps (mmengine/mmcv/mmsegmentation) that clipseg/groupvit don't.

## Layout

```
clipseg/
groupvit/
sclip/
  requirements.txt              pip-installable deps for this model only
  setup_env.sh                  sclip/ only -- builds its venv via mim (mmcv needs this,
                                 not plain pip) and vendors the SCLIP repo
  base.py                       BaseOVSModel interface
  clipseg.py / groupvit.py / sclip.py   model wrapper (get_text_embedding / predict_with_embeddings)
  SCLIP/                        sclip/ only -- vendored, code-only copy of
                                 https://github.com/wangf3014/SCLIP (ECCV 2024), used
                                 directly per its own license; proper citation/
                                 contribution reference to be added later
  expanded_benchmark_helpers.py   dataset loaders, WordNet/embedding utilities
  benchmark_data.py             loads ../../../benchmark/ (LVIS/ADE20K/PascalVOC),
                                 GT mask resolution incl. shared-hypernym mask merging
  approaches.py                 the 5 query-transform methods (baseline/ours/shine/
                                 waffleclip/llm_descriptor)
  waffleclip_word_list.pkl      approaches.py's WaffleCLIP dependency
  alpha_value_experiment.py     COCO alpha sweep (coarse then fine grid)
  positive_set_experiment.py    IoU, all 5 approaches, on the benchmark's positive set
  negative_set_experiment.py    false-positive rate, all 5 approaches, on the negative set
```

`expanded_benchmark_helpers.py` / `benchmark_data.py` / `approaches.py` are point-in-time
copies of `../experiments_for_cluster/{expanded_benchmark_helpers,benchmark_data,
approaches}.py` (that directory is the source of truth / "common library" these are copied
from -- kept in sync manually, not imported live). `base.py` / `<model>.py` are copies of
the corresponding wrapper in `../ovs_eval/models/`.

## Running

```bash
cd clipseg   # or groupvit, or sclip
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # sclip/ instead: bash setup_env.sh

# 1. derive this model's alpha on COCO, full scale
python3 alpha_value_experiment.py --coco-dir ./coco_data

# 2. run the full approaches comparison on the real benchmark, using that alpha
python3 positive_set_experiment.py --alpha 0.71
python3 negative_set_experiment.py --alpha 0.71
```

Each script is independently checkpointed/resumable and writes
`results/<script_name>_<model>_{detail,summary}.csv`. Pass `--limit-categories`/
`--limit-images` (and for the alpha sweep, `--n-categories`/`--n-images`) for a quick local
check -- omit them (or pass `-1`) for the real, full-scale run.

## Known limitation

SCLIP's joint-softmax interface only accepts one embedding per class per call (see
`sclip/sclip.py`'s module docstring), so its scripts issue one forward pass per tag instead
of one batched call like clipseg/groupvit do. Expect SCLIP to run substantially slower than
the other two models at full scale.
