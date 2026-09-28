#!/bin/bash
# One batch job that finishes the CAT-Seg validation, so it does not need an interactive
# session (which keeps dying with the SSH connection at fairshare 0.11).
#
# It answers the one open question: the alpha sweep reports orig IoU ~0.12 at alpha=0,
# but check_native_iou.py measured 0.31 on the first 8 COCO classes. With desc=False,
# alpha=0 IS the templated embedding, so those should agree -- unless the sweep simply
# picked harder (WordNet-eligible) categories. This runs check_native_iou.py on the
# EXACT categories the sweep used, making the two directly comparable.
#
# USAGE
#   sbatch diagnose_catseg.sbatch <ALPHA_RUN_DIR>
# e.g.
#   sbatch diagnose_catseg.sbatch ~/workspace/OVS/tmp/alpha_catseg_dry2
#
#SBATCH --job-name=diag_catseg
#SBATCH --partition=htc         # short job; htc backfills far better than public
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G               # GPU jobs have a 24000 MB floor on SOL
#SBATCH --time=0-01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

RUNDIR=${1:?usage: sbatch diagnose_catseg.sbatch <alpha_run_dir>}
REPO=$HOME/workspace/OVS/taxonomy-driven-ovs-models
EXPDIR=$REPO/model_experiments/catseg
PY=$EXPDIR/catseg_venv/bin/python
CONFIG=${CATSEG_CONFIG:-$EXPDIR/CAT-Seg/configs/vitl_336.yaml}
WEIGHTS=${CATSEG_WEIGHTS:-/scratch/$USER/catseg_ckpt/catseg_model_large.pth}

export BENCHMARK_DIR=$HOME/workspace/OVS/benchmark
export HF_HOME=$HOME/workspace/OVS/hf_cache
export NLTK_DATA=$HOME/workspace/OVS/nltk_data
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
# SOL's /lib64/libstdc++.so.6 predates CXXABI_1.3.15 that this env's libicu needs
export LD_LIBRARY_PATH="$EXPDIR/catseg_venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$EXPDIR"
DETAIL=$(ls "$RUNDIR"/*_detail.csv 2>/dev/null | head -1)
[ -n "$DETAIL" ] || { echo "ABORT: no *_detail.csv in $RUNDIR"; exit 1; }
[ -f "$CONFIG" ]  || { echo "ABORT: config not found: $CONFIG";  exit 1; }
[ -f "$WEIGHTS" ] || { echo "ABORT: weights not found: $WEIGHTS"; exit 1; }

echo "=================================================="
echo "node   = $SLURMD_NODENAME   job = $SLURM_JOB_ID"
echo "detail = $DETAIL"
echo "=================================================="

echo
echo "############ 1. per-variant means from the sweep ############"
$PY - "$DETAIL" <<'PYEOF'
import csv, collections, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for a in ("0.0", "0.5", "1.0"):
    d = collections.defaultdict(list)
    for r in rows:
        if r["alpha"] == a and r["weighted"] == "True":
            d[r["variant"]].append(float(r["iou"]))
    means = {k: round(sum(v)/len(v), 4) for k, v in sorted(d.items())}
    ns = {k: len(v) for k, v in sorted(d.items())}
    print(f"  alpha={a}: {means}  n={ns}")
PYEOF

echo
echo "############ 2. per-category orig IoU at alpha=0 ############"
CATS=$($PY - "$DETAIL" <<'PYEOF'
import csv, collections, sys
rows = list(csv.DictReader(open(sys.argv[1])))
d = collections.defaultdict(list)
for r in rows:
    if r["alpha"] == "0.0" and r["weighted"] == "True" and r["variant"] == "orig":
        d[r["category"]].append(float(r["iou"]))
import sys as _s
for c, v in sorted(d.items()):
    print(f"  {c:<22} {sum(v)/len(v):.4f}  (n={len(v)})", file=_s.stderr)
print(",".join(sorted(d)))
PYEOF
)
echo "  sweep categories: $CATS"

echo
echo "############ 3. NATIVE vs BARE vs TEMPLATED on those SAME categories ############"
echo "   (this is the comparison that decides whether the sweep path is sound)"
$PY check_native_iou.py \
    --config "$CONFIG" --weights "$WEIGHTS" \
    --coco-dir "$HOME/workspace/OVS/coco_data" \
    --categories "$CATS" --n-images 3

echo
echo "############ 4. injection probe (independent sanity) ############"
$PY probe_injection.py --config "$CONFIG" --weights "$WEIGHTS" || true

echo
echo "DONE. Read section 3: if NATIVE on these categories ~= the sweep's orig@a=0,"
echo "the pipeline is sound and CAT-Seg simply scores lower than CLIPSeg (joint argmax"
echo "over 80 classes is a harder task than independent per-prompt sigmoid). If NATIVE"
echo "is much higher, the sweep's own path still differs and must be fixed first."
