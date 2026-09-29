#!/bin/bash
# Benchmark experiments for CAT-Seg -- 5 approaches x 4 variants over the 326-category
# LVIS/ADE20K/PascalVOC benchmark.
#
# USAGE
#   sbatch benchmark_catseg.sbatch positive                 # IoU on positive images
#   sbatch benchmark_catseg.sbatch negative "" 30           # FPR, 30 images/category
#   sbatch benchmark_catseg.sbatch positive <OUT_DIR>       # resume an interrupted run
#
# Resumable: checkpointed on (category, img_id). Resubmit the SAME command with the SAME
# OUT_DIR to continue. summarize()/meta.finish() run only at the end, so a job that hits
# the time limit leaves a complete-so-far detail CSV but no summary.
#
# On resume, pass the SAME arg3 (--limit-images) as the original run: the sample is seeded
# per (seed, category), so a different cap reshuffles the draw and mixes two image sets
# into one CSV.
#
#SBATCH --job-name=bench_catseg
#SBATCH --partition=public      # NOT 'general': as of the May 2026 SOL maintenance,
                                # 'general' holds only privately-owned nodes and
                                # 'public' is the partition for public ones.
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

# $HOME is not always set inside `srun --pty bash` on SOL, and every path below
# depends on it -- unguarded, BENCHMARK_DIR becomes "/workspace/OVS/benchmark" and the
# run dies with FileNotFoundError on word_sets_v2.json. Recover it from passwd.
if [ -z "${HOME:-}" ] || [ ! -d "${HOME:-}" ]; then
    HOME=$(getent passwd "$(id -un)" | cut -d: -f6); export HOME
    echo "HOME was unset -- recovered as $HOME"
fi

SET=${1:?usage: sbatch benchmark_catseg.sbatch <positive|negative> [out_dir] [limit_images]}
ALPHA=${ALPHA:-0.80}        # CAT-Seg's OWN sweep value (not CLIPSeg's 0.69):
                            #   coarse grid peak alpha=0.80  mean_iou 0.336930  (n=20917)
                            #   fine   0.74-0.86 step 0.02, peak alpha=0.80    0.336980
                            # The whole fine range spans 0.000401, so this is a very flat
                            # optimum -- the exact value barely matters. WEIGHTED centroid
                            # beats unweighted by 0.001215, hence --weighted below.
                            # Override per submission with:  ALPHA=0.78 sbatch ...
REPO=$HOME/workspace/OVS/taxonomy-driven-ovs-models
EXPDIR=$REPO/model_experiments/catseg
PY=$EXPDIR/catseg_venv/bin/python
DRY_TAG=""
[ -n "${LIMIT_CATEGORIES:-}" ] && DRY_TAG="_DRYRUN"
OUT=${2:-$HOME/workspace/OVS/results/${SET}_catseg${DRY_TAG}_$(date +%Y%m%d_%H%M%S)}
LIMIT_IMAGES=${3:-}

# Override from the command line without editing this file, e.g.
#   CATSEG_WEIGHTS=/scratch/$USER/model_base.pth \
#   CATSEG_CONFIG=$PWD/CAT-Seg/configs/vitb_384.yaml \
#   sbatch benchmark_catseg.sbatch ...
CONFIG=${CATSEG_CONFIG:-$EXPDIR/CAT-Seg/configs/vitl_336.yaml}
WEIGHTS=${CATSEG_WEIGHTS:-$HOME/workspace/OVS/checkpoints/catseg_model_large.pth}

export BENCHMARK_DIR=$HOME/workspace/OVS/benchmark
export HF_HOME=$HOME/workspace/OVS/hf_cache
export TORCH_HOME=$HOME/workspace/OVS/torch_cache
export NLTK_DATA=$HOME/workspace/OVS/nltk_data
export PYTHONNOUSERSITE=1
# SOL's /lib64/libstdc++.so.6 predates CXXABI_1.3.15, which the env's libicu
# needs -- without this every import dies with:
#   ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found
export LD_LIBRARY_PATH="$EXPDIR/catseg_venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p "$OUT" "$HF_HOME" "$TORCH_HOME" "$NLTK_DATA"
cd "$EXPDIR"
echo "=================================================="
echo "node   = $SLURMD_NODENAME   job = $SLURM_JOB_ID"
echo "set    = $SET   alpha = $ALPHA (weighted)   threshold = 0.0 (argmax)"
echo "OUT    = $OUT"
echo "  (pass this OUT as arg 2 to resume, with the SAME arg3)"
echo "=================================================="

[ -d "$BENCHMARK_DIR" ] || { echo "ABORT: BENCHMARK_DIR does not exist: $BENCHMARK_DIR"; \
                            echo "  (a leading /workspace instead of $HOME/workspace means HOME was unset)"; exit 1; }
for f in word_sets_v2.json positive_set_v2.json negative_set_v2.json img_metadata.pkl; do
    [ -f "$BENCHMARK_DIR/$f" ] || { echo "ABORT: missing $BENCHMARK_DIR/$f"; exit 1; }
done
[ -f "$CONFIG" ]  || { echo "ABORT: config not found: $CONFIG"; exit 1; }
[ -f "$WEIGHTS" ] || { echo "ABORT: weights not found: $WEIGHTS"; exit 1; }
echo "benchmark files + checkpoint OK"

$PY - <<'PYEOF'
import sys, torch, detectron2, nltk
print("python    :", sys.executable)
print("torch     :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")
print("detectron2:", detectron2.__version__)
try: nltk.data.find("corpora/wordnet")
except LookupError: nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
if not torch.cuda.is_available():
    sys.exit("ABORT: no CUDA device visible.")
print("deps ok")
PYEOF

if curl -sf --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama: REACHABLE -- llm_descriptor will use the LLM"
else
    echo "Ollama: NOT reachable -- llm_descriptor FALLS BACK to WordNet glosses."
    echo "        Valid run, but report that approach as 'gloss', not 'LLM'."
fi

ARGS=(--alpha "$ALPHA" --weighted --config "$CONFIG" --weights "$WEIGHTS" --out-dir "$OUT")
[ -n "$LIMIT_IMAGES" ] && ARGS+=(--limit-images "$LIMIT_IMAGES")
# LIMIT_CATEGORIES makes a DRY RUN queueable instead of needing an interactive GPU:
#   LIMIT_CATEGORIES=40 sbatch benchmark_catseg.sbatch positive "" 2
# Use 40, not 20: sorted() puts uppercase first, so a low cap gives mostly
# proper-noun artifacts (Frisbee, New Jersey) and no hypo rows at all.
[ -n "${LIMIT_CATEGORIES:-}" ] && ARGS+=(--limit-categories "$LIMIT_CATEGORIES")

$PY "${SET}_set_experiment.py" "${ARGS[@]}"

echo "DONE ($SET) -> $OUT"
ls -la "$OUT"
