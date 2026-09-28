#!/bin/bash
# Benchmark experiments for CLIPSeg -- 5 approaches x 4 variants over the 346-category
# LVIS/ADE20K/PascalVOC benchmark.
#
# USAGE
#   sbatch benchmark_clipseg.sbatch positive                    # IoU on positive images
#   sbatch benchmark_clipseg.sbatch negative                    # FPR on negative images
#   sbatch benchmark_clipseg.sbatch positive <OUT_DIR>          # resume an interrupted run
#   sbatch benchmark_clipseg.sbatch negative <OUT_DIR> 30       # arg3 = --limit-images
#
# Resumable: checkpointed on (category, img_id). Resubmit the SAME command with the SAME
# OUT_DIR to continue. NOTE: summarize()/meta.finish() only run at the end, so a job that
# hits the time limit leaves a complete-so-far detail CSV but no summary.
#
#SBATCH --job-name=bench_clipseg
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

SET=${1:?usage: sbatch benchmark_clipseg.sbatch <positive|negative> [out_dir] [limit_images]}
ALPHA=0.69          # from alpha_value_experiment.py fine sweep -- CLIPSeg specific
REPO=$HOME/workspace/OVS/taxonomy-driven-ovs-models
EXPDIR=$REPO/model_experiments/clipseg
PY=/home/sschaud5/.conda/envs/tasa/bin/python
OUT=${2:-$HOME/workspace/OVS/results/${SET}_clipseg_$(date +%Y%m%d_%H%M%S)}
LIMIT_IMAGES=${3:-}

export BENCHMARK_DIR=$HOME/workspace/OVS/benchmark
export HF_HOME=$HOME/workspace/OVS/hf_cache
export TORCH_HOME=$HOME/workspace/OVS/torch_cache
export NLTK_DATA=$HOME/workspace/OVS/nltk_data
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p "$OUT" "$HF_HOME" "$TORCH_HOME" "$NLTK_DATA"
cd "$EXPDIR"
echo "=================================================="
echo "node   = $SLURMD_NODENAME   job = $SLURM_JOB_ID"
echo "set    = $SET   alpha = $ALPHA (weighted)"
echo "OUT    = $OUT"
echo "  (pass this OUT as arg 2 to resume)"
echo "=================================================="

# ---------- the 4 benchmark files must exist ----------
for f in word_sets_v2.json positive_set_v2.json negative_set_v2.json img_metadata.pkl; do
    [ -f "$BENCHMARK_DIR/$f" ] || { echo "ABORT: missing $BENCHMARK_DIR/$f"; exit 1; }
done
echo "benchmark files OK in $BENCHMARK_DIR"

# ---------- sanity ----------
$PY - <<'PYEOF'
import sys, torch, transformers, nltk
print("python :", sys.executable)
print("torch  :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")
print("transformers:", transformers.__version__)
try: nltk.data.find("corpora/wordnet")
except LookupError: nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
if not torch.cuda.is_available():
    sys.exit("ABORT: no CUDA device visible.")
print("deps ok")
PYEOF

# ---------- llm_descriptor needs a local Ollama, else it falls back to the WordNet gloss ----------
if curl -sf --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama: REACHABLE -- llm_descriptor will use the LLM"
else
    echo "Ollama: NOT reachable -- llm_descriptor FALLS BACK to WordNet glosses."
    echo "        This is a valid run, but report that approach as 'gloss', not 'LLM'."
fi

ARGS=(--alpha "$ALPHA" --weighted --out-dir "$OUT")
[ -n "$LIMIT_IMAGES" ] && ARGS+=(--limit-images "$LIMIT_IMAGES")

$PY "${SET}_set_experiment.py" "${ARGS[@]}"

echo "DONE ($SET) -> $OUT"
ls -la "$OUT"
