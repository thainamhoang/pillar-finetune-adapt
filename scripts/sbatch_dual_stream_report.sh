#!/bin/bash -l
#SBATCH --job-name=pillar-dual-stream-report
#SBATCH --output=/home/thahoa/PET/Pillar-0/pillar-finetune-adapt/logs/slurm/%x-%j.out
#SBATCH --error=/home/thahoa/PET/Pillar-0/pillar-finetune-adapt/logs/slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
# Per-GPU peak ~55 GB at bs=4 (45 GB activations + 10 GB eval KV cache),
# under H100's 80 GB ceiling. H200 only needed if pushing bs to 8 per
# GPU; with 2206 train samples eff=32 is the sweet spot and bs=4 is
# enough to hit it on 4 GPUs with grad_accum=2.
#SBATCH --gpus=H100:4

# Phase B: Dual-stream PET/CT -> LLM report generator.
#
# Stack (per configs/vimed_chest_dual_stream_report.yaml):
#   * Frozen Pillar0-ChestCT (W_B) encoders loaded from Phase A export
#   * Trainable Perceiver Resamplers (128 queries x 1152, 6 layers, per
#     modality)
#   * Trainable vision -> LLM projection MLPs (per modality)
#   * Frozen LLM (MedGemma 1.5 4B-IT default) + LoRA r=8 alpha=32
#
# REQUIRED BEFORE LAUNCHING:
#   1. Phase A artifact at
#      ./logs/dual-stream-pillar/fusion/dual_stream_encoder.pt
#      Generate via:
#        python scripts/export_dual_stream_encoder.py \
#           --ct-ckpt  logs/dual-stream-pillar/ct/<run>/checkpoints/best.ckpt \
#           --pet-ckpt logs/dual-stream-pillar/pet-from-ct/<run>/checkpoints/best.ckpt \
#           --out      logs/dual-stream-pillar/fusion/dual_stream_encoder.pt \
#           --smoke    /scratch/thahoa/PET/ViMed_prep_v2/manifest_splits.csv
#
#   2. HF access for the LLM. For MedGemma you must accept the Health AI
#      Developer Foundations terms once at
#      https://huggingface.co/google/medgemma-1.5-4b-it
#      Then `huggingface-cli login` and ensure HF_TOKEN is set.
#
#   3. peft, sacrebleu, rouge-score installed (`uv sync` after pyproject
#      update).

set -euo pipefail

module load miniforge3 cuda h100 dev2025a cmake

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

eval "$(mamba shell hook --shell bash)"
mamba activate runai

CONFIG_FILE=${1:-configs/vimed_chest_dual_stream_report.yaml}
PROJECT_ROOT=/home/thahoa/PET/Pillar-0/pillar-finetune-adapt
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs/slurm"

[[ -f "$PROJECT_ROOT/.env" ]] && set -a && source "$PROJECT_ROOT/.env" && set +a

export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
# HF tokenizers warns when DataLoader workers fork after a tokenizer has
# been used in the main process. Setting this avoids the noisy warning
# and is the documented HF recommendation when using multi-worker loading.
export TOKENIZERS_PARALLELISM=false

# HF cache lives on /scratch so it doesn't fill the home quota with the
# ~9 GB MedGemma weights.
export HF_HOME=/scratch/thahoa/hf_cache
export HUGGINGFACE_HUB_CACHE=/scratch/thahoa/hf_cache/hub
export TRANSFORMERS_CACHE=/scratch/thahoa/hf_cache/transformers
mkdir -p "$HF_HOME"

echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv

( while sleep 60; do
    PIDS=$(pgrep -f "scripts/train.py" || true)
    if [[ -z "$PIDS" ]]; then continue; fi
    for p in $PIDS; do
        awk -v pid=$p '
            /^VmHWM:/ { hwm = $2 }
            /^VmRSS:/ { rss = $2 }
            END { printf "[mem-probe] pid=%s VmHWM=%.1fGB VmRSS=%.1fGB\n",
                         pid, hwm/1024/1024, rss/1024/1024 }
        ' /proc/$p/status 2>/dev/null
    done
  done ) &
MEM_PROBE_PID=$!
trap "kill $MEM_PROBE_PID 2>/dev/null || true" EXIT

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | wc -l)
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + ${SLURM_JOB_ID:-0} % 50000))

echo "Using $NUM_GPUS GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Torch rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"

set +e
torchrun \
  --nnodes=1 \
  --nproc_per_node="$NUM_GPUS" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$PROJECT_ROOT/scripts/train.py" "$CONFIG_FILE"
TORCHRUN_EXIT=$?

exit $TORCHRUN_EXIT
