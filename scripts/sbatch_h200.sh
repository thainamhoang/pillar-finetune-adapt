#!/bin/bash -l
#SBATCH --job-name=pillar-dual-stream-report
#SBATCH --output=/home/thahoa/PET/Pillar-0/pillar-finetune-adapt/logs/slurm/%x-%j.out
#SBATCH --error=/home/thahoa/PET/Pillar-0/pillar-finetune-adapt/logs/slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gpus=H200:1

set -euo pipefail

module load miniforge3 cuda h200 dev2025a cmake

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
# Default: torchrun launches NUM_GPUS python processes inside a single
# SLURM task. Works on any cluster.
torchrun \
  --nnodes=1 \
  --nproc_per_node="$NUM_GPUS" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$PROJECT_ROOT/scripts/train.py" "$CONFIG_FILE" \
    --opts dataloader.num_workers 12 \
           dataloader.batch_size 4 \
           engine.kwargs.accumulate_grad_batches 8
TORCHRUN_EXIT=$?

exit $TORCHRUN_EXIT
