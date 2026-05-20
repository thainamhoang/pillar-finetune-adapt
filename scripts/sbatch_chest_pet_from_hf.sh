#!/bin/bash -l
#SBATCH --job-name=pillar-chest-pet-from-hf
#SBATCH --output=/home/thahoa/PET/Pillar-0/pillar-finetune-adapt/logs/slurm/%x-%j.out
#SBATCH --error=/home/thahoa/PET/Pillar-0/pillar-finetune-adapt/logs/slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gpus=H100:1

# Phase 2-A: PET-only fine-tune starting from HF Pillar0-ChestCT (W_A).
# Baseline against which we evaluate the W_B (CT-trained init) variant.

set -euo pipefail

module load miniforge3 cuda h100 dev2025a cmake

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

eval "$(mamba shell hook --shell bash)"
mamba activate runai

CONFIG_FILE=${1:-configs/vimed_chest_pet_from_hf.yaml}
PROJECT_ROOT=/home/thahoa/PET/Pillar-0/pillar-finetune-adapt
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs/slurm"

export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Background memory probe -- watches python train.py processes every 30s.
( while sleep 30; do
    PIDS=$(pgrep -f "scripts/train.py" || true)
    if [[ -z "$PIDS" ]]; then
        echo "[mem-probe] no train.py process yet"
        continue
    fi
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
