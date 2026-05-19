#!/bin/bash -l
#SBATCH --job-name=pillar-chest-ct-adapt
#SBATCH --output=/home/thahoa/PET/Pillar-0/pillar-finetune-adapt/logs/slurm/%x-%j.out
#SBATCH --error=/home/thahoa/PET/Pillar-0/pillar-finetune-adapt/logs/slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=H200:1

set -euo pipefail

module load miniforge3 cuda h200 dev2025a cmake

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

eval "$(mamba shell hook --shell bash)"
mamba activate runai

CONFIG_FILE=${1:-configs/vimed_chest_ct_only.yaml}
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
  "$PROJECT_ROOT/scripts/train.py" "$CONFIG_FILE" \
  --opts \
      dataloader.batch_size 32 \
      dataloader.eval_batch_size 32 \
      # engine.kwargs.accumulate_grad_batches 2 \
      # optimizer.kwargs.lr 4.0e-5 \
      # optimizer.scheduler.kwargs.warmup_epochs 3 \
      optimizer.kwargs.lr 2.8e-5 \
      engine.max_epochs 30 \
      optimizer.scheduler.kwargs.max_epochs 30
TORCHRUN_EXIT=$?

exit $TORCHRUN_EXIT