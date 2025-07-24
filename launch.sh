#!/bin/bash
#SBATCH --job-name=distributed_training
#SBATCH --nodes=32
#SBATCH --cpus-per-gpu=12
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=6-00:00:00
#SBATCH --partition=high

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1 | xargs -I{} getent hosts {} | awk '{ print $1 }')
export MASTER_PORT=29500

echo "Master node IP: $MASTER_ADDR"
echo "Total nodes: $SLURM_NNODES"

# Run on each node with conda env! 
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 bash -c '
  source ~/.bashrc
  eval "$(conda shell.bash hook)"
  conda activate simple-pp
  
  echo "[$SLURM_PROCID] Starting on $(hostname)"
  
  export NCCL_DEBUG=INFO
  export LD_LIBRARY_PATH=$CONDA_LIB_PATH/nvidia/nccl/lib:$LD_LIBRARY_PATH

  NCCL_NET_PLUGIN=none python -m torch.distributed.run \
    --nproc-per-node=8 \
    --nnodes='$SLURM_NNODES' \
    --node_rank=$SLURM_PROCID \
    --master-addr='$MASTER_ADDR' \
    --master-port='$MASTER_PORT' \
    train.py --config configs/train.yaml
'