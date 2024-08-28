#!/bin/bash
#SBATCH --gpus=2
module load anaconda3/2024.02-1 go/1.18.2 cmake/3.22.0 cuda/11.8 cudnn/8.1.1_cuda11.x gcc/11.2

source activate deepspeed

# Number of nodes
NUM_NODES=1
# Number of GPUs per node
NUM_GPUS=2
# Size of expert parallel world (should be less than total world size)
EP_SIZE=2
# Number of total experts
EXPERTS=2

        #   --bind_cores_to_rank \

deepspeed --num_nodes=${NUM_NODES}\
          --num_gpus=${NUM_GPUS} \
        cifar10_deepspeed.py \
	--log-interval 100 \
	--deepspeed \
	--moe \
	--ep-world-size ${EP_SIZE} \
	--num-experts ${EXPERTS} \
	--top-k 1 \
	--noisy-gate-policy 'RSample' \
	--moe-param-group
