#!/bin/bash
#SBATCH --gpus=2

module load anaconda3/2024.02-1 go/1.18.2 cmake/3.22.0 cuda/11.8 cudnn/8.1.1_cuda11.x gcc/11.2

source activate deepspeed


deepspeed train_bert_ds.py --checkpoint_dir experiment_deepspeed $@

# # Number of nodes
# NUM_NODES=1
# # Number of GPUs per node
# NUM_GPUS=2
# deepspeed --num_nodes=${NUM_NODES} --num_gpus=${NUM_GPUS} train_bert_ds.py --checkpoint_dir experiment_deepspeed $@
# deepspeed --bind_cores_to_rank train_bert_ds.py --checkpoint_dir experiment_deepspeed $@
