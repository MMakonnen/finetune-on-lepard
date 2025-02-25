#!/bin/bash

#SBATCH --output=logs/eval_model/slurm-%j.out
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=30G
#SBATCH --gres=gpumem:23G
#SBATCH --time=00:30:00

module load stack/2024-06 cuda/12.4.1 eth_proxy
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lpr-ra

# ...
python src/eval_finetuned_model.py