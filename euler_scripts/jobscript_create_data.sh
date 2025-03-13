#!/bin/bash

#SBATCH --output=logs/create_data/slurm-%j.out
#SBATCH --mem-per-cpu=30G
#SBATCH --time=00:30:00

module load stack/2024-06 cuda/12.4.1 eth_proxy
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finetune-lpr-env

# ...
python src/create_finetuning_dataset.py