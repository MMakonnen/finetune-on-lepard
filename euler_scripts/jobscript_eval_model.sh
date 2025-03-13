#!/bin/bash

#SBATCH --output=logs/eval_model/slurm-%j.out
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=40G
#SBATCH --gres=gpumem:38G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user="ENTER_YOUR_MAIL_HERE"

module load stack/2024-06 cuda/12.4.1 eth_proxy
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finetune-lpr-env

# ...
python src/eval_finetuned_model.py