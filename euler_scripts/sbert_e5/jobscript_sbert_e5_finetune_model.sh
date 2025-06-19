#!/bin/bash

#SBATCH --output=logs/sbert_e5/finetune_model/slurm-%j.out
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=40G
#SBATCH --gres=gpumem:34G
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user="ENTER_YOUR_MAIL_HERE"

module load stack/2024-06 cuda/12.4.1 eth_proxy
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lpr-ra

# ...
python src_sbert/sbert_finetune_e5.py