#!/bin/bash

#SBATCH --output=logs/sbert/finetune_model/slurm-%j.out
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=30G
#SBATCH --gres=gpumem:24G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user="ENTER_YOUR_MAIL_HERE"

module load stack/2024-06 cuda/12.4.1 eth_proxy
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lpr-ra

# Hugging Face authentication
export HUGGINGFACE_TOKEN=$(<access_tokens/NV_Embed_v2/.hf_token)
huggingface-cli login --token "$HUGGINGFACE_TOKEN" > /dev/null

# Redirect HF caches onto scratch
export HF_HOME=/cluster/scratch/mmakonnen/hf_home
export HF_DATASETS_CACHE=/cluster/scratch/mmakonnen/hf_datasets
export HF_TRANSFORMERS_CACHE=/cluster/scratch/mmakonnen/hf_models

# ...
python src_sbert/sbert_finetune.py