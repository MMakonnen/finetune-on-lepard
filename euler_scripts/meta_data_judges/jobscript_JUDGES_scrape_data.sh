#!/bin/bash

#SBATCH --output=logs/meta_judges/scrape_data/slurm-%j.out
#SBATCH --mem-per-cpu=10G
#SBATCH --time=25:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user="ENTER_YOUR_MAIL_HERE"

module load stack/2024-06 cuda/12.4.1 eth_proxy
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finetune-lpr-env 

# ...
python meta_data_scraping/scrape_meta_data.py