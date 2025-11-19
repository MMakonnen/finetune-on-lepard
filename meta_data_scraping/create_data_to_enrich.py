from datasets import load_dataset
import os
import sys

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from src.config import config
from src.data_prep_utils import prep_contexts, sample_data_with_all_passages, stratified_split, build_data_suffix

# Load Dataset
file_name = f"top_{config['dataset']}000_data.csv.gz"
contexts = load_dataset("rmahari/LePaRD", data_files=file_name)


cols_to_keep = ['passage_id', 'destination_context', 'dest_cite']

print('preping contexts')
contexts = prep_contexts(contexts, cols_to_keep)

print('sampling data with all passages')
contexts = sample_data_with_all_passages(contexts, config['data_usage_fraction'], config['seed'])

print('creating stratified splits')
# Stratified Split => Returns DatasetDict
contexts = stratified_split(
    dataset=contexts,
    splits=config['train_test_val_split'],
    stratify_col='passage_id',
    seed=config['seed']
)

data_suffix = build_data_suffix(config)

# make sure directory exists
os.makedirs("finetuning_data_judge", exist_ok=True)

train_filename = os.path.join("finetuning_data_judge", f"train_JUDGES_{data_suffix}.csv")

contexts["train"].to_pandas().to_csv(train_filename, index=False, encoding='utf-8')
print(f"Saved train split to {train_filename}")
