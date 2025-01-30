from datasets import load_dataset

# Local imports
from config import config
from utils import validate_config, count_unique_values
from data_prep import prep_contexts, sample_data_with_all_passages, stratified_split, create_json_files

# For reproducibility
seed = 42

# === Display Configuration ===
print("=== Configuration ===")
print(f"Dataset: {config['dataset']}k")
print(f"Train split: {config['train_test_val_split']['train']}")
print(f"Validation split: {config['train_test_val_split']['valid']}")
print(f"Test split: {config['train_test_val_split']['test']}")
print(f"Data usage fraction: {config['data_usage_fraction']}")
print("======================\n")

# Validate configuration
validate_config(config)

# === Load Dataset ===
file_name = f"top_{config['dataset']}000_data.csv.gz"
contexts = load_dataset("rmahari/LePaRD", data_files=file_name)

# === Preprocess and Sample Subset of Contexts ===
cols_to_keep = ['passage_id', 'destination_context']
contexts = prep_contexts(contexts, cols_to_keep)
contexts = sample_data_with_all_passages(contexts, config['data_usage_fraction'], seed)

# === Stratified Split => Returns DatasetDict ===
contexts = stratified_split(
    dataset=contexts,
    splits=config['train_test_val_split'],
    stratify_col='passage_id',
    seed=seed
)

# === Final Information ===
print("=== Final Splits ===")
print(f"Train: {len(contexts['train'])} samples")
print(f"Validation: {len(contexts['valid'])} samples")
print(f"Test: {len(contexts['test'])} samples")
print("====================\n")

print("=== Unique Passage IDs ===")
print(f"Train: {count_unique_values(contexts['train'], 'passage_id')}")
print(f"Validation: {count_unique_values(contexts['valid'], 'passage_id')}")
print(f"Test: {count_unique_values(contexts['test'], 'passage_id')}")
print("==========================\n")

# # === Sanity Check ===
# # Uncomment if needed:
# # import pandas as pd
# # print("=== Sanity Check (distribution of passage ids in train/valid/test) ===")
# # print(pd.Series(contexts['train']['passage_id']).value_counts(True).head(5))
# # print("############")
# # print(pd.Series(contexts['valid']['passage_id']).value_counts(True).head(5))
# # print("############")
# # print(pd.Series(contexts['test']['passage_id']).value_counts(True).head(5))
# # print("====================")

# ==============================================================================
# Create JSON files with special tokens
# ==============================================================================

# Collect passage_ids from the train split
passage_ids = list(set(contexts['train']['passage_id']))

# Map each passage_id to a custom special token
num_special_tok = len(passage_ids)
special_tokens = [f"<special_token_{i}>" for i in range(1, num_special_tok + 1)]
passage_to_token = dict(zip(passage_ids, special_tokens))

# Specify output folder on one line
output_folder = "finetuning_data"

# Create JSON files
create_json_files(contexts, passage_to_token, output_folder, config, seed)
