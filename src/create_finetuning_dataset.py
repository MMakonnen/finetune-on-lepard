# ADJUSTMENTS MADE FOR USING ALREADY CREATED ENRICHED META DATA,
# places of changes indicated via "HERE"

from datasets import load_dataset

# HERE
import pandas as pd
import os, sys
from datasets import Dataset, DatasetDict

# HERE
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Local imports
from config import config
from utils import validate_config, count_unique_values
from data_prep_utils import (
    prep_contexts,
    sample_data_with_all_passages,
    stratified_split,
    generate_special_tokens,
    create_json_files
)

# Display Configuration
print("=== Configuration ===")
print(f"Dataset: {config['dataset']}k")
print(f"Extended data format: {config['use_enriched_context']}")
print(f"Train split: {config['train_test_val_split']['train']}")
print(f"Validation split: {config['train_test_val_split']['valid']}")
print(f"Test split: {config['train_test_val_split']['test']}")
print(f"Data usage fraction: {config['data_usage_fraction']}")
print(f"Random seed: {config['seed']}")
print("======================\n")

# Validate configuration
validate_config(config)

# Load Dataset
file_name = f"top_{config['dataset']}000_data.csv.gz"
contexts_hf = load_dataset("rmahari/LePaRD", data_files=file_name)

# HERE
# Read enriched judge meta from both train-scraped and test-scraped files
# HERE
# Read enriched judge meta from both train-scraped and test-scraped files
paths_enriched = [
    "meta_data_scraping/meta_judge_data/meta_judge_N_full_10k_percent20_split801010_seed42.csv",
    "meta_data_scraping/meta_judge_data/meta_judge_N_test_not_in_scraped_full_10k_percent20_split801010_seed42.csv",
]

dfs = [pd.read_csv(p, usecols=["dest_cite", "judge_names"]) for p in paths_enriched]
df_enriched = pd.concat(dfs, ignore_index=True)


# HERE
# Convert HF to pandas, merge judges on dest_cite (LEFT join to keep all base rows)
df_base = contexts_hf["train"].to_pandas()
df_merged = df_base.merge(df_enriched, on="dest_cite", how="left")

# HERE
# Back to HF format for the rest of the pipeline
contexts = DatasetDict({
    "train": Dataset.from_pandas(df_merged, preserve_index=False)
})

# Preprocess and Sample Subset of Contexts
if config.get("use_enriched_context", False):
    cols_to_keep = ['passage_id', 'destination_context', 'judge_names']
    # cols_to_keep = ['passage_id', 'destination_context', 'dest_court', 'source_date', 'source_court']
else:
    cols_to_keep = ['passage_id', 'destination_context']
contexts = prep_contexts(contexts, cols_to_keep)
contexts = sample_data_with_all_passages(contexts, config['data_usage_fraction'], config['seed'])


# Stratified Split => Returns DatasetDict
contexts = stratified_split(
    dataset=contexts,
    splits=config['train_test_val_split'],
    stratify_col='passage_id',
    seed=config['seed']
)

# Final Information
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

# # Sanity Check (UNCOMMENT IF WANTED)
# import pandas as pd
# print("=== Sanity Check (distribution of passage ids in train/valid/test) ===")
# print(pd.Series(contexts['train']['passage_id']).value_counts(True).head(5))
# print("############")
# print(pd.Series(contexts['valid']['passage_id']).value_counts(True).head(5))
# print("############")
# print(pd.Series(contexts['test']['passage_id']).value_counts(True).head(5))
# print("====================")

# ==============================================================================
# Create JSON files (train/val/test data) with special tokens
# ==============================================================================

# Collect passage_ids from the train split
passage_ids = list(set(contexts['train']['passage_id']))

# Map each passage_id to a custom special token and save the dict
output_folder_token_dic = "passage_special_token_map"
passage_to_token, special_tokens = generate_special_tokens(passage_ids, output_folder_token_dic)

# Create JSON files and save them
output_folder_data = "finetuning_data"
create_json_files(contexts, passage_to_token, output_folder_data, config)
