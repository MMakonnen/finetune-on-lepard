from datasets import load_dataset
import os
import sys

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# local imports
from distilbert_config import config
from src.utils import validate_config
from src.data_prep_utils import prep_contexts, sample_data_with_all_passages, stratified_split, build_data_suffix
from distilbert_data_utils import format_extended_example


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
contexts = load_dataset("rmahari/LePaRD", data_files=file_name)


# Preprocess and Sample Subset of Contexts
if config.get("use_enriched_context", False):
    cols_to_keep = ['passage_id', 'destination_context', 'dest_court', 'source_date', 'source_court']
else:
    cols_to_keep = ['passage_id', 'destination_context']
contexts = prep_contexts(contexts, cols_to_keep)
contexts = sample_data_with_all_passages(contexts, config['data_usage_fraction'], config['seed'])


# format data for extended format
if config.get("use_enriched_context", False):
    contexts = contexts.map(format_extended_example)
    cols_to_drop = ["destination_context", "dest_court", "source_date", "source_court"]
    contexts = contexts.remove_columns(cols_to_drop)

# Stratified Split => Returns DatasetDict
contexts = stratified_split(
    dataset=contexts,
    splits=config['train_test_val_split'],
    stratify_col='passage_id',
    seed=config['seed']
)

data_suffix = build_data_suffix(config)

# Create the folder if it doesn't exist
output_dir = "finetuning_data_distilbert"

# Add optional 'extended_' prefix to the filename if enriched format is used
format_prefix = "extended_" if config.get("use_enriched_context", False) else ""

# Build file names based on the naming suffix
train_filename = os.path.join(output_dir, f"train_distilbert_{format_prefix}{data_suffix}.csv")
valid_filename = os.path.join(output_dir, f"valid_distilbert_{format_prefix}{data_suffix}.csv")
test_filename = os.path.join(output_dir, f"test_distilbert_{format_prefix}{data_suffix}.csv")

# create folder if it doesnt exist already
os.makedirs(output_dir, exist_ok=True)

# Convert each split to a Pandas DataFrame and save as CSV with UTF-8 encoding
contexts["train"].to_pandas().to_csv(train_filename, index=False, encoding='utf-8')
contexts["valid"].to_pandas().to_csv(valid_filename, index=False, encoding='utf-8')
contexts["test"].to_pandas().to_csv(test_filename, index=False, encoding='utf-8')

print(f"Saved train split to {train_filename}")
print(f"Saved valid split to {valid_filename}")
print(f"Saved test split to {test_filename}")