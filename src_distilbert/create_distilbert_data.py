from datasets import load_dataset
import os
import sys

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# local imports
from distilbert_config import config
from src.utils import validate_config
from src.data_prep_utils import prep_contexts, sample_data_with_all_passages, stratified_split


# Display Configuration
print("=== Configuration ===")
print(f"Dataset: {config['dataset']}k")
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

dataset_size = config["dataset"]  # e.g. "10" for 10k
fraction_percentage = int(config["data_usage_fraction"] * 100)
# To ensure a consistent order, we order the splits as train, valid, test:
split_order = ["train", "valid", "test"]
splits_dict = config["train_test_val_split"]
split_str = "".join(str(int(splits_dict[k] * 100)).zfill(2) for k in split_order)
data_suffix = f"{dataset_size}k_percent{fraction_percentage}_split{split_str}_seed{config['seed']}"

# Create the folder if it doesn't exist
output_dir = "finetuning_data_distilbert"

# Build file names based on the naming suffix
train_filename = os.path.join(output_dir, f"train_distilbert_{data_suffix}.csv")
valid_filename = os.path.join(output_dir, f"valid_distilbert_{data_suffix}.csv")
test_filename = os.path.join(output_dir, f"test_distilbert_{data_suffix}.csv")

# create folder if it doesnt exist already
os.makedirs(output_dir, exist_ok=True)

# Convert each split to a Pandas DataFrame and save as CSV with UTF-8 encoding
contexts["train"].to_pandas().to_csv(train_filename, index=False, encoding='utf-8')
contexts["valid"].to_pandas().to_csv(valid_filename, index=False, encoding='utf-8')
contexts["test"].to_pandas().to_csv(test_filename, index=False, encoding='utf-8')

print(f"Saved train split to {train_filename}")
print(f"Saved valid split to {valid_filename}")
print(f"Saved test split to {test_filename}")