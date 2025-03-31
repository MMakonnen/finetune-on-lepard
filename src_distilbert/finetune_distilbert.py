import os
import sys
from transformers import AutoTokenizer

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# local imports
from src.data_prep_utils import build_data_suffix
from distilbert_config import config
from distilbert_utils import set_seed
from distilbert_data_class import LegalClassificationDataset
from distilbert_data_utils import load_data_csv, load_special_token_map
from distilbert_train_eval_utils import train_model, evaluate_epoch


# Set seed for reproducibility
set_seed(config["seed"])

# Build data_suffix using the config parameters
data_suffix = build_data_suffix(config)

# Load special token mapping
special_token_map_path = os.path.join("passage_special_token_map", "passage_to_token.json")
special_token_map, label_to_zero_index, zero_index_to_label = load_special_token_map(special_token_map_path)

# Sanity check: ensure config["n_labels"] matches the actual number of labels
expected_n_labels = len(label_to_zero_index)
if config.get("n_labels") is not None and config["n_labels"] != expected_n_labels:
    raise ValueError(f"Mismatch in number of labels: config['n_labels']={config['n_labels']} vs expected={expected_n_labels}")

# Load training data with label mapping
print("Loading training data...")
train_examples = load_data_csv(config["train_file"], label_to_zero_index)
print(f"Loaded {len(train_examples)} training examples.")

# Load development data with same label mapping
print("Loading development data...")
dev_examples = load_data_csv(config["dev_file"], label_to_zero_index)
print(f"Loaded {len(dev_examples)} development examples.")

# Prepare tokenizer and datasets
tokenizer = AutoTokenizer.from_pretrained(config["model_name"], truncation_side="left")
train_dataset = LegalClassificationDataset(train_examples, tokenizer)
dev_dataset = LegalClassificationDataset(dev_examples, tokenizer)

print("Starting training...")
model = train_model(train_dataset, config["model_name"], tokenizer, config)

print("Evaluating on development data...")
dev_accuracy = evaluate_epoch(model, dev_dataset, config["per_device_train_batch_size"])
print(f"Development set top-1 accuracy: {dev_accuracy:.4f}")


# # # # # # # #
# SAVE MODEL #
# # # # # # # #

# Create model identifier
model_identifier = (
    f"{os.path.basename(config['model_name'])}_"
    f"{data_suffix}"
)

base_path = "finetuned_distilbert_models"

# Create full save path
save_path = os.path.join(base_path, model_identifier)
os.makedirs(save_path, exist_ok=True)

# Save model and tokenizer
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model saved to: {save_path}")