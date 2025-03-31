from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

from distilbert_config import config
from distilbert_data_utils import load_data_csv, load_special_token_map
from distilbert_data_class import LegalClassificationDataset
from distilbert_train_eval_utils import evaluate_top_k

# device
device = "cuda"

# Load the model
model_path = f"finetuned_distilbert_models/{config['finetuned_model']}"

print(f"Loading model and tokenizer from {model_path}...")
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side="left")

# Load special token mapping
special_token_map_path = os.path.join("passage_special_token_map", "passage_to_token.json")
special_token_map, label_to_zero_index, zero_index_to_label = load_special_token_map(special_token_map_path)

# Sanity check: ensure config["n_labels"] matches the actual number of labels
expected_n_labels = len(label_to_zero_index)
if config.get("n_labels") is not None and config["n_labels"] != expected_n_labels:
    raise ValueError(f"Mismatch in number of labels: config['n_labels']={config['n_labels']} vs expected={expected_n_labels}")


# Load test data
print("Loading test data...")
test_examples = load_data_csv(config["test_file"], label_to_zero_index)
print(f"Loaded {len(test_examples)} training examples.")

test_dataset = LegalClassificationDataset(test_examples, tokenizer)


# Evaluate the model for top-1, top-5, and top-10 accuracy
batch_size = config["per_device_train_batch_size"]
accuracies = evaluate_top_k(model, test_dataset, batch_size, ks=[1, 5, 10])
print("Test Retrieval Accuracies:")
for k in sorted(accuracies):
    print(f"Top-{k} Accuracy: {accuracies[k]:.4f}")

# Note: The accuracy returned in the training script (dev set) is top-1 accuracy.