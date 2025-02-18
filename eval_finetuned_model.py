import json

# local imports
from config import config
from eval_utils import evaluate_model, load_finetuned_model


json_file_path = "passage_special_token_map/passage_to_token.json"

with open(json_file_path, "r") as file:
    passage_to_token = json.load(file)

# Extract special tokens from the mapping
special_tokens = list(passage_to_token.values())

# Load the model
model_path = f"finetuned_models/{config['finetuned_model']}"
model, tokenizer = load_finetuned_model(model_path, config)

# Load test data
test_path = f"finetuning_data/{config['test_data']}"
with open(test_path, 'r') as f:
    test_data = json.load(f)

# Run evaluation
results = evaluate_model(model, tokenizer, test_data, special_tokens,
                        num_samples=config['num_samples_to_evaluate'], random_seed=config['seed'])

# Print results
print("\nEvaluation Results:")
print(f"Number of samples evaluated: {results['samples_evaluated']}")
print(f"Top-1 Accuracy: {results['top_1_accuracy']:.4f}")
print(f"Top-5 Accuracy: {results['top_5_accuracy']:.4f}")
print(f"Top-10 Accuracy: {results['top_10_accuracy']:.4f}")