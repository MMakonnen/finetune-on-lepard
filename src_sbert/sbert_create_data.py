import json
import os
import sys
from datasets import load_dataset
from huggingface_hub import hf_hub_download

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from sbert_config import config
from src.utils import validate_config
from src.data_prep_utils import prep_contexts, sample_data_with_all_passages, stratified_split, build_data_suffix
from sbert_data_utils import prepare_sbert_pairs

# Load precedent passage texts and keep only those relevant to contexts
special_token_map_path = "passage_special_token_map/passage_to_token.json"
with open(special_token_map_path, "r", encoding="utf-8") as f:
    special_token_map = json.load(f)

# Load passage_dict.json from Hugging Face
passage_dict_path = hf_hub_download(
    repo_id="rmahari/LePaRD",
    filename="passage_dict.json",
    repo_type="dataset"
)
with open(passage_dict_path, "r", encoding="utf-8") as f:
    passage_dict = json.load(f)["data"]

# or alternatively load passage data locally
# passage_dict_path = "passage_dict/passage_dict.json"
# with open(passage_dict_path, "r", encoding="utf-8") as f:
#     passage_dict = json.load(f)["data"]

# Keep only relevant passages
passage_dict = {k: v for k, v in passage_dict.items() if k in special_token_map}

print("=== Configuration ===")
print(f"Dataset: {config['dataset']}k")
print(f"Train split: {config['train_test_val_split']['train']}")
print(f"Validation split: {config['train_test_val_split']['valid']}")
print(f"Test split: {config['train_test_val_split']['test']}")
print(f"Data usage fraction: {config['data_usage_fraction']}")
print(f"Random seed: {config['seed']}")
print("======================\n")

validate_config(config)

# load context data from Hugging Face
file_name = f"top_{config['dataset']}000_data.csv.gz"
contexts = load_dataset("rmahari/LePaRD", data_files=file_name)

# Preprocess and Sample Subset of Contexts
if config.get("use_enriched_context", False):
    cols_to_keep = ['passage_id', 'destination_context', 'dest_court', 'source_date', 'source_court']
else:
  cols_to_keep = ['passage_id', 'destination_context']
contexts = prep_contexts(contexts, cols_to_keep)
contexts = sample_data_with_all_passages(contexts, config['data_usage_fraction'], config['seed'])

# create splits
contexts = stratified_split(
    dataset=contexts,
    splits=config['train_test_val_split'],
    stratify_col='passage_id',
    seed=config['seed']
)

data_suffix = build_data_suffix(config)
format_prefix = "extended_" if config.get("use_enriched_context", False) else ""

output_dir = "finetuning_data_sbert"
os.makedirs(output_dir, exist_ok=True)

# Save each split in JSON format
for split_name in ["train", "valid", "test"]:
    
    queries, paragraphs = prepare_sbert_pairs(contexts[split_name], passage_dict, config)
    split_data = {"queries": queries, "paragraphs": paragraphs}
    
    split_filename = os.path.join(
        output_dir,
        f"{split_name}_sbert_{format_prefix}{data_suffix}.json"
    )    
    
    with open(split_filename, "w", encoding="utf-8") as f:
        json.dump(split_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {split_name} split to {split_filename}")