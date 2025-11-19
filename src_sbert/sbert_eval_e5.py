import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Local imports
from src.data_prep_utils import build_data_suffix
from src_sbert.sbert_config_e5 import config
from sbert_data_utils import load_sbert_dataset

# Check GPU availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading configuration...")
print(f"Dataset: {config['dataset']}k passages")
print(f"Data usage fraction: {config['data_usage_fraction']}")
print(f"Use enriched context: {config.get('use_enriched_context', False)}")

# Build model path from config
base_path = config['save_path']
suffix = build_data_suffix(config)
prefix = 'extended_' if config.get('use_enriched_context', False) else ''
model_id = f"{os.path.basename(config['model_name'])}_{prefix}{suffix}"
model_path = os.path.join(base_path, model_id)

print(f"Loading model from: {model_path}")

# Load the fine-tuned model
model = SentenceTransformer(model_path, device=device)
print("✓ Model loaded successfully")

# Load test data
print("Loading test data...")
test_path = config['finetuning_data']['test_path']
test_dataset = load_sbert_dataset(test_path)

# Add instruction prefix to queries
if config.get("use_enriched_context", False):
    task_desc = (
        "Given a legal context and the court it appeared in, "
        "retrieve the most relevant precedent passage."
    )
else:
    task_desc = (
        "Given a legal context, retrieve the most relevant precedent passage."
    )

prompt_prefix = f"Instruct: {task_desc}\nQuery: "

test_dataset = test_dataset.map(
    lambda ex: {
        "question": prompt_prefix + " ".join(ex["question"].split())
    },
    desc="Adding instruction prefix to test queries"
)

queries = test_dataset['question']
passages = test_dataset['answer']
print(f"Test set size: {len(queries)} query-passage pairs")

# Encode queries in batches (keep on GPU)
print("Encoding queries...")
batch_size = 256  # Increased batch size for GPU
query_embeddings = []

for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
    batch = queries[i:i + batch_size]
    batch_embeddings = model.encode(batch, convert_to_tensor=True, device=device)
    query_embeddings.append(batch_embeddings)

query_embeddings = torch.cat(query_embeddings, dim=0)

# Encode passages in batches (keep on GPU)
print("Encoding passages...")
passage_embeddings = []

for i in tqdm(range(0, len(passages), batch_size), desc="Encoding passages"):
    batch = passages[i:i + batch_size]
    batch_embeddings = model.encode(batch, convert_to_tensor=True, device=device)
    passage_embeddings.append(batch_embeddings)

passage_embeddings = torch.cat(passage_embeddings, dim=0)

# Normalize embeddings for faster cosine similarity computation
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

print(f"Query embeddings shape: {query_embeddings.shape}")
print(f"Passage embeddings shape: {passage_embeddings.shape}")

# Compute similarity matrix on GPU using matrix multiplication (faster than cosine_similarity)
print("Computing similarities...")
similarities = torch.mm(query_embeddings, passage_embeddings.t())

# Get rankings on GPU
_, ranked_indices = torch.sort(similarities, dim=1, descending=True)

# Compute top-k accuracies using GPU tensors
num_queries = query_embeddings.shape[0]
k_values = [1, 5, 10]
accuracies = {}

# Create a tensor of correct indices (0, 1, 2, ..., num_queries-1)
correct_indices = torch.arange(num_queries, device=device).unsqueeze(1)

for k in k_values:
    # Get top-k predictions for each query
    top_k_predictions = ranked_indices[:, :k]
    
    # Check if correct index is in top-k for each query
    correct_in_topk = torch.any(top_k_predictions == correct_indices, dim=1)
    
    # Calculate accuracy
    accuracy = correct_in_topk.float().mean().item()
    accuracies[k] = accuracy

# Print results
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

for k in k_values:
    acc = accuracies[k]
    print(f"TOP-{k} Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# Show some example retrievals
print("\n" + "-"*80)
print("EXAMPLE RETRIEVALS")
print("-"*80)

num_examples = 5
example_indices = np.random.choice(num_queries, min(num_examples, num_queries), replace=False)

for i, query_idx in enumerate(example_indices):
    print(f"\nExample {i+1}:")
    print(f"Query {query_idx}: {queries[query_idx][:200]}...")
    
    correct_passage_idx = query_idx
    similarity_score = similarities[query_idx, correct_passage_idx].item()  # Convert to Python float
    
    # Move tensor to CPU before using with numpy
    ranked_indices_cpu = ranked_indices[query_idx].cpu().numpy()
    rank_matches = np.where(ranked_indices_cpu == correct_passage_idx)[0]
    rank = rank_matches[0] + 1 if len(rank_matches) > 0 else -1
    
    print(f"Correct passage (rank {rank}, similarity {similarity_score:.4f}):")
    print(f"  {passages[correct_passage_idx][:200]}...")
    
    # Show top-3 retrieved passages
    print("Top-3 retrieved passages:")
    for j in range(min(3, len(ranked_indices_cpu))):
        passage_idx = int(ranked_indices_cpu[j])  # Convert to Python int for indexing
        sim_score = similarities[query_idx, passage_idx].item()  # Convert to Python float
        marker = "✓" if passage_idx == correct_passage_idx else "✗"
        print(f"  {j+1}. {marker} (sim: {sim_score:.4f}) {passages[passage_idx][:150]}...")

print("\n" + "="*80)