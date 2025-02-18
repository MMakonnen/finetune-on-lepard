import torch
import numpy as np
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
import os
import json

def get_special_token_predictions(model, tokenizer, context, special_tokens, top_k=10):
    """
    Get top-k predictions among special tokens for a given context.

    Args:
        model: The fine-tuned language model
        tokenizer: The tokenizer
        context: Input context string
        special_tokens: List of special tokens to consider
        top_k: Number of top predictions to return

    Returns:
        list: Top-k special token predictions
        dict: Probabilities for all special tokens
    """
    # Pre-compute special token IDs
    special_token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(token) for token in special_tokens],
                                   device="cuda")

    # Tokenize input context
    inputs = tokenizer(context, return_tensors="pt").to("cuda")

    # Get logits ONLY for special tokens directly
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract just the last token's logits and only for special tokens
        logits = outputs.logits[:, -1:, special_token_ids]

    # Compute softmax only on the special tokens' logits
    probabilities = torch.nn.functional.softmax(logits.squeeze(1), dim=-1)[0]

    # Get top-k predictions
    top_k_probs, top_k_indices = torch.topk(probabilities, min(top_k, len(special_tokens)))
    top_k_tokens = [special_tokens[idx] for idx in top_k_indices.cpu().numpy()]

    return top_k_tokens, {token: prob.item() for token, prob in zip(special_tokens, probabilities)}


def evaluate_model(model, tokenizer, test_data, special_tokens, num_samples=None, random_seed):
    """
    Evaluate model performance on test data.

    Args:
        model: The fine-tuned language model
        tokenizer: The tokenizer
        test_data: Test dataset
        special_tokens: List of special tokens
        num_samples: Number of samples to evaluate (None for all samples)
        random_seed: Seed for random sampling
    """
    top_1_correct = 0
    top_5_correct = 0
    top_10_correct = 0
    total = 0

    # Enable evaluation mode
    model.eval()

    # Get random subset of test data if num_samples is specified
    test_messages = test_data['messages']
    if num_samples is not None:
        num_samples = min(num_samples, len(test_messages))
        # Random sampling
        np.random.seed(random_seed)
        random_indices = np.random.choice(len(test_messages), num_samples, replace=False)
        test_messages = [test_messages[i] for i in random_indices]
        print(f"\nEvaluating on {num_samples} randomly sampled examples out of {len(test_data['messages'])} total samples")

    for item in test_messages:
        # Extract context and ground truth
        context = item[0]['content']
        ground_truth = item[1]['content'].strip()

        # Get predictions
        top_10_predictions, _ = get_special_token_predictions(
            model, tokenizer, context, special_tokens, top_k=10
        )

        # Check accuracies
        if ground_truth == top_10_predictions[0]:
            top_1_correct += 1
        if ground_truth in top_10_predictions[:5]:
            top_5_correct += 1
        if ground_truth in top_10_predictions:
            top_10_correct += 1

        total += 1

    # Calculate accuracies
    results = {
        'top_1_accuracy': top_1_correct / total,
        'top_5_accuracy': top_5_correct / total,
        'top_10_accuracy': top_10_correct / total,
        'samples_evaluated': total
    }

    return results


def save_finetuned_model(model, tokenizer, config, base_path="finetuned_models"):
    """
    Save the fine-tuned model and tokenizer.

    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        config: Configuration dictionary
        base_path: Base directory for saving models
    """
    # Create data suffix
    data_suffix = (
        f"{config['dataset']}k_"
        f"percent{int(config['data_usage_fraction'] * 100)}_"
        f"split{''.join(str(int(v * 10)) for v in config['train_test_val_split'].values())}_"
        f"seed{config['seed']}"
    )

    # Create model identifier
    model_identifier = (
        f"{os.path.basename(config['model'])}_"
        f"{data_suffix}"
    )

    # Create full save path
    save_path = os.path.join(base_path, model_identifier)
    os.makedirs(save_path, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model saved to: {save_path}")
    return save_path



def load_finetuned_model(model_path, config, max_seq_length=2048):
    """
    Load a fine-tuned model for evaluation.

    Args:
        model_path: Path to the saved model
        max_seq_length: Maximum sequence length

    Returns:
        model, tokenizer
    """

    # Load base model and tokenizer
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"],
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True
    )

    # Load saved tokenizer and get its vocabulary
    saved_tokenizer = AutoTokenizer.from_pretrained(model_path)
    new_tokens = list(set(saved_tokenizer.get_vocab().keys()) - set(tokenizer.get_vocab().keys()))

    # Add the new tokens to current tokenizer
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        base_model.resize_token_embeddings(len(tokenizer))

    # Load LoRA adapters
    model = FastLanguageModel.get_peft_model(
        base_model,
        r=config["lora_rank_approx"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"]
    )

    # Load the trained weights
    model.load_adapter(model_path, "default")
    FastLanguageModel.for_inference(model)

    return model, tokenizer