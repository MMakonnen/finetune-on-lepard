import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score


def train_model(train_dataset, model_name, tokenizer, config):
    """
    Fine-tune DistilBERT for legal classification.
    This implementation updates once per batch.
    """
    device = "cuda"
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = config["n_labels"]

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config).to(device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    total_steps = len(train_dataloader) * config["num_train_epochs"]

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 8e-6,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config["learning_rate"],
        eps=config["adam_epsilon"]
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.train()
    for epoch in range(config["num_train_epochs"]):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            outputs = model(**batch["model_inputs"], labels=batch["label"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} finished with average loss: {avg_loss:.4f}")
    return model


def evaluate_epoch(model, dataset, batch_size):
    """
    Evaluate the model on the given dataset.
    Returns the accuracy over the dataset.
    """
    model.eval()
    all_labels = []
    all_preds = []
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch["model_inputs"])
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
    return accuracy_score(all_labels, all_preds)


def evaluate_top_k(model, dataset, batch_size, ks=[1, 5, 10]):
    """
    Evaluate the model to compute top-k retrieval accuracy.
    For each sample, if the true label is among the top k predictions, count it as correct.
    """
    model.eval()
    # Counters for correct predictions per k
    correct_counts = {k: 0 for k in ks}
    total = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            outputs = model(**batch["model_inputs"])
            logits = outputs.logits  # shape: (batch_size, num_labels)
            # Get sorted indices (highest logits first)
            sorted_indices = torch.argsort(logits, dim=1, descending=True)
            true_labels = batch["label"]
            batch_size_current = true_labels.size(0)
            total += batch_size_current
            for k in ks:
                # For each sample, check if the true label is within top k predictions
                topk = sorted_indices[:, :k]
                # Compare true label with topk predictions for each sample
                match = (topk == true_labels.unsqueeze(1)).any(dim=1)
                correct_counts[k] += match.sum().item()
    # Calculate accuracies for each k
    accuracies = {k: correct_counts[k] / total for k in ks}
    return accuracies