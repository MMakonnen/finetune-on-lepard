import torch
from torch.utils.data import Dataset

class LegalClassificationDataset(Dataset):
    """
    Dataset for legal text classification.
    Each example is a tuple of (input_text, label).
    """
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
        self.device = "cuda"
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]
    def collate_fn(self, batch):
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}
        labels = torch.tensor(labels).to(self.device)
        return {"model_inputs": model_inputs, "label": labels}