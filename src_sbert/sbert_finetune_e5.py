
# TODO:
# - 
# - ask about hyperpar: 2 train epochs???, dont just use 1
# - python version, what packages and versions to install...
# - first test with subset of data
# - inference script
# - adjust text feeding in format
# - solid reference: https://github.com/kamalkraj/e5-mistral-7b-instruct/blob/master/lora.json
# - max seq length extend and test out -> ask domnik about this
# - test on subset of the data, e.g. 25% and compare to prior performance
# - potentially WANDB disable (os.environ["WANDB_DISABLED"] = "True")
# - move some stuff into conda file
# - later: make unified script for testing different models
# - how do eval with test train val, we know correct passage and has to be able to tell this ...
# - make file more modular + move more stuff in to config file, make it easier to split between models
# - push everything to github
# - share to euler cluster ...


import os
import sys
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# local imports
from src.data_prep_utils import build_data_suffix
from src_sbert.sbert_config_e5 import config
from sbert_data_utils import load_sbert_dataset
from sbert_eval_utils import get_ir_evaluator

def is_bfloat16_supported():
    # Checks if the current GPU supports BF16
    return torch.cuda.is_bf16_supported()

os.environ["WANDB_DISABLED"] = "True"

# Paths & dynamic save directory
model_name = config['model_name']
train_path = config['finetuning_data']['train_path']
valid_path = config['finetuning_data']['valid_path']
base_path  = config['save_path']
suffix     = build_data_suffix(config)
prefix     = 'extended_' if config.get('use_enriched_context', False) else ''
model_id   = f"{os.path.basename(model_name)}_{prefix}{suffix}"


# final model saved here
save_path  = os.path.join(base_path, model_id)
os.makedirs(save_path, exist_ok=True)

# scratch for all the intermediate checkpoints
ckpt_dir   = "/cluster/scratch/mmakonnen/checkpoints_model_sbert_mistral"
os.makedirs(ckpt_dir, exist_ok=True)


# Hyperparameters
BATCH_SIZE    = config.get('batch_size', 4)
NUM_EPOCHS    = config.get('num_train_epochs', 2)
LEARNING_RATE = config.get('learning_rate', 2e-5)
WARMUP_RATIO  = config.get('warmup_ratio', 0.1)
LORA_R        = config.get('lora_r', 32)
LORA_ALPHA    = config.get('lora_alpha', 128)
LORA_DROPOUT  = config.get('lora_dropout', 0.1)
USE_BF16      = config.get('use_bf16', True)


model_kwargs = {"torch_dtype":torch.bfloat16}
model = SentenceTransformer(model_name, model_kwargs=model_kwargs)

model.max_seq_length = 512 # POTENTIALLY ADJUST THIS ???, model can have larger !!!


# train split
train_ds = load_sbert_dataset(train_path)

if config.get("use_enriched_context", False):
    task_desc = (
        "Given a legal context and the court it appeared in, "
        "retrieve the most relevant precedent passage."
    )
else:
    task_desc = (
        "Given a legal context, retrieve the most relevant precedent passage."
    )

# Final prefix in the E5-style “Instruct … Query …”
prompt_prefix = f"Instruct: {task_desc}\nQuery: "

# Add the prefix without changing the answer column.
train_ds = train_ds.map(
    lambda ex: {
        "question": prompt_prefix + " ".join(ex["question"].split())  # strip weird spacing
    },
    desc="Prefix train queries with E5 instruction"
)

# valid split
valid_ds = load_sbert_dataset(valid_path)

valid_ds = valid_ds.map(
    lambda ex: {
        "question": prompt_prefix + " ".join(ex["question"].split())
    },
    desc="Prefix validation queries with E5 instruction"
)

# evaluator for dev set
dev_eval = get_ir_evaluator(valid_ds, name="dev-eval")


peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "down_proj", "up_proj", "gate_proj"
    ],
    bias="none"
)
model.add_adapter(peft_config)

loss = CachedMultipleNegativesRankingLoss(model)

training_args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=ckpt_dir,
    # Optional training parameters:
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE*2,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    fp16=not is_bfloat16_supported(),  # Set to False if you get an error that your GPU can't run on FP16
    bf16=is_bfloat16_supported(),  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=50,
    logging_first_step=True,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    loss=loss,
    evaluator=dev_eval,
    eval_dataset=valid_ds,
    #collator=DataCollatorWithPadding(model.tokenizer, padding=True, pad_to_multiple_of=8)
)

# train model
trainer_stats = trainer.train()
print(trainer_stats)

model.save(save_path)
print(f"Training complete. Final model saved to {save_path}")
