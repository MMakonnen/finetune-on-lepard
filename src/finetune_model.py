
# ...
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template


# local imports
from config import config
from data_prep_utils import load_and_tokenize_finetune_data
from eval_utils import save_finetuned_model



max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


# ... download pretrained model ...
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config['model'],
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# Define and set a chat template for the LLama3 model
tokenizer = get_chat_template(
    tokenizer,
    chat_template=config["chat_template"],
)


# Extend vocabulary tokenizer by special tokens and resize tokenizer embedding

# Load previously used passage id special token dictionary to ensure consistent mapping
json_file_path = "passage_special_token_map/passage_to_token.json"
with open(json_file_path, "r") as file:
    passage_to_token = json.load(file)

# Extract special tokens from the mapping
special_tokens = list(passage_to_token.values())

# extend vocab tokenizer by special tokens and resize tokenizer embedding
tokenizer.add_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

print(f"Added {len(special_tokens)} special tokens to the tokenizer.")


# add LoRA adapters so only 1 to 10% of all par need to be updated
# NOTE: importantly this has to be done after having extended the tokenizer vocab &
model = FastLanguageModel.get_peft_model(
    model,
    r = config['lora_rank_approx'], # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = config['seed'],
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# Load train data
train_path = f"finetuning_data/{config['train_data']}"
train = load_and_tokenize_finetune_data(train_path, tokenizer)
dataset = Dataset.from_dict(train)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        warmup_steps=5,
        num_train_epochs = config['num_train_epochs'] if config.get('full_epoch', False) else -1,
        max_steps = -1 if config.get('full_epoch', False) else config['max_steps'],
        learning_rate=config['learning_rate'],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=config['seed'],
        output_dir=config['checkpoint_path'],
        report_to=[],
    ),
)

# train model
trainer_stats = trainer.train()
print(trainer_stats)

# Save the model after training
save_path = save_finetuned_model(model, tokenizer, config)
print(f"Model saved to: {save_path}")