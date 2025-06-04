"""
Fine-tune NV-Embed-v2 with LoRA adapters for legal passage retrieval.
Includes:
  - NV-specific prompt prefix + EOS, with enriched context option
  - Right-side padding for causal model
  - LoRA adapter with FEATURE_EXTRACTION and NV target modules
  - CachedMultipleNegativesRankingLoss with NO_DUPLICATES batch sampler
  - In-epoch dev evaluation (recall@1/5/10)
  - DataCollatorWithPadding for pad-to-multiple-of-8
  - Dynamic save paths from config + build_data_suffix
"""
import os
import sys
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# local imports
from src.data_prep_utils import build_data_suffix
from sbert_config import config
from sbert_data_utils import load_sbert_dataset
from sbert_eval_utils import get_ir_evaluator


def main():
    # Paths & dynamic save directory
    model_name = config['model_name']
    train_path = config['finetuning_data']['train_path']
    valid_path = config['finetuning_data']['valid_path']
    base_path  = config.get('save_path', 'sbert_finetuned_model')
    suffix     = build_data_suffix(config)
    prefix     = 'extended_' if config.get('use_enriched_context', False) else ''
    model_id   = f"{os.path.basename(model_name)}_{prefix}{suffix}"

    # final-model goes here:
    save_path  = os.path.join(base_path, model_id)
    os.makedirs(save_path, exist_ok=True)

    # scratch for all the intermediate checkpoints
    ckpt_dir   = "/cluster/scratch/mmakonnen/checkpoints_model_nv"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Hyperparameters
    BATCH_SIZE    = config.get('batch_size', 16)
    NUM_EPOCHS    = config.get('num_train_epochs', 2)
    LEARNING_RATE = config.get('learning_rate', 2e-5)
    WARMUP_RATIO  = config.get('warmup_ratio', 0.1)
    LORA_R        = config.get('lora_r', 32)
    LORA_ALPHA    = config.get('lora_alpha', 128)
    LORA_DROPOUT  = config.get('lora_dropout', 0.1)
    USE_BF16      = config.get('use_bf16', True)

    # 1) Load NV-Embed-v2 model
    model_kwargs = {'torch_dtype': torch.bfloat16} if USE_BF16 else {}
    model = SentenceTransformer(
        model_name,
        trust_remote_code='nv' in model_name.lower(),
        model_kwargs=model_kwargs
    )
    # Right-side padding for causal attention
    model.tokenizer.padding_side = 'right'
    model.max_seq_length = config.get('max_seq_length', 512)

    # # 2) Optionally load existing LoRA adapter
    # if config.get('lora_name'):
    #     model.load_adapter(config['lora_name'])

    # 3) Load & prefix train data
    train_ds = load_sbert_dataset(train_path)
    # Determine instruction prefix based on enriched context flag
    if config.get('use_enriched_context', False):
        instruction = (
            "Given a legal context and the court it appeared in, "
            "retrieve the most relevant precedent passage for which we also know "
            "the court it came from and its date.\nContext: "
        )
    else:
        instruction = (
            "Given a legal context, retrieve the most relevant precedent passage.\n"
            "Context: "
        )
    eos = model.tokenizer.eos_token
    pref_qs = [instruction + q + eos for q in train_ds['question']]
    train_ds = Dataset.from_dict({'question': pref_qs, 'answer': train_ds['answer']})

    # 4) Load & prefix validation data (same formatting)
    valid_ds = load_sbert_dataset(valid_path)
    pref_qs_val = [instruction + q + eos for q in valid_ds['question']]
    valid_ds = Dataset.from_dict({'question': pref_qs_val, 'answer': valid_ds['answer']})

    # 5) Build evaluator for dev-set
    dev_eval = get_ir_evaluator(valid_ds, name='dev-eval')

    # 6) Configure LoRA adapter for NV layers
    peft_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=['q_proj','k_proj','v_proj','o_proj'],
        bias='none'
    )
    model.add_adapter(peft_cfg)

    # 7) Loss and collator
    loss = CachedMultipleNegativesRankingLoss(model)
    # collator = model.smart_batching_collate

    # 8) Training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE*2,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        fp16=False,
        bf16=USE_BF16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        logging_steps=50,
        logging_first_step=True,
        run_name=model_id
    )

    # 9) Instantiate Trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        evaluator=dev_eval,
        loss=loss,
        #data_collator=collator
    )

    trainer.train()

    # 10) Final save
    model.save(save_path)
    print(f"Training complete. Final model saved to {save_path}")

if __name__ == '__main__':
    main()
