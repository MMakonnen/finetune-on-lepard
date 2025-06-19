# CHECK:
# - model saved with name that does not contain pre name thingy
# - check the code fully
# ...

import os
from sentence_transformers import SentenceTransformer

# local imports
from sbert_config_e5 import config
from sbert_data_utils import load_sbert_dataset
from sbert_eval_utils import get_ir_evaluator


# 1) Reconstruct the path where your fine-tuned model was saved
base_path  = config['save_path']
suffix     = __import__('src.data_prep_utils', fromlist=['build_data_suffix']).build_data_suffix(config)
prefix     = 'extended_' if config.get('use_enriched_context', False) else ''
model_id   = f"{os.path.basename(config['model_name'])}_{prefix}{suffix}"
save_path  = os.path.join(base_path, model_id)

# 2) Load the SBERT pipeline (includes the LoRA adapters you saved)
model = SentenceTransformer(save_path, trust_remote_code=True)

# 3) Recreate the same prefix you used during training
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

# 4) Load & prefix the test split
test_ds = load_sbert_dataset(config['finetuning_data']['test_path'])
test_ds = test_ds.map(
    lambda ex: {"question": prompt_prefix + " ".join(ex["question"].split())},
    desc="Prefix test queries"
)

# 5) Build the evaluator and run
evaluator = get_ir_evaluator(test_ds, name="test-eval")
metrics   = evaluator(model)

# 6) Print Recall@1, @5, @10
print(f"Test Recall@1:  {metrics['test-eval_recall@1']:.4f}")
print(f"Test Recall@5:  {metrics['test-eval_recall@5']:.4f}")
print(f"Test Recall@10: {metrics['test-eval_recall@10']:.4f}")
