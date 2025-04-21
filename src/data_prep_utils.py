from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import os
import json
import numpy as np
import sys

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# local imports
from src_distilbert.distilbert_data_utils import format_extended_example


def prep_contexts(contexts, cols_to_keep):
    """
    Remove duplicates and keep selected columns.
    
    Args:
        contexts (DatasetDict): Input dataset.
        cols_to_keep (list): Columns to retain.
    
    Returns:
        Dataset: Processed dataset.
    """
    df = contexts['train'].to_pandas()
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[cols_to_keep]
    return Dataset.from_pandas(df)


def sample_data_with_all_passages(dataset, fraction, seed):
    """
    Subsample dataset while keeping all passage IDs.
    
    Args:
        dataset (Dataset): Input dataset.
        fraction (float): Sample fraction.
        seed (int): Random seed.
    
    Returns:
        Dataset: Sampled dataset.
    """
    df = dataset.to_pandas()
    df = df.groupby('passage_id', group_keys=False).apply(
        lambda x: x.sample(frac=fraction, random_state=seed)
    ).reset_index(drop=True)
    return Dataset.from_pandas(df)


def stratified_split(dataset, splits, stratify_col, seed):
    """
    Perform stratified dataset split.
    
    Args:
        dataset (Dataset): Input dataset.
        splits (dict): Train/validation/test split ratios.
        stratify_col (str): Column for stratification.
        seed (int): Random seed.
    
    Returns:
        DatasetDict: Train, validation, and test splits.
    """

    df = dataset.to_pandas()
    train_split, val_split, test_split = splits.values()
    remaining_split = val_split + test_split

    # Initial split: train vs remaining (val + test)
    train_df, remaining_df = train_test_split(
        df,
        test_size=remaining_split,
        stratify=df[stratify_col],
        random_state=seed
    )

    # Relative proportion of test in remaining data
    test_ratio_in_remaining = test_split / remaining_split
    
    # Split remaining into validation and test
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=test_ratio_in_remaining,
        stratify=remaining_df[stratify_col],
        random_state=seed
    )
    
    return DatasetDict({
        'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
        'valid': Dataset.from_pandas(val_df.reset_index(drop=True)),
        'test': Dataset.from_pandas(test_df.reset_index(drop=True))
    })


def generate_special_tokens(passage_ids, output_folder=None):
    """
    Map unique passage IDs to special tokens.
    
    Args:
        passage_ids (list): Unique passage IDs.
        output_folder (str, optional): Folder to save the mapping.
    
    Returns:
        dict: Passage ID to token mapping.
        list: Generated special tokens.
    """
    passage_ids = sorted(set(passage_ids))  # Ensure consistent order of unique passage IDs
    
    special_tokens = [f"<special_token_{i}>" for i in range(1, len(passage_ids) + 1)]
    passage_to_token = dict(zip(passage_ids, special_tokens))
    
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)  # Create folder if it does not exist
        output_path = os.path.join(output_folder, "passage_to_token.json")
        with open(output_path, "w") as f:
            json.dump(passage_to_token, f, indent=2)
        print(f"JSON file saved to {output_path}")
    
    return passage_to_token, special_tokens


def build_data_suffix(config):
    """
    Build a filename suffix from config parameters.

    Format: "{dataset}k_percent{data_usage_percentage}_split{split_str}_seed{seed}"

    Args:
        config (dict): Must include:
            - "dataset": Dataset size in thousands (e.g., "10" for 10k)
            - "data_usage_fraction": Fraction of data used (float)
            - "train_test_val_split": Dict of split ratios
            - "seed": Random seed

    Returns:
        str: Suffix string, e.g., "10k_percent20_split811_seed42"
    """
    dataset_size = config['dataset']
    fraction_percentage = int(config['data_usage_fraction'] * 100)
    splits_dict = config['train_test_val_split']
    split_str = "".join(str(int(v * 100)).zfill(2) for v in splits_dict.values())
    data_suffix = f"{dataset_size}k_percent{fraction_percentage}_split{split_str}_seed{config['seed']}"
    return data_suffix



def create_json_files(context_data_split, passage_to_token, output_folder, config) -> None:
    """
    Generate JSON files for training, validation, and test datasets.

    Args:
        context_data_split (DatasetDict): A Hugging Face DatasetDict containing train, validation, and test splits.
        passage_to_token (dict): Mapping of passage IDs to special tokens.
        output_folder (str): Directory where JSON files will be saved.
        config (dict): Configuration dictionary with dataset size, data usage fraction, and train-test-validation split ratios.
        seed (int): Random seed for reproducibility.

    Returns:
        None: Saves JSON files with user-assistant message interactions.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Build naming suffix (e.g. "10k_percent20_split811_seed42")
    data_suffix = build_data_suffix(config)

    system_prompt = "You are an expert legal-precedent selector."

    # Process each split if it exists
    for split in ['train', 'valid', 'test']:
        if split not in context_data_split:
            continue

        df = context_data_split[split].to_pandas()
        all_messages = []

        # Build user-assistant messages
        for row in df.itertuples(index=False):

            if config.get("use_enriched_context", False):
                example = {
                    "passage_id": "placeholder",  # required by function signature but not used
                    "dest_court": row.dest_court,
                    "source_court": row.source_court,
                    "source_date": row.source_date,
                    "destination_context": row.destination_context,
                }
                enriched = format_extended_example(example)
                payload = enriched["input_text"]

                middle = (
                    "You are given the following destination court, source court, "
                    "source date and legal context:"
                )
            else:
                # just wrap the bare context in a tag
                payload = f"<preceding_context>{row.destination_context}</preceding_context>"
                middle = "You are given the following legal context:"


            # inject payload under a unified instruction
            user_content = (
                f"{system_prompt}\n\n"
                f"{middle}\n"
                f"{payload}\n\n"
                "Please respond with the corresponding special token."
            )

            assistant_content = passage_to_token[row.passage_id]
            all_messages.append([
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ])

        # Save JSON
        final_dict = {"messages": all_messages}
        format_prefix = "extended_" if config.get("use_enriched_context", False) else ""
        output_filename = f"{split}_{format_prefix}{data_suffix}.json"
        output_path = os.path.join(output_folder, output_filename)
        with open(output_path, "w") as f:
            json.dump(final_dict, f, indent=2)

        print(f"JSON file saved to {output_path}")


def load_and_tokenize_finetune_data(file_path, tokenizer):
    """
    Loads fine-tuning data from a JSON file and applies chat template formatting.
    NOTE:  Store lengths of tokenized examples to analyze message distribution, helping with batching, optimization, etc.

    Args:
        file_path (str): Path to the JSON file containing chat messages.
        tokenizer (object): Tokenizer with an `apply_chat_template` method.

    Returns:
        dict: A dictionary with the key "text" containing a list of formatted chat messages.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    if "messages" not in data:
        raise KeyError("The JSON file must contain a 'messages' key.")

    messages = data["messages"]

    # Tokenization and length analysis
    tokenized_messages = []
    message_lenths = []

    for message in messages:
        formatted_message = tokenizer.apply_chat_template(
            message, add_generation_prompt=False, tokenize=False
        )
        tokenized_messages.append(formatted_message)
        message_lenths.append(len(formatted_message.split()))

    # Print mean and median length statistics for the messages
    print(f"Mean message length: {np.mean(message_lenths):.2f}")
    print(f"Median message length: {np.median(message_lenths)}")

    return {"text": tokenized_messages}