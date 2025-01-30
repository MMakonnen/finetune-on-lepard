from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import os
import json


def prep_contexts(contexts, cols_to_keep):
    """Remove duplicates from contexts and keep specific columns."""
    df = contexts['train'].to_pandas()
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[cols_to_keep]
    return Dataset.from_pandas(df)


def sample_data_with_all_passages(dataset, fraction, seed):
    """Subsample data while ensuring all passage_ids are retained."""
    df = dataset.to_pandas()
    df = df.groupby('passage_id', group_keys=False).apply(
        lambda x: x.sample(frac=fraction, random_state=seed)
    ).reset_index(drop=True)
    return Dataset.from_pandas(df)


def stratified_split(dataset, splits, stratify_col, seed):
    """Split dataset into a DatasetDict with stratified 'train', 'validation', and 'test' subsets."""
    df = dataset.to_pandas()
    train_split, val_split, test_split = splits.values()
    remaining_split = val_split + test_split

    # First split: train vs. temp (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=remaining_split,
        stratify=df[stratify_col],
        random_state=seed
    )

    # Second split: val vs. test
    val_test_ratio = test_split / remaining_split
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_test_ratio,
        stratify=temp_df[stratify_col],
        random_state=seed
    )

    # Convert DataFrames back to HF Datasets
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

    # Return a DatasetDict with train/validation/test splits
    return DatasetDict({
        'train': train_dataset,
        'valid': val_dataset,
        'test': test_dataset
    })


def create_json_files(context_data_split, passage_to_token, output_folder, config, seed) -> None:
    """
    Create minimal JSON files for train/validation/test. 
    'context_data_split' is a Hugging Face DatasetDict with splits.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Build naming suffix (e.g. "10k_percent20_split811_seed42")
    dataset_size = config['dataset']
    fraction_percentage = int(config['data_usage_fraction'] * 100)
    splits_dict = config['train_test_val_split']
    split_str = "".join(str(int(v * 10)) for v in splits_dict.values())
    data_suffix = (
        f"{dataset_size}k_"
        f"percent{fraction_percentage}_"
        f"split{split_str}_"
        f"seed{seed}"
    )

    system_prompt = "You are a helpful legal assistant."

    # Process each split if it exists
    for split in ['train', 'valid', 'test']:
        if split not in context_data_split:
            continue

        df = context_data_split[split].to_pandas()
        all_messages = []

        # Build user-assistant messages
        for _, row in df.iterrows():
            passage_id = row['passage_id']
            destination_context = row['destination_context']
            user_content = (
                f"{system_prompt} You are given the following legal context:\n"
                f"<preceding context>{destination_context}</preceding context>\n\n"
                "Please respond with the corresponding special token."
            )
            assistant_content = passage_to_token[passage_id]
            all_messages.append([
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ])

        # Save JSON
        final_dict = {"messages": all_messages}
        output_filename = f"{split}_{data_suffix}.json"
        output_path = os.path.join(output_folder, output_filename)
        with open(output_path, "w") as f:
            json.dump(final_dict, f, indent=2)

        print(f"JSON file saved to {output_path}")