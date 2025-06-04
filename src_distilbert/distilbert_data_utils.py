import pandas as pd
import json

def load_data_csv(file_path, text_col, label_to_zero_index=None):
    """
    Load data from a CSV file with optional label mapping.
    
    Args:
        file_path (str): Path to the CSV file
        label_to_zero_index (dict, optional): Mapping of original labels to zero-indexed labels
    
    Returns:
        list: List of (input_text, label) tuples
    """
    df = pd.read_csv(file_path)
    examples = []
    for _, row in df.iterrows():
        input_text = str(row[text_col]).strip()
        original_label = f"{row['passage_id']}"
        
        if label_to_zero_index is not None:
            label = label_to_zero_index.get(original_label)
            if label is None:
                continue
        else:
            label = original_label
        
        examples.append((input_text, label))
    return examples


def load_special_token_map(json_path):
    """
    Load the special token mapping from JSON file.
    
    Args:
        json_path (str): Path to the JSON file containing special token mapping
    
    Returns:
        dict: Mapping of original labels to special tokens
        dict: Mapping of original labels to zero-indexed labels
        dict: Reverse mapping of zero-indexed labels to original labels
    """
    with open(json_path, 'r') as f:
        special_token_map = json.load(f)
    
    numeric_keys = [k for k in special_token_map.keys()]
    sorted_keys = sorted(numeric_keys, key=lambda x: int(x.split('_')[-1]))
    
    label_to_zero_index = {label: idx for idx, label in enumerate(sorted_keys)}
    zero_index_to_label = {v: k for k, v in label_to_zero_index.items()}
    
    return special_token_map, label_to_zero_index, zero_index_to_label


# def format_extended_example(example):
#     return {
#         'passage_id': example['passage_id'],
#         'input_text': (
#             f"<DEST COURT>{example['dest_court']}</DEST COURT> "
#             f"<SOURCE COURT>{example['source_court']}</SOURCE COURT> "
#             f"<SOURCE DATE>{example['source_date']}</SOURCE DATE> "
#             f"<DEST CONTEXT>{example['destination_context']}</DEST CONTEXT>"
#         )
#     }

def format_extended_example(example):
    return {
        'passage_id': example['passage_id'],
        'input_text': (
            f"<DEST COURT>{example['dest_court']}</DEST COURT> "
            f"<DEST CONTEXT>{example['destination_context']}</DEST CONTEXT>"
        )
    }