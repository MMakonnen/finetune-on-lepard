import json
import os
import sys
from datasets import Dataset

# Add parent directory to system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# local imports
from src.data_prep_utils import build_data_suffix


def format_query(item, config):
    if config.get("use_enriched_context", False):
        return (
            f"<DEST COURT>{item['dest_court']}</DEST COURT> "
            f"<CONTEXT>{item['destination_context']}</CONTEXT>"
        )
    else:
        return item["destination_context"]

def format_paragraph(item, passage, config):
    if config.get("use_enriched_context", False):
        return (
            f"<SOURCE COURT>{item['source_court']}</SOURCE COURT> "
            f"<SOURCE DATE>{item['source_date']}</SOURCE DATE> "
            f"<PARAGRAPH>{passage}</PARAGRAPH>"
        )
    else:
        return passage


def prepare_sbert_pairs(dataset_split, passage_dict, config):
    queries, paragraphs = [], []
    for item in dataset_split:
        pid = item["passage_id"]
        passage = passage_dict.get(pid)

        q = format_query(item, config)
        p = format_paragraph(item, passage, config)

        queries.append(q)
        paragraphs.append(p)

    return queries, paragraphs


def load_data(split, config):
    data_suffix = build_data_suffix(config)
    path = f"finetuning_data_sbert/{split}_sbert_{data_suffix}.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"], data["paragraphs"]


def load_sbert_dataset(path: str) -> Dataset:
    """
    Load JSON file with 'queries' and 'paragraphs' lists into HF Dataset.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    qs = data['queries']
    ps = data['paragraphs']
    assert len(qs) == len(ps), 'Queries and paragraphs length mismatch'
    return Dataset.from_dict({'question': qs, 'answer': ps})