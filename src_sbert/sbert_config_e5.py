config = {
    "seed": 42,

    # Dataset options: '10', '20', '50' corresponding to top 10k, 20k, or 50k passages
    "dataset": "10",

    # For a subset of the data set to <1
    "data_usage_fraction": 0.25,

    "train_test_val_split": {
        "train": 0.9,
        "valid": 0.05,
        "test": 0.05
    },

    # Use additional legal metadata (source/destination court and source date) alongside context
    "use_enriched_context": True,

    "model_name": "intfloat/e5-mistral-7b-instruct",

    "finetuning_data": {
        "train_path": "finetuning_data_sbert/train_sbert_extended_10k_percent25_split900505_seed42.json",
        "valid_path": "finetuning_data_sbert/valid_sbert_extended_10k_percent25_split900505_seed42.json",
        "test_path": "finetuning_data_sbert/test_sbert_extended_10k_percent25_split900505_seed42.json"
    },

    # Training duration
    "num_train_epochs": 1,

    # Directory to save trained model and checkpoints
    "save_path": "sbert_finetuned_model",
}