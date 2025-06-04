config = {
    # Random seed for reproducibility
    "seed": 42,
    
    # Dataset specification
    # Options: '10', '20', '50' corresponding to 10k, 20k, or 50k top passages
    "dataset": "10",

    # Use additional legal metadata (source/destination court and source date) alongside context
    "use_enriched_context": True,
    
    # Fraction of the dataset to use (for quick iterations, set <1 to use a subset)
    "data_usage_fraction": 1,
    
    # Train-Validation-Test Split proportions
    "train_test_val_split": {
        "train": 0.9,
        "valid": 0.05,
        "test": 0.05
    },
    
    # Model specifications: using DistilBERT in this case
    "model_name": "distilbert-base-uncased",
    
    # Number of target classes (i.e. the total number of special tokens)
    "n_labels": 10000,
    
    # Learning rate and optimizer settings
    "learning_rate": 2e-5,
    "adam_epsilon": 1e-8,
    
    # Training duration
    "num_train_epochs": 3,
    
    # Training batch settings
    "per_device_train_batch_size": 64,
    "gradient_accumulation_steps": 1,
    
    # Paths to training and development (validation) data JSON files
    # These files should follow the expected format (see load_data function)
    "train_file": "finetuning_data_distilbert/train_distilbert_extended_10k_percent100_split900505_seed42.csv",
    "dev_file": "finetuning_data_distilbert/valid_distilbert_extended_10k_percent100_split900505_seed42.csv",
    "test_file": "finetuning_data_distilbert/test_distilbert_extended_10k_percent100_split900505_seed42.csv",

    # model folder of finetuned distilbert model to be used for evaluation 
    # NOTE: first need to finetune distilbert model, hence entry here does not matter until eval
    "finetuned_model": "distilbert-base-uncased_extended_10k_percent100_split900505_seed42"
}
