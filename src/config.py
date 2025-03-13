config = {
    # Random seed for reproducibility
    "seed": 42,  
    
    # Dataset Selection
    # Options: '10', '20', '50' for 10k, 20k, or 50k dataset (top passages of precedent)
    "dataset": "10",       

    # Data Usage Fraction
    # Fraction of the complete dataset to use for faster iterations
    "data_usage_fraction": 0.2,       

    # Train-Test-Validation Split
    "train_test_val_split": {
        "train": 0.8,
        "valid": 0.1,
        "test": 0.1
    },

    # Model options (choose based on available compute)
    # - "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    # - "unsloth/Meta-Llama-3.1-70B-bnb-4bit"
    # - "unsloth/Meta-Llama-3.1-405B-bnb-4bit"
    "model": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",

    # path to where model checkpoints can be saved (can get large for many training steps, 
    # hence for example move to temporary storage)
    "checkpoint_path": "/cluster/scratch/mmakonnen/checkpoints_model",

    # LoRA settings
    # Controls trainable LoRA parameters; suggested values: 8, 16, 32, 64, 128
    "lora_rank_approx": 128,

    # set learning rate to be used for finetuning
    "learning_rate": 2e-4,

    # Training mode
    # If True, runs for 'num_train_epochs'; otherwise, runs for 'max_steps'
    "full_epoch": True,

    # Training duration
    "num_train_epochs": 1,  # Used if 'full_epoch' is True
    "max_steps": 5000,  # Used if 'full_epoch' is False

    # Training batch settings
    "per_device_train_batch_size": 8,  # Increase if memory allows; affects speed and stability
    "gradient_accumulation_steps": 4,  # Higher values reduce VRAM usage by simulating larger batches

    # number of samples from test set to test (useful for faster iterations and sanity checks)
    "num_samples_to_evaluate": None, # None if want to eval all test samples

    # train and test data to be used 
    # NOTE: first need to create these through create_finetuning_dataset script, hence first 
    # entry here does not matter until those datasets created
    "train_data": "train_10k_percent20_split811_seed42.json",
    "test_data": "test_10k_percent20_split811_seed42.json",

    # set chat template
    "chat_template": "llama-3.1",

    # model folder of finetuned model to be used for evaluation 
    # NOTE: first need to finetune model, hence first entry here does not matter until eval
    "finetuned_model": "Meta-Llama-3.1-8B-bnb-4bit_10k_percent20_split811_seed42",
}                       