config = {

    # Dataset Selection
    # Options: '10', '20', '50' for 10k, 20k, or 50k dataset
    'dataset_size': '10',              

    # Train-Test-Validation Split Ratios
    'train_test_val_split': {
        'train': 0.8,
        'validation': 0.1,
        'test': 0.1
    },

    # Data Usage Fraction
    # Fraction of the complete dataset to use for faster iterations
    'data_usage_fraction': 0.2
}
