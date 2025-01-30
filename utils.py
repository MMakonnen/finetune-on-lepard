

def validate_config(config):
    """Sanity checks for configuration values."""
    # Validate dataset selection
    allowed_datasets = {'10', '20', '50'}
    assert config['dataset'] in allowed_datasets, \
        f"Invalid dataset: {config['dataset']}. Must be one of {allowed_datasets}"

    # Validate splits
    splits = config['train_test_val_split']
    assert set(splits.keys()) == {'train', 'valid', 'test'}, \
        "Split keys must be 'train', 'valid', 'test'"

    for name, frac in splits.items():
        assert 0 <= frac <= 1, \
            f"Invalid {name} fraction {frac} - must be between 0 and 1"

    total = sum(splits.values())
    assert abs(total - 1.0) < 1e-3, \
        f"Splits sum to {total:.2f} - must sum to 1.0"

    # Validate data usage fraction
    assert 0 < config['data_usage_fraction'] <= 1, \
        f"Invalid data_usage_fraction {config['data_usage_fraction']} - must be in (0, 1]"

    print("Config validation passed")



def count_unique_values(dataset, column):
    """Return count of unique values in a column."""
    return len(set(dataset[column]))