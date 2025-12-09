import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random
import os
import yaml

def split_date(dates, cutoff):
    train_date = [x for x in dates if x <= cutoff]
    test_date = sorted(list(set(dates) - set(train_date)))
    return train_date, test_date


def sample_nested_config(search_space_dict):
    """
    Recursively sample from a nested search space dictionary, preserving structure.
    Example: {"lstm": {"input_size": [10]}} -> {"lstm": {"input_size": 10}}
    """
    sampled = {}
    for key, value in search_space_dict.items():
        if isinstance(value, dict):
            # Recursively sample nested dictionaries
            sampled[key] = sample_nested_config(value)
        elif isinstance(value, list):
            # Sample from list of values
            sampled[key] = random.choice(value)
        else:
            # Keep as is (non-list values)
            sampled[key] = value
    return sampled


def save_yaml_config(path_dir, **config):
    os.makedirs(path_dir, exist_ok=True)
    yaml.dump(config, open(os.path.join(path_dir, "config.yaml"), "w"))