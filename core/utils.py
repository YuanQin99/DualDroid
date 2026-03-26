import os
import json
from pathlib import Path
import pickle


def build_opcode_vocab(root_dir: str):
    """Scan malicious/benign folders and build a global opcode-to-index mapping."""
    if os.path.exists(os.path.join(root_dir, "opcode_vocab.pkl")):
        return load_data(os.path.join(root_dir, "opcode_vocab.pkl"))

    vocab = set()
    for label in ("malicious", "benign"):
        d = os.path.join(root_dir, label)
        for fn in os.listdir(d):
            if not fn.endswith(".pkl"):
                continue
            data = load_data(os.path.join(d, fn))
            for _, attrs in data["call_graph"]["nodes"]:
                vocab.update(attrs["opcodes"])

    vocab = {opcode: i for i, opcode in enumerate(sorted(vocab))}
    save_data(vocab, os.path.join(root_dir, "opcode_vocab.pkl"))
    return vocab


def build_permission_vocab(root_dir: str):
    """Scan malicious/benign folders and build a global permission-to-index mapping."""
    if os.path.exists(os.path.join(root_dir, "permission_vocab.pkl")):
        return load_data(os.path.join(root_dir, "permission_vocab.pkl"))

    vocab = set()
    for label in ("malicious", "benign"):
        d = os.path.join(root_dir, label)
        for fn in os.listdir(d):
            if not fn.endswith(".pkl"):
                continue
            data = load_data(os.path.join(d, fn))
            for _, attrs in data["call_graph"]["nodes"]:
                vocab.update(attrs["permissions"])

    vocab = {permission: i for i, permission in enumerate(sorted(vocab))}
    save_data(vocab, os.path.join(root_dir, "permission_vocab.pkl"))
    return vocab


def save_features(features, output_path):
    """Save features to the specified path."""
    save_data(features, output_path)


def load_features(input_path):
    """Load features from the specified path."""
    return load_data(input_path)


def save_data(data, output_path):
    """Save data to the specified path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def load_data(input_path):
    """Load data from the specified path."""
    input_path = Path(input_path)
    with open(input_path, "rb") as f:
        return pickle.load(f)

def set_seed(seed=42):
    """Set all random seeds to ensure reproducibility."""
    import random
    import numpy as np
    import torch

    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU
    
    # Ensure deterministic CUDA behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable (optional, stricter determinism)
    os.environ['PYTHONHASHSEED'] = str(seed)
     