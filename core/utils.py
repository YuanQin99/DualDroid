import os
import json
from pathlib import Path
import pickle


def build_opcode_vocab(root_dir: str):
    """扫描恶意/良性文件夹，构建全局 opcode->索引 映射"""
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
    """扫描恶意/良性文件夹，构建全局 权限->索引 映射"""
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
    """保存特征到指定路径"""
    save_data(features, output_path)


def load_features(input_path):
    """从指定路径加载特征"""
    return load_data(input_path)


def save_data(data, output_path):
    """保存数据到指定路径"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def load_data(input_path):
    """从指定路径加载数据"""
    input_path = Path(input_path)
    with open(input_path, "rb") as f:
        return pickle.load(f)

def set_seed(seed=42):
    """设置所有随机种子以确保可复现性"""
    import random
    import numpy as np
    import torch

    # Python 内置随机数
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量（可选，更严格的确定性）
    os.environ['PYTHONHASHSEED'] = str(seed)
     