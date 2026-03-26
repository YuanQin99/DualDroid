# Python libraries
import torch
from torch.utils.data import random_split
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import os

# Local imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from core.utils import load_features, build_opcode_vocab, build_permission_vocab
from config.logging_config import set_logging

# Configure logging
logger = set_logging(__file__)


# Dataset class
class APKGraphDataset(Dataset):
    """
        A PyG Dataset for loading call-graph data in `.pkl` format.
        Directory structure:
      root_dir/
        malicious/*.pkl
        benign/*.pkl
        All scanning, feature construction, and `Data` object creation
        are done in `load_data()`, and created `Data` objects are cached
        in memory to speed up indexing.
    """

    def __init__(self, root_dir: str, opcode_vocab: dict, permission_vocab: dict, transform=None, pre_transform=None):
        # Call parent constructor; `root_dir` will be assigned to `self.root`
        super().__init__(root=root_dir, transform=transform, pre_transform=pre_transform)
        # Store opcode and permission vocabularies
        self.opcode_vocab = opcode_vocab
        self.permission_vocab = permission_vocab
        # Preload and process all samples, then cache into a list
        self.cache_path = os.path.join(self.root, "data.pkl")
        self.data_list = self.load_data()

    def len(self):
        # Return total number of graphs
        return len(self.data_list)

    def get(self, idx):
        # Return the corresponding `Data` object by index
        return self.data_list[idx]

    def load_data(self):
        """
        Scan `.pkl` files under the root directory,
        build PyG `Data` objects, and return them as a list.
        """
        # Load directly if cache file exists
        if os.path.exists(self.cache_path):
            logger.info(f"Loading cached data from {self.cache_path}")
            return torch.load(self.cache_path, weights_only=False)

        data_list = []
        # Process malicious and benign samples separately
        for label_str, lbl in (("malicious", 1), ("benign", 0)):
            folder = os.path.join(self.root, label_str)
            if not os.path.isdir(folder):
                continue
            for fn in os.listdir(folder):
                # Only process `.pkl` files
                if not fn.endswith(".pkl"):
                    continue

                path = os.path.join(folder, fn)
                # Load feature dictionary
                features = load_features(path)
                cg = features.get("call_graph", {})
                nodes = cg.get("nodes", [])
                # Skip graphs with <= 1 node
                if len(nodes) <= 1:
                    continue

                num_nodes = len(nodes)
                # Feature dim = opcode vocab size + permission vocab size + 1 (`is_sensitive`)
                feat_dim = len(self.opcode_vocab) + len(self.permission_vocab) + 1
                # Initialize node feature matrix [num_nodes, feat_dim]
                x = torch.zeros((num_nodes, feat_dim), dtype=torch.float)

                # Build mapping from node ID to index
                id_map = {}
                for i, (nid, attrs) in enumerate(nodes):
                    id_map[nid] = i
                    # One-hot encode opcodes
                    for op in attrs.get("opcodes", []):
                        idx = self.opcode_vocab.get(op)
                        if idx is not None:
                            x[i, idx] = 1.0
                    # One-hot encode permissions, offset after opcode vocab
                    base = len(self.opcode_vocab)
                    for per in attrs.get("permissions", []):
                        pidx = self.permission_vocab.get(per)
                        if pidx is not None:
                            x[i, base + pidx] = 1.0
                    # `is_sensitive` flag
                    x[i, -1] = 1.0 if attrs.get("is_sensitive", False) else 0.0

                # Build `edge_index` tensor
                edges = cg.get("edges", [])
                edge_indices = [(id_map[src], id_map[dst]) for src, dst in edges if src in id_map and dst in id_map]
                # Skip graph if there are no edges
                if len(edge_indices) == 0:
                    continue
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

                # Graph-level label `y`
                y = torch.tensor([lbl], dtype=torch.long)

                # Build and append `Data` object
                data_list.append(Data(x=x, edge_index=edge_index, y=y))

        # Save cache file
        torch.save(data_list, self.cache_path)
        logger.info(f"[INFO] Data cached at {self.cache_path}")
        return data_list


def get_datasets(root_dir: str, batch_size=32, seed=42):
    """
    Get an `APKGraphDataset` instance and split it into train/val/test sets.
    :param root_dir: Dataset root directory
    :param batch_size: Batch size for `DataLoader`
    :return: Train/validation/test `DataLoader`s
    """

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"数据集目录 {root_dir} 不存在，请检查路径。")

    # Build opcode and permission vocabularies
    opcode_vocab = build_opcode_vocab(root_dir)
    permission_vocab = build_permission_vocab(root_dir)

    logger.info("Building vocabularies...")
    logger.info(f"{len(opcode_vocab)} opcodes found")
    logger.info(f"{len(permission_vocab)} permissions found")

    if not opcode_vocab or not permission_vocab:
        raise ValueError("Opcode 词表或 Permission 词表为空，请检查数据集。")

    dataset = APKGraphDataset(root_dir=root_dir, opcode_vocab=opcode_vocab, permission_vocab=permission_vocab)

    # Dataset split ratios
    train_ratio = 0.7  # Training split ratio
    val_ratio = 0.15  # Validation split ratio

    # Total number of samples
    num_samples = len(dataset)

    # Compute train/val/test sizes
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size  # Remaining part goes to test set

    # Use `random_split` to split dataset by ratio
    # Random split each run; results can vary
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # Ensure reproducibility with a fixed random seed
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# dataset =  APKGraphDataset(root_dir="/home/yuan/workspace/MaskEdgeDroid/data/output-with-init", opcode_vocab=None, permission_vocab=None)
# print(f"Dataset has {len(dataset)} samples.")