# python 库
import torch
from torch.utils.data import random_split
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import os

# 自定义
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from core.utils import load_features, build_opcode_vocab, build_permission_vocab
from config.logging_config import set_logging

# 设置日志记录
logger = set_logging(__file__)


# Dataset 类
class APKGraphDataset(Dataset):
    """
    一个用于读取 .pkl 格式调用图数据的 PyG Dataset
    目录结构:
      root_dir/
        malicious/*.pkl
        benign/*.pkl
    所有扫描、特征生成和 Data 创建都在 load_data() 中执行，
    并将生成的 Data 对象缓存至内存以加速索引。
    """

    def __init__(self, root_dir: str, opcode_vocab: dict, permission_vocab: dict, transform=None, pre_transform=None):
        # 调用父类构造函数，root_dir 由父类赋给 self.root
        super().__init__(root=root_dir, transform=transform, pre_transform=pre_transform)
        # 保存 opcode 与 permission 词表
        self.opcode_vocab = opcode_vocab
        self.permission_vocab = permission_vocab
        # 预先加载并处理所有样本，结果缓存到列表中
        self.cache_path = os.path.join(self.root, "data.pkl")
        self.data_list = self.load_data()

    def len(self):
        # 返回图的总数
        return len(self.data_list)

    def get(self, idx):
        # 根据索引返回对应的 Data 对象
        return self.data_list[idx]

    def load_data(self):
        """
        扫描 root 目录下的 .pkl 文件，
        构造 PyG Data 对象并返回列表
        """
        # 如果缓存文件存在，直接加载
        if os.path.exists(self.cache_path):
            logger.info(f"Loading cached data from {self.cache_path}")
            return torch.load(self.cache_path, weights_only=False)

        data_list = []
        # 分别处理恶意和良性样本
        for label_str, lbl in (("malicious", 1), ("benign", 0)):
            folder = os.path.join(self.root, label_str)
            if not os.path.isdir(folder):
                continue
            for fn in os.listdir(folder):
                # 仅处理 .pkl 文件
                if not fn.endswith(".pkl"):
                    continue

                path = os.path.join(folder, fn)
                # 加载特征字典
                features = load_features(path)
                cg = features.get("call_graph", {})
                nodes = cg.get("nodes", [])
                # 跳过节点数小于等于1的图
                if len(nodes) <= 1:
                    continue

                num_nodes = len(nodes)
                # 特征维度 = opcode 词表长度 + permission 词表长度 + 1（is_sensitive）
                feat_dim = len(self.opcode_vocab) + len(self.permission_vocab) + 1
                # 初始化节点特征矩阵 [num_nodes, feat_dim]
                x = torch.zeros((num_nodes, feat_dim), dtype=torch.float)

                # 构建节点ID到索引的映射
                id_map = {}
                for i, (nid, attrs) in enumerate(nodes):
                    id_map[nid] = i
                    # opcode 独热编码
                    for op in attrs.get("opcodes", []):
                        idx = self.opcode_vocab.get(op)
                        if idx is not None:
                            x[i, idx] = 1.0
                    # permission 独热编码，偏移到词表后面
                    base = len(self.opcode_vocab)
                    for per in attrs.get("permissions", []):
                        pidx = self.permission_vocab.get(per)
                        if pidx is not None:
                            x[i, base + pidx] = 1.0
                    # is_sensitive 标志位
                    x[i, -1] = 1.0 if attrs.get("is_sensitive", False) else 0.0

                # 构建 edge_index 张量
                edges = cg.get("edges", [])
                edge_indices = [(id_map[src], id_map[dst]) for src, dst in edges if src in id_map and dst in id_map]
                # 如果没有边，可选择跳过
                if len(edge_indices) == 0:
                    continue
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

                # 图级标签 y
                y = torch.tensor([lbl], dtype=torch.long)

                # 构造并添加 Data 对象
                data_list.append(Data(x=x, edge_index=edge_index, y=y))

        # 保存缓存文件
        torch.save(data_list, self.cache_path)
        logger.info(f"[INFO] Data cached at {self.cache_path}")
        return data_list


def get_datasets(root_dir: str, batch_size=32, seed=42):
    """
    获取 APKGraphDataset 实例, 并划分为训练集、验证集和测试集
    :param root_dir: 数据集根目录
    :param batch_size: DataLoader 的批大小
    :return: 训练集、验证集和测试集的 DataLoader
    """

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"数据集目录 {root_dir} 不存在，请检查路径。")

    # 构建 opcode 和 permission 词表
    opcode_vocab = build_opcode_vocab(root_dir)
    permission_vocab = build_permission_vocab(root_dir)

    logger.info("Building vocabularies...")
    logger.info(f"{len(opcode_vocab)} opcodes found")
    logger.info(f"{len(permission_vocab)} permissions found")

    if not opcode_vocab or not permission_vocab:
        raise ValueError("Opcode 词表或 Permission 词表为空，请检查数据集。")

    dataset = APKGraphDataset(root_dir=root_dir, opcode_vocab=opcode_vocab, permission_vocab=permission_vocab)

    # 数据集划分比例
    train_ratio = 0.7  # 训练集比例
    val_ratio = 0.15  # 验证集比例

    # 样本总数
    num_samples = len(dataset)

    # 计算训练、验证和测试集大小
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size  # 剩余部分分配给测试集

    # 使用 random_split 按比例划分数据集
    # 每次运行随机划分，结果不同
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # 保证实验可重复，结果稳定
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# dataset =  APKGraphDataset(root_dir="/home/yuan/workspace/MaskEdgeDroid/data/output-with-init", opcode_vocab=None, permission_vocab=None)
# print(f"Dataset has {len(dataset)} samples.")