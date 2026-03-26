import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.utils import subgraph, negative_sampling, to_networkx
from torch_geometric.data import Data

import networkx as nx

# ===== Project dependencies =====
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from config.logging_config import set_logging
from config.path_config import OUTPUT_PATH
from core.dataset import get_datasets

logger = set_logging(__file__)


# ============================================================
# 1) Adaptive per-graph node+edge dual masking (DualMask, structure + semantics)
# ============================================================
@torch.no_grad()
def mask_edges_and_nodes(
    data,
    edge_mask_ratio: float = 0.15,
    node_mask_ratio: float = 0.15,
    augment: bool = False,
    adaptive: bool = True,
    protect_bridges: bool = True,
):
    """
        Apply per-graph node+edge masking to PyG batched `Data`:
            - Node importance: imp = 0.5*degree + 0.5*||x||
            - Edge importance: mean of the endpoint node importances
            - Lower-importance items are masked first (can be reversed)
            - Optional bridge-edge protection
            - Negative sampling is performed within each graph
        Returned fields:
      data.masked_edge_index, data.masked_edges, data.masked_edges_for_pred, data.masked_edge_label
      data.original_x, data.masked_x, data.masked_nodes, data.masked_nodes_for_pred, data.masked_node_label
    """
    device = data.x.device
    assert hasattr(data, "batch"), "Expected batched Data with `data.batch`."

    batch = data.batch
    num_graphs = int(batch.max()) + 1

    # Save original features + masked features to be written
    data.original_x = data.x.clone()
    data.masked_x = data.x.clone()

    kept_edge_chunks, pos_edge_chunks, edges4pred_chunks, edge_label_chunks = [], [], [], []
    masked_nodes_chunks, masked_feats_chunks, nodes4pred_chunks, node_label_chunks = [], [], [], []

    for gid in range(num_graphs):
        node_sel = batch == gid
        nodes_global = node_sel.nonzero(as_tuple=False).view(-1)  # Global node indices of current graph in the batched graph
        if nodes_global.numel() == 0:
            continue

        # Extract local subgraph for this graph (with relabeled nodes)
        e_local, _ = subgraph(nodes_global, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)  # Local edge indices of current graph
        n_local = nodes_global.size(0)  # Number of nodes in current graph
        x_local = data.x[nodes_global]  # Node features of current graph

        # Structural + semantic importance
        deg = torch.zeros(n_local, device=device)
        if e_local.numel() > 0:
            deg.scatter_add_(0, e_local[0], torch.ones(e_local.size(1), device=device))
        deg_s = (deg - deg.min()) / (deg.max() - deg.min() + 1e-9) if deg.max() > 0 else deg

        sem = x_local.norm(p=2, dim=1)
        sem_s = (sem - sem.min()) / (sem.max() - sem.min() + 1e-9) if sem.max() > 0 else sem

        imp = 0.5 * deg_s + 0.5 * sem_s  # [0,1], larger means more important

        # Lightweight augmentation: dynamic perturbation of mask ratios
        # if augment:
        #     edge_mask_ratio = edge_mask_ratio * (0.8 + 0.4 * torch.rand(1, device=device).item())
        #     node_mask_ratio = node_mask_ratio * (0.8 + 0.4 * torch.rand(1, device=device).item())
        edge_ratio = edge_mask_ratio * (0.8 + 0.4 * torch.rand(1, device=device).item()) if augment else edge_mask_ratio
        node_ratio = node_mask_ratio * (0.8 + 0.4 * torch.rand(1, device=device).item()) if augment else node_mask_ratio

        # ===== Adaptive edge masking =====
        num_edges = e_local.size(1)
        # num_edges_to_mask = int(num_edges * edge_mask_ratio)
        num_edges_to_mask = int(num_edges * edge_ratio)

        # By default all edges are removable; bridge protection is optional
        prunable = torch.ones(num_edges, dtype=torch.bool, device=device)
        if protect_bridges and num_edges > 0:
            try:
                G = to_networkx(Data(x=x_local.cpu(), edge_index=e_local.cpu()), to_undirected=True)
                bridges = {tuple(sorted(b)) for b in nx.bridges(G)}
                loc_tuples = [tuple(sorted((int(u), int(v)))) for u, v in e_local.t().tolist()]
                for i, t in enumerate(loc_tuples):
                    if t in bridges:
                        prunable[i] = False
            except Exception:
                pass  # If failed, skip protection

        pr_idx = prunable.nonzero(as_tuple=False).view(-1)
        chosen_e = torch.empty((2, 0), dtype=torch.long, device=device)

        if num_edges_to_mask > 0 and pr_idx.numel() > 0:
            ei_imp = ((imp[e_local[0]] + imp[e_local[1]]) / 2.0)[pr_idx]
            if adaptive:
                # Mask lower-importance edges first
                probs = (1.0 - ei_imp).clamp(min=1e-9)
                probs = probs / probs.sum()
                pick = pr_idx[torch.multinomial(probs, num_samples=min(num_edges_to_mask, pr_idx.numel()), replacement=False)]
            else:
                perm = torch.randperm(pr_idx.numel(), device=device)
                pick = pr_idx[perm[:num_edges_to_mask]]
            chosen_e = e_local[:, pick]

        # Construct kept edges
        if chosen_e.size(1) == 0:
            keep_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
        else:
            chosen_set = {tuple(sorted((int(u), int(v)))) for u, v in chosen_e.t().tolist()}
            loc_tuples = [tuple(sorted((int(u), int(v)))) for u, v in e_local.t().tolist()]
            keep_mask = torch.tensor([t not in chosen_set for t in loc_tuples], device=device)

        e_local_kept = e_local[:, keep_mask]

        # Map back to global indices
        kept_global = (
            torch.stack([nodes_global[e_local_kept[0]], nodes_global[e_local_kept[1]]], dim=0)
            if e_local_kept.numel()
            else torch.empty((2, 0), dtype=torch.long, device=device)
        )

        pos_global = (
            torch.stack([nodes_global[chosen_e[0]], nodes_global[chosen_e[1]]], dim=0)
            if chosen_e.numel()
            else torch.empty((2, 0), dtype=torch.long, device=device)
        )

        # In-graph negative sampling (same count as positive samples)
        neg_local = (
            negative_sampling(e_local, num_nodes=n_local, num_neg_samples=pos_global.size(1), method="sparse")
            if pos_global.size(1) > 0
            else torch.empty((2, 0), dtype=torch.long, device=device)
        )

        neg_global = (
            torch.stack([nodes_global[neg_local[0]], nodes_global[neg_local[1]]], dim=0)
            if neg_local.numel()
            else torch.empty((2, 0), dtype=torch.long, device=device)
        )

        edges4pred = torch.cat([pos_global, neg_global], dim=1)
        edge_labels = torch.cat([torch.ones(pos_global.size(1), device=device), torch.zeros(neg_global.size(1), device=device)], dim=0)

        kept_edge_chunks.append(kept_global)
        pos_edge_chunks.append(pos_global)
        edges4pred_chunks.append(edges4pred)
        edge_label_chunks.append(edge_labels)

        # ===== Adaptive node masking =====
        # k_nodes = max(1, int(n_local * node_mask_ratio))
        k_nodes = max(1, int(n_local * node_ratio))
        if adaptive:
            p = (1.0 - imp).clamp(min=1e-9)
            p = p / p.sum()
            chosen_nodes_local = torch.multinomial(p, num_samples=min(k_nodes, n_local), replacement=False)
        else:
            chosen_nodes_local = torch.randperm(n_local, device=device)[:k_nodes]

        chosen_nodes_global = nodes_global[chosen_nodes_local]
        masked_feats = data.x[chosen_nodes_global]

        # Write mask token (zero / small noise)
        if augment and torch.rand(1, device=device).item() < 0.3:
            data.masked_x[chosen_nodes_global] = torch.randn_like(masked_feats) * 0.1
        else:
            data.masked_x[chosen_nodes_global] = 0.0

        neg_feats = torch.randn_like(masked_feats) * 0.3
        nodes4pred = torch.cat([masked_feats, neg_feats], dim=0)
        node_labels = torch.cat([torch.ones(masked_feats.size(0), device=device), torch.zeros(neg_feats.size(0), device=device)], dim=0)

        masked_nodes_chunks.append(chosen_nodes_global)
        masked_feats_chunks.append(masked_feats)
        nodes4pred_chunks.append(nodes4pred)
        node_label_chunks.append(node_labels)

    # Merge results back into batch
    data.masked_edge_index = torch.cat(kept_edge_chunks, dim=1) if kept_edge_chunks else data.edge_index
    data.masked_edges = torch.cat(pos_edge_chunks, dim=1) if pos_edge_chunks else torch.empty((2, 0), dtype=torch.long, device=device)
    data.masked_edges_for_pred = torch.cat(edges4pred_chunks, dim=1) if edges4pred_chunks else torch.empty((2, 0), dtype=torch.long, device=device)
    data.masked_edge_label = torch.cat(edge_label_chunks, dim=0) if edge_label_chunks else torch.empty(0, dtype=torch.float, device=device)

    data.masked_nodes = torch.cat(masked_nodes_chunks, dim=0) if masked_nodes_chunks else torch.empty(0, dtype=torch.long, device=device)
    data.masked_node_features = torch.cat(masked_feats_chunks, dim=0) if masked_feats_chunks else torch.empty((0, data.x.size(1)), device=device)
    data.masked_nodes_for_pred = torch.cat(nodes4pred_chunks, dim=0) if nodes4pred_chunks else torch.empty((0, data.x.size(1)), device=device)
    data.masked_node_label = torch.cat(node_label_chunks, dim=0) if node_label_chunks else torch.empty(0, device=device)

    return data


# ============================================================
# 2) Model: Encoder/Decoders/Classifier (all using LayerNorm)
# ============================================================
class ImprovedGNNEncoder(nn.Module):
    """GCN backbone with LayerNorm (replacing BN for more stable single-sample inference)."""

    def __init__(self, in_dim, hidden_dim, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x, edge_index):
        for i, (conv, ln, do) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            x = conv(x, edge_index)
            x = ln(x)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = do(x)
        return x


class ImprovedEdgeDecoder(nn.Module):
    """Edge decoder: LayerNorm replaces BN."""

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_emb, edge_pairs):
        if edge_pairs.numel() == 0:
            return torch.empty(0, device=node_emb.device)
        src = node_emb[edge_pairs[0]]
        dst = node_emb[edge_pairs[1]]
        edge_feat = torch.cat([src, dst], dim=1)
        return self.mlp(edge_feat).squeeze(-1)


class ImprovedNodeDecoder(nn.Module):
    """Node reconstruction decoder: LayerNorm replaces BN."""

    def __init__(self, hidden_dim, feature_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, node_emb):
        if node_emb.numel() == 0:
            return torch.empty(0, node_emb.size(-1), device=node_emb.device)
        return self.mlp(node_emb)


class ImprovedNodeSimilarityDecoder(nn.Module):
    """Node semantic similarity discriminator: LayerNorm replaces BN."""

    def __init__(self, feature_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, anchor_feat, candidate_feats):
        # anchor_feat: (F,) or scalar-like (F); candidate_feats: (K, F)
        if candidate_feats.numel() == 0:
            return torch.empty(0, device=candidate_feats.device)
        if anchor_feat.dim() == 1:
            anchor_feat = anchor_feat.unsqueeze(0).expand(candidate_feats.size(0), -1)
        elif anchor_feat.dim() == 0:
            anchor_feat = anchor_feat.view(1, 1).expand(candidate_feats.size(0), -1)
        pair = torch.cat([anchor_feat, candidate_feats], dim=1)
        return self.mlp(pair).squeeze(-1)


class SuperEnhancedGraphClassifier(nn.Module):
    """Encoder + (optional) GAT + multi-pooling + classification head (LayerNorm version)."""

    def __init__(self, encoder, hidden_dim, num_classes, num_heads=4, dropout_rate=0.15, use_gat=True, pooling_method="multi"):
        super().__init__()
        self.encoder = encoder
        self.use_gat = use_gat
        self.pooling_method = pooling_method

        self.layer_norm = nn.LayerNorm(hidden_dim)

        if use_gat:
            self.gat_layers = nn.ModuleList(
                [
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout_rate, concat=True),
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout_rate, concat=True),
                ]
            )
            self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
            self.gat_norm = nn.LayerNorm(hidden_dim)

        final_dim = hidden_dim
        if pooling_method == "multi":
            final_dim = hidden_dim * 4
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim // 2, 1)
            )
            self.weight_pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 4), nn.ReLU(), nn.Linear(hidden_dim // 4, 1), nn.Sigmoid())

        self.cls_head = nn.Sequential(
            nn.Linear(final_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        node_emb = self.encoder(x, edge_index)
        residual = node_emb
        node_emb = self.layer_norm(node_emb)

        if self.use_gat:
            for i, gat in enumerate(self.gat_layers):
                g = gat(node_emb, edge_index)
                g = self.dropout(g)
                if i == 0:
                    node_emb = g + self.residual_proj(residual)
                else:
                    node_emb = g + node_emb
                node_emb = F.relu(self.gat_norm(node_emb))

        if self.pooling_method == "multi":
            mean_pool = global_mean_pool(node_emb, batch)
            max_pool = global_max_pool(node_emb, batch)
            att_pool = self._attention_pooling_per_graph(node_emb, batch)
            w_pool = self._weighted_pooling_per_graph(node_emb, batch)
            graph_emb = torch.cat([mean_pool, max_pool, att_pool, w_pool], dim=1)
        else:
            graph_emb = global_mean_pool(node_emb, batch)

        return self.cls_head(graph_emb)

    def _attention_pooling_per_graph(self, node_emb, batch):
        outs = []
        for gid in range(int(batch.max()) + 1):
            mask = batch == gid
            if mask.sum() == 0:
                continue
            gnodes = node_emb[mask]
            att = self.attention_pool(gnodes)
            w = F.softmax(att, dim=0)
            outs.append((gnodes * w).sum(dim=0))
        return torch.stack(outs)

    def _weighted_pooling_per_graph(self, node_emb, batch):
        outs = []
        for gid in range(int(batch.max()) + 1):
            mask = batch == gid
            if mask.sum() == 0:
                continue
            gnodes = node_emb[mask]
            w = self.weight_pool(gnodes)
            outs.append((gnodes * w).sum(dim=0) / (w.sum(dim=0) + 1e-8))
        return torch.stack(outs)


# ============================================================
# 3) Training loops (pretraining: edge/node reconstruction + semantic consistency; then classification finetuning)
# ============================================================
def train_edge_and_node_pred_improved(
    encoder, edge_decoder, node_decoder, node_similarity_decoder, loader, optimizer, device, edge_mask_ratio, node_mask_ratio, epoch
):
    encoder.train()
    edge_decoder.train()
    node_decoder.train()
    node_similarity_decoder.train()

    total_edge_loss = 0.0
    total_node_recon_loss = 0.0
    total_node_sim_loss = 0.0
    total_loss = 0.0

    # Dynamic weights
    edge_w = 1.0
    node_rec_w = 1.0
    node_sim_w = 0.5 + 0.5 * min(epoch / 20.0, 1.0)

    for data in loader:
        data = data.to(device)
        # augment = torch.rand(1).item() < 0.5
        augment = False

        data = mask_edges_and_nodes(
            data, edge_mask_ratio=edge_mask_ratio, node_mask_ratio=node_mask_ratio, augment=augment, adaptive=True, protect_bridges=True
        )
        node_emb = encoder(data.masked_x, data.masked_edge_index)

        loss = torch.tensor(0.0, device=device)
        edge_loss = torch.tensor(0.0, device=device)
        node_rec_loss = torch.tensor(0.0, device=device)
        node_sim_loss = torch.tensor(0.0, device=device)

        # Edge prediction (binary classification)
        if data.masked_edges_for_pred.numel() > 0:
            edge_pred = edge_decoder(node_emb, data.masked_edges_for_pred)
            edge_loss = F.binary_cross_entropy_with_logits(edge_pred, data.masked_edge_label)
            loss = loss + edge_w * edge_loss

        # Node reconstruction + similarity
        if data.masked_nodes.numel() > 0:
            recon = node_decoder(node_emb[data.masked_nodes])
            node_rec_loss = F.mse_loss(recon, data.masked_node_features)
            loss = loss + node_rec_w * node_rec_loss

            # Semantic consistency: use mean reconstruction as anchor
            if recon.size(0) > 1:
                anchor = recon.mean(dim=0)
            else:
                anchor = recon.squeeze(0)
            sim_logit = node_similarity_decoder(anchor, data.masked_nodes_for_pred)
            node_sim_loss = F.binary_cross_entropy_with_logits(sim_logit, data.masked_node_label)
            loss = loss + node_sim_w * node_sim_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(
            list(encoder.parameters())
            + list(edge_decoder.parameters())
            + list(node_decoder.parameters())
            + list(node_similarity_decoder.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += float(loss.item())
        total_edge_loss += float(edge_loss.item())
        total_node_recon_loss += float(node_rec_loss.item())
        total_node_sim_loss += float(node_sim_loss.item())

    n = len(loader)
    return (
        total_loss / n,
        total_edge_loss / n,
        total_node_recon_loss / n,
        total_node_sim_loss / n,
    )


def train_cls_improved(model, loader, optimizer, device, epoch=None):
    model.train()
    y_true, y_pred = [], []
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)

        # Label smoothing
        smoothing = 0.1
        confidence = 1.0 - smoothing
        num_classes = out.size(-1)
        true_dist = torch.zeros_like(out)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, data.y.unsqueeze(1), confidence)

        loss = F.kl_div(F.log_softmax(out, dim=1), true_dist, reduction="batchmean")

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        y_true.append(data.y)
        y_pred.append(out.argmax(dim=1))
        total_loss += float(loss.item())

    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    return total_loss / len(loader), acc


@torch.no_grad()
def validate_cls(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        total_loss += float(F.cross_entropy(out, data.y).item())
        y_true.append(data.y)
        y_pred.append(out.argmax(dim=1))
    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    return total_loss / len(loader), acc


@torch.no_grad()
def test_cls(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        y_true.append(data.y)
        y_pred.append(out.argmax(dim=1))
        y_prob.append(out)

    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()
    y_prob = torch.cat(y_prob).softmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        auc = float("nan")

    logger.info("Test Results:")
    logger.info(f"Acc: {acc:.4f}")
    logger.info(f"F1: {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"AUC: {auc if not math.isnan(auc) else 'nan'}")


# ============================================================
# 4) Main pipeline
# ============================================================
def main():
    from core.utils import set_seed

    set_seed(42)

    # ---------- Hyperparameters ----------
    root_dir = OUTPUT_PATH 
    batch_size = 48
    hidden_dim = 160
    num_classes = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # edge_mask_ratio = 0.2
    # node_mask_ratio = 0.1
    edge_mask_ratio = 0.1
    node_mask_ratio = 0.1

    dropout = 0.15
    num_heads = 4
    use_gat = True

    # Number of training epochs (can also be read from CLI)
    train_edge_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    train_cls_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    # Learning rates
    train_pretrain_lr = 8e-4
    train_pretrain_weight_decay = 5e-6

    train_cls_lr = 3e-4
    train_cls_weight_decay = 5e-5

    gat_lr = 3e-4
    gat_weight_decay = 5e-5

    finetune_lr = 5e-6
    finetune_weight_decay = 1e-6

    # ---------- Data ----------
    train_loader, val_loader, test_loader = get_datasets(root_dir, batch_size, seed=30)
    input_dim = train_loader.dataset[0].x.size(1)

    # ---------- Self-supervised pretraining ----------
    encoder = ImprovedGNNEncoder(input_dim, hidden_dim, num_layers=4, dropout=0.1).to(device)
    edge_decoder = ImprovedEdgeDecoder(hidden_dim, dropout=0.1).to(device)
    node_decoder = ImprovedNodeDecoder(hidden_dim, input_dim, dropout=0.1).to(device)
    node_similarity_decoder = ImprovedNodeSimilarityDecoder(input_dim, dropout=0.1).to(device)

    optimizer_pretrain = torch.optim.AdamW(
        list(encoder.parameters()) + list(edge_decoder.parameters()) + list(node_decoder.parameters()) + list(node_similarity_decoder.parameters()),
        lr=train_pretrain_lr,
        weight_decay=train_pretrain_weight_decay,
    )
    scheduler_pretrain = CosineAnnealingLR(optimizer_pretrain, T_max=train_edge_epochs, eta_min=1e-6)

    logger.info("==== DualMask Self-supervised pretraining ====")
    for epoch in range(1, train_edge_epochs + 1):
        total_loss, edge_loss, node_recon_loss, node_sim_loss = train_edge_and_node_pred_improved(
            encoder,
            edge_decoder,
            node_decoder,
            node_similarity_decoder,
            train_loader,
            optimizer_pretrain,
            device,
            edge_mask_ratio,
            node_mask_ratio,
            epoch,
        )
        scheduler_pretrain.step()
        logger.info(
            f"Epoch {epoch:02d} | Total: {total_loss:.4f} | Edge: {edge_loss:.4f} | "
            f"NodeRecon: {node_recon_loss:.4f} | NodeSim: {node_sim_loss:.4f} | "
            f"LR: {scheduler_pretrain.get_last_lr()[0]:.6f}"
        )

    # ---------- Classification finetuning ----------
    logger.info("==== Graph classification finetune ====")
    model = SuperEnhancedGraphClassifier(encoder, hidden_dim, num_classes, num_heads, dropout, use_gat, pooling_method="multi").to(device)

    optim_groups = [
        {"params": model.cls_head.parameters(), "lr": train_cls_lr, "weight_decay": train_cls_weight_decay},
        {"params": model.encoder.parameters(), "lr": finetune_lr, "weight_decay": finetune_weight_decay},
    ]
    if use_gat:
        optim_groups.extend(
            [
                {"params": model.gat_layers.parameters(), "lr": gat_lr, "weight_decay": gat_weight_decay},
                {"params": model.residual_proj.parameters(), "lr": gat_lr, "weight_decay": gat_weight_decay},
                {"params": model.layer_norm.parameters(), "lr": gat_lr},
                {"params": model.gat_norm.parameters(), "lr": gat_lr},
                {"params": model.attention_pool.parameters(), "lr": gat_lr},
                {"params": model.weight_pool.parameters(), "lr": gat_lr},
            ]
        )

    optimizer_cls = torch.optim.AdamW(optim_groups)
    scheduler = ReduceLROnPlateau(optimizer_cls, mode="max", factor=0.8, patience=6, min_lr=1e-7)

    best_val_acc, epochs_no_improve, patience, min_delta = 0.0, 0, 25, 5e-5
    best_model_path = "best_dualmask_model_2018_2021.pth"

    for epoch in range(1, train_cls_epochs + 1):
        train_loss, train_acc = train_cls_improved(model, train_loader, optimizer_cls, device, epoch)
        val_loss, val_acc = validate_cls(model, val_loader, device)
        scheduler.step(val_acc)

        logger.info(
            f"Finetune Epoch {epoch:02d} | Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f} | LR: {optimizer_cls.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"🎉 New best model saved | Val Acc: {best_val_acc:.4f} @ epoch {epoch}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping after {patience} epochs with no improvement.")
            break

        # Overfitting warning
        if (train_acc - val_acc) > 0.04:
            logger.warning(f"⚠️ Potential overfitting: gap={train_acc - val_acc:.4f}")

    logger.info(f"Finetuning finished. Best Val Acc: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(best_model_path, weights_only=True, map_location=device))

    # ---------- Testing ----------
    test_cls(model, test_loader, device)


if __name__ == "__main__":
    for i in range(5):
        logger.info(f"=================== Run {i+1} ===================")
        main()
