import torch
import torch.nn as nn
from typing import Tuple
from .lnn_core import CfCCell

class HybridGNNLNN(nn.Module):
    """
    GNN(2-capas) + LNN(CfC) + Output
    - 2-layer GCN captures multi-hop risk patterns (corridors, amplification)
    - LNN models continuous temporal dynamics
    - Trained with RILL loss for spatial coherence
    """
    def __init__(self, in_features: int, gnn_hidden: int, lnn_hidden: int):
        super().__init__()
        self.gnn_hidden = gnn_hidden
        self.lnn_hidden = lnn_hidden
        
        # GCN Layer 1: aggregates direct neighbors
        self.gcn1_w = nn.Parameter(torch.randn(in_features, gnn_hidden) * 0.1)
        self.gcn1_b = nn.Parameter(torch.zeros(gnn_hidden))
        
        # GCN Layer 2: aggregates neighbor-of-neighbors (captures corridors)
        self.gcn2_w = nn.Parameter(torch.randn(gnn_hidden, gnn_hidden) * 0.1)
        self.gcn2_b = nn.Parameter(torch.zeros(gnn_hidden))
        
        # LNN for temporal dynamics
        self.lnn = CfCCell(gnn_hidden, lnn_hidden)
        
        # Output
        self.out = nn.Linear(lnn_hidden, 1)

    def _norm_adjacency(self, edge_index, num_nodes):
        """Compute D^(-1/2) * A_hat * D^(-1/2)"""
        A = torch.zeros(num_nodes, num_nodes)
        A[edge_index[0], edge_index[1]] = 1.0
        A = A + torch.eye(num_nodes)  # self-loops
        deg = A.sum(dim=1).clamp(min=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
        return D_inv_sqrt @ A @ D_inv_sqrt

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        seq_len, num_nodes, _ = x_seq.shape
        
        norm_A = self._norm_adjacency(edge_index, num_nodes)
        h = torch.zeros(num_nodes, self.lnn_hidden)
        
        for t in range(seq_len):
            x_t = x_seq[t]
            
            # GCN Layer 1: spatial aggregation from direct neighbors
            h1 = norm_A @ x_t @ self.gcn1_w + self.gcn1_b
            h1 = torch.relu(h1)
            
            # GCN Layer 2: multi-hop aggregation (captures corridors)
            h2 = norm_A @ h1 @ self.gcn2_w + self.gcn2_b
            h2 = torch.relu(h2)
            
            # LNN temporal step
            h = self.lnn(h2, h)
        
        return torch.sigmoid(self.out(h))
