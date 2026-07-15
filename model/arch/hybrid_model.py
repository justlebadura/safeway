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
        
        self.gcn1_w = nn.Parameter(torch.randn(in_features, gnn_hidden) * 0.1)
        self.gcn1_b = nn.Parameter(torch.zeros(gnn_hidden))
        
        self.gcn2_w = nn.Parameter(torch.randn(gnn_hidden, gnn_hidden) * 0.1)
        self.gcn2_b = nn.Parameter(torch.zeros(gnn_hidden))
        
        self.lnn = CfCCell(gnn_hidden, lnn_hidden)
        self.out = nn.Linear(lnn_hidden, 1)

    def _sparse_gcn(self, x: torch.Tensor, edge_index: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Sparse GCN layer: aggregate neighbor features, multiply by weight, add bias."""
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        
        # Compute node degrees (undirected)
        deg = torch.zeros(N, device=x.device).scatter_add(0, src, torch.ones(len(src), device=x.device))
        deg = deg + 1.0  # self-loop
        deg_inv_sqrt = 1.0 / torch.sqrt(deg.clamp(min=1))
        
        # Normalization: 1/sqrt(deg[src] * deg[dst])
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        
        # Aggregate neighbor features
        msg = x[src] * norm.unsqueeze(1)
        out = torch.zeros(N, x.shape[1], device=x.device).scatter_add(0, dst.unsqueeze(1).expand(-1, x.shape[1]), msg)
        
        # Self-loop
        self_norm = deg_inv_sqrt * deg_inv_sqrt
        out = out + x * self_norm.unsqueeze(1)
        
        return out @ w + b

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        seq_len, num_nodes, _ = x_seq.shape
        h = torch.zeros(num_nodes, self.lnn_hidden, device=x_seq.device)
        
        for t in range(seq_len):
            x_t = x_seq[t]
            
            h1 = self._sparse_gcn(x_t, edge_index, self.gcn1_w, self.gcn1_b)
            h1 = torch.relu(h1)
            
            h2 = self._sparse_gcn(h1, edge_index, self.gcn2_w, self.gcn2_b)
            h2 = torch.relu(h2)
            
            h = self.lnn(h2, h)
        
        return torch.sigmoid(self.out(h))
