import torch
import torch.nn as nn
from typing import Tuple
from backend.model.arch.lnn_core import CfCCell

class HybridGNNLNN(nn.Module):
    """
    Hybrid architecture combining GNN for spatial propagation and 
    LNN for continuous temporal dynamics.
    """
    def __init__(self, in_features: int, gnn_hidden: int, lnn_hidden: int):
        super().__init__()
        self.gnn_hidden = gnn_hidden
        self.lnn_hidden = lnn_hidden
        
        # Simple GNN Layer (simplified GCN)
        self.gnn_w = nn.Parameter(torch.randn(in_features, gnn_hidden))
        
        # LNN Layer
        self.lnn = CfCCell(gnn_hidden, lnn_hidden)
        
        # Output
        self.out = nn.Linear(lnn_hidden, 1)
        
    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Processes temporal sequence through GNN and LNN layers.
        
        Args:
            x_seq (torch.Tensor): Input sequence (seq_len, num_nodes, in_features)
            edge_index (torch.Tensor): Graph edge indices (2, num_edges)
            
        Returns:
            torch.Tensor: Risk score prediction (num_nodes, 1)
        """
        seq_len, num_nodes, _ = x_seq.shape
        
        h = torch.zeros(num_nodes, self.lnn_hidden)
        
        # Precompute normalized adjacency for GNN (Simplified)
        A = torch.zeros(num_nodes, num_nodes)
        A[edge_index[0], edge_index[1]] = 1.0
        A = A + torch.eye(num_nodes)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(A.sum(dim=1)))
        norm_A = D_inv_sqrt @ A @ D_inv_sqrt
        
        for t in range(seq_len):
            x_t = x_seq[t] # (num_nodes, in_features)
            
            # GNN step: Aggregate neighbor features
            spatial = norm_A @ x_t @ self.gnn_w
            
            # LNN step
            h = self.lnn(spatial, h)
            
        return torch.sigmoid(self.out(h))
