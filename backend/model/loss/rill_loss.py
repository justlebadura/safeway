import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    """
    LTN Loss with 3 logic predicates:
    1. Smoothness: connected nodes → similar risk
    2. Degree propagation: high-degree intersections spread more risk
    3. Severity dominance: MUERTO nodes must have risk >= LESIONADO nodes
    """
    def __init__(self, lambda_logic: float = 0.1):
        super().__init__()
        self.lambda_logic = lambda_logic
        
    def forward(self, pred: torch.Tensor, y_true: torch.Tensor, 
                features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        
        data_loss = F.mse_loss(pred, y_true)
        
        u = edge_index[0]; v = edge_index[1]
        N = pred.shape[0]
        
        # ── Predicate 1: Spatial smoothness ──
        # ∀(u,v)∈E: |risk(u) - risk(v)| ≈ 0
        smoothness = torch.mean((pred[u] - pred[v]) ** 2)
        
        # ── Predicate 2: Degree propagation ──
        # high-degree nodes (big intersections) influence neighbors more
        deg = features[-1, :, 3].clamp(min=0.02)
        deg_factor = (deg[u] + deg[v]) / 2.0
        degree_prop = torch.mean(deg_factor.unsqueeze(1) * (pred[u] - pred[v]) ** 2)
        
        # ── Predicate 3: Severity dominance (sampled) ──
        # Nodes with MUERTO must have risk >= nodes with only LESIONADO
        sev = features[-1, :, 1]
        K = 2000
        idx_i = torch.randint(0, N, (K,), device=pred.device)
        idx_j = torch.randint(0, N, (K,), device=pred.device)
        sev_diff = (sev[idx_i] - sev[idx_j]).clamp(min=0)
        pred_gap = pred[idx_i].squeeze() - pred[idx_j].squeeze()
        severity_violation = torch.mean(sev_diff * F.relu(-pred_gap) ** 2)
        
        logic_loss = 0.40 * smoothness + 0.35 * degree_prop + 0.25 * severity_violation
        
        return data_loss + self.lambda_logic * logic_loss
