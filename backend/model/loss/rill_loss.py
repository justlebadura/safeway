import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    """
    LTN Loss with 4 logic predicates:
    1. Smoothness: connected nodes -> similar risk
    2. Degree propagation: high-degree intersections spread more risk
    3. Severity dominance: MUERTO nodes must have risk >= LESIONADO nodes
    4. Climate intensification: heavy rain implies higher risk than light rain
    """
    def __init__(self, lambda_logic: float = 0.1):
        super().__init__()
        self.lambda_logic = lambda_logic
        
    def forward(self, pred: torch.Tensor, y_true: torch.Tensor, 
                features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        
        data_loss = F.mse_loss(pred, y_true)
        
        u = edge_index[0]; v = edge_index[1]
        N = pred.shape[0]
        
        # P1: Spatial smoothness
        smoothness = torch.mean((pred[u] - pred[v]) ** 2)
        
        # P2: Degree propagation
        deg = features[-1, :, 3].clamp(min=0.02)
        deg_factor = (deg[u] + deg[v]) / 2.0
        degree_prop = torch.mean(deg_factor.unsqueeze(1) * (pred[u] - pred[v]) ** 2)
        
        # P3: Severity dominance (sampled)
        sev = features[-1, :, 1]
        K = 2000
        i = torch.randint(0, N, (K,), device=pred.device)
        j = torch.randint(0, N, (K,), device=pred.device)
        sev_diff = (sev[i] - sev[j]).clamp(min=0)
        pred_gap = pred[i].squeeze() - pred[j].squeeze()
        severity_violation = torch.mean(sev_diff * F.relu(-pred_gap) ** 2)
        
        # P4: Climate intensification
        # Heavy rain days (>15mm) should have higher aggregate risk
        rain = features[-1, :, 0]  # rain_intensity (mm/30, normalized)
        heavy = (rain > 0.5).float()
        if heavy.sum() > 0 and (1-heavy).sum() > 0:
            risk_heavy = (pred.squeeze() * heavy).sum() / heavy.sum()
            risk_light = (pred.squeeze() * (1-heavy)).sum() / (1-heavy).sum()
            climate_violation = F.relu(risk_light - risk_heavy) ** 2
        else:
            climate_violation = 0.0
        
        logic_loss = 0.28*smoothness + 0.24*degree_prop + 0.24*severity_violation + 0.24*climate_violation
        
        return data_loss + self.lambda_logic * logic_loss
