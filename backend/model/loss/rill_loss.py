import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    """
    Combines Data Loss (MSE) with Reduced Implication-bias Logic Loss (RILL)
    to ensure spatial and logical consistency in predictions.
    """
    def __init__(self, lambda_logic: float = 0.1):
        super().__init__()
        self.lambda_logic = lambda_logic
        
    def forward(self, pred: torch.Tensor, y_true: torch.Tensor, 
                features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined loss.
        
        Args:
            pred (torch.Tensor): Model predictions
            y_true (torch.Tensor): Ground truth targets
            features (torch.Tensor): Input features (not used in this simplified version)
            edge_index (torch.Tensor): Graph edge indices for logic constraints
            
        Returns:
            torch.Tensor: Combined loss
        """
        # Data loss
        data_loss = F.mse_loss(pred, y_true)
        
        # Vectorized Logic Loss (RILL approximation: spatial consistency constraint)
        # u and v are tensors of indices of connected nodes
        u = edge_index[0]
        v = edge_index[1]
        
        # Calculate mean squared differences in parallel using prediction tensor
        smoothness_loss = torch.mean((pred[u] - pred[v]) ** 2)
        
        return data_loss + self.lambda_logic * smoothness_loss
