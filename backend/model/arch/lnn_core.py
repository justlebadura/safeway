import torch
import torch.nn as nn

class CfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) cell approximation for Liquid Neural Networks.
    Models neural dynamics as an ODE-inspired continuous system.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Linear layer for computing gates (Input, Forget, Cell, Output)
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Computes the next hidden state based on continuous-time dynamics.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_size)
            h (torch.Tensor): Hidden state tensor of shape (batch, hidden_size)
            
        Returns:
            torch.Tensor: New hidden state
        """
        combined = torch.cat([x, h], dim=1)
        gates = self.linear(combined)
        
        i, f, c, o = torch.split(gates, self.hidden_size, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c = torch.tanh(c)
        o = torch.sigmoid(o)
        
        # h_new calculation based on CfC dynamics approximation
        h_new = o * torch.tanh(f * h + i * c)
        return h_new
