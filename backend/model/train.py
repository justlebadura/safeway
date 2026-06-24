import torch
import torch.optim as optim
from backend.model.hybrid_model import HybridGNNLNN
from backend.model.rill_loss import HybridLoss
from backend.model.dataset import TemporalGraphDataset

def train_model(nodes, adjacency, epochs=100):
    # Dataset
    dataset = TemporalGraphDataset(nodes, adjacency)
    edge_index = dataset.edge_index
    
    # Model
    model = HybridGNNLNN(in_features=5, gnn_hidden=16, lnn_hidden=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = HybridLoss(lambda_logic=0.1)
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get data
        x_seq, y_true = dataset.generate_temporal_sequence()
        
        # Forward pass
        pred = model(x_seq, edge_index)
        
        # Loss (includes RILL component)
        loss = criterion(pred, y_true, x_seq, edge_index)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
    return model

if __name__ == "__main__":
    # This is a dummy training script setup.
    # In a real scenario, we'd pass real node objects and adjacency.
    print("Train script initialized. Dataset and Model components linked.")
