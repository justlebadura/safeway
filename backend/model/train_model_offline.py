import sys
import os
import math
import torch
import torch.optim as optim

sys.path.append("/home/lebadura/Documentos/GitHub/safeway")
sys.path.append("/home/lebadura/Documentos/GitHub/safeway/backend")

from backend.model.loader.dataset import TemporalGraphDataset
from backend.microservices.grapher import MapGrapher
from backend.microservices.api_soda_cleaner import get_combined_datasets_snapshot
from backend.model.arch.hybrid_model import HybridGNNLNN
from backend.model.loss.rill_loss import HybridLoss
from backend.microservices.routing import GraphNode

def train_offline():
    print("Loading real dataset for offline training...")
    snapshot = get_combined_datasets_snapshot("7cci-nqqb", max_rows=1500)
    records = snapshot["tables"]["records"]
    
    grapher = MapGrapher()
    nodes = grapher.build_structural_graph(records)
    num_nodes = len(nodes)
    print(f"Loaded {num_nodes} positive nodes.")
    
    # --- Spatial Negative Sampling (Positive-Only Learning Correction) ---
    import random
    lats = [n.lat for n in nodes]
    lngs = [n.lng for n in nodes]
    lat_min, lat_max = min(lats), max(lats)
    lng_min, lng_max = min(lngs), max(lngs)
    
    num_negatives = len(nodes)  # 1:1 Positive-to-Negative ratio
    neg_nodes = []
    neg_counter = 1
    
    random.seed(42)
    attempts = 0
    while len(neg_nodes) < num_negatives and attempts < 2000:
        attempts += 1
        rand_lat = random.uniform(lat_min, lat_max)
        rand_lng = random.uniform(lng_min, lng_max)
        
        # Check distance to all positive nodes (at least 150m / 0.0013 degrees away)
        too_close = False
        for n in nodes:
            if abs(n.lat - rand_lat) < 0.0013 and abs(n.lng - rand_lng) < 0.0013:
                too_close = True
                break
        
        if not too_close:
            node_id = f"node_neg_{neg_counter}"
            neg_counter += 1
            # Negative node has 0 accidents
            node = GraphNode(node_id, rand_lat, rand_lng, label="Zona Residencial Segura")
            neg_nodes.append(node)
            
    nodes.extend(neg_nodes)
    num_nodes = len(nodes)
    print(f"Added {len(neg_nodes)} negative control nodes. Total nodes for GNN training: {num_nodes}")
    
    # Build edge index
    sources, targets = [], []
    for i, node_a in enumerate(nodes):
        for j, node_b in enumerate(nodes):
            if i != j:
                dist = math.sqrt((node_a.lat - node_b.lat)**2 + (node_a.lng - node_b.lng)**2)
                if dist < 0.005:
                    sources.append(i)
                    targets.append(j)
    if not sources:
        sources = list(range(num_nodes))
        targets = list(range(num_nodes))
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    
    # We will generate a list of training pairs (x_seq, y_true) for different conditions
    training_data = []
    for rain_active in [False, True]:
        for hour in range(24):
            # Generate features sequence
            sequences = []
            for t in range(5):
                h_seq = (hour - 4 + t) % 24
                r_seq = rain_active if t == 4 else False
                
                features = torch.zeros((num_nodes, 5), dtype=torch.float32)
                for idx, node in enumerate(nodes):
                    features[idx, 0] = 1.0 if r_seq else 0.0
                    features[idx, 1] = 0.0 if r_seq else 1.0
                    features[idx, 2] = h_seq / 24.0
                    
                    num_acc = len(node.accidents)
                    features[idx, 4] = float(num_acc) / 50.0  # Normalize count
                    if num_acc > 0:
                        sev_sum = 0.0
                        for acc in node.accidents:
                            v = str(acc.get("vehicles", "")).upper()
                            if "MUERTO" in v or "FALLECIDO" in v:
                                sev_sum += 4.0
                            elif "HERIDO" in v or "LESIONADO" in v:
                                sev_sum += 2.0
                            else:
                                sev_sum += 1.0
                        features[idx, 3] = (sev_sum / num_acc) / 4.0  # Normalize severity
                    else:
                        features[idx, 3] = 0.0
                sequences.append(features)
            x_seq = torch.stack(sequences, dim=0)
            
            # Target output
            y_true = torch.zeros((num_nodes, 1), dtype=torch.float32)
            for idx, node in enumerate(nodes):
                y_true[idx, 0] = min(1.0, node.calculate_risk(2012, rain_active, hour) / 10.0)
                
            training_data.append((x_seq, y_true))
            
    # Initialize model
    model = HybridGNNLNN(in_features=5, gnn_hidden=8, lnn_hidden=16)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = HybridLoss(lambda_logic=0.05)
    
    # Offline training loop
    epochs = 150
    print(f"Starting offline training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        optimizer.zero_grad()
        
        # Accumulate loss over all diverse weather/hour sequences
        for x_seq, y_true in training_data:
            pred = model(x_seq, edge_index)
            loss = criterion(pred, y_true, x_seq, edge_index)
            total_loss += loss
            
        total_loss = total_loss / len(training_data)
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} | Average Loss: {total_loss.item():.6f}")
            
    # Save the trained model
    model_dir = "backend/model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Offline training complete. Model saved to {model_path}")

if __name__ == "__main__":
    train_offline()
