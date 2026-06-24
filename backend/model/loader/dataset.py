import torch
import os
from typing import Dict, List, Any, Tuple
from backend.microservices.api_soda_cleaner import get_combined_datasets_snapshot

class TemporalGraphDataset:
    """
    Clase encargada de estructurar los datos del grafo y generar secuencias temporales.
    Utiliza la lógica de limpieza centralizada en 'api_soda_cleaner'.
    """
    def __init__(self):
        self.node_list = []
        self.node_to_idx = {}
        self.num_nodes = 0
        self.edge_index = None

    def load_from_json(self, dataset_ids: str = "stq8-drvp,7cci-nqqb,3v2w-chcq,sjpx-eqfp,dr5c-eewa", max_rows: int = 200):
        """
        Carga datos procesados usando el servicio central de limpieza.
        """
        # Obtener datos limpios a través del servicio central
        snapshot = get_combined_datasets_snapshot(dataset_ids, max_rows=max_rows)
        records = snapshot["tables"]["records"]
        
        # Agrupar por ubicación (nodo)
        unique_locations = {}
        for acc in records:
            loc = acc.get("location", "unknown")
            if loc not in unique_locations:
                unique_locations[loc] = []
            unique_locations[loc].append(acc)
            
        self.node_ids = list(unique_locations.keys())
        self.node_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}
        self.num_nodes = len(self.node_ids)
        
        class Node:
            def __init__(self, id, accidents):
                self.id = id
                self.accidents = accidents
        
        self.node_list = [Node(nid, unique_locations[nid]) for nid in self.node_ids]
        
        # Construir grafo simple
        sources, targets = [], []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                sources.extend([i, j])
                targets.extend([j, i])
        self.edge_index = torch.tensor([sources, targets], dtype=torch.long)

    def extract_node_features(self, target_hour: int, rain_active: bool) -> torch.Tensor:
        """
        Extrae las características de los nodos en un instante específico.
        """
        features = torch.zeros((self.num_nodes, 5), dtype=torch.float32)
        for idx, node in enumerate(self.node_list):
            features[idx, 0] = 1.0 if rain_active else 0.0
            features[idx, 1] = 0.0 if rain_active else 1.0
            features[idx, 2] = target_hour / 24.0
            
            num_acc = len(node.accidents)
            features[idx, 4] = float(num_acc)
            if num_acc > 0:
                sev_sum = 0.0
                for acc in node.accidents:
                    vehicles = str(acc.get("vehicles", "")).upper()
                    if "MUERTO" in vehicles or "FALLECIDO" in vehicles:
                        sev_sum += 4.0
                    elif "HERIDO" in vehicles or "LESIONADO" in vehicles:
                        sev_sum += 2.0
                    else:
                        sev_sum += 1.0
                features[idx, 3] = sev_sum / num_acc
            else:
                features[idx, 3] = 0.0
        return features

    def generate_temporal_sequence(self, history_steps: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Genera una secuencia temporal para entrenamiento.
        """
        sequences = []
        for t in range(history_steps):
            hour = (12 + t) % 24
            rain = (t % 3 == 0)
            X_t = self.extract_node_features(target_hour=hour, rain_active=rain)
            sequences.append(X_t)
        
        Y = torch.zeros((self.num_nodes, 1), dtype=torch.float32)
        for idx, node in enumerate(self.node_list):
            Y[idx, 0] = min(1.0, len(node.accidents) * 0.2)
            
        return torch.stack(sequences, dim=0), Y
