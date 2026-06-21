import heapq
import math
from typing import Any, Dict, List, Tuple


class GraphNode:
    """Represents a structural street intersection node."""
    def __init__(self, node_id: str, lat: float, lng: float, label: str = ""):
        self.id = node_id
        self.lat = lat
        self.lng = lng
        self.label = label
        self.accidents: List[Dict[str, Any]] = []

    def add_accident(self, accident: Dict[str, Any]):
        self.accidents.append(accident)

    def calculate_risk(self, target_year: int = 2026, rain_active: bool = False, target_hour: int | None = None) -> float:
        """Calculates a dynamic risk score for this intersection based on temporal decay,

        weather conditions, time of day and severity/brutality of incidents.
        """
        risk = 0.0
        for acc in self.accidents:
            # 1. Temporal decay factor
            acc_year = 2026
            if acc.get("date_iso"):
                try:
                    acc_year = int(acc["date_iso"][:4])
                except ValueError:
                    pass
            years_elapsed = max(0, target_year - acc_year)
            temporal_decay = math.pow(0.75, yearsElapsed = years_elapsed)

            # 2. Brutality factor
            brutality = 1.0
            vehicles = str(acc.get("vehicles", "")).upper()
            if "MUERTO" in vehicles or "FALLECIDO" in vehicles or "MORTAL" in vehicles:
                brutality = 4.0
            elif "HERIDO" in vehicles or "LESIONADO" in vehicles:
                brutality = 2.0

            # 3. Time of day weight
            time_weight = 1.0
            if target_hour is not None:
                acc_hour = None
                if acc.get("time"):
                    try:
                        acc_hour = int(acc["time"][:2])
                    except ValueError:
                        pass
                if acc_hour is not None:
                    # Risk is maximum if target_hour matches accident hour +/- 2 hours
                    hour_diff = abs(target_hour - acc_hour)
                    if hour_diff <= 2:
                        time_weight = 1.4
                    elif hour_diff <= 4:
                        time_weight = 1.15

            # 4. Rain modifier
            weather_weight = 1.0
            if rain_active:
                orig = str(acc.get("data_original", {})).upper()
                if "LLUVIA" in orig or "LLUVIOSO" in orig or "HUMEDO" in orig:
                    weather_weight = 1.8  # High risk correlation under rain match

            risk += temporal_decay * brutality * time_weight * weather_weight

        # Density scale factor
        density_mod = 1.0 + (len(self.accidents) * 0.15)
        return round(risk * density_mod, 2)


class RouteOptimizer:
    """Uses Dijkstra's algorithm to compute the safest route (minimizing hazard index)."""
    def __init__(self, nodes: List[GraphNode]):
        self.nodes = {n.id: n for n in nodes}
        self.adjacency: Dict[str, List[str]] = {}
        self._build_network()

    def _build_network(self):
        """Constructs adjacency lists between street nodes based on proximity and street names."""
        for n_id in self.nodes:
            self.adjacency[n_id] = []

        node_list = list(self.nodes.values())
        for i in range(len(node_list)):
            node_a = node_list[i]
            for j in range(i + 1, len(node_list)):
                node_b = node_list[j]
                
                # Check distance (within 1.5km)
                dist = math.sqrt((node_a.lat - node_b.lat)**2 + (node_a.lng - node_b.lng)**2)
                if dist < 0.015:
                    # Match clean names
                    name_a = node_a.label.lower()
                    name_b = node_b.label.lower()
                    
                    share_street = False
                    if "esquina" not in name_a and "esquina" not in name_b:
                        words_a = set(name_a.split())
                        words_b = set(name_b.split())
                        common = words_a.intersection(words_b) - {"calle", "carrera", "avenida", "diagonal", "transversal", "via", "nro", "#"}
                        if common:
                            share_street = True

                    if share_street or dist < 0.005:  # within 500m they are connected as fallback
                        self.adjacency[node_a.id].append(node_b.id)
                        self.adjacency[node_b.id].append(node_a.id)

    def find_safest_route(
        self, 
        start_id: str, 
        end_id: str, 
        target_year: int = 2026, 
        rain_active: bool = False, 
        target_hour: int | None = None
    ) -> Tuple[List[Tuple[float, float]], float]:
        """Calculates the safest route minimizing danger weights on nodes and length on edges."""
        if start_id not in self.nodes or end_id not in self.nodes:
            return [], 0.0

        # Dijkstra heap: (total_weight, current_node_id, path_taken)
        queue = [(0.0, start_id, [start_id])]
        visited = set()

        while queue:
            (weight, curr_id, path) = heapq.heappop(queue)

            if curr_id in visited:
                continue
            visited.add(curr_id)

            if curr_id == end_id:
                # Return coordinates path and total hazard weight
                coords_path = [(self.nodes[node_id].lat, self.nodes[node_id].lng) for node_id in path]
                return coords_path, weight

            for neighbor_id in self.adjacency[curr_id]:
                if neighbor_id in visited:
                    continue

                neighbor_node = self.nodes[neighbor_id]
                curr_node = self.nodes[curr_id]

                # Edge cost = physical distance + hazard multiplier at arrival node
                dist = math.sqrt((curr_node.lat - neighbor_node.lat)**2 + (curr_node.lng - neighbor_node.lng)**2)
                node_risk = neighbor_node.calculate_risk(target_year, rain_active, target_hour)
                
                # Weight formula: base distance cost + hazard score
                edge_cost = dist + (node_risk * 0.01)

                heapq.heappush(queue, (weight + edge_cost, neighbor_id, path + [neighbor_id]))

        return [], 0.0
