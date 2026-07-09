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
            temporal_decay = math.pow(0.75, years_elapsed)

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


class StreetSegmentNode:
    """
    Represents a directed street segment (an edge in the original graph).
    In the Edge-Based Graph (Line Graph), this is our Node.
    """
    def __init__(self, seg_id: str, u: GraphNode, v: GraphNode):
        self.id = seg_id
        self.u = u  # start intersection
        self.v = v  # end intersection
        self.lat = (u.lat + v.lat) / 2.0
        self.lng = (u.lng + v.lng) / 2.0
        self.distance = math.sqrt((u.lat - v.lat)**2 + (u.lng - v.lng)**2)
        # Combine labels to describe the segment
        self.label = f"{u.label} -> {v.label}"


class RouteOptimizer:
    """
    Uses an Edge-Based Graph (Line Graph / Dual Graph) and Dijkstra to compute safest routes,
    natively supporting turn penalties and road hierarchy factor weights just like Google Maps/Waze.
    """
    def __init__(self, nodes: List[GraphNode]):
        self.nodes = {n.id: n for n in nodes}
        self.segments: Dict[str, StreetSegmentNode] = {}
        self.adjacency: Dict[str, List[str]] = {}  # Adjacency between segments (maneuvers)
        self._build_edge_based_graph()

    def _build_edge_based_graph(self):
        """Builds the dual line graph where original connections are nodes, and turn maneuvers are edges."""
        node_list = list(self.nodes.values())
        connections = []

        # Find original intersection connections (within 500m proximity or sharing name)
        for i in range(len(node_list)):
            node_a = node_list[i]
            for j in range(i + 1, len(node_list)):
                node_b = node_list[j]
                dist = math.sqrt((node_a.lat - node_b.lat)**2 + (node_a.lng - node_b.lng)**2)

                name_a = node_a.label.lower()
                name_b = node_b.label.lower()
                share_street = False
                if "esquina" not in name_a and "esquina" not in name_b:
                    words_a = set(name_a.split())
                    words_b = set(name_b.split())
                    common = words_a.intersection(words_b) - {"calle", "carrera", "avenida", "diagonal", "transversal", "via", "nro", "#"}
                    if common:
                        share_street = True

                if share_street or dist < 0.005:
                    # Directed segments (both directions)
                    connections.append((node_a, node_b))
                    connections.append((node_b, node_a))

        # Create Line Graph vertices (StreetSegmentNode)
        for u, v in connections:
            seg_id = f"seg_{u.id}_{v.id}"
            self.segments[seg_id] = StreetSegmentNode(seg_id, u, v)

        # Create Line Graph edges (Transitions / Turn maneuvers)
        # seg_1 = (A, B) connects to seg_2 = (B, C) if seg_1.v == seg_2.u
        for seg_id in self.segments:
            self.adjacency[seg_id] = []

        for seg_id_1, seg_1 in self.segments.items():
            for seg_id_2, seg_2 in self.segments.items():
                # Avoid immediate U-turns (e.g. A -> B -> A)
                if seg_1.v.id == seg_2.u.id and seg_1.u.id != seg_2.v.id:
                    self.adjacency[seg_id_1].append(seg_id_2)

    def find_safest_route(
        self, 
        start_id: str, 
        end_id: str, 
        target_year: int = 2026, 
        rain_active: bool = False, 
        target_hour: int | None = None,
        use_hazard: bool = True
    ) -> Tuple[List[Tuple[float, float]], float]:
        """Runs Dijkstra on the Edge-Based Graph (Street Segments) to find safest paths."""
        # Find all segments that originate from start_id
        start_segs = [s_id for s_id, seg in self.segments.items() if seg.u.id == start_id]
        if not start_segs:
            # Fallback if graph is empty or node is isolated
            if start_id in self.nodes and end_id in self.nodes:
                return [
                    (self.nodes[start_id].lat, self.nodes[start_id].lng),
                    (self.nodes[end_id].lat, self.nodes[end_id].lng)
                ], 0.0
            return [], 0.0

        def get_road_hierarchy_factor(label: str) -> float:
            lbl = label.lower()
            if any(k in lbl for k in ["avenida", "autopista", "viaducto", "carrera 27", "carrera 33", "diagonal 15", "boulevard"]):
                return 1.0
            elif "calle" in lbl or "carrera" in lbl:
                return 1.3
            else:
                return 1.9

        # Dijkstra queue: (total_cost, current_segment_id, path_taken_segment_ids, visited_intersection_ids)
        queue = []
        for s_id in start_segs:
            seg = self.segments[s_id]
            h_factor = get_road_hierarchy_factor(seg.label)
            init_cost = seg.distance * h_factor
            if use_hazard:
                risk_u = getattr(seg.u, "predicted_risk", None) or seg.u.calculate_risk(target_year, rain_active, target_hour)
                risk_v = getattr(seg.v, "predicted_risk", None) or seg.v.calculate_risk(target_year, rain_active, target_hour)
                init_cost += ((risk_u + risk_v) / 2.0 * 0.01)
            heapq.heappush(queue, (init_cost, s_id, [s_id], [seg.u.id, seg.v.id]))

        visited = set()

        while queue:
            (weight, curr_seg_id, path, visited_nodes) = heapq.heappop(queue)

            if curr_seg_id in visited:
                continue
            visited.add(curr_seg_id)

            curr_seg = self.segments[curr_seg_id]

            # Reached destination intersection
            if curr_seg.v.id == end_id:
                # Reconstruct path coordinates: starting node of first segment, and ending node of all segments
                coords_path = [(self.segments[path[0]].u.lat, self.segments[path[0]].u.lng)]
                for s_id in path:
                    coords_path.append((self.segments[s_id].v.lat, self.segments[s_id].v.lng))
                return coords_path, weight

            for neighbor_seg_id in self.adjacency[curr_seg_id]:
                if neighbor_seg_id in visited:
                    continue

                neighbor_seg = self.segments[neighbor_seg_id]

                # Prevent loops by checking if the end node of neighbor segment is already visited
                if neighbor_seg.v.id in visited_nodes:
                    continue

                # 1. Base segment distance
                dist = neighbor_seg.distance
                
                # 2. Road Hierarchy Penalization
                h_factor = get_road_hierarchy_factor(neighbor_seg.label)
                base_cost = dist * h_factor

                # 3. Turn Penalty (calculates vector angle between incoming and outgoing segments)
                # Current vector: curr_seg.u -> curr_seg.v
                v1_lat = curr_seg.v.lat - curr_seg.u.lat
                v1_lng = curr_seg.v.lng - curr_seg.u.lng
                # Neighbor vector: neighbor_seg.u -> neighbor_seg.v
                v2_lat = neighbor_seg.v.lat - neighbor_seg.u.lat
                v2_lng = neighbor_seg.v.lng - neighbor_seg.u.lng
                
                turn_penalty = 0.0
                dot = v1_lat * v2_lat + v1_lng * v2_lng
                m1 = math.sqrt(v1_lat**2 + v1_lng**2)
                m2 = math.sqrt(v2_lat**2 + v2_lng**2)
                if m1 > 0 and m2 > 0:
                    cos_angle = max(-1.0, min(1.0, dot / (m1 * m2)))
                    angle = math.acos(cos_angle)
                    if angle > 0.78:  # Turn > 45 degrees
                        turn_penalty = 0.0015  # Virtual distance penalty (~150m)

                edge_cost = base_cost + turn_penalty
                
                if use_hazard:
                    risk_v = getattr(neighbor_seg.v, "predicted_risk", None)
                    if risk_v is None:
                        risk_v = neighbor_seg.v.calculate_risk(target_year, rain_active, target_hour)
                    edge_cost += (risk_v * 0.01)

                heapq.heappush(queue, (weight + edge_cost, neighbor_seg_id, path + [neighbor_seg_id], visited_nodes + [neighbor_seg.v.id]))

        # Fallback if no path is found
        if start_id in self.nodes and end_id in self.nodes:
            return [
                (self.nodes[start_id].lat, self.nodes[start_id].lng),
                (self.nodes[end_id].lat, self.nodes[end_id].lng)
            ], 0.0
        return [], 0.0
