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
        self.predicted_risk: float | None = None

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
                    weather_weight = 1.8

            risk += temporal_decay * brutality * time_weight * weather_weight

        # Density scale factor
        density_mod = 1.0 + (len(self.accidents) * 0.15)
        return round(risk * density_mod, 2)


class RouteOptimizer:
    """
    Uses NetworkX A* with a Euclidean heuristic and safety-weighted edges to compute
    safest and fastest routes. Implements bounding-box pruning to eliminate nodes
    outside the origin-destination corridor before running the search.

    Edge weight formula:
        peso = distancia × jerarquía × (1.0 + riesgo × 0.5)

    Where riesgo comes from GraphNode.calculate_risk() — a time/weather/severity
    heuristic with real variance across the city — ensuring meaningful route differences.
    """

    _HIERARCHY: List[Tuple[List[str], float]] = [
        (["avenida", "autopista", "viaducto", "carrera 27", "carrera 33", "diagonal 15", "boulevard"], 1.0),
        (["calle", "carrera", "transversal", "diagonal"], 1.3),
    ]
    _HIERARCHY_DEFAULT = 1.9

    def __init__(self, nodes: List[GraphNode]):
        self.node_map: Dict[str, GraphNode] = {n.id: n for n in nodes}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _euclidean(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        return math.sqrt((lat1 - lat2) ** 2 + (lng1 - lng2) ** 2)

    def _hierarchy_factor(self, label: str) -> float:
        lbl = label.lower()
        for keywords, factor in self._HIERARCHY:
            if any(k in lbl for k in keywords):
                return factor
        return self._HIERARCHY_DEFAULT

    def _build_graph(
        self,
        target_year: int,
        rain_active: bool,
        target_hour: int | None,
        use_hazard: bool,
        bbox: Tuple[float, float, float, float] | None,
    ):
        """Build a directed NetworkX DiGraph with safety-weighted edges.

        bbox = (lat_min, lat_max, lng_min, lng_max) — nodes outside are pruned.
        """
        import networkx as nx

        G = nx.DiGraph()

        # Add nodes inside bounding box
        for nid, node in self.node_map.items():
            if bbox:
                lat_min, lat_max, lng_min, lng_max = bbox
                if not (lat_min <= node.lat <= lat_max and lng_min <= node.lng <= lng_max):
                    continue
            G.add_node(nid, lat=node.lat, lng=node.lng)

        node_ids = list(G.nodes)

        # Add directed edges for proximate nodes (< 0.005° ≈ 500 m)
        PROXIMITY = 0.005
        for i, aid in enumerate(node_ids):
            a = self.node_map[aid]
            for bid in node_ids[i + 1:]:
                b = self.node_map[bid]
                if abs(a.lat - b.lat) > PROXIMITY or abs(a.lng - b.lng) > PROXIMITY:
                    continue

                dist = self._euclidean(a.lat, a.lng, b.lat, b.lng)
                hier = (self._hierarchy_factor(a.label) + self._hierarchy_factor(b.label)) / 2.0

                # Forward A→B: weight penalises risk at destination node B
                risk_b = getattr(b, "predicted_risk", None)
                if risk_b is None:
                    risk_b = b.calculate_risk(target_year, rain_active, target_hour)
                if not use_hazard:
                    risk_b = 0.0
                G.add_edge(aid, bid, weight=dist * hier * (1.0 + risk_b * 3.0))

                # Reverse B→A: weight penalises risk at destination node A
                risk_a = getattr(a, "predicted_risk", None)
                if risk_a is None:
                    risk_a = a.calculate_risk(target_year, rain_active, target_hour)
                if not use_hazard:
                    risk_a = 0.0
                G.add_edge(bid, aid, weight=dist * hier * (1.0 + risk_a * 3.0))

        return G

    def _bounding_box(
        self, start_id: str, end_id: str, margin_factor: float = 0.35
    ) -> Tuple[float, float, float, float] | None:
        """Compute a generous bounding box around origin and destination."""
        if start_id not in self.node_map or end_id not in self.node_map:
            return None
        s = self.node_map[start_id]
        e = self.node_map[end_id]
        lat_span = abs(s.lat - e.lat)
        lng_span = abs(s.lng - e.lng)
        margin_lat = max(lat_span * margin_factor, 0.01)
        margin_lng = max(lng_span * margin_factor, 0.01)
        return (
            min(s.lat, e.lat) - margin_lat,
            max(s.lat, e.lat) + margin_lat,
            min(s.lng, e.lng) - margin_lng,
            max(s.lng, e.lng) + margin_lng,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_safest_route(
        self,
        start_id: str,
        end_id: str,
        target_year: int = 2026,
        rain_active: bool = False,
        target_hour: int | None = None,
        use_hazard: bool = True,
    ) -> Tuple[List[Tuple[float, float]], float]:
        """Find a route using A* with SafeWay risk weights and bounding-box pruning."""
        import networkx as nx

        if start_id not in self.node_map or end_id not in self.node_map:
            return [], 0.0

        bbox = self._bounding_box(start_id, end_id)
        G = self._build_graph(target_year, rain_active, target_hour, use_hazard, bbox)

        # Guarantee start and end are always in the graph
        for nid in (start_id, end_id):
            if nid not in G:
                node = self.node_map[nid]
                G.add_node(nid, lat=node.lat, lng=node.lng)

        if not nx.has_path(G, start_id, end_id):
            # Retry without bounding box
            G = self._build_graph(target_year, rain_active, target_hour, use_hazard, bbox=None)
            for nid in (start_id, end_id):
                if nid not in G:
                    node = self.node_map[nid]
                    G.add_node(nid, lat=node.lat, lng=node.lng)

        if not nx.has_path(G, start_id, end_id):
            s, e = self.node_map[start_id], self.node_map[end_id]
            return [(s.lat, s.lng), (e.lat, e.lng)], 0.0

        end_node = self.node_map[end_id]

        def heuristic(u: str, _v: str) -> float:
            nu = self.node_map.get(u)
            if nu is None:
                return 0.0
            return self._euclidean(nu.lat, nu.lng, end_node.lat, end_node.lng)

        try:
            path_ids = nx.astar_path(G, start_id, end_id, heuristic=heuristic, weight="weight")
            total_cost = nx.astar_path_length(G, start_id, end_id, heuristic=heuristic, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            s, e = self.node_map[start_id], self.node_map[end_id]
            return [(s.lat, s.lng), (e.lat, e.lng)], 0.0

        coords = [(self.node_map[nid].lat, self.node_map[nid].lng) for nid in path_ids]
        return coords, round(total_cost, 4)
