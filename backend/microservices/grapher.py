from typing import Any, Dict, List
from microservices.routing import GraphNode


class MapGrapher:
    """Microservice responsible for organizing and snapping geocoded accidents onto structural nodes/edges."""
    def __init__(self, proximity_threshold: float = 0.0004):
        self.proximity_threshold = proximity_threshold

    def build_structural_graph(self, accidents: List[Dict[str, Any]]) -> List[GraphNode]:
        """Snaps geocoded accidents into structural intersection nodes and returns them."""
        corners: Dict[str, GraphNode] = {}
        node_counter = 1

        for acc in accidents:
            # Skip if it is a fallback coordinate (we don't snap fallback coords to structural graph)
            if acc.get("is_fallback_coord"):
                continue

            lat = acc.get("latitude")
            lng = acc.get("longitude")
            if lat is None or lng is None:
                continue

            street_name = (
                acc.get("data_limpia", {}).get("lugar") or 
                acc.get("data_limpia", {}).get("barrio") or 
                acc.get("data_limpia", {}).get("DIRECCION") or 
                acc.get("data_limpia", {}).get("sitio_exacto_accidente") or
                acc.get("data_limpia", {}).get("Dirección_Reporte") or
                "Esquina"
            )

            # Check snapping to existing structural node
            snapped = False
            for c in corners.values():
                lat_diff = abs(c.lat - lat)
                lng_diff = abs(c.lng - lng)
                if lat_diff < self.proximity_threshold and lng_diff < self.proximity_threshold:
                    c.add_accident(acc)
                    # Slightly adjust centroid to reflect the average of snapped incidents
                    count = len(c.accidents)
                    c.lat = (c.lat * (count - 1) + lat) / count
                    c.lng = (c.lng * (count - 1) + lng) / count
                    snapped = True
                    break
            
            if not snapped:
                node_id = f"node_{node_counter}"
                node_counter += 1
                node = GraphNode(node_id, lat, lng, label=street_name)
                node.add_accident(acc)
                corners[node_id] = node

        return list(corners.values())
