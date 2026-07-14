import heapq, math
from typing import Any, Dict, List, Tuple

class GraphNode:
    def __init__(self, node_id: str, lat: float, lng: float, label: str = "", is_fallback: bool = False):
        self.id = node_id
        self.lat = lat
        self.lng = lng
        self.label = label
        self.is_fallback = is_fallback
        self.accidents: List[Dict[str, Any]] = []
        self.predicted_risk: float | None = None

    def add_accident(self, accident: Dict[str, Any]):
        self.accidents.append(accident)

    def calculate_risk(self, target_year: int = 2026, rain_active: bool = False, target_hour: int | None = None) -> float:
        risk = 0.0
        for acc in self.accidents:
            years_elapsed = max(0, target_year - int(str(acc.get("date_iso", "2026"))[:4])) if acc.get("date_iso") else 0
            decay = math.pow(0.75, years_elapsed)

            v = str(acc.get("vehicles", "")).upper()
            brutality = 4.0 if any(k in v for k in ("MUERTO","FALLECIDO","MORTAL")) else (2.0 if any(k in v for k in ("HERIDO","LESIONADO")) else 1.0)

            time_w = 1.0
            if target_hour is not None and acc.get("time"):
                try:
                    ah = int(acc["time"][:2])
                    diff = abs(target_hour - ah)
                    time_w = 1.4 if diff <= 2 else (1.15 if diff <= 4 else 1.0)
                except: pass

            risk += decay * brutality * time_w

        return round(risk * (1.0 + len(self.accidents) * 0.05), 2)


class RouteOptimizer:
    def __init__(self, nodes: List[GraphNode]):
        self.node_map: Dict[str, GraphNode] = {n.id: n for n in nodes}

    @staticmethod
    def _dist(lat1, lng1, lat2, lng2):
        return math.sqrt((lat1 - lat2)**2 + (lng1 - lng2)**2)

    def _build_graph(self, target_year, rain_active, target_hour, use_hazard: bool):
        import networkx as nx
        G = nx.DiGraph()
        nids = list(self.node_map.keys())
        PROX = 0.02
        K = 5  # k-nearest neighbors for dense connectivity

        for nid, node in self.node_map.items():
            G.add_node(nid)

        for i, aid in enumerate(nids):
            a = self.node_map[aid]
            # Get K nearest neighbors
            neighbors = []
            for j, bid in enumerate(nids):
                if aid == bid: continue
                b = self.node_map[bid]
                d = self._dist(a.lat, a.lng, b.lat, b.lng)
                if d < PROX or len(neighbors) < K:
                    neighbors.append((d, bid, b))
            neighbors.sort(key=lambda x: x[0])
            
            for d, bid, b in neighbors[:K]:
                km = d * 111.0
                time_s = (km / 30.0) * 3600.0
                risk_b = b.predicted_risk if b.predicted_risk is not None else b.calculate_risk(target_year, rain_active, target_hour)
                risk_a = a.predicted_risk if a.predicted_risk is not None else a.calculate_risk(target_year, rain_active, target_hour)
                if not use_hazard: risk_b = risk_a = 0.0
                G.add_edge(aid, bid, weight=time_s + risk_b * 30.0)
                G.add_edge(bid, aid, weight=time_s + risk_a * 30.0)

        # Bridge disconnected components
        UG = G.to_undirected()
        comps = list(nx.connected_components(UG))
        if len(comps) > 1:
            comps.sort(key=len, reverse=True)
            main = comps[0]
            for other in comps[1:]:
                md = float('inf'); best = None
                for u in main:
                    for v in other:
                        d = self._dist(self.node_map[u].lat, self.node_map[u].lng, self.node_map[v].lat, self.node_map[v].lng)
                        if d < md: md = d; best = (u, v)
                if best:
                    u, v = best
                    km = md * 111.0; time_s = (km / 30.0) * 3600.0
                    G.add_edge(u, v, weight=time_s)
                    G.add_edge(v, u, weight=time_s)
                    main = set(list(main) + list(other))

        return G

    def find_safest_route(self, start_id, end_id, target_year=2026, rain_active=False, target_hour=None, use_hazard=True):
        if start_id not in self.node_map or end_id not in self.node_map:
            return [], 0.0

        G = self._build_graph(target_year, rain_active, target_hour, use_hazard)
        
        if start_id not in G: G.add_node(start_id)
        if end_id not in G: G.add_node(end_id)

        if start_id == end_id:
            n = self.node_map[start_id]
            return [(n.lat, n.lng)], 0.0

        # Bidirectional Dijkstra
        fw_q = [(0.0, start_id)]
        bw_q = [(0.0, end_id)]
        fw_d = {start_id: 0.0}
        bw_d = {end_id: 0.0}
        fw_p = {start_id: None}
        bw_p = {end_id: None}
        fw_v = set()
        bw_v = set()
        best = float('inf')
        meet = None

        while fw_q and bw_q:
            if fw_q[0][0] + bw_q[0][0] >= best:
                break

            if len(fw_q) <= len(bw_q):
                d, u = heapq.heappop(fw_q)
                if d > fw_d.get(u, float('inf')) or u in fw_v: continue
                fw_v.add(u)
                if u in bw_d and fw_d[u] + bw_d[u] < best:
                    best, meet = fw_d[u] + bw_d[u], u
                for v in G.successors(u):
                    nd = fw_d[u] + G[u][v].get('weight', 1.0)
                    if nd < fw_d.get(v, float('inf')):
                        fw_d[v] = nd; fw_p[v] = u; heapq.heappush(fw_q, (nd, v))
            else:
                d, u = heapq.heappop(bw_q)
                if d > bw_d.get(u, float('inf')) or u in bw_v: continue
                bw_v.add(u)
                if u in fw_d and fw_d[u] + bw_d[u] < best:
                    best, meet = fw_d[u] + bw_d[u], u
                for v in G.predecessors(u):
                    nd = bw_d[u] + G[v][u].get('weight', 1.0)
                    if nd < bw_d.get(v, float('inf')):
                        bw_d[v] = nd; bw_p[v] = u; heapq.heappush(bw_q, (nd, v))

        if meet is None:
            s, e = self.node_map[start_id], self.node_map[end_id]
            return [(s.lat, s.lng), (e.lat, e.lng)], 0.0

        path = []
        cur = meet
        while cur is not None:
            path.append(cur); cur = fw_p[cur]
        path.reverse()
        cur = bw_p.get(meet)
        while cur is not None:
            path.append(cur); cur = bw_p.get(cur)

        return [(self.node_map[nid].lat, self.node_map[nid].lng) for nid in path], round(best, 4)
