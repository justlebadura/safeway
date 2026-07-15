"""
OSM Graph Module — Real street graph for GNN training and routing.
Replaces the synthetic Cr×Cl grid with actual OpenStreetMap topology.

Imports are lazy to avoid crashing on serverless (Vercel) where torch/scipy
may not be installed. Core routing uses pure networkx + math.
"""
from __future__ import annotations
import math
import networkx as nx
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import defaultdict

from backend.microservices.routing import GraphNode

# Heavy imports — lazy loaded when available
_torch = None
_ox = None
_cKDTree = None
_np = None

def _get_torch():
    global _torch
    if _torch is None:
        try:
            import torch as _t; _torch = _t
        except ImportError:
            raise RuntimeError("torch not installed — needed for GNN inference")
    return _torch

def _get_osmnx():
    global _ox
    if _ox is None:
        import osmnx as _o; _ox = _o
    return _ox

def _get_spatial():
    global _cKDTree
    if _cKDTree is None:
        from scipy.spatial import cKDTree as _c; _cKDTree = _c
    return _cKDTree

def _get_np():
    global _np
    if _np is None:
        import numpy as _n; _np = _n
    return _np


def load_osm_graph(graphml_path: str) -> nx.MultiDiGraph:
    """Load OSM street graph from GraphML file. Downloads from OSM if missing."""
    path = Path(graphml_path)
    if not path.exists():
        # Auto-download from OpenStreetMap
        place_names = {
            'palmira_streets.graphml': 'Palmira, Valle del Cauca, Colombia',
            'bga_streets.graphml': 'Bucaramanga, Santander, Colombia',
            'pereira_streets.graphml': 'Pereira, Risaralda, Colombia',
        }
        name = place_names.get(path.name)
        if name:
            import urllib.request, time
            try:
                ox = _get_osmnx()
                G = ox.graph_from_place(name, network_type='drive')
                path.parent.mkdir(parents=True, exist_ok=True)
                ox.save_graphml(G, str(path))
                return G
            except Exception:
                raise RuntimeError(f"Cannot load or download OSM graph: {graphml_path}")
        else:
            raise FileNotFoundError(f"OSM graph not found: {graphml_path}")
    return _get_osmnx().load_graphml(str(path))


def build_edge_index(G: nx.MultiDiGraph):
    """Build edge_index tensor from OSM graph. Returns torch tensor if available."""
    src, tgt = [], []
    node_to_idx = {n: i for i, n in enumerate(G.nodes())}
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            src.append(node_to_idx[u])
            tgt.append(node_to_idx[v])
    if not src:
        src, tgt = [0], [0]
    return (src, tgt)  # return tuple, caller handles tensor conversion if needed


def _build_spatial_index(G: nx.MultiDiGraph):
    """Build spatial index for fast nearest-neighbor lookup. Returns (tree, ids, coords)."""
    node_ids = []
    coords = []
    for nid, data in G.nodes(data=True):
        if 'y' in data and 'x' in data:
            node_ids.append(nid)
            coords.append([data['y'], data['x']])
    if not coords:
        return None, [], []
    try:
        KD = _get_spatial()
        tree = KD(_get_np().array(coords))
        return tree, node_ids, _get_np().array(coords)
    except (ImportError, RuntimeError):
        # Fallback: store raw coords for linear search
        return coords, node_ids, coords


def snap_to_osm_node(lat: float, lng: float, G: nx.MultiDiGraph, max_dist_m: float = 200) -> Optional[str]:
    """Find nearest OSM node. Uses KDTree if available, otherwise linear search."""
    cache_key = f"spatial_{id(G)}"
    cache = getattr(snap_to_osm_node, '_cache', {})
    if cache_key not in cache:
        cache[cache_key] = _build_spatial_index(G)
        snap_to_osm_node._cache = cache
    index, node_ids, _ = cache[cache_key]
    if index is None:
        return None
    try:
        # KDTree fast path
        dist, idx = index.query([lat, lng], k=1)
        if dist * 111000 < max_dist_m:
            return node_ids[idx]
    except AttributeError:
        # Linear search fallback (no scipy)
        best_dist = float('inf')
        best_node = None
        for i, (cy, cx) in enumerate(index):
            d = math.sqrt((cy - lat)**2 + (cx - lng)**2) * 111000
            if d < best_dist and d < max_dist_m:
                best_dist = d
                best_node = node_ids[i]
        return best_node
    return None


def build_osm_nodes(
    accidents: List[Dict],
    G: nx.MultiDiGraph,
    modes: List[str] = None,
    temporal_window: int = None,
) -> Tuple[List[GraphNode], Tuple, int]:
    """
    Build GraphNode list from OSM graph + snap accidents.
    Also computes neighbor features for GNN training.
    Returns:
        (nodes, edge_index_tuple, num_snapped)
    """
    if modes is None:
        modes = ['MOTOCICLISTA', 'CONDUCTOR', 'PEATON', 'ACOMPAÑANTE MOTOCICLISTA',
                 'ACOMPAÑANTE CONDUCTOR', 'CICLISTA', 'PASAJERO', 'ACOMPAÑANTE CICLISTA']
    
    filtered = []
    for acc in accidents:
        if temporal_window is not None:
            try:
                year = int(str(acc.get('date_iso', '2022'))[:4])
                if year > temporal_window:
                    continue
            except:
                pass
        filtered.append(acc)
    
    node_accidents = defaultdict(list)
    snapped = 0
    for acc in filtered:
        lat = acc.get('latitude')
        lng = acc.get('longitude')
        if lat is None or lng is None:
            continue
        osm_nid = snap_to_osm_node(lat, lng, G, max_dist_m=200)
        if osm_nid:
            node_accidents[osm_nid].append(acc)
            snapped += 1
    
    # Build nodes with base features
    nodes = []
    osm_to_idx = {}
    for i, (nid, data) in enumerate(G.nodes(data=True)):
        accs = node_accidents.get(nid, [])
        lat = data.get('y', 0)
        lng = data.get('x', 0)
        label = data.get('name', str(nid))
        street_count = data.get('street_count', 1)
        
        nd = GraphNode(f"osm_{nid}", lat, lng, label=label, is_fallback=(len(accs) == 0))
        nd.osm_id = nid
        nd.degree = street_count
        nd.betweenness = 0.0
        nd.clustering = 0.0
        for acc in accs:
            nd.add_accident(acc)
        nodes.append(nd)
        osm_to_idx[nid] = i
    
    # Precompute neighbor statistics (1-hop aggregation)
    for i, nd in enumerate(nodes):
        nid = nd.osm_id
        neighbor_accs = []
        neighbor_sevs = []
        for neighbor in G.neighbors(nid):
            if neighbor in osm_to_idx:
                nb = nodes[osm_to_idx[neighbor]]
                neighbor_accs.append(len(nb.accidents))
                if nb.accidents:
                    sevs = []
                    for a in nb.accidents:
                        orig = a.get('data_original', {})
                        lm = str(orig.get('lesionados_y_muertos', '')).upper()
                        g = str(orig.get('gravedad', '')).upper()
                        if 'MUERTO' in lm or 'MUERT' in g: sevs.append(1.0)
                        elif 'LESIONADO' in lm or 'HERIDO' in g: sevs.append(0.5)
                        else: sevs.append(0.25)
                    neighbor_sevs.append(sum(sevs)/len(sevs))
        nd._neighbor_acc = sum(neighbor_accs)/max(len(neighbor_accs),1) if neighbor_accs else 0.0
        nd._neighbor_sev = sum(neighbor_sevs)/max(len(neighbor_sevs),1) if neighbor_sevs else 0.0
    
    edge_index = build_edge_index(G)
    return nodes, edge_index, snapped

def prepare_features(
    nodes: List[GraphNode],
    mode: str = 'all',
) -> torch.Tensor:
    """
    Build feature tensor for GNN input (10 dims).
    
    Features:
        0: lluvia_real (0/1) — Open-Meteo Archive API (precipitation_sum > 0 ese día)
        1: lat_norm — OpenStreetMap (lat/90°)
        2: lng_norm — OpenStreetMap (lng/180°)
        3: severity — Palmira lesionados_y_muertos (MUERTO=1, LESIONADO=0.5, else=0.25)
        4: acc_density — Palmira GPS → snap OSM (count / 20)
        5: degree — OpenStreetMap street_count / 9
        6: neighbor_acc — Palmira + OSM (promedio accidentes en vecinos / 20)
        7: neighbor_sev — Palmira + OSM (promedio severidad en vecinos)
        8: betweenness — OpenStreetMap (centralidad de intermediación)
        9: mode_match — Palmira condicion_de_la_victima (fracción que coincide con modo)
    """
    N = len(nodes)
    feats = torch.zeros((N, 10), dtype=torch.float32)
    
    for idx, nd in enumerate(nodes):
        feats[idx, 1] = nd.lat / 90.0 if nd.lat != 0 else 0.0
        feats[idx, 2] = nd.lng / 180.0 if nd.lng != 0 else 0.0
        feats[idx, 5] = getattr(nd, 'degree', 1) / 9.0
        feats[idx, 6] = min(1.0, getattr(nd, '_neighbor_acc', 0) / 20.0)
        feats[idx, 7] = getattr(nd, '_neighbor_sev', 0.5)
        feats[idx, 8] = getattr(nd, 'betweenness', 0.0)
        
        if nd.accidents:
            sev_total = 0.0
            mode_hits = 0
            lluvia_hits = 0
            total_acc = len(nd.accidents)
            for acc in nd.accidents:
                orig = acc.get('data_original', {})
                g = str(orig.get('lesionados_y_muertos', '')).upper()
                cond = str(orig.get('condicion_de_la_victima', '')).upper()
                if 'MUERTO' in g: sev_total += 1.0
                elif 'LESIONADO' in g: sev_total += 0.5
                else: sev_total += 0.25
                if mode == 'all' or _mode_match(mode, cond): mode_hits += 1
                if orig.get('lluvia_real', False): lluvia_hits += 1
            
            feats[idx, 0] = lluvia_hits / total_acc if lluvia_hits > 0 else 0.0
            feats[idx, 3] = sev_total / total_acc
            feats[idx, 4] = min(1.0, total_acc / 20.0)
            feats[idx, 9] = mode_hits / total_acc
    
    return feats


def _mode_match(mode: str, condicion: str) -> bool:
    """Check if transport mode matches victim condition."""
    mode = mode.upper().strip()
    cond = condicion.upper().strip()
    if mode == 'CARRO' or mode == 'CONDUCTOR':
        return 'CONDUCTOR' in cond and 'MOTOCICLISTA' not in cond and 'ACOMPA' not in cond
    if mode == 'MOTO' or mode == 'MOTOCICLISTA':
        return 'MOTOCICLISTA' in cond
    if mode == 'PEATON':
        return 'PEAT' in cond or 'PEATON' in cond
    if mode == 'BICI' or mode == 'CICLISTA':
        return 'CICLISTA' in cond
    return False


def compute_targets(nodes: List[GraphNode], target_year: int, mode: str = 'all') -> torch.Tensor:
    """Binary target: did this node have an accident in target_year matching the mode?"""
    targets = []
    for nd in nodes:
        hit = False
        for acc in nd.accidents:
            try:
                year = int(str(acc.get('date_iso', '2000'))[:4])
            except:
                continue
            if year == target_year:
                orig = acc.get('data_original', {})
                cond = str(orig.get('condicion_de_la_victima', '')).upper()
                if mode == 'all' or mode.upper() in cond or _mode_match(mode, cond):
                    hit = True
                    break
        targets.append(1.0 if hit else 0.0)
    return torch.tensor(targets, dtype=torch.float32).unsqueeze(1)


def get_edge_weights(
    G: nx.MultiDiGraph,
    node_risks: Dict[str, float],
    penalty: float = 30.0,
) -> Dict:
    """Compute edge weights for routing: street_length + risk_dest × penalty."""
    weights = {}
    for u, v, data in G.edges(data=True):
        length = data.get('length', 0)
        risk = node_risks.get(v, 0.0)
        weights[(u, v)] = length + risk * penalty
    return weights


def find_safest_route_osm(
    G: nx.MultiDiGraph,
    start_lat: float, start_lng: float,
    end_lat: float, end_lng: float,
    node_risks: Dict[str, float],
    penalty: float = 30.0,
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Find safest path on OSM graph using risk-weighted Dijkstra.
    
    Returns:
        (path_as_lat_lng_pairs, total_distance_meters)
    """
    # Snap start/end to nearest OSM nodes
    start_node = snap_to_osm_node(start_lat, start_lng, G, max_dist_m=500)
    end_node = snap_to_osm_node(end_lat, end_lng, G, max_dist_m=500)
    
    if start_node is None or end_node is None:
        return [(start_lat, start_lng), (end_lat, end_lng)], 0.0
    
    if start_node == end_node:
        nd = G.nodes[start_node]
        return [(nd['y'], nd['x'])], 0.0
    
    # Build weighted graph
    G_weighted = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        length = data.get('length', 10.0)
        risk = node_risks.get(v, 0.0)
        weight = length + risk * penalty
        G_weighted.add_edge(u, v, weight=weight, length=length)
        G_weighted.add_edge(v, u, weight=length + node_risks.get(u, 0.0) * penalty, length=length)
    
    # Run Dijkstra
    try:
        path_nodes = nx.shortest_path(G_weighted, source=start_node, target=end_node, weight='weight')
    except nx.NetworkXNoPath:
        return [(start_lat, start_lng), (end_lat, end_lng)], 0.0
    
    # Convert to lat/lng coordinates
    path = []
    total_dist = 0.0
    for nid in path_nodes:
        nd = G.nodes[nid]
        path.append((nd['y'], nd['x']))
        if len(path) > 1 and G_weighted.has_edge(path_nodes[path_nodes.index(nid)-1], nid):
            total_dist += G_weighted[path_nodes[path_nodes.index(nid)-1]][nid].get('length', 0)
    
    return path, round(total_dist, 2)


def compute_risk_scores(
    nodes: List['GraphNode'],
    G: nx.MultiDiGraph,
    rain_active: bool = False,
    target_hour: int = 12,
    mode: str = 'all',
) -> Dict[str, float]:
    """
    Compute risk scores for all OSM nodes using symbolic formula.
    Returns dict mapping osm_node_id -> risk_score.
    """
    from backend.api import _load_symbolic_formula, _symbolic_risk
    
    cf = _load_symbolic_formula()
    risks = {}
    for nd in nodes:
        n = len(nd.accidents)
        sev = 0.0; lluvia = 0.0; mode_match_frac = 0.0
        if n > 0:
            for acc in nd.accidents:
                orig = acc.get('data_original', {})
                g = str(orig.get('lesionados_y_muertos', '')).upper()
                if 'MUERTO' in g: sev += 1.0
                elif 'LESIONADO' in g: sev += 0.5
                else: sev += 0.25
                if orig.get('lluvia_real', False): lluvia += 1
            sev /= n
            lluvia = lluvia / n
            conds = [str(a.get('data_original',{}).get('condicion_de_la_victima','')).upper() for a in nd.accidents]
            if mode == 'all': mode_match_frac = 1.0
            elif mode == 'moto': mode_match_frac = sum(1 for c in conds if 'MOTOCICLISTA' in c) / n
            elif mode == 'carro': mode_match_frac = sum(1 for c in conds if 'CONDUCTOR' in c and 'MOTOCICLISTA' not in c and 'ACOMPA' not in c) / n
            elif mode == 'peaton': mode_match_frac = sum(1 for c in conds if 'PEAT' in c) / n
        
        risks[nd.osm_id] = _symbolic_risk(
            rain=lluvia,
            lat_norm=nd.lat/90.0 if nd.lat!=0 else 0,
            lng_norm=nd.lng/180.0 if nd.lng!=0 else 0,
            severity=sev,
            acc_density=min(1.0, n/20.0),
            degree=getattr(nd,'degree',1)/9.0,
            neighbor_acc=min(1.0, getattr(nd,'_neighbor_acc',0)/20.0),
            neighbor_sev=getattr(nd,'_neighbor_sev',0.5),
            betweenness=getattr(nd,'betweenness',0.0),
            mode_match=mode_match_frac
        )
    
    # Percentile normalization
    sorted_risks = sorted(risks.values())
    n_r = len(sorted_risks)
    for key, val in risks.items():
        rank = sorted_risks.index(val) + 1
        risks[key] = round(10.0 * rank / n_r, 3)
    
    return risks
