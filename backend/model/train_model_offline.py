#!/usr/bin/env python3
"""
SafeWay v9 — 8 structural features + 6 quarterly timesteps + LTN predicates
GNN(2-GCN) + CfC(LNN) + Enhanced RILL loss
"""
import sys, os, math, time, json, warnings
warnings.filterwarnings("ignore")

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_repo_root = os.path.dirname(_backend_dir)
for p in [_repo_root, _backend_dir]:
    if p not in sys.path: sys.path.insert(0, p)

import numpy as np, torch, torch.nn as nn, torch.optim as optim

from backend.model.arch.hybrid_model import HybridGNNLNN
from backend.model.loss.rill_loss import HybridLoss
from backend.microservices.api_soda_cleaner import cache_service
from backend.microservices.osm_graph import (
    load_osm_graph, build_osm_nodes, prepare_features,
    compute_targets, build_edge_index
)

GNN_H, LNN_H = 32, 64
EPOCHS, LR, WD, LAMBDA = 400, 0.002, 1e-4, 0.05
# 4 semi-annual timesteps for richer LTN signal
WINDOWS = [2022.0, 2022.5, 2023.0, 2023.5]
TARGET_YEAR = 2024

def load_data():
    print(f"Loading Palmira accidents...", flush=True)
    cache_service._entries.clear()
    snap = cache_service.get_snapshot('sjpx-eqfp', max_rows=50000, force_refresh=False)
    all_records = [dict(r) for r in snap.processed if r.get('latitude') is not None]
    print(f"  Total: {len(all_records)}", flush=True)
    G = load_osm_graph(os.path.join(_repo_root, 'data', 'palmira_streets.graphml'))
    print(f"  OSM: {len(G.nodes)}n {len(G.edges)}e", flush=True)
    return all_records, G

def gen_temporal_samples(all_records, G):
    nodes_by_window = []
    for wy in WINDOWS:
        nds, ei, _ = build_osm_nodes(all_records, G, temporal_window=wy)
        nodes_by_window.append(nds)
        print(f"  Window {wy}: {sum(1 for n in nds if n.accidents)} nodes with accidents", flush=True)
    full_nodes, _, _ = build_osm_nodes(all_records, G)
    
    samples = []
    for mode in ['all', 'moto', 'carro', 'peaton']:
        seq = []
        for nodes_t in nodes_by_window:
            seq.append(prepare_features(nodes_t, mode=mode))
        y = compute_targets(full_nodes, TARGET_YEAR, mode=mode)
        samples.append({'x': torch.stack(seq, dim=0), 'y': y, 'mode': mode})
    return samples, full_nodes, ei

def compute_metrics(preds, targets, threshold=0.5):
    pb = (preds > threshold).float()
    tp = (pb * targets).sum().item(); fp = (pb * (1-targets)).sum().item()
    fn = ((1-pb) * targets).sum().item(); tn = ((1-pb) * (1-targets)).sum().item()
    p = tp/(tp+fp) if (tp+fp)>0 else 0; r = tp/(tp+fn) if (tp+fn)>0 else 0
    return {'accuracy':(tp+tn)/len(targets),'precision':p,'recall':r,'f1':2*p*r/(p+r) if(p+r)>0 else 0,
            'tp':tp,'fp':fp,'fn':fn,'tn':tn}

def symbolic_regression(model, nodes, ei, samples):
    print("\n--- Symbolic Regression ---", flush=True)
    model.eval(); N = len(nodes)
    A = torch.zeros(N,N); A[ei[0],ei[1]]=1.0
    D = A.sum(dim=1).clamp(min=1); NA = torch.diag(1.0/D.sqrt()) @ A @ torch.diag(1.0/D.sqrt())
    Xf, yp = [], []
    for s in samples:
        with torch.no_grad(): pr = model(s['x'], ei)
        lf = s['x'][-1]
        for i in range(N):
            Xf.append([lf[i,j].item() for j in range(8)])
            yp.append(pr[i,0].item())
    X = np.array(Xf,dtype=np.float64); y = np.array(yp,dtype=np.float64)
    from sklearn.linear_model import Ridge
    rd = Ridge(alpha=0.01); rd.fit(X,y); r2 = rd.score(X,y)
    c = rd.coef_
    names = ['lluvia','sev','acc','deg','nb_acc','nb_sev','btwn','mode']
    formula = f"risk={rd.intercept_:.6f}"
    for i in range(len(c)): formula += f"+{c[i]:.6f}*{names[i]}"
    print(f"  R² = {r2:.4f}", flush=True)
    for n,v in zip(names,c): print(f"    {n:8s}: {v:+.6f}", flush=True)
    return formula, r2, rd

def train():
    print("\n"+"="*60, flush=True)
    print(f"SafeWay v9 — 8 structural features + 4 timesteps + LTN", flush=True)
    print(f"   City: Palmira | Target: {TARGET_YEAR} | Windows: {len(WINDOWS)}", flush=True)
    print("="*60, flush=True)
    
    all_records, G = load_data()
    samples, full_nodes, ei = gen_temporal_samples(all_records, G)
    N = len(full_nodes)
    print(f"\n  Nodes: {N} | Samples: {len(samples)} | Edges: {ei.shape[1]}", flush=True)
    
    model = HybridGNNLNN(in_features=8, gnn_hidden=GNN_H, lnn_hidden=LNN_H)
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    crit = HybridLoss(lambda_logic=LAMBDA)
    
    all_yt = torch.cat([samples[i]['y'] for i in range(len(samples))])
    pos = all_yt.sum().item(); neg = len(all_yt)-pos
    pw = neg/max(pos,1)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
    npar = sum(p.numel() for p in model.parameters())
    print(f"  Params: {npar:,} | Pos/Neg: {pos:.0f}/{neg:.0f} (w={pw:.1f})", flush=True)
    
    ns = len(samples)
    best_f1 = 0.0; t0 = time.time()
    for ep in range(1, EPOCHS+1):
        model.train(); el = 0.0
        for i in range(ns):
            opt.zero_grad()
            pr = model(samples[i]['x'], ei)
            ls = bce(pr, samples[i]['y']) + LAMBDA * crit(pr, samples[i]['y'], samples[i]['x'], ei)
            ls.backward(); opt.step(); el += ls.item()
        
        if ep % 50 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                ap = torch.cat([model(samples[i]['x'], ei) for i in range(ns)])
                ay = torch.cat([samples[i]['y'] for i in range(ns)])
            m = compute_metrics(ap, ay)
            if m['f1'] > best_f1:
                best_f1 = m['f1']
                torch.save(model.state_dict(), os.path.join(_backend_dir, "model", "model.pth"))
            print(f"Ep{ep:3d} L:{el/ns:.4f} P:{m['precision']:.2f} R:{m['recall']:.2f} F1:{m['f1']:.2f}", flush=True)
    
    print(f"\nDone in {time.time()-t0:.1f}s | Best F1: {best_f1:.3f}", flush=True)
    
    model.eval()
    with torch.no_grad():
        ap = torch.cat([model(samples[i]['x'], ei) for i in range(ns)])
        ay = torch.cat([samples[i]['y'] for i in range(ns)])
    m = compute_metrics(ap, ay)
    print(f"Final: Acc={m['accuracy']:.3f} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}", flush=True)
    print(f"TP={m['tp']:.0f} FP={m['fp']:.0f} FN={m['fn']:.0f} TN={m['tn']:.0f}", flush=True)
    
    for mode in ['all','moto','carro','peaton']:
        mi = [i for i,s in enumerate(samples) if s['mode']==mode]
        if mi:
            mp = torch.cat([model(samples[i]['x'], ei) for i in mi])
            my = torch.cat([samples[i]['y'] for i in mi])
            mm = compute_metrics(mp, my)
            print(f"  {mode:8s}: P={mm['precision']:.2f} R={mm['recall']:.2f} F1={mm['f1']:.2f}", flush=True)
    
    formula, r2, ridge = symbolic_regression(model, full_nodes, ei, samples)
    if formula:
        fp = os.path.join(_backend_dir, "model", "symbolic_formula.txt")
        with open(fp, "w") as f:
            f.write(f"# SafeWay v9 — 8 structural features + 6 timesteps + LTN\n")
            f.write(f"# City: Palmira | Target: {TARGET_YEAR} | R²={r2:.4f}\n")
            f.write(f"# P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}\n\n")
            f.write(formula)
        print(f"\nFormula: R²={r2:.4f}", flush=True)
    
    return model, full_nodes, ei

if __name__ == '__main__':
    train()
