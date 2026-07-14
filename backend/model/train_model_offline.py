#!/usr/bin/env python3
"""
SafeWay v6 — LNN Temporal Real + Prediccion de Accidentes Nuevos
- Timesteps reales acumulativos (2014, 2016, 2018, 2020, 2022)
- Target binario: tuvo accidente en 2023?
- GCN(2-capas) propaga riesgo vecinal creciente
- LNN(CfC) aprende trayectorias temporales de riesgo
- Metricas: Precision, Recall, F1 sobre prediccion de nuevos accidentes
"""
import sys, os, math, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from backend.model.arch.hybrid_model import HybridGNNLNN
from backend.model.loss.rill_loss import HybridLoss
from backend.microservices.api_soda_cleaner import get_combined_datasets_snapshot, cache_service
from backend.microservices.routing import GraphNode
from backend.api import BGA_COORDS

GNN_H, LNN_H = 32, 64
EPOCHS, LR, WD, LAMBDA = 200, 0.003, 1e-4, 0.1
HOLDOUT_YEAR = 2023
# Temporal windows for LNN timesteps
WINDOWS = [2014, 2016, 2018, 2020, 2022]

def _sev(v):
    v = str(v).upper()
    if any(k in v for k in ("MUERTO","FALLECIDO","MORTAL")): return 4.0
    if any(k in v for k in ("HERIDO","LESIONADO")): return 2.0
    return 1.0

def _filter_by_year(records, max_year):
    return [r for r in records if int(str(r.get('date_iso','2012'))[:4]) <= max_year]

def _build_grid(records):
    """Build 1560-intersection grid with accumulated accidents up to given records."""
    from collections import defaultdict
    
    BGA_CR0, BGA_CL0, UNIT = -73.1220, 7.1190, 0.0007
    CR_MIN, CR_MAX, CL_MIN, CL_MAX = 15, 40, 1, 60
    
    grid = defaultdict(list)
    for acc in records:
        b = acc.get('data_original', {}).get('barrio', '').strip()
        key = b.split(',')[0].strip().upper()
        coords = (BGA_CL0, BGA_CR0)
        for bname, bcoord in BGA_COORDS.items():
            if bname in key: coords = bcoord; break
        cr = max(CR_MIN, min(CR_MAX, int(round(27 + (coords[1] - BGA_CR0) / UNIT))))
        cl = max(CL_MIN, min(CL_MAX, int(round(36 + (coords[0] - BGA_CL0) / UNIT))))
        grid[(cr, cl)].append(acc)
    
    nodes = []
    nid = 1
    for cr in range(CR_MIN, CR_MAX + 1):
        for cl in range(CL_MIN, CL_MAX + 1):
            lat = BGA_CL0 + (cl - 36) * UNIT
            lng = BGA_CR0 - (cr - 27) * UNIT
            nd = GraphNode(f"n_{nid}", lat, lng, f"Cr {cr} c/ Cl {cl}", is_fallback=False)
            nd.cr = cr; nd.cl = cl
            for a in grid.get((cr, cl), []): nd.add_accident(a)
            # Topological features
            neighbors = 0
            for dc, dl in [(1,0),(-1,0),(0,1),(0,-1)]:
                if CR_MIN <= cr+dc <= CR_MAX and CL_MIN <= cl+dl <= CL_MAX:
                    neighbors += 1
            nd.degree = neighbors
            nd.dist_center = math.sqrt((cr-27)**2 + (cl-36)**2) / 50.0
            nodes.append(nd); nid += 1
    return nodes

def _build_edge_index(nodes):
    src, tgt = [], []
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i >= j: continue
            d = math.sqrt((a.lat-b.lat)**2 + (a.lng-b.lng)**2)
            if d < 0.0015: src.append(i); tgt.append(j); src.append(j); tgt.append(i)
    if not src: return torch.tensor([list(range(len(nodes)))]*2, dtype=torch.long)
    return torch.tensor([src, tgt], dtype=torch.long)

def load_data():
    print(f"Loading data for temporal LNN training...")
    cache_service._entries.clear()
    os.system('rm -f data/raw_7cci-nqqb.json')
    snap = get_combined_datasets_snapshot("7cci-nqqb", max_rows=50000, force_refresh=True)
    all_records = snap["tables"]["records"]
    
    # Split: train uses 2012-2022, target is 2023
    train_records = _filter_by_year(all_records, 2022)
    holdout_records = _filter_by_year(all_records, 2023)
    holdout_2023 = [r for r in holdout_records if int(str(r.get('date_iso','2012'))[:4]) == HOLDOUT_YEAR]
    
    print(f"  Train (2012-2022): {len(train_records)} | Holdout ({HOLDOUT_YEAR}): {len(holdout_2023)}")
    
    # Build grid for target computation
    full_nodes = _build_grid(all_records)
    
    # Build temporal sequences: each timestep = accumulated data up to that year
    temporal_nodes = []
    for wy in WINDOWS:
        subset = _filter_by_year(all_records, wy)
        temporal_nodes.append(_build_grid(subset))
    
    # Target: which nodes had accidents in 2023?
    ei = _build_edge_index(temporal_nodes[0])
    
    # Compute target for all nodes
    node_targets = []
    for nd in full_nodes:
        # Count accidents in holdout period
        holdout_count = sum(1 for a in nd.accidents if int(str(a.get('date_iso','2012'))[:4]) == HOLDOUT_YEAR)
        node_targets.append(1 if holdout_count > 0 else 0)
    
    print(f"  Nodes with accidents in {HOLDOUT_YEAR}: {sum(node_targets)}/{len(full_nodes)}")
    return full_nodes, temporal_nodes, ei, node_targets

def gen_samples(nodes_by_window, ei, node_targets, conditions):
    """Generate training samples: 5 timesteps of real temporal data → binary target."""
    samples = []
    for rain in conditions.get('rain', [False, True]):
        for hour in conditions.get('hour', range(0, 24, 3)):
            seq = []
            for t, nodes_t in enumerate(nodes_by_window):
                N = len(nodes_t)
                f = torch.zeros((N, 7), dtype=torch.float32)
                for idx, nd in enumerate(nodes_t):
                    n = len(nd.accidents)
                    f[idx,0] = 1.0 if rain else 0.0
                    f[idx,1] = 0.0 if rain else 1.0
                    f[idx,2] = hour/24.0
                    f[idx,4] = float(n)/50.0
                    f[idx,3] = sum(_sev(a.get("vehicles","")) for a in nd.accidents)/max(n,1)/4.0
                    f[idx,5] = getattr(nd,'degree',4)/4.0
                    f[idx,6] = min(1.0, getattr(nd,'dist_center',0))
                seq.append(f)
            
            y = torch.tensor([[t] for t in node_targets], dtype=torch.float32)
            samples.append((torch.stack(seq, dim=0), y, {"rain": rain, "hour": hour}))
    return samples

def flops(model, x_seq, ei):
    in_f, gh, lh = 7, model.gnn_hidden, model.lnn_hidden
    sl, N, _ = x_seq.shape; E = ei.shape[1]
    g1 = E*in_f*2 + N*in_f*gh*2
    g2 = E*gh*2 + N*gh*gh*2
    lf = N*(gh+lh)*(4*lh)*2 + N*lh*8
    of = N*lh*2 + N*4
    return g1+g2+lf*sl+of, (g1+g2+lf*sl+of)*3

def symbolic_regression(model, nodes, ei, samples):
    print("\n--- Regresion Simbolica ---")
    if len(nodes) < 2: return None, 0, None, None, None
    model.eval()
    Xf, yp = [], []
    
    A = torch.zeros(len(nodes), len(nodes))
    A[ei[0], ei[1]] = 1.0
    D = A.sum(dim=1).clamp(min=1)
    NA = torch.diag(1.0/D.sqrt()) @ A @ torch.diag(1.0/D.sqrt())
    
    for xs, _, cnd in samples:
        r = cnd["rain"]; h = cnd["hour"]
        last_t = xs[-1]  # final timestep features
        sev = last_t[:,3].numpy(); acc = last_t[:,4].numpy()
        sn = (NA @ torch.tensor(sev,dtype=torch.float32).unsqueeze(1)).squeeze().numpy()
        an = (NA @ torch.tensor(acc,dtype=torch.float32).unsqueeze(1)).squeeze().numpy()
        with torch.no_grad(): pr = model(xs, ei)
        for i in range(len(nodes)):
            Xf.append([1.0 if r else 0.0, h/24.0, sev[i], acc[i], sn[i], an[i],
                       np.sin(6.283*h/24.0), np.cos(6.283*h/24.0)])
            yp.append(pr[i,0].item())
    
    X = np.array(Xf,dtype=np.float64); y = np.array(yp,dtype=np.float64)
    from sklearn.linear_model import Ridge
    rd = Ridge(alpha=0.01); rd.fit(X, y); r2 = rd.score(X, y)
    c = rd.coef_; ic = rd.intercept_
    print(f"  R² = {r2:.4f}")
    return (f"risk={ic:.6f}+{c[0]:.6f}*rain+{c[1]:.6f}*h+{c[2]:.6f}*sev+{c[3]:.6f}*acc+{c[4]:.6f}*sevN+{c[5]:.6f}*accN+{c[6]:.6f}*sin+{c[7]:.6f}*cos", r2, X, y, rd)

def compute_metrics(model, samples, ei, threshold=0.5):
    """Precision, Recall, F1 for binary accident prediction."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xs, yt, _ in samples:
            pr = model(xs, ei)
            all_preds.extend((pr > threshold).float().squeeze().tolist())
            all_targets.extend(yt.squeeze().tolist())
    
    tp = sum(1 for p, t in zip(all_preds, all_targets) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(all_preds, all_targets) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(all_preds, all_targets) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(all_preds, all_targets) if p == 0 and t == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(all_targets) if all_targets else 0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def train():
    print("\n" + "="*60)
    print(f"SafeWay v6 — LNN Temporal Real + Prediccion {HOLDOUT_YEAR}")
    print("   GCN(2-capas) + LNN(CfC) + LTN(RILL)")
    print("   Timesteps reales: " + ", ".join(str(w) for w in WINDOWS))
    print("="*60)
    
    full_nodes, temporal_nodes, ei, targets = load_data()
    N = len(temporal_nodes[0])
    
    conditions = {'rain': [False, True], 'hour': range(0, 24, 3)}
    samples = gen_samples(temporal_nodes, ei, targets, conditions)
    
    # Shuffle and split by condition
    np.random.seed(42)
    idx = np.random.permutation(len(samples))
    sp = int(len(samples) * 0.8)
    ti, vi = idx[:sp], idx[sp:]
    print(f"\nTrain: {len(ti)} | Val: {len(vi)} conditions")
    
    model = HybridGNNLNN(in_features=7, gnn_hidden=GNN_H, lnn_hidden=LNN_H)
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    crit = HybridLoss(lambda_logic=LAMBDA)
    npar = sum(p.numel() for p in model.parameters())
    fwd, fwd_bw = flops(model, samples[0][0], ei)
    tfl = fwd_bw * len(ti) * EPOCHS
    
    # Use weighted loss: penalize false negatives more (missing accidents is worse)
    pos_weight = (N - sum(targets)) / max(sum(targets), 1)  # balance classes
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    
    print(f"Params: {npar:,} | Fwd: {fwd/1e6:.1f}M | Train: {tfl/1e9:.1f}G FLOPs")
    print(f"Class balance: {sum(targets)} positive / {N-sum(targets)} negative (weight={pos_weight:.1f})")
    
    tl, vl = [], []
    best = float('inf'); t0 = time.time()
    
    for ep in range(1, EPOCHS + 1):
        model.train(); el = 0.0
        for i in np.random.permutation(ti):
            xs, yt, _ = samples[i]; opt.zero_grad()
            pr = model(xs, ei)
            # Combine BCE (binary) + RILL (spatial smoothness)
            ls = bce(pr, yt) + LAMBDA * crit(pr, yt, xs, ei)
            ls.backward(); opt.step()
            el += ls.item()
        tl.append(el/len(ti))
        
        model.eval(); vl_ = 0.0
        with torch.no_grad():
            for i in vi:
                xs, yt, _ = samples[i]; pr = model(xs, ei)
                vl_ += (bce(pr, yt) + LAMBDA * crit(pr, yt, xs, ei)).item()
        vl.append(vl_/len(vi))
        
        if vl[-1] < best: best = vl[-1]; torch.save(model.state_dict(), "backend/model/model.pth")
        if ep % 30 == 0 or ep == 1:
            m = compute_metrics(model, [samples[i] for i in vi], ei)
            print(f"Ep{ep:3d} TrL:{tl[-1]:.4f} VlL:{vl[-1]:.4f} | P:{m['precision']:.2f} R:{m['recall']:.2f} F1:{m['f1']:.2f}")
    
    tt = time.time() - t0
    train_metrics = compute_metrics(model, [samples[i] for i in ti], ei)
    val_metrics = compute_metrics(model, [samples[i] for i in vi], ei)
    all_metrics = compute_metrics(model, samples, ei)
    
    print(f"\n--- Final ---")
    print(f"Train: Acc={train_metrics['accuracy']:.3f} P={train_metrics['precision']:.2f} R={train_metrics['recall']:.2f} F1={train_metrics['f1']:.2f}")
    print(f"Val:   Acc={val_metrics['accuracy']:.3f} P={val_metrics['precision']:.2f} R={val_metrics['recall']:.2f} F1={val_metrics['f1']:.2f}")
    print(f"All:   Acc={all_metrics['accuracy']:.3f} P={all_metrics['precision']:.2f} R={all_metrics['recall']:.2f} F1={all_metrics['f1']:.2f}")
    print(f"TP={all_metrics['tp']} FP={all_metrics['fp']} FN={all_metrics['fn']} TN={all_metrics['tn']}")
    
    # Check specific nodes
    print(f"\n--- Predicciones especificas ---")
    model.eval()
    with torch.no_grad():
        pred = model(samples[0][0], ei)
    for nd in full_nodes:
        if nd.label == 'Cr 16 c/ Cl 36':
            idx = full_nodes.index(nd)
            prob = pred[idx, 0].item()
            actual = targets[idx]
            print(f"  {nd.label}: pred={prob:.3f} (>{0.5}=peligroso) actual={'SI' if actual else 'NO'}")
        if nd.label == 'Cr 27 c/ Cl 36':
            idx = full_nodes.index(nd)
            prob = pred[idx, 0].item()
            actual = targets[idx]
            print(f"  {nd.label}: pred={prob:.3f} actual={'SI' if actual else 'NO'}")
    
    # Show top predicted danger nodes
    top = sorted([(i, pred[i,0].item()) for i in range(N)], key=lambda x:-x[1])[:10]
    print(f"\n  Top 10 predicciones de peligro:")
    for i, p in top:
        nd = full_nodes[i]
        print(f"    {nd.label:22s} pred={p:.3f} actual={'SI' if targets[i] else 'NO'} acc={len(nd.accidents):4d}")
    
    # Symbolic regression
    fm, r2, Xs, ys, rd = symbolic_regression(model, temporal_nodes[-1], ei, samples)
    sf, nf = 34, fwd
    
    if fm and rd is not None:
        with open("backend/model/symbolic_formula.txt","w") as f:
            f.write(f"# SafeWay v6 — Temporal LNN {HOLDOUT_YEAR} Prediction\n")
            f.write(f"# R²={r2:.4f} P={val_metrics['precision']:.2f} R={val_metrics['recall']:.2f} F1={val_metrics['f1']:.2f}\n\n")
            f.write(fm)
            f.write(f"\n\n# Python:\n")
            f.write("def symbolic_risk(rain, hour, severity, acc, sevN, accN):\n")
            f.write("    import math; h=hour/24.0\n")
            f.write(f"    return max(0.0, {rd.intercept_:.6f} +\n")
            f.write(f"        {rd.coef_[0]:.6f}*rain + {rd.coef_[1]:.6f}*h +\n")
            f.write(f"        {rd.coef_[2]:.6f}*severity + {rd.coef_[3]:.6f}*acc +\n")
            f.write(f"        {rd.coef_[4]:.6f}*sevN + {rd.coef_[5]:.6f}*accN +\n")
            f.write(f"        {rd.coef_[6]:.6f}*math.sin(6.283*h) +\n")
            f.write(f"        {rd.coef_[7]:.6f}*math.cos(6.283*h))\n")
    
    # Charts
    os.makedirs("backend/model", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0,0].plot(tl, c='#2563eb', label='Train'); axes[0,0].plot(vl, c='#ef4444', label='Val')
    axes[0,0].set_title('Loss (BCE + RILL λ=0.1)'); axes[0,0].legend()
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    train_vals = [train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall'], train_metrics['f1']]
    val_vals = [val_metrics['accuracy'], val_metrics['precision'], val_metrics['recall'], val_metrics['f1']]
    x = np.arange(len(metrics_names))
    axes[0,1].bar(x-0.2, train_vals, 0.4, label='Train', color='#2563eb')
    axes[0,1].bar(x+0.2, val_vals, 0.4, label='Val', color='#ef4444')
    axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(metrics_names)
    axes[0,1].set_title('Prediccion de Accidentes Nuevos'); axes[0,1].legend()
    axes[0,1].set_ylim(0, 1)
    
    axes[0,2].bar(['TP','FP','FN','TN'], [all_metrics['tp'], all_metrics['fp'], all_metrics['fn'], all_metrics['tn']],
                  color=['#10b981','#ef4444','#f59e0b','#94a3b8'])
    axes[0,2].set_title('Matriz de Confusion')
    
    axes[1,0].bar(['Red Neuronal\n(1 forward)', 'Formula\nSimbolica'], [nf, sf], color=['#8b5cf6','#10b981'])
    axes[1,0].set_title('Costo Computacional (FLOPs)')
    for i,v in enumerate([nf, sf]): axes[1,0].text(i, v*1.5, f'{v:,}', ha='center', fontsize=9)
    
    if Xs is not None:
        ism = np.random.choice(len(Xs), min(5000, len(Xs)), replace=False)
        axes[1,1].scatter(ys[ism], rd.predict(Xs[ism]), alpha=0.2, s=1, c='#6366f1')
        axes[1,1].plot([0,1],[0,1],'r--',alpha=0.5)
        axes[1,1].set_title(f'Regresion Simbolica (R²={r2:.3f})')
    
    axes[1,2].axis('off')
    txt = (f"GNN(2-capas)+LNN(CfC)+RILL\n"
           f"Prediccion temporal {HOLDOUT_YEAR}\n\n"
           f"Grid: {N} intersecciones\n"
           f"Timesteps reales: {len(WINDOWS)}\n"
           f"Hasta: {WINDOWS[-1]}\n"
           f"Target: accidente en {HOLDOUT_YEAR}\n\n"
           f"Params: {npar:,}\n"
           f"Precision: {val_metrics['precision']:.2f}\n"
           f"Recall: {val_metrics['recall']:.2f}\n"
           f"F1: {val_metrics['f1']:.2f}\n\n"
           f"NN: {nf/1e6:.1f}M FLOPs\n"
           f"Simbolica: {sf} FLOPs\n"
           f"R²: {r2 if r2 else 0:.3f}\n"
           f"Time: {tt:.1f}s")
    axes[1,2].text(0.05, 0.95, txt, transform=axes[1,2].transAxes, fontsize=10,
                   va='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='#f8fafc', alpha=0.9))
    plt.tight_layout(); plt.savefig("backend/model/training_report.png", dpi=150, bbox_inches='tight'); plt.close()
    
    st = {"params": npar, "accuracy": val_metrics['accuracy'], "precision": val_metrics['precision'],
          "recall": val_metrics['recall'], "f1": val_metrics['f1'],
          "tp": all_metrics['tp'], "fp": all_metrics['fp'], "fn": all_metrics['fn'], "tn": all_metrics['tn'],
          "train_flops": int(tfl), "infer_flops": int(nf), "sym_flops": sf,
          "sym_r2": float(r2) if r2 else 0, "time": tt, "nodes": N,
          "target_year": HOLDOUT_YEAR, "windows": WINDOWS}
    with open("backend/model/training_stats.json","w") as f: json.dump(st,f,indent=2)
    print("Charts & stats saved.\n")
    return model, st

if __name__ == "__main__":
    train()
