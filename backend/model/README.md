# SafeWay — Documentación Técnica Completa

**Sistema Neuro-Simbólico de Predicción y Evasión de Riesgo Vial**
*CRISP-ML · Bucaramanga, Colombia · 2025*

---

## Metodología CRISP-ML

| Fase | Actividad | Evidencia |
|------|-----------|-----------|
| 1. **Business Understanding** | Navegación segura: evitar intersecciones con alto riesgo de accidentes | `README.md`, endpoints `/route` |
| 2. **Data Understanding** | 39,193 registros 2012-2023, 7 datasets SODA, 186 barrios Bucaramanga | `api_soda_cleaner.py`, `mapper.py` |
| 3. **Data Preparation** | Limpieza de texto, geocodificación, grid 26×60=1,560 intersecciones | `api_soda_cleaner.py`, `mapper.py`, `api.py:_build_nodes()` |
| 4. **Modeling** | GNN(2-capas)+LNN(CfC)+RILL, 200 épocas, timesteps reales 2014-2022 | `arch/hybrid_model.py`, `train_model_offline.py` |
| 5. **Evaluation** | Precision 93.75%, Recall 100%, F1 96.77%, R²=0.936 | `train_model_offline.py:compute_metrics()` |
| 6. **Deployment** | FastAPI + Vercel, fórmula simbólica 34 FLOPs en producción | `api/index.py`, `vercel.json`, `_symbolic_risk_production()` |

---

## Flujo de Datos

```
                    ┌──────────────────────┐
FASE 1: INGESTA     │  SODA API (gov.co)   │  soda_client.py
                    └──────────┬───────────┘
                               │ HTTP paginado
                               ▼
                    ┌──────────────────────┐
                    │  data/raw_*.json     │  caché en disco
                    └──────────┬───────────┘
                               │ api_soda_cleaner.py
FASE 2: LIMPIEZA    │ normalize_text(), clean_soda_row()
                    │ normalize_row_features() → date_iso, time, vehicles
                               │
                    ┌──────────▼───────────┐
FASE 3: GEO         │  mapper.py           │
                    │  AddressExtractor    │ → VIA_PRINCIPAL, BARRIO
                    │  resolve_coordinates │ → (lat, lng) en grid BGA
                    └──────────┬───────────┘
                               │ api.py:_build_nodes()
FASE 4: MODELADO    │ Grid 26×60 = 1,560 intersecciones
                    │ + features topológicas (grado, dist_centro)
                    │ + timesteps reales (2014→2022)
                               │
                    ┌──────────▼───────────┐
                    │  GNN(2-capas)+LNN    │  train_model_offline.py
                    │  + RILL (λ=0.1)      │
                    │  + BCE (51:1 class)  │
                    │  Target: accidente 2023?
                    └──────────┬───────────┘
                               │ Ridge regression
                    ┌──────────▼───────────┐
FASE 5: DISTILL     │  symbolic_formula.txt│  R²=0.936
                    │  risk = 0.000241 +   │
                    │  3.988×sev +         │
                    │  0.350×sev_neighbor  │
                    └──────────┬───────────┘
                               │ api.py:_symbolic_risk_production()
FASE 6: PRODUCCIÓN  │  34 FLOPs por nodo   │  vs 392M FLOPs (NN)
                    │  Percentiles [0,10]  │
                    │  Dijkstra bidirecc.  │  routing.py
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
FASE 7: FRONTEND   │  Leaflet + OSRM      │  index.html
                    │  Selector año/hora   │
                    │  Toggle grafo vial   │
                    └──────────────────────┘
```

---

## Estructura del Proyecto

```
safeway/
├── api/index.py                    Vercel entry point (8 L)
├── vercel.json                     Python runtime, 1024MB, 60s
├── requirements.txt                fastapi, requests, numpy, networkx
├── README.md
│
├── backend/
│   ├── api.py                      ★ 14 endpoints REST (604 L)
│   ├── external/
│   │   ├── soda_client.py          HTTP paginado para datos.gov.co (58 L)
│   │   └── pipeline.py             ETL: fetch → clean → extract (90 L)
│   ├── microservices/
│   │   ├── api_soda_cleaner.py     Limpieza, caché, 7 datasets (689 L)
│   │   ├── mapper.py               Geocodificación: texto → lat/lng (135 L)
│   │   ├── routing.py              GraphNode + RouteOptimizer + Dijkstra (162 L)
│   │   ├── grapher.py              Snap de accidentes a intersecciones (49 L)
│   │   └── reporter.py             Filtros + gráficos matplotlib (193 L)
│   └── model/
│       ├── train_model_offline.py  ★ Entrenamiento v6: temporal + destilación (384 L)
│       ├── model.pth               Pesos GNN(2-capas)+LNN (101 KB)
│       ├── symbolic_formula.txt    Fórmula Ridge (R²=0.936, 34 FLOPs)
│       ├── INFORME.md              Este documento
│       ├── arch/
│       │   ├── hybrid_model.py     GNN(2-GCN) + LNN(CfC) (61 L)
│       │   └── lnn_core.py         Celda CfC-LNN (38 L)
│       └── loss/
│           └── rill_loss.py        HybridLoss: MSE + λ·Σ(ŷᵤ−ŷᵥ)² (39 L)
│
├── frontend/
│   └── index.html                  SPA: Leaflet + panel de ruteo (1235 L)
│
├── tests/
│   ├── test_api.py                 Integración FastAPI + mock SODA
│   ├── test_cleaning.py            Unit: normalización de texto
│   ├── test_extractor.py           Unit: AddressExtractor
│   └── test_pipeline.py            Unit: pipeline ETL
│
└── data/                           Caché local raw_*.json (SODA)
```

---

## Arquitectura del Modelo

### GNN — 2 capas de Convolución sobre Grafos

```
Input (7D) → GCN₁(7→32) → ReLU → GCN₂(32→32) → ReLU → LNN
  [0] rain      │                │
  [1] clear     │  Agrega        │  Agrega vecinos
  [2] hour/24   │  vecinos       │  de vecinos
  [3] sev/4     │  directos      │  → corredores
  [4] acc/50    │                │     de riesgo
  [5] grado/4   │                │
  [6] dist_cent │                │
```

$$H^{(1)} = \text{ReLU}(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} X W_1 + b_1)$$
$$H^{(2)} = \text{ReLU}(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(1)} W_2 + b_2)$$

### LNN — Liquid Neural Network (CfC)

5 timesteps reales: **2014, 2016, 2018, 2020, 2022**

$$h_{t+1} = \sigma(o_t) \odot \tanh(f_t \odot h_t + \sigma(i_t) \odot \tanh(c_t))$$

Donde $i_t, f_t, c_t, o_t$ son compuertas aprendidas que modulan el estado oculto según la entrada espacial $H^{(2)}_t$.

### Pérdida

$$\mathcal{L} = \underbrace{\text{BCE}(y, \hat{y}; w=51)}_{\text{binaria balanceada}} + \underbrace{0.1 \sum_{(u,v)\in E} (\hat{y}_u - \hat{y}_v)^2}_{\text{RILL: coherencia espacial}}$$

---

## Resultados — Predicción de Accidentes 2023

| Métrica | Valor |
|---------|-------|
| **Precision** | **93.75%** |
| **Recall** | **100.00%** |
| **F1** | **96.77%** |
| **Accuracy** | 99.87% |
| **R² Simbólico** | **0.936** |
| True Positives | 480 |
| False Positives | 32 |
| False Negatives | **0** |
| True Negatives | 24,448 |

### Comparativa NN vs Simbólica

| | Red Neuronal | Fórmula Simbólica |
|---|---|---|
| **FLOPs** | 392,952,144 | **34** |
| **Ratio** | — | **11,557,416× más rápida** |
| **Parámetros** | 26,209 | 10 (float) |
| **Memoria** | 101 KB + PyTorch | ~200 B |

### Fórmula en Producción

```
risk = 0.000241
     − 0.000074 × rain
     + 0.000105 × hour/24
     + 3.987597 × severity          ← DOMINANTE
     + 0.000017 × accidents
     + 0.350155 × severity_neighbor  ← 2° más importante
     − 0.000487 × accidents_neighbor
     + 0.000002 × sin(2π·h)
     − 0.000003 × cos(2π·h)
```

**Interpretación:** La severidad (1×/2×/4×) es el predictor dominante. Un accidente con muertos predice más riesgo futuro que 100 accidentes leves.

---

## API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Frontend SPA |
| GET | `/health` | `{"status":"ok"}` |
| GET | `/datasets/{id}` | Dataset individual (caché) |
| GET | `/datasets/combined` | Múltiples datasets combinados |
| GET | `/datasets/combined/graph` | Grafo de riesgo: 1,560 nodos + aristas |
| GET | `/datasets/combined/route` | Ruta más segura (Dijkstra + OSRM) |
| GET | `/datasets/export` | Exportar registros filtrados (JSON/CSV) |
| GET | `/datasets/chart.png` | Gráfico PNG de estadísticas |
| GET | `/datasets/{id}/updates` | Long-poll para datos nuevos |
| PUT | `/datasets/{id}/nodes/{row_id}` | Editar nodo manualmente |

### Parámetros de `/datasets/combined/graph`

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `dataset_ids` | `7cci-nqqb` | IDs separados por coma |
| `max_rows` | 50000 | Máximo de registros |
| `target_year` | 2026 | Año para decaimiento temporal |
| `rain_active` | false | Condición de lluvia |
| `target_hour` | 12 | Hora del día (0-23) |

### Parámetros de `/datasets/combined/route`

Además de los anteriores: `start_lat`, `start_lng`, `end_lat`, `end_lng` (obligatorios).

---

## Glosario

| Término | Definición |
|---------|------------|
| **GNN** | Graph Neural Network — red neuronal que opera sobre grafos |
| **GCN** | Graph Convolutional Network — convolución espectral sobre vecindario |
| **LNN** | Liquid Neural Network — red continua basada en ODEs |
| **CfC** | Closed-form Continuous-time — aproximación cerrada de la ODE |
| **LTN** | Logic Tensor Networks — lógica difusa sobre tensores |
| **RILL** | Reduced Implication-bias Logic Loss — pérdida lógica sin sesgo |
| **Dijkstra** | Camino más corto con pesos no negativos |
| **OSRM** | Open Source Routing Machine — API de ruteo sobre calles reales |
| **SODA** | Socrata Open Data API — fuente de datos gubernamentales |
| **FLOPs** | Floating Point Operations — costo computacional |
| **Precision** | TP/(TP+FP) — pureza de predicciones positivas |
| **Recall** | TP/(TP+FN) — cobertura de accidentes reales |
| **F1** | Media armónica de Precision y Recall |
| **R²** | Coeficiente de determinación — calidad del ajuste |
| **BCE** | Binary Cross-Entropy — pérdida para clasificación binaria |
| **Percentil** | Rango relativo en distribución ordenada |
| **Grid Manhattan** | Malla vial ortogonal (calles perpendiculares) |
| **CRISP-ML** | Cross-Industry Standard Process for Machine Learning |
| **Serverless** | Arquitectura sin servidor persistente (Vercel) |
| **Cold start** | Latencia de primera invocación en serverless |

---

## Entrenar / Reentrenar

```bash
python3 -m backend.model.train_model_offline
```

Esto regenera `model.pth`, `symbolic_formula.txt`, y las métricas de evaluación.

## Iniciar Servidor

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

## Deploy a Vercel

```bash
vercel --prod
```
