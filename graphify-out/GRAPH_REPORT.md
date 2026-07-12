# Graph Report - safeway  (2026-07-12)

## Corpus Check
- 39 files · ~247,362 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 172 nodes · 305 edges · 19 communities (16 shown, 3 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.67)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9b44c52d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 18|Community 18]]

## God Nodes (most connected - your core abstractions)
1. `GraphNode` - 13 edges
2. `AddressExtractor` - 11 edges
3. `HybridGNNLNN` - 11 edges
4. `clean_location_text()` - 10 edges
5. `DatasetCacheService` - 10 edges
6. `get_combined_datasets_snapshot()` - 10 edges
7. `MapGrapher` - 10 edges
8. `HybridLoss` - 10 edges
9. `SodaClient` - 9 edges
10. `RouteOptimizer` - 9 edges

## Surprising Connections (you probably didn't know these)
- `test_clean_location_text_normalizes_case_accents_and_spaces()` --calls--> `clean_location_text()`  [EXTRACTED]
  tests/test_cleaning.py → backend/microservices/api_soda_cleaner.py
- `test_clean_location_text_replaces_no_token_with_hash()` --calls--> `clean_location_text()`  [EXTRACTED]
  tests/test_cleaning.py → backend/microservices/api_soda_cleaner.py
- `test_extract_detects_la_playa_alias()` --calls--> `clean_location_text()`  [EXTRACTED]
  tests/test_extractor.py → backend/microservices/api_soda_cleaner.py
- `test_extract_detects_ocana_without_accent()` --calls--> `clean_location_text()`  [EXTRACTED]
  tests/test_extractor.py → backend/microservices/api_soda_cleaner.py
- `test_extract_returns_required_keys_with_confidence()` --calls--> `clean_location_text()`  [EXTRACTED]
  tests/test_extractor.py → backend/microservices/api_soda_cleaner.py

## Import Cycles
- None detected.

## Communities (19 total, 3 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (20): HybridGNNLNN, Processes temporal sequence through GNN and LNN layers.                  Args:, Hybrid architecture combining GNN for spatial propagation and      LNN for conti, CfCCell, Computes the next hidden state based on continuous-time dynamics., Closed-form Continuous-time (CfC) cell approximation for Liquid Neural Networks., get_combined_datasets(), Carga datos procesados usando el servicio central de limpieza. (+12 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (22): clean_location_text(), clean_soda_value(), _load_edits(), _normalize_date(), normalize_row_features(), normalize_text(), _normalize_whitespace(), Helper method to parse raw dates, times, and vehicles from SODA schemas. (+14 more)

### Community 2 - "Community 2"
Cohesion: 0.21
Nodes (6): Build a directed NetworkX DiGraph with safety-weighted edges.          bbox = (l, Calculates a dynamic risk score for this intersection based on temporal decay,, Compute a generous bounding box around origin and destination., Runs Bidirectional Dijkstra with a dynamic Highway Hierarchy penalty system., Uses NetworkX A* with a Euclidean heuristic and safety-weighted edges to compute, RouteOptimizer

### Community 3 - "Community 3"
Cohesion: 0.21
Nodes (13): Any, get_dataset(), poll_dataset_updates(), main(), parse_args(), process_rows(), run_pipeline(), clean_soda_row() (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.12
Nodes (15): add_community_report(), export_dataset_records(), get_chart_image(), get_graph_data(), get_routing_graph_nodes(), get_safest_route(), update_node(), MapGrapher (+7 more)

### Community 5 - "Community 5"
Cohesion: 0.29
Nodes (3): SodaClient, CacheEntry, DatasetCacheService

### Community 6 - "Community 6"
Cohesion: 0.33
Nodes (6): build_service(), SequencedSodaClient, test_combined_route_with_use_symbolic(), test_get_dataset_uses_cache_and_returns_processed_rows(), test_long_poll_returns_updated_data_when_version_changes(), test_long_poll_times_out_without_change()

### Community 7 - "Community 7"
Cohesion: 0.29
Nodes (6): API y Funcionalidades, Configuración del Entorno de Desarrollo, Desarrollo y ML, Estructura del Proyecto, Iniciando el Sistema, SafeWay Backend

### Community 8 - "Community 8"
Cohesion: 0.40
Nodes (4): Datos Utilizados, Estructura de Archivos, Implementación de Arquitecturas, Modelo de Riesgo (SafeWay ML)

## Knowledge Gaps
- **10 isolated node(s):** `safeway`, `rewrites`, `Estructura del Proyecto`, `Configuración del Entorno de Desarrollo`, `Iniciando el Sistema` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_safest_route()` connect `Community 4` to `Community 0`, `Community 2`, `Community 3`?**
  _High betweenness centrality (0.171) - this node is a cross-community bridge._
- **Why does `get_combined_datasets_snapshot()` connect `Community 0` to `Community 1`, `Community 3`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `HybridGNNLNN` connect `Community 0` to `Community 4`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `HybridGNNLNN` (e.g. with `CfCCell` and `get_safest_route()`) actually correct?**
  _`HybridGNNLNN` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Normaliza texto para facilitar reglas de extracción.`, `Helper method to parse raw dates, times, and vehicles from SODA schemas.`, `Microservice responsible for organizing and snapping geocoded accidents onto str` to the rest of the system?**
  _31 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07957957957957958 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.13105413105413105 - nodes in this community are weakly interconnected._