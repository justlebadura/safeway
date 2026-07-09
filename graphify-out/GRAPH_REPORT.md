# Graph Report - safeway  (2026-07-09)

## Corpus Check
- 36 files · ~215,226 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 162 nodes · 275 edges · 18 communities (16 shown, 2 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.67)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0828e0a9`
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

## God Nodes (most connected - your core abstractions)
1. `AddressExtractor` - 11 edges
2. `clean_location_text()` - 10 edges
3. `DatasetCacheService` - 10 edges
4. `SodaClient` - 9 edges
5. `GraphNode` - 9 edges
6. `HybridGNNLNN` - 9 edges
7. `get_combined_datasets_snapshot()` - 8 edges
8. `HybridLoss` - 8 edges
9. `process_rows()` - 7 edges
10. `run_pipeline()` - 7 edges

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

## Communities (18 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (18): HybridGNNLNN, Processes temporal sequence through GNN and LNN layers.                  Args:, Hybrid architecture combining GNN for spatial propagation and      LNN for conti, CfCCell, Computes the next hidden state based on continuous-time dynamics., Closed-form Continuous-time (CfC) cell approximation for Liquid Neural Networks., get_safest_route(), Carga datos procesados usando el servicio central de limpieza. (+10 more)

### Community 1 - "Community 1"
Cohesion: 0.15
Nodes (19): clean_location_text(), clean_soda_value(), _normalize_date(), normalize_row_features(), normalize_text(), _normalize_whitespace(), Helper method to parse raw dates, times, and vehicles from SODA schemas., Normaliza texto para facilitar reglas de extracción. (+11 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (12): MapGrapher, Snaps geocoded accidents into structural intersection nodes and returns them., Microservice responsible for organizing and snapping geocoded accidents onto str, GraphNode, Builds the dual line graph where original connections are nodes, and turn maneuv, Runs Dijkstra on the Edge-Based Graph (Street Segments) to find safest paths., Calculates a dynamic risk score for this intersection based on temporal decay,, Represents a structural street intersection node. (+4 more)

### Community 3 - "Community 3"
Cohesion: 0.23
Nodes (13): Any, update_node(), main(), parse_args(), process_rows(), run_pipeline(), clean_soda_row(), _load_edits() (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.21
Nodes (10): export_dataset_records(), get_chart_image(), get_combined_datasets(), get_dataset(), poll_dataset_updates(), get_combined_datasets_snapshot(), serialize_entry(), generate_report_chart() (+2 more)

### Community 5 - "Community 5"
Cohesion: 0.29
Nodes (3): SodaClient, CacheEntry, DatasetCacheService

### Community 6 - "Community 6"
Cohesion: 0.36
Nodes (5): build_service(), SequencedSodaClient, test_get_dataset_uses_cache_and_returns_processed_rows(), test_long_poll_returns_updated_data_when_version_changes(), test_long_poll_times_out_without_change()

### Community 7 - "Community 7"
Cohesion: 0.29
Nodes (6): API y Funcionalidades, Configuración del Entorno de Desarrollo, Desarrollo y ML, Estructura del Proyecto, Iniciando el Sistema, SafeWay Backend

### Community 8 - "Community 8"
Cohesion: 0.40
Nodes (4): Datos Utilizados, Estructura de Archivos, Implementación de Arquitecturas, Modelo de Riesgo (SafeWay ML)

## Knowledge Gaps
- **8 isolated node(s):** `Estructura del Proyecto`, `Configuración del Entorno de Desarrollo`, `Iniciando el Sistema`, `API y Funcionalidades`, `Desarrollo y ML` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_safest_route()` connect `Community 0` to `Community 2`, `Community 3`, `Community 4`?**
  _High betweenness centrality (0.247) - this node is a cross-community bridge._
- **Why does `DatasetCacheService` connect `Community 5` to `Community 1`, `Community 6`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `SodaClient` (e.g. with `CacheEntry` and `DatasetCacheService`) actually correct?**
  _`SodaClient` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Normaliza texto para facilitar reglas de extracción.`, `Helper method to parse raw dates, times, and vehicles from SODA schemas.`, `Microservice responsible for organizing and snapping geocoded accidents onto str` to the rest of the system?**
  _29 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.0784313725490196 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.14855072463768115 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.11462450592885376 - nodes in this community are weakly interconnected._