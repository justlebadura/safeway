# Modelo de Riesgo (SafeWay ML)

Este directorio contiene la arquitectura del modelo de aprendizaje automático utilizada para la evaluación de riesgos urbanos.

## Datos Utilizados
El sistema utiliza archivos JSON reales ubicados en `/data/`. El cargador (`dataset.py`) procesa estos archivos para extraer ubicaciones únicas, accidentes históricos y atributos climáticos/temporales, estructurándolos como grafos para su procesamiento.

## Implementación de Arquitecturas
La arquitectura principal es un modelo **Hybrid (GNN+LNN+LTN+RILL)**:
- **GNN:** Implementada en `hybrid_model.py` usando convoluciones de grafos simplificadas para propagar la influencia de accidentes entre calles vecinas.
- **LNN:** Implementada en `lnn_core.py` (capa `CfCCell`), que modela la dinámica temporal continua mediante una aproximación de Ecuaciones Diferenciales Ordinarias (CfC).
- **Lógica (LTN+RILL):** La pérdida `HybridLoss` (`rill_loss.py`) integra una penalización lógica que fuerza la coherencia espacial y física de las predicciones, mitigando el sesgo de implicación mediante técnicas de optimización iterativa.

## Estructura de Archivos
- `arch/`:
    - `lnn_core.py`: Núcleo de la LNN (dinámica continua).
    - `hybrid_model.py`: Modelo híbrido GNN+LNN principal.
- `loss/`:
    - `rill_loss.py`: Implementación de `HybridLoss` (RILL+MSE).
- `loader/`:
    - `dataset.py`: Cargador y procesador de datos reales.
- `runner/`:
    - `experiment_runner.py`: Script de entrenamiento, evaluación de KPIs y generación de gráficos (`efficiency.png`).

*Nota: Los resultados (`results.json`, `efficiency.png`) se generan localmente tras ejecutar el script del runner.*
