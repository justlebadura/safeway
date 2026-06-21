# Safeway - Extracción y Trazado de Rutas Seguras de Accidentabilidad

Safeway es una plataforma web y API HTTP modular (diseñada en Programación Orientada a Objetos) para consumir datos abiertos por API SODA/CSV, estructurar direcciones físicas con NLP, georreferenciar incidentes con coordenadas de fallback inteligentes, y trazar rutas de mínimo riesgo utilizando un motor de optimización basado en Dijkstra.

---

## 🏛️ Arquitectura del Proyecto

El backend se encuentra desacoplado bajo una arquitectura limpia de microservicios:

```
safeway/
├── backend/
│   ├── api.py                     # Controlador HTTP (Capa de entrega con FastAPI)
│   ├── external/
│   │   ├── pipeline.py            # Orquestador del flujo ETL por consola
│   │   └── soda_client.py         # Cliente HTTP genérico para APIs SODA
│   └── microservices/
│       ├── api_soda_cleaner.py    # CLEANER: Sanitización, normalización de datos y caché en disco/RAM
│       ├── grapher.py             # GRAPHER: Agrupación y snapping de accidentes en nodos viales
│       ├── mapper.py              # LOCATOR: Geolocalización, NLP NER para direcciones y fallbacks de coordenadas
│       └── routing.py             # ROUTER: Cálculo de peligro dinámico y optimizador de rutas (Dijkstra)
├── data/                          # Caché de datasets descargados y ediciones manuales
├── frontend/
│   └── index.html                 # Interfaz interactiva de mapas con Leaflet.js
├── tests/                         # Suite de pruebas automatizadas (pytest)
└── requirements.txt               # Dependencias del proyecto
```

---

## ⚙️ Descripción de Microservicios

### 1. 🧹 Cleaner (`api_soda_cleaner.py`)
*   **Función**: Normaliza formatos de fechas, horas y tipos de vehículos en los esquemas SODA.
*   **Caché inteligente**: Almacena en memoria y en disco local (`data/raw_*.json`) las respuestas remotas para optimizar tiempos de respuesta.
*   **Edición**: Almacena ediciones manuales (`data/edits_*.json`) y las fusiona de forma reactiva invalidando la caché.

### 2. 📍 Locator (`mapper.py`)
*   **Función**: Utiliza modelos de procesamiento de lenguaje natural local (`spaCy EntityRuler`) y expresiones regulares para estructurar direcciones físicas en 4 claves:
    1.  `VIA_PRINCIPAL`
    2.  `NUMERO_O_KM`
    3.  `REFERENCIA_SEMANTICA`
    4.  `BARRIO_O_MUNICIPIO`
*   **Manejo de Fallbacks**: Si un registro no posee georreferencia original válida, calcula una coordenada pseudo-aleatoria dispersa alrededor del centro del municipio correspondiente, marcándola con la propiedad `is_fallback_coord: true`.

### 3. 🗺️ Grapher (`grapher.py`)
*   **Función**: Toma los accidentes geolocalizados y los agrupa espacialmente utilizando un umbral de proximidad de ~40 metros (intersecciones estructurales).
*   **Control de Fallbacks**: Excluye explícitamente los accidentes cuya coordenada es estimada (`is_fallback_coord`) para evitar distorsiones viales en el mapa de grafos.

### 4. 🔀 Router & Optimizer (`routing.py`)
*   **Función**: Implementa un algoritmo de Dijkstra que calcula el trayecto más seguro entre dos nodos viales.
*   **Cálculo de Riesgo Dinámico**: La peligrosidad de un tramo se calcula en base a:
    *   **Decaimiento temporal**: Accidentes más recientes (ej. del 2026) tienen más peso.
    *   **Gravedad**: Accidentes con heridos o víctimas fatales aumentan drásticamente el riesgo.
    *   **Horarios y Clima**: Modificadores dinámicos si se indica que es de noche, hora pico, o si hay lluvia activa.

### 5. 📊 Reporter (`reporter.py`)
*   **Función**: Filtra datos espaciales e históricos por variables múltiples y genera gráficos estadísticos consolidados en formato de imagen de alta calidad.

---

## 🚀 Instalación y Configuración

1.  **Crear el entorno virtual e instalar dependencias**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Iniciar el Servidor de Desarrollo**:
    ```bash
    PYTHONPATH=backend .venv/bin/uvicorn api:app --host 0.0.0.0 --port 8080 --reload
    ```
    El servidor estará disponible en [http://localhost:8080/](http://localhost:8080/).

3.  **Ejecutar Pruebas Unitarias**:
    ```bash
    PYTHONPATH=backend .venv/bin/pytest
    ```

---

## 📈 Nuevas APIs de Reporte y Exportación

### 1. 📥 Exportar Datos Filtrados (`/datasets/export`)
Permite exportar los accidentes filtrados según múltiples condiciones en formato **JSON** o **CSV**.

*   **Ruta**: `GET /datasets/export`
*   **Parámetros de consulta (Query params)**:
    *   `dataset_ids`: IDs de datasets separados por comas (por defecto todos).
    *   `max_rows`: Límite de filas a recuperar.
    *   `start_year` / `end_year`: Rango de años a filtrar.
    *   `rain_only`: `true` filtra incidentes bajo lluvia, `false` clima seco.
    *   `vehicle_type`: Filtra por tipo de vehículo (ej: `moto`, `bus`).
    *   `city`: Filtra por municipio (ej: `cucuta`, `cali`, `bogota`).
    *   `export_format`: `json` (por defecto) o `csv` (descarga directa de archivo).
*   **Ejemplo**: `GET http://localhost:8080/datasets/export?export_format=csv&start_year=2024&rain_only=true`

### 2. 🖼️ Generar Gráfico Estadístico PNG (`/datasets/chart.png`)
Genera dinámicamente un gráfico consolidado de 4 cuadrantes (Accidentes por Año, Top 5 Ciudades, Top 5 Vehículos, Proporción de Lluvia) utilizando un diseño oscuro premium y lo sirve directamente como una imagen PNG.

*   **Ruta**: `GET /datasets/chart.png`
*   **Parámetros de consulta**: Admite los mismos filtros de filtrado temporal, climático, de ciudad y vehículos que el endpoint de exportación.
*   **Ejemplo**: Accede desde tu navegador o etiqueta HTML a `http://localhost:8080/datasets/chart.png?city=cucuta&start_year=2020`

---

## 🌐 Interfaz Gráfica (Frontend)

Ubicada en [frontend/index.html](file:///home/lebadura/Documentos/GitHub/safeway/frontend/index.html), ofrece:
*   **Mapa de Calor e Incidentes**: Puntos rojos indican zonas de alta densidad de accidentalidad.
*   **Visualización de Fallbacks**: Los marcadores que cayeron en coordenadas estimadas muestran una advertencia amarilla `⚠️ Fallback` tanto en el mapa como en la barra lateral.
*   **Modo de Grafos**: Dibuja la red estructurada de calles y el índice de riesgo vicular de cada esquina.
*   **Buscador de Rutas Seguras**: Introduce el nodo de origen y destino para que el backend trace la ruta minimizando riesgos.
*   **Editor en Vivo**: Haz click en cualquier marcador para corregir manualmente su dirección o moverlo arrastrándolo por el mapa.
