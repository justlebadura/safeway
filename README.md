# SafeWay Backend

Este proyecto es el núcleo analítico de **SafeWay**, un sistema diseñado para la evaluación, categorización y visualización de riesgos urbanos basándose en infraestructura, condiciones climáticas e historial de accidentes.

## Estructura del Proyecto

```text
/
├── backend/            # Lógica central del sistema
│   ├── api.py          # Punto de entrada de la API REST (FastAPI)
│   ├── external/       # Clientes para fuentes de datos externas (SODA)
│   ├── microservices/  # Servicios dedicados: limpieza, grafos, reportes
│   └── model/          # Modelos de ML (LNN, GNN, RILL)
├── data/               # Conjunto de datos crudos (JSON/CSV)
├── frontend/           # Interfaz de usuario (página estática)
└── tests/              # Suite de pruebas unitarias
```

## Configuración del Entorno de Desarrollo

Para comenzar a desarrollar, configura un entorno virtual de Python:

1.  **Crear entorno virtual:**
    ```bash
    python3 -m venv .venv
    ```
2.  **Activar entorno:**
    ```bash
    source .venv/bin/activate
    ```
3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## Iniciando el Sistema

El backend utiliza **FastAPI**. Para ejecutar el servidor en modo desarrollo:

```bash
# Desde la raíz del proyecto
uvicorn backend.api:app --reload
```

Esto iniciará el servidor (usualmente en `http://127.0.0.1:8000`).

## API y Funcionalidades

El archivo `backend/api.py` contiene los endpoints principales:

- `GET /datasets/combined`: Obtiene una instantánea de los datasets combinados.
- `GET /datasets/export`: **Exportación de datos**. Permite obtener registros de accidentes filtrados por año, clima, tipo de vehículo o ciudad. Puede exportar en formato `json` (por defecto) o `csv` (usando `export_format=csv`).
- `PUT /datasets/{dataset_id}/nodes/{row_id}`: Permite actualizar registros específicos de un dataset.
- `GET /`: Sirve la interfaz estática (`frontend/index.html`).

## Desarrollo y ML

*   **Modelos de ML:** Los modelos (`HybridGNNLNN` con `RILL`) se encuentran en `backend/model/`. Para verificar el rendimiento antes de subir cambios, ejecuta la suite de experimentos:
    ```bash
    python3 -m backend.model.experiment_runner
    ```
*   **Tests:** Ejecuta la suite de pruebas regularmente:
    ```bash
    pytest tests/
    ```
