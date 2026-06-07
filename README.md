# Safeway - Extraccion estructurada de direcciones imprecisas

Pipeline en Python para consumir datos abiertos por API SODA, limpiar texto de direcciones y extraer entidades espaciales con confianza para Norte de Santander.

## Objetivo

Implementar una cadena dinamica:

1. Descargar datos de SODA.
2. Limpiar campo textual de ubicacion.
3. Extraer direccion estructurada con 4 claves obligatorias.
4. Retornar metricas de confianza por entidad y ` ["UNKNOWN"] ` cuando no hay referencia espacial valida.

## Dataset

- Portal: https://www.datos.gov.co
- Dataset: `stq8-drvp`
- Recurso API: `https://www.datos.gov.co/resource/stq8-drvp.json`
- Campo de ubicacion usado: `lugar`

## Como funciona la API

Este proyecto usa la API SODA de datos abiertos en dos niveles:

1. API externa (SODA):
	- Se consulta `https://www.datos.gov.co/resource/stq8-drvp.json`.
	- Se usa paginacion con `$limit` y `$offset` para traer lotes de filas.
	- Opcionalmente se envia `X-App-Token` (variable `SODA_APP_TOKEN`) para mejorar limites.

2. Procesamiento interno (pipeline de este repo):
	- Cada fila recibida se limpia automaticamente en todos sus campos (`data_limpia`).
	- Si el campo tiene `fecha` en su nombre, se intenta normalizar a formato ISO.
	- El campo `lugar` se normaliza con reglas de direccion para extraccion.
	- Se genera `extraccion` con entidades estructuradas y confianza.
	- Si no hay senal espacial valida, retorna ` ["UNKNOWN"] `.

Flujo resumido:

`SODA -> data_original -> data_limpia -> extraccion`

## Uso de la API (paso a paso)

1. Configurar dependencias:

```bash
pip install -r requirements.txt
```

2. Opcional: definir token SODA:

```bash
export SODA_APP_TOKEN="tu_token"
```

3. Ejecutar pipeline:

```bash
PYTHONPATH=src python -m safeway.pipeline --dataset-id stq8-drvp --max-rows 200 --output outputs/extracciones_full.json
```

4. Revisar salida JSON en `outputs/extracciones_full.json`:
	- `data_original`: fila completa tal cual llega desde SODA.
	- `data_limpia`: fila completa limpiada automaticamente.
	- `extraccion`: direccion estructurada + `confidence` por entidad.

## API HTTP con cache y long polling

Ademas del pipeline por consola, el proyecto expone una API HTTP para clientes que necesiten leer datos cacheados y enterarse cuando el dataset cambie.

### Levantar el servidor

```bash
PYTHONPATH=src uvicorn safeway.api:app --host 0.0.0.0 --port 8000
```

### Endpoints principales

1. Salud del servicio:

```bash
curl http://localhost:8000/health
```

2. Snapshot del dataset con cache:

```bash
curl "http://localhost:8000/datasets/stq8-drvp?max_rows=200"
```

Este endpoint:
	- descarga desde SODA si no hay cache o el cache vencio
	- devuelve la ultima version cacheada si no hubo cambios
	- responde con `processed`, que contiene todas las filas limpias y su extraccion

Ejemplo de respuesta resumida:

```json
{
	"dataset_id": "stq8-drvp",
	"max_rows": 200,
	"version": "a1b2c3d4...",
	"fetched_at": 1717777777.0,
	"processed": [
		{
			"comparendo": "123456",
			"data_original": {"lugar": "CALLE 10 #5-20 CUCUTA"},
			"data_limpia": {"lugar": "CALLE 10 # 5-20 CUCUTA"},
			"extraccion": {
				"VIA_PRINCIPAL": {"value": "CALLE 10", "confidence": 0.84},
				"NUMERO_O_KM": {"value": "10 # 5-20", "confidence": 0.88},
				"REFERENCIA_SEMANTICA": {"value": null, "confidence": 0.0},
				"BARRIO_O_MUNICIPIO": {"value": "CUCUTA", "confidence": 0.95}
			}
		}
	],
	"cached": true
}
```

3. Forzar refresco del cache:

```bash
curl "http://localhost:8000/datasets/stq8-drvp?max_rows=200&force_refresh=true"
```

4. Long polling para detectar actualizaciones:

```bash
curl "http://localhost:8000/datasets/stq8-drvp/updates?max_rows=200&last_version=VERSION_ACTUAL&timeout_seconds=30"
```

Este endpoint:
	- espera hasta `timeout_seconds` por una nueva version del dataset
	- si detecta cambio, responde con `changed: true` y devuelve `data`
	- si no detecta cambio, responde con `changed: false` y `timed_out: true`

Ejemplo cuando hubo cambio:

```json
{
	"dataset_id": "stq8-drvp",
	"max_rows": 200,
	"version": "nueva-version-hash",
	"changed": true,
	"timed_out": false,
	"fetched_at": 1717777777.0,
	"data": []
}
```

Ejemplo cuando no hubo cambio:

```json
{
	"dataset_id": "stq8-drvp",
	"max_rows": 200,
	"version": "misma-version-hash",
	"changed": false,
	"timed_out": true,
	"fetched_at": 1717777777.0
}
```

### Como funciona el cache

- El cache vive en memoria dentro del proceso de la API.
- Cada dataset se guarda por combinacion de `dataset_id` y `max_rows`.
- La version del dataset se calcula con un hash del contenido procesado.
- Si el contenido cambia en SODA, cambia la version y el long polling notifica a los clientes en la siguiente consulta.
- Los clientes conectados deben volver a consultar el endpoint de updates al recibir una respuesta, que es el patron normal de long polling.

## Estructura del proyecto

```text
.
├── data/
│   └── fewshot_50_norte_santander.json
├── src/
│   └── safeway/
│       ├── api.py
│       ├── cleaning.py
│       ├── extractor.py
│       ├── pipeline.py
│       └── soda_client.py
├── tests/
│   ├── test_api.py
│   ├── test_cleaning.py
│   ├── test_extractor.py
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Opcional para mejorar limites de consulta:

```bash
export SODA_APP_TOKEN="tu_token"
```

## Ejecucion

```bash
PYTHONPATH=src python -m safeway.pipeline --dataset-id stq8-drvp --max-rows 200 --output outputs/extracciones.json
```

Salida:

- Archivo JSON con objetos por fila, incluyendo:
	- `comparendo`
	- `data_original` (fila completa recibida desde SODA)
	- `data_limpia` (fila completa con limpieza automatica de texto/fechas)
	- `extraccion`

## Pruebas automaticas

Para validar que la limpieza y la extraccion funcionan automaticamente en cada cambio:

```bash
PYTHONPATH=src pytest -q
```

Las pruebas cubren:

- Normalizacion de texto en limpieza.
- Estructura obligatoria y confianza en extraccion.
- Respuesta ` ["UNKNOWN"] ` cuando no hay referencia espacial valida.
- Flujo del pipeline sin depender de red (mock de cliente SODA).

## Formato de salida de extraccion

Cuando hay referencia espacial:

```json
{
	"VIA_PRINCIPAL": {"value": "CALLE 10", "confidence": 0.84},
	"NUMERO_O_KM": {"value": "10 # 5-20", "confidence": 0.88},
	"REFERENCIA_SEMANTICA": {"value": "FRENTE A TERMINAL", "confidence": 0.76},
	"BARRIO_O_MUNICIPIO": {"value": "CUCUTA", "confidence": 0.95}
}
```

Cuando no hay referencia espacial valida:

```json
["UNKNOWN"]
```

## Cobertura de criterios de aceptacion

- [x] Extraccion estructurada con claves obligatorias:
	- `VIA_PRINCIPAL`
	- `NUMERO_O_KM`
	- `REFERENCIA_SEMANTICA`
	- `BARRIO_O_MUNICIPIO`
- [x] Metricas de confianza por entidad (rango 0.0 a 1.0).
- [x] Manejo de alucinaciones con token ` ["UNKNOWN"] `.
- [x] Conexion dinamica descargar -> procesar -> usar datos (API SODA + pipeline).

## Criterios tecnicos aplicados

- Libreria NLP usada: `spaCy` (pipeline local `spacy.blank("es")` + `EntityRuler`).
- Reglas hibridas: regex + entidades reconocidas por `EntityRuler`.
- Few-shot / set critico: `data/fewshot_50_norte_santander.json` con 50 ejemplos.

## Notas tecnicas

- Endpoint de metadatos del dataset:
	- `https://www.datos.gov.co/api/views/stq8-drvp.json`
- Consulta principal usada en pipeline:
	- `$select=*`
	- sin `$where` para no perder columnas ni filas en origen
	- paginacion con `$limit` y `$offset`
- Formula simple de confianza (heuristica):
	- coincidencia fuerte regex + entidad spaCy: alta ($\sim 0.90$)
	- coincidencia regex media: media-alta ($\sim 0.74-0.88$)
	- no encontrado: $0.0$
- Si al ejecutar falla con error de DNS/NameResolutionError, el problema es de conectividad hacia `www.datos.gov.co`, no de la logica del pipeline.

## Avance #1

- Se creo un pipeline funcional end-to-end con API SODA.
- Se implemento limpieza y normalizacion del campo `lugar`.
- Se implemento extractor estructurado con las 4 claves obligatorias y score por entidad.
- Se incluyo manejo de ` ["UNKNOWN"] ` para evitar alucinaciones.
- Se agrego set de 50 ejemplos criticos de Norte de Santander.

Hechos relevantes:

- El dataset `stq8-drvp` contiene la ubicacion en la columna `lugar`.
- El enfoque actual es hibrido (NLP local + reglas), sin dependencia de un LLM remoto.

## Detalles de revision

- Proceso calificativo sugerido:
	1. Ejecutar pipeline sobre muestra ($n=200$ o mayor).
	2. Medir cobertura de estructura valida vs `UNKNOWN`.
	3. Muestrear manualmente errores por tipo de entidad.
	4. Ajustar reglas/patrones y recalibrar confianza.
- Responsable de calificacion:
	- Pendiente de asignar (QA/analista de datos del equipo).

## Entrega

Artefactos entregables:

1. Codigo fuente del pipeline (`src/safeway`).
2. Dependencias (`requirements.txt`).
3. Set critico de 50 ejemplos (`data/fewshot_50_norte_santander.json`).
4. Documento tecnico y trazabilidad de criterios (este `README.md`).