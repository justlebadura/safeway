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

## Estructura del proyecto

```text
.
├── data/
│   └── fewshot_50_norte_santander.json
├── src/
│   └── safeway/
│       ├── cleaning.py
│       ├── extractor.py
│       ├── pipeline.py
│       └── soda_client.py
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
	- `lugar_original`
	- `lugar_limpio`
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
	- `$select=comparendo,lugar`
	- `$where=lugar is not null`
	- paginacion con `$limit` y `$offset`
- Formula simple de confianza (heuristica):
	- coincidencia fuerte regex + entidad spaCy: alta ($\sim 0.90$)
	- coincidencia regex media: media-alta ($\sim 0.74-0.88$)
	- no encontrado: $0.0$

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