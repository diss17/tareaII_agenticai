# Sistema Multiagente con LangGraph + Ollama

Sistema multiagente construido con **LangGraph** y un modelo LLM local (`gemma3:4b`) servido mediante **Ollama**.

## Arquitectura

- **Orquestador**: analiza la solicitud del usuario y delega en el subagente adecuado.
- **Calculador**: resuelve operaciones matemáticas mediante la herramienta `safe_eval`.
- **Organizador**: gestiona un calendario local persistente en SQLite.
- **Experto**: responde consultas conceptuales usando búsqueda web con DuckDuckGo.
- **Crítico**: evalúa la respuesta final y decide si aprueba o solicita retroalimentación.

## Requisitos

- Python 3.10+
- Ollama instalado y corriendo
- Modelo `gemma3:4b` descargado en Ollama

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

1. Asegúrate de que Ollama esté ejecutándose:

```bash
ollama serve
```

2. Descarga el modelo local utilizado por el sistema:

```bash
ollama pull gemma3:4b
```

3. Ejecuta el sistema:

```bash
python main.py
```

## Ejemplos de consulta

- `"¿Cuánto es 15 * 23 + sqrt(9)?"` → delega al calculador
- `"Agrega una reunión mañana a las 10am llamada Planning"` → delega al organizador
- `"¿Qué es la computación cuántica?"` → delega al experto

## Estructura del proyecto

```
.
├── agents/           # Nodos del grafo
├── tools/            # Herramientas disponibles
├── state/            # Estado compartido
├── graph/            # Construcción del grafo
├── models/           # Configuración del LLM
├── utils/            # Utilidades (parser JSON)
├── config.py         # Variables de configuración
├── main.py           # CLI interactiva
└── requirements.txt
```

## Configuración

Puedes modificar `config.py` o usar variables de entorno:

- `OLLAMA_MODEL`: modelo a usar (default: `gemma3:4b`)
- `OLLAMA_BASE_URL`: URL del servidor Ollama (default: `http://localhost:11434`)
- `OLLAMA_TEMPERATURE`: temperatura del modelo (default: `0.0`)
- `MAX_CRITIC_ITERATIONS`: máximo de iteraciones de retroalimentación (default: `3`)
- `CALENDAR_DB_PATH`: ruta de la base de datos SQLite (default: `calendar.db`)
