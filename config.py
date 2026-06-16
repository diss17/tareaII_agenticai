"""Configuración global del sistema multiagente."""

import os

# Configuración del modelo LLM local via Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

# Límite de iteraciones de retroalimentación del crítico
MAX_CRITIC_ITERATIONS = int(os.getenv("MAX_CRITIC_ITERATIONS", "3"))

# Base de datos SQLite para el calendario
CALENDAR_DB_PATH = os.getenv("CALENDAR_DB_PATH", "calendar.db")

# Modo verbose: muestra el flujo interno entre agentes
VERBOSE = os.getenv("VERBOSE", "true").lower() in {"true", "1", "yes"}
