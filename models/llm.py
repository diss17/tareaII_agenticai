"""Configuración del modelo LLM local via Ollama."""

from typing import TypeVar

from langchain_ollama import ChatOllama
from pydantic import BaseModel
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE


T = TypeVar("T", bound=BaseModel)


def get_llm(json_mode: bool = False) -> ChatOllama:
    """Devuelve una instancia de ChatOllama configurada.

    Args:
        json_mode: Si es True, fuerza la salida en formato JSON.
    """
    kwargs = {"format": "json"} if json_mode else {}
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
        num_predict=1024,
        **kwargs,
    )


def get_structured_llm(schema: type[T]) -> ChatOllama:
    """Devuelve un LLM con salida estructurada según el esquema Pydantic.

    Para modelos locales es más robusto que pedir JSON libre, porque fuerza
    la generación a respetar los nombres de campo y los tipos definidos,
    reduciendo alucinaciones de parámetros.
    """
    return get_llm(json_mode=True).with_structured_output(schema, include_raw=False)
