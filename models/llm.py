"""Configuración del modelo LLM local via Ollama."""

from langchain_ollama import ChatOllama
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE


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
        num_predict=512,
        **kwargs,
    )
