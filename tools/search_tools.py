"""Herramientas de búsqueda web para el agente experto."""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from utils import make_tool_error


# Instancia reutilizable para evitar recrear el objeto en cada llamada
_duckduckgo_search: DuckDuckGoSearchRun | None = None


def _get_search() -> DuckDuckGoSearchRun:
    """Devuelve (o crea) la instancia de búsqueda de DuckDuckGo."""
    global _duckduckgo_search
    if _duckduckgo_search is None:
        _duckduckgo_search = DuckDuckGoSearchRun()
    return _duckduckgo_search


@tool
def web_search(query: str) -> str:
    """Realiza una búsqueda web usando DuckDuckGo.

    Args:
        query: Consulta de búsqueda.

    Returns:
        Resultados de la búsqueda como texto, o un mensaje [TOOL_ERROR] si falla.
    """
    try:
        return _get_search().run(query)
    except Exception as exc:
        return make_tool_error(f"La búsqueda web falló: {exc}")
