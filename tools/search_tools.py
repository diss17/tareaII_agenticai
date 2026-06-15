"""Herramientas de búsqueda web para el agente experto."""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


@tool
def web_search(query: str) -> str:
    """Realiza una búsqueda web usando DuckDuckGo.

    Args:
        query: Consulta de búsqueda.

    Returns:
        Resultados de la búsqueda como texto.
    """
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as exc:
        return f"Error en la búsqueda web: {exc}"
