"""Utilidades para detectar y construir mensajes de error de herramientas.

Centralizar este mecanismo permite que todos los agentes identifiquen de la
misma forma cuando una herramienta falló técnicamente, evitando que generen
respuestas inventadas a partir de un resultado de búsqueda o cálculo fallido.
"""

TOOL_ERROR_PREFIX = "[TOOL_ERROR]"


def is_tool_error(result: object) -> bool:
    """Devuelve True si el resultado de una herramienta indica un fallo técnico."""
    return isinstance(result, str) and result.startswith(TOOL_ERROR_PREFIX)


def make_tool_error(message: str) -> str:
    """Construye un mensaje de error de herramienta estandarizado."""
    return f"{TOOL_ERROR_PREFIX} {message}"


def get_tool_error_detail(result: object) -> str:
    """Extrae el detalle del mensaje de error sin el prefijo.

    Si el resultado no es un error de herramienta, lo convierte a string.
    """
    if is_tool_error(result):
        return str(result)[len(TOOL_ERROR_PREFIX) :].strip()
    return str(result)
