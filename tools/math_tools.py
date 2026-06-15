"""Herramientas matemáticas seguras para el agente calculador."""

import math
from langchain_core.tools import tool


@tool
def safe_eval(expression: str) -> str:
    """Evalúa una expresión matemática de forma segura.

    Soporta operaciones aritméticas básicas, potencias, raíces,
    logaritmos y funciones trigonométricas.

    Args:
        expression: Expresión matemática en formato string.
            Ejemplo: "15 * 23 + sqrt(9)" o "sin(pi/2) + log(10, 10)"

    Returns:
        El resultado de la evaluación como string.
    """
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "pow": pow,
        "abs": abs,
        "round": round,
    }

    # Reemplaza potencias tipo 2^3 por 2**3 para eval
    expression = expression.replace("^", "**")

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as exc:
        return f"Error al evaluar '{expression}': {exc}"
