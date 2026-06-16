"""Utilidades de logging/traza para el sistema multiagente.

Este módulo expone funciones para mostrar en pantalla el flujo interno del
sistema multiagente: qué nodo se ejecuta, qué mensajes intercambian los
agentes, qué herramientas se invocan, qué decide el crítico, etc.

La traza se imprime SIEMPRE por consola (independientemente de la variable
VERBOSE), porque el enunciado exige que el usuario pueda observar los pasos
intermedios de ejecución.
"""

from config import VERBOSE


def log(message: str) -> None:
    """Imprime un mensaje de log si el modo verbose está activado.

    Úsalo para detalles de depuración que no forman parte de la traza
    obligatoria del sistema.
    """
    if VERBOSE:
        print(f"  [debug] {message}")


def trace(node: str, message: str) -> None:
    """Imprime una línea de traza de ejecución visible siempre.

    Args:
        node: Nombre del nodo/agente que genera el mensaje.
        message: Descripción del paso intermedio.
    """
    print(f"  [TRAZA] [{node}] {message}")


def trace_header(title: str) -> None:
    """Imprime un encabezado de sección en la traza."""
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def trace_message(sender: str, recipient: str, content: str) -> None:
    """Registra el envío de un mensaje entre dos agentes/nodos."""
    preview = content.replace("\n", " ")[:120]
    if len(content) > 120:
        preview += "..."
    print(f"  [TRAZA] [mensaje] {sender} -> {recipient}: {preview}")


def trace_tool(node: str, tool_name: str, arguments: object, result: object = None) -> None:
    """Registra la invocación (y opcionalmente el resultado) de una herramienta."""
    print(f"  [TRAZA] [{node}] Invocando herramienta: {tool_name}")
    print(f"  [TRAZA] [{node}]   argumentos: {arguments}")
    if result is not None:
        preview = str(result).replace("\n", " ")[:200]
        if len(str(result)) > 200:
            preview += "..."
        print(f"  [TRAZA] [{node}]   resultado: {preview}")


def trace_critic(decision: str, feedback: str) -> None:
    """Registra la evaluación del crítico."""
    print(f"  [TRAZA] [crítico] Decisión: {decision}")
    if feedback:
        preview = feedback.replace("\n", " ")[:200]
        if len(feedback) > 200:
            preview += "..."
        print(f"  [TRAZA] [crítico] Retroalimentación: {preview}")


def trace_state(state: dict) -> None:
    """Imprime un resumen del estado relevante para el usuario."""
    print("  [TRAZA] [estado]")
    print(f"    - agente asignado: {state.get('assigned_agent', 'ninguno')}")
    print(f"    - iteración: {state.get('iteration_count', 0)}")
    print(f"    - decisión del crítico: {state.get('critic_decision', 'pendiente')}")
    result = state.get('agent_result', '')
    if result:
        preview = str(result).replace("\n", " ")[:120]
        if len(str(result)) > 120:
            preview += "..."
        print(f"    - resultado del agente: {preview}")
