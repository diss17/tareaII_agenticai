"""Nodo calculador: resuelve operaciones matemáticas."""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state import AgentState
from models import get_llm, get_structured_llm, CalculatorOutput
from tools import safe_eval
from utils import is_tool_error, trace, trace_message, trace_tool


SYSTEM_PROMPT = """Eres un agente calculador experto en matemáticas.

Debes resolver operaciones matemáticas usando ÚNICAMENTE la herramienta `safe_eval`.
Responde con el nombre de la herramienta y los argumentos exactos.

Reglas:
- La expresión debe usar funciones matemáticas compatibles: sqrt, sin, cos, tan, log, log10, exp, pi, e, pow, abs, round.
- Ejemplo: para "raíz cuadrada de 9 más 5", usa "sqrt(9) + 5".
- No añadas explicaciones fuera de la estructura.
"""

FINAL_PROMPT = """Eres un agente calculador. Dada la siguiente operación y su resultado,
responde al usuario de forma clara y breve en español.

Operación: {expression}
Resultado: {result}
"""


def calculator_node(state: AgentState) -> dict:
    """Ejecuta el agente calculador y guarda el resultado.

    Usa salida estructurada para forzar al modelo local a emitir únicamente
    la llamada a herramienta esperada, evitando parámetros inventados.
    """
    trace("calculador", "Iniciando resolución matemática")
    task = state.get("task_description", state["user_input"])
    trace("calculador", f"Tarea recibida: {task}")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=task),
    ]

    trace_message("calculador", "modelo LLM", "Solicitando selección de herramienta")

    try:
        parsed = get_structured_llm(CalculatorOutput).invoke(messages)
    except Exception as exc:
        error_msg = f"Error: el calculador no pudo generar una llamada válida: {exc}"
        trace("calculador", error_msg)
        return {
            "agent_result": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "last_error": error_msg,
        }

    trace_message("modelo LLM", "calculador", str(parsed.model_dump_json()))
    trace("calculador", f"Herramienta seleccionada: {parsed.tool}")

    if parsed.tool != "safe_eval":
        error_msg = "Error: el calculador no seleccionó la herramienta correcta."
        trace("calculador", error_msg)
        return {
            "agent_result": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "last_error": error_msg,
        }

    expression = parsed.arguments.get("expression", "")
    trace("calculador", f"Expresión a evaluar: {expression}")
    raw_result = safe_eval.invoke({"expression": expression})
    trace_tool("calculador", "safe_eval", parsed.arguments, raw_result)

    if is_tool_error(raw_result):
        error_msg = (
            "No pude resolver la operación matemática. "
            f"Error técnico: {raw_result}"
        )
        trace("calculador", f"La herramienta safe_eval falló: {raw_result}")
        return {
            "agent_result": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "last_error": raw_result,
        }

    final_messages = [
        SystemMessage(content=FINAL_PROMPT.format(expression=expression, result=raw_result)),
        HumanMessage(content="Genera la respuesta final."),
    ]
    trace_message("calculador", "modelo LLM", "Solicitando respuesta final al usuario")
    final_response = get_llm().invoke(final_messages)
    trace_message("modelo LLM", "calculador", str(final_response.content))

    trace("calculador", f"Resultado final generado: {final_response.content}")

    return {
        "agent_result": str(final_response.content),
        "messages": [AIMessage(content=str(parsed.model_dump_json())), final_response],
        "last_error": "",
    }
