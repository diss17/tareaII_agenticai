"""Nodo calculador: resuelve operaciones matemáticas."""

from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from models import get_llm
from tools import safe_eval
from utils import parse_json_output


SYSTEM_PROMPT = """Eres un agente calculador experto en matemáticas.

Debes resolver operaciones matemáticas usando ÚNICAMENTE la herramienta `safe_eval`.
Responde con un objeto JSON válido con este formato exacto:
{
  "tool": "safe_eval",
  "arguments": {
    "expression": "expresión matemática a evaluar"
  }
}

Reglas:
- No añadas explicaciones fuera del JSON.
- La expresión debe usar funciones matemáticas compatibles: sqrt, sin, cos, tan, log, log10, exp, pi, e, pow, abs, round.
- Ejemplo: para "raíz cuadrada de 9 más 5", usa "sqrt(9) + 5".
"""

FINAL_PROMPT = """Eres un agente calculador. Dada la siguiente operación y su resultado,
responde al usuario de forma clara y breve en español.

Operación: {expression}
Resultado: {result}
"""


def calculator_node(state: AgentState) -> dict:
    """Ejecuta el agente calculador y guarda el resultado."""
    llm = get_llm(json_mode=True)
    task = state.get("task_description", state["user_input"])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=task),
    ]

    response = llm.invoke(messages)
    parsed = parse_json_output(response.content)

    tool_name = parsed.get("tool")
    arguments = parsed.get("arguments", {})

    if tool_name != "safe_eval":
        return {
            "agent_result": "Error: el calculador no seleccionó la herramienta correcta.",
            "messages": [response],
        }

    expression = arguments.get("expression", "")
    raw_result = safe_eval.invoke({"expression": expression})

    final_messages = [
        SystemMessage(content=FINAL_PROMPT.format(expression=expression, result=raw_result)),
        HumanMessage(content="Genera la respuesta final."),
    ]
    final_response = get_llm().invoke(final_messages)

    return {
        "agent_result": str(final_response.content),
        "messages": [response, final_response],
    }
