"""Nodo organizador: gestiona el calendario del usuario."""

from datetime import datetime, timedelta

from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from models import get_llm
from tools import add_event, get_events, update_event, delete_event
from utils import parse_json_output


SYSTEM_PROMPT_TEMPLATE = """Eres un agente organizador especializado en gestión del tiempo.

Tienes acceso a un calendario local persistente. Debes usar una de estas herramientas:
- "add_event": agrega un evento. Argumentos: title (str), event_datetime (str), description (str, opcional).
- "get_events": consulta eventos. Argumentos: date (str opcional, formato YYYY-MM-DD).
- "update_event": modifica un evento. Argumentos: event_id (int), title (str, opcional), event_datetime (str, opcional), description (str, opcional).
- "delete_event": elimina un evento. Argumentos: event_id (int).

Responde ÚNICAMENTE con un objeto JSON válido con este formato exacto:
{{
  "tool": "nombre_de_la_herramienta",
  "arguments": {{
    "arg1": "valor1",
    "arg2": "valor2"
  }}
}}

Reglas:
- No añadas explicaciones fuera del JSON.
- La fecha de hoy es {today}.
- Si el usuario dice "mañana", usa {tomorrow}.
- Si el usuario dice "pasado mañana", usa {day_after_tomorrow}.
- Si no especifica hora, usa "00:00".
- La fecha y hora deben estar en formato "YYYY-MM-DD HH:MM".
- Si la consulta es para ver eventos sin fecha específica, omite el argumento date.
"""

FINAL_PROMPT = """Eres un agente organizador. Dada la acción realizada en el calendario,
responde al usuario de forma clara y breve en español.

Acción: {tool_name}
Argumentos: {arguments}
Resultado: {result}
"""

TOOLS = {
    "add_event": add_event,
    "get_events": get_events,
    "update_event": update_event,
    "delete_event": delete_event,
}


def _build_system_prompt() -> str:
    """Construye el system prompt con las fechas relativas resueltas."""
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    day_after = today + timedelta(days=2)
    return SYSTEM_PROMPT_TEMPLATE.format(
        today=today.strftime("%Y-%m-%d"),
        tomorrow=tomorrow.strftime("%Y-%m-%d"),
        day_after_tomorrow=day_after.strftime("%Y-%m-%d"),
    )


def organizer_node(state: AgentState) -> dict:
    """Ejecuta el agente organizador y guarda el resultado."""
    llm = get_llm(json_mode=True)
    task = state.get("task_description", state["user_input"])

    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=task),
    ]

    response = llm.invoke(messages)
    parsed = parse_json_output(response.content)

    tool_name = parsed.get("tool")
    arguments = parsed.get("arguments", {})

    if tool_name not in TOOLS:
        return {
            "agent_result": f"Error: herramienta de calendario desconocida '{tool_name}'.",
            "messages": [response],
        }

    try:
        raw_result = TOOLS[tool_name].invoke(arguments)
    except Exception as exc:
        raw_result = f"Error al ejecutar {tool_name}: {exc}"

    final_messages = [
        SystemMessage(
            content=FINAL_PROMPT.format(tool_name=tool_name, arguments=arguments, result=raw_result)
        ),
        HumanMessage(content="Genera la respuesta final."),
    ]
    final_response = get_llm().invoke(final_messages)

    return {
        "agent_result": str(final_response.content),
        "messages": [response, final_response],
    }
