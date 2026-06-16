"""Nodo organizador: gestiona el calendario del usuario."""

from datetime import datetime, timedelta

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state import AgentState
from models import get_llm, get_structured_llm, OrganizerOutput
from tools import add_event, get_events, update_event, delete_event
from utils import is_tool_error, make_tool_error, trace, trace_message, trace_tool, extract_date


SYSTEM_PROMPT_TEMPLATE = """Eres un agente organizador especializado en gestión del tiempo.

Tienes acceso a un calendario local persistente. Debes usar una de estas herramientas:
- "add_event": agrega un evento. Argumentos: title (str), event_datetime (str), description (str, opcional).
- "get_events": consulta eventos. Argumentos: date (str opcional, formato YYYY-MM-DD).
- "update_event": modifica un evento. Argumentos: event_id (int), title (str, opcional), event_datetime (str, opcional), description (str, opcional).
- "delete_event": elimina un evento. Argumentos: event_id (int).

Reglas CRÍTICAS para fechas:
- La fecha de hoy es {today}.
- Si el usuario dice "mañana", usa {tomorrow}.
- Si el usuario dice "pasado mañana", usa {day_after_tomorrow}.
- Si no especifica hora, usa "00:00".
- La fecha y hora deben estar en formato "YYYY-MM-DD HH:MM".

Reglas para get_events:
- Si el usuario pregunta por eventos en una FECHA ESPECÍFICA (ej: "el 16 de junio", "el 25 de diciembre de 2026", "mañana"), SIEMPRE debes pasar el argumento "date" en formato YYYY-MM-DD.
- Si la consulta es GENÉRICA sin fecha específica (ej: "qué eventos tengo", "ver mi agenda"), omite el argumento date.
- Ejemplo: "¿Tengo algo el 16 de junio de 2026?" → get_events con date="2026-06-16"
- Ejemplo: "¿Qué eventos tengo mañana?" → get_events con date="{tomorrow}"
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
    """Ejecuta el agente organizador y guarda el resultado.

    Usa salida estructurada para que el modelo local produzca exactamente
    la herramienta de calendario y los argumentos esperados.

    Incluye detección determinista de fechas con regex como FALLBACK para
    garantizar que el parámetro 'date' se pase correctamente a get_events,
    incluso si el LLM local no extrae la fecha del input del usuario.
    """
    trace("organizador", "Iniciando gestión de calendario")
    task = state.get("task_description", state["user_input"])
    original_input = state["user_input"]
    trace("organizador", f"Tarea recibida: {task}")
    trace("organizador", f"Input original del usuario: {original_input}")

    # Detección determinista de fecha (fallback robusto)
    detected_date = extract_date(original_input)
    if detected_date:
        trace("organizador", f"Fecha detectada en input: {detected_date}")

    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=task),
    ]

    trace_message("organizador", "modelo LLM", "Solicitando selección de herramienta de calendario")

    try:
        parsed = get_structured_llm(OrganizerOutput).invoke(messages)
    except Exception as exc:
        error_msg = f"Error: el organizador no pudo generar una llamada válida: {exc}"
        trace("organizador", error_msg)
        return {
            "agent_result": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "last_error": error_msg,
        }

    trace_message("modelo LLM", "organizador", str(parsed.model_dump_json()))
    trace("organizador", f"Herramienta seleccionada: {parsed.tool}")

    # FALLBACK: Si el LLM eligió get_events sin fecha pero detectamos una fecha
    # en el input original, forzar el uso de esa fecha
    if (
        parsed.tool == "get_events"
        and detected_date
        and not parsed.arguments.get("date")
    ):
        trace(
            "organizador",
            f"LLM no incluyó 'date'. Inyectando fecha detectada: {detected_date}",
        )
        parsed.arguments["date"] = detected_date

    if parsed.tool not in TOOLS:
        error_msg = f"Error: herramienta de calendario desconocida '{parsed.tool}'."
        trace("organizador", error_msg)
        return {
            "agent_result": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "last_error": error_msg,
        }

    try:
        raw_result = TOOLS[parsed.tool].invoke(parsed.arguments)
    except Exception as exc:
        raw_result = make_tool_error(f"Error al ejecutar {parsed.tool}: {exc}")

    trace_tool("organizador", parsed.tool, parsed.arguments, raw_result)

    if is_tool_error(raw_result):
        error_msg = (
            "No pudo completarse la operación del calendario. "
            f"Error técnico: {raw_result}"
        )
        trace("organizador", f"La herramienta {parsed.tool} falló: {raw_result}")
        return {
            "agent_result": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "last_error": raw_result,
        }

    final_messages = [
        SystemMessage(
            content=FINAL_PROMPT.format(tool_name=parsed.tool, arguments=parsed.arguments, result=raw_result)
        ),
        HumanMessage(content="Genera la respuesta final."),
    ]
    trace_message("organizador", "modelo LLM", "Solicitando respuesta final al usuario")
    final_response = get_llm().invoke(final_messages)
    trace_message("modelo LLM", "organizador", str(final_response.content))

    trace("organizador", f"Resultado final generado: {final_response.content}")

    return {
        "agent_result": str(final_response.content),
        "messages": [AIMessage(content=str(parsed.model_dump_json())), final_response],
        "last_error": "",
    }
