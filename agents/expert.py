"""Nodo experto: responde consultas conceptuales usando búsqueda web."""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state import AgentState
from models import get_llm, get_structured_llm, ExpertOutput
from tools import web_search
from utils import is_tool_error, trace, trace_message, trace_tool


SYSTEM_PROMPT = """Eres un agente experto en conocimiento general.

Para responder consultas conceptuales o preguntas sobre hechos del mundo real,
debes usar la herramienta "web_search".

Reglas:
- La consulta debe ser clara y en español si la pregunta del usuario está en español.
- Si la pregunta es puramente filosófica u opinable y no requiere datos actuales,
  aún así puedes buscar información de contexto.
- No añadas explicaciones fuera de la estructura.
"""

FINAL_PROMPT = """Eres un agente experto. Usa la información de búsqueda para responder
la pregunta del usuario de forma clara, precisa y breve en español.

Pregunta: {question}
Resultado de búsqueda: {search_result}

Responde directamente a la pregunta. No inventes información que no esté en el resultado de búsqueda.
"""


def expert_node(state: AgentState) -> dict:
    """Ejecuta el agente experto y guarda el resultado.

    Usa salida estructurada para forzar al modelo local a emitir la llamada
    a web_search con los argumentos correctos, evitando parámetros inventados.
    """
    trace("experto", "Iniciando consulta de conocimiento")
    task = state.get("task_description", state["user_input"])
    trace("experto", f"Tarea recibida: {task}")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=task),
    ]

    trace_message("experto", "modelo LLM", "Solicitando selección de herramienta de búsqueda")

    try:
        parsed = get_structured_llm(ExpertOutput).invoke(messages)
    except Exception as exc:
        error_msg = f"Error: el experto no pudo generar una llamada válida: {exc}"
        trace("experto", error_msg)
        return {
            "agent_result": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "last_error": error_msg,
        }

    trace_message("modelo LLM", "experto", str(parsed.model_dump_json()))
    trace("experto", f"Herramienta seleccionada: {parsed.tool}")

    if parsed.tool != "web_search":
        error_msg = "Error: el experto no seleccionó la herramienta de búsqueda."
        trace("experto", error_msg)
        return {
            "agent_result": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "last_error": error_msg,
        }

    query = parsed.arguments.get("query", task)
    trace("experto", f"Consulta de búsqueda: {query}")
    search_result = web_search.invoke({"query": query})
    trace_tool("experto", "web_search", parsed.arguments, search_result)

    if is_tool_error(search_result):
        error_msg = (
            "No pude realizar la búsqueda web solicitada. "
            f"Error técnico: {search_result}"
        )
        trace("experto", f"La herramienta web_search falló: {search_result}")
        return {
            "agent_result": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "last_error": search_result,
        }

    final_messages = [
        SystemMessage(
            content=FINAL_PROMPT.format(question=task, search_result=search_result)
        ),
        HumanMessage(content="Genera la respuesta final."),
    ]
    trace_message("experto", "modelo LLM", "Solicitando respuesta final al usuario")
    final_response = get_llm().invoke(final_messages)
    trace_message("modelo LLM", "experto", str(final_response.content))

    trace("experto", f"Resultado final generado: {final_response.content}")

    return {
        "agent_result": str(final_response.content),
        "messages": [AIMessage(content=str(parsed.model_dump_json())), final_response],
        "last_error": "",
    }
