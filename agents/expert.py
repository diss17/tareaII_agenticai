"""Nodo experto: responde consultas conceptuales usando búsqueda web."""

from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from models import get_llm
from tools import web_search
from utils import parse_json_output


SYSTEM_PROMPT = """Eres un agente experto en conocimiento general.

Para responder consultas conceptuales o preguntas sobre hechos del mundo real,
debes usar la herramienta "web_search".

Responde ÚNICAMENTE con un objeto JSON válido con este formato exacto:
{
  "tool": "web_search",
  "arguments": {
    "query": "consulta de búsqueda en español"
  }
}

Reglas:
- No añadas explicaciones fuera del JSON.
- La consulta debe ser clara y en español si la pregunta del usuario está en español.
- Si la pregunta es puramente filosófica u opinable y no requiere datos actuales,
  aún así puedes buscar información de contexto.
"""

FINAL_PROMPT = """Eres un agente experto. Usa la información de búsqueda para responder
la pregunta del usuario de forma clara, precisa y breve en español.

Pregunta: {question}
Resultado de búsqueda: {search_result}

Responde directamente a la pregunta. No inventes información que no esté en el resultado de búsqueda.
"""


def expert_node(state: AgentState) -> dict:
    """Ejecuta el agente experto y guarda el resultado."""
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

    if tool_name != "web_search":
        return {
            "agent_result": "Error: el experto no seleccionó la herramienta de búsqueda.",
            "messages": [response],
        }

    query = arguments.get("query", task)
    search_result = web_search.invoke({"query": query})

    final_messages = [
        SystemMessage(
            content=FINAL_PROMPT.format(question=task, search_result=search_result)
        ),
        HumanMessage(content="Genera la respuesta final."),
    ]
    final_response = get_llm().invoke(final_messages)

    return {
        "agent_result": str(final_response.content),
        "messages": [response, final_response],
    }
