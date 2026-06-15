"""Nodo orquestador: decide a qué subagente delegar la tarea."""

from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from models import get_llm
from utils import parse_json_output


SYSTEM_PROMPT = """Eres un orquestador inteligente. Tu trabajo es analizar la solicitud del usuario y decidir a qué agente especializado delegar la tarea.

Agentes disponibles:
- "calculator": para operaciones matemáticas, cálculos numéricos, fórmulas.
- "organizer": para gestionar el calendario, eventos, citas, agenda del usuario.
- "expert": para consultas conceptuales, preguntas de conocimiento general, hechos del mundo.
- "none": si la solicitud no requiere ninguna herramienta especializada (saludo, despedida, charla casual).

Debes responder ÚNICAMENTE con un objeto JSON válido con este formato exacto:
{
  "assigned_agent": "calculator|organizer|expert|none",
  "task_description": "descripción clara y específica de la tarea a realizar"
}

Reglas:
- No añadas explicaciones, solo el JSON.
- Si hay retroalimentación del crítico, úsala para reformular la tarea.
- El task_description debe incluir toda la información necesaria para que el agente ejecute sin preguntar.
"""


def orchestrator_node(state: AgentState) -> dict:
    """Ejecuta el orquestador y actualiza el estado."""
    llm = get_llm(json_mode=True)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Incluimos historial previo del grafo
    messages.extend(state["messages"])

    if state.get("critic_feedback"):
        messages.append(
            HumanMessage(
                content=(
                    f"El crítico solicitó mejoras: {state['critic_feedback']}. "
                    "Reformula la tarea y reasigna al agente adecuado."
                )
            )
        )

    response = llm.invoke(messages)
    parsed = parse_json_output(response.content)

    assigned_agent = parsed.get("assigned_agent", "none")
    task_description = parsed.get("task_description", state["user_input"])

    return {
        "assigned_agent": assigned_agent,
        "task_description": task_description,
        "messages": [response],
    }


def orchestrator_router(state: AgentState) -> str:
    """Devuelve el siguiente nodo según la decisión del orquestador."""
    agent = state.get("assigned_agent", "none")
    if agent in {"calculator", "organizer", "expert"}:
        return agent
    return "critic"  # Si es "none", va directo al crítico para evaluar respuesta directa
