"""Nodo crítico: evalúa la respuesta final del sistema."""

from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from models import get_llm
from utils import parse_json_output
from config import MAX_CRITIC_ITERATIONS


SYSTEM_PROMPT = """Eres un crítico imparcial. Tu trabajo es evaluar si la respuesta generada por el sistema responde adecuadamente a la solicitud del usuario.

Debes responder ÚNICAMENTE con un objeto JSON válido con este formato exacto:
{
  "decision": "approved|feedback",
  "feedback": "explicación breve de por qué se aprueba o qué debe corregirse"
}

Criterios:
- "approved": la respuesta aborda la solicitud del usuario de manera razonable y útil.
  Usa "approved" por defecto si la respuesta es correcta y contiene la información solicitada.
- "feedback": solo si la respuesta es claramente incorrecta, no responde la pregunta,
  o contiene errores graves que impiden su utilidad.

Reglas:
- No seas excesivamente exigente con detalles secundarios.
- No añadas explicaciones fuera del JSON.
- El feedback debe ser constructivo y específico para que el orquestador pueda mejorar.
"""


def critic_node(state: AgentState) -> dict:
    """Ejecuta el crítico y actualiza el estado."""
    llm = get_llm(json_mode=True)

    user_input = state["user_input"]
    agent_result = state.get("agent_result", "")

    content = f"""Solicitud del usuario:
{user_input}

Respuesta del sistema:
{agent_result}

Evalúa la respuesta."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]

    response = llm.invoke(messages)
    parsed = parse_json_output(response.content)

    decision = parsed.get("decision", "feedback")
    feedback = parsed.get("feedback", "")

    updates = {
        "critic_decision": decision,
        "critic_feedback": feedback,
        "messages": [response],
    }

    if decision == "approved":
        updates["final_response"] = agent_result

    return updates


def critic_router(state: AgentState) -> str:
    """Decide si terminar o volver al orquestador con retroalimentación."""
    decision = state.get("critic_decision", "feedback")
    iterations = state.get("iteration_count", 0)

    if decision == "approved" or iterations >= MAX_CRITIC_ITERATIONS:
        return "__end__"

    return "orchestrator"
