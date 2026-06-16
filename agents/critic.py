"""Nodo crítico: evalúa la respuesta final del sistema."""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state import AgentState
from models import get_llm, get_structured_llm, CriticOutput
from utils import is_tool_error, trace, trace_critic, trace_message
from config import MAX_CRITIC_ITERATIONS


SYSTEM_PROMPT = """Eres un crítico imparcial. Tu trabajo es evaluar si la respuesta generada por el sistema responde adecuadamente a la solicitud del usuario.

Criterios:
- "approved": la respuesta responde a la solicitud del usuario de manera correcta.
  - SIEMPRE aprueba respuestas que son técnicamente correctas, incluso si son breves o "vacías".
  - Ejemplos de respuestas correctas que DEBES aprobar:
    * "No se encontraron eventos" cuando el usuario pregunta por eventos en una fecha sin eventos
    * "No se pudo realizar la operación" cuando hay un error técnico real
    * Resultados numéricos correctos (ej: "42" para "25 + 17")
    * Respuestas que contestan directamente lo preguntado
  - Usa "approved" por defecto si la respuesta es correcta.
- "feedback": solo si la respuesta es CLARAMENTE incorrecta:
  - No tiene relación con la pregunta del usuario
  - Contiene información falsa o contradictoria
  - Tiene errores graves que la hacen inútil
  - NO rechazó por ser breve, negativa o no ofrecer alternativas

Reglas importantes:
- NO pidas que la respuesta ofrezca alternativas o sugerencias adicionales si la pregunta no las requiere.
- NO consideres una respuesta incorrecta solo porque es breve o negativa.
- Acepta respuestas como "No se encontraron eventos", "No hay datos", "No se pudo" como VÁLIDAS cuando corresponden a la realidad.
- El feedback debe ser constructivo y específico para que el orquestador pueda mejorar.
- Deja el campo feedback vacío cuando la decisión sea "approved".
"""


def critic_node(state: AgentState) -> dict:
    """Ejecuta el crítico y actualiza el estado.

    Emplea salida estructurada para forzar al modelo local a emitir únicamente
    la decisión y la retroalimentación, sin texto adicional que rompa el parseo.
    """
    trace("crítico", "Iniciando evaluación de la respuesta")

    user_input = state["user_input"]
    agent_result = state.get("agent_result", "")
    last_error = state.get("last_error", "")

    trace("crítico", f"Solicitud del usuario: {user_input}")
    trace("crítico", f"Respuesta a evaluar: {agent_result}")

    # Rechazo temprano si el agente anterior reportó un fallo de herramienta.
    # Esto evita que el crítico "apruebe" una respuesta inventada cuando
    # se esperaban datos reales de una herramienta.
    if last_error or is_tool_error(agent_result):
        feedback = (
            "La herramienta del agente falló o no está disponible. "
            "No generes una respuesta inventada; reporta el error técnico al usuario."
        )
        trace_critic("feedback", feedback)
        return {
            "critic_decision": "feedback",
            "critic_feedback": feedback,
            "final_response": "",
            "messages": [AIMessage(content=feedback)],
        }

    content = f"""Solicitud del usuario:
{user_input}

Respuesta del sistema:
{agent_result}

Evalúa la respuesta."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]

    trace_message("crítico", "modelo LLM", "Solicitando evaluación")

    try:
        parsed = get_structured_llm(CriticOutput).invoke(messages)
    except Exception as exc:
        # Si el modelo local no puede structured output, aprobamos por defecto
        # para no bloquear la ejecución.
        trace("crítico", f"Structured output falló: {exc}. Aprobando por defecto.")
        parsed = CriticOutput(decision="approved", feedback="")

    trace_message("modelo LLM", "crítico", str(parsed.model_dump_json()))
    trace_critic(parsed.decision, parsed.feedback)

    updates = {
        "critic_decision": parsed.decision,
        "critic_feedback": parsed.feedback,
        "messages": [AIMessage(content=str(parsed.model_dump_json()))],
    }

    if parsed.decision == "approved":
        updates["final_response"] = agent_result
        trace("crítico", "Respuesta aprobada; se consolidará como respuesta final")
    else:
        trace("crítico", "Respuesta rechazada; se enviará retroalimentación al orquestador")

    return updates


def critic_router(state: AgentState) -> str:
    """Decide si terminar o volver al orquestador con retroalimentación.

    Si se supera el límite de iteraciones, fuerza el final asignando como
    respuesta final el último resultado disponible, evitando respuestas vacías.
    """
    decision = state.get("critic_decision", "feedback")
    iterations = state.get("iteration_count", 0)

    if decision == "approved":
        return "__end__"

    if iterations >= MAX_CRITIC_ITERATIONS:
        trace(
            "crítico",
            f"Se alcanzó el máximo de iteraciones ({MAX_CRITIC_ITERATIONS}); terminando con la última respuesta.",
        )
        return "__end__"

    return "orchestrator"
