"""Nodo orquestador: decide a qué subagente delegar la tarea."""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state import AgentState
from models import get_llm, get_structured_llm, OrchestratorOutput
from utils import trace, trace_message


SYSTEM_PROMPT = """Eres un orquestador inteligente. Tu trabajo es analizar la solicitud del usuario y decidir a qué agente especializado delegar la tarea.

Agentes disponibles:
- "calculator": para operaciones matemáticas, cálculos numéricos, fórmulas.
- "organizer": para gestionar el calendario, eventos, citas, agenda del usuario.
- "expert": para consultas conceptuales, preguntas de conocimiento general, hechos del mundo.

Reglas ESTRICTAS:
- SOLO puedes asignar a uno de los 3 agentes anteriores (calculator, organizer, expert).
- Si la solicitud está FUERA del alcance (charla casual, saludos, despedidas, o temas que ningún agente puede manejar), asigna "out_of_scope".
- Si hay retroalimentación del crítico, úsala para reformular la tarea y reasigna a un agente válido.
- CRÍTICO: El task_description debe preservar TODA la información del usuario, especialmente:
  * Fechas y horas específicas (ej: "25 de diciembre de 2026", "mañana a las 10:00")
  * Números, cantidades y unidades
  * Nombres propios y títulos
  * IDs de eventos
- El task_description debe ser COPIA o PARÁFRASIS COMPLETA de la solicitud original, sin omitir detalles.
- Para "out_of_scope", puedes dejar task_description vacío.
"""


def orchestrator_node(state: AgentState) -> dict:
    """Ejecuta el orquestador y actualiza el estado.

    Utiliza salida estructurada (Pydantic) para forzar al modelo local a
    generar exactamente los campos esperados, evitando alucinaciones de
    parámetros y eliminando el parseo manual de JSON.
    """
    trace("orquestador", "Iniciando análisis de la solicitud")
    trace("orquestador", f"Entrada del usuario: {state['user_input']}")

    llm = get_structured_llm(OrchestratorOutput)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    # Incluimos historial previo del grafo (acumulado por add_messages)
    messages.extend(state["messages"])

    if state.get("critic_feedback"):
        feedback_msg = (
            f"El crítico solicitó mejoras: {state['critic_feedback']}. "
            "Reformula la tarea y reasigna al agente adecuado."
        )
        messages.append(HumanMessage(content=feedback_msg))
        trace_message("crítico", "orquestador", feedback_msg)

    trace_message("orquestador", "modelo LLM", "Solicitando decisión de enrutamiento")

    try:
        parsed = llm.invoke(messages)
    except Exception as exc:
        # Fallback: si el modelo local falla en structured output, usamos JSON plano
        trace("orquestrador", f"Structured output falló: {exc}. Usando fallback JSON.")
        raw = get_llm(json_mode=True).invoke(messages)
        from utils import parse_json_output
        data = parse_json_output(raw.content)
        parsed = OrchestratorOutput(
            assigned_agent=data.get("assigned_agent", "out_of_scope"),
            task_description=data.get("task_description", state["user_input"]),
        )

    assigned_agent = parsed.assigned_agent
    task_description = parsed.task_description or state["user_input"]

    trace("orquestrador", f"Agente asignado: {assigned_agent}")
    trace("orquestrador", f"Descripción de tarea: {task_description}")

    updates = {
        "assigned_agent": assigned_agent,
        "task_description": task_description,
        "last_error": "",
    }

    # Para solicitudes fuera de alcance, establecer mensaje de error estático
    if assigned_agent == "out_of_scope":
        out_of_scope_message = (
            "Lo siento, esta solicitud está fuera del alcance del sistema. "
            "Solo puedo ayudarte con: cálculos matemáticos, gestión de calendario, "
            "y consultas de conocimiento general."
        )
        trace("orquestrador", f"Solicitud fuera de alcance: {out_of_scope_message}")
        updates["final_response"] = out_of_scope_message
        updates["agent_result"] = out_of_scope_message
        updates["critic_decision"] = "approved"  # No necesita crítico
        updates["messages"] = [AIMessage(content=str(parsed.model_dump_json()))]
    else:
        updates["agent_result"] = ""
        updates["messages"] = [AIMessage(content=str(parsed.model_dump_json()))]

    return updates


def orchestrator_router(state: AgentState) -> str:
    """Devuelve el siguiente nodo según la decisión del orquestador."""
    from langgraph.graph import END
    
    agent = state.get("assigned_agent", "out_of_scope")
    if agent in {"calculator", "organizer", "expert"}:
        return agent
    # "out_of_scope" va directamente al END sin pasar por crítico
    return END
