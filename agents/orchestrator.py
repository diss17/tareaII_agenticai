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
- "none": si la solicitud no requiere ninguna herramienta especializada (saludo, despedida, charla casual).

Reglas:
- Si hay retroalimentación del crítico, úsala para reformular la tarea.
- El task_description debe incluir toda la información necesaria para que el agente ejecute sin preguntar.
- Para saludos o charla casual, asigna "none" y deja task_description vacío.
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
        trace("orquestador", f"Structured output falló: {exc}. Usando fallback JSON.")
        raw = get_llm(json_mode=True).invoke(messages)
        from utils import parse_json_output
        data = parse_json_output(raw.content)
        parsed = OrchestratorOutput(
            assigned_agent=data.get("assigned_agent", "none"),
            task_description=data.get("task_description", state["user_input"]),
        )

    assigned_agent = parsed.assigned_agent
    task_description = parsed.task_description or state["user_input"]

    trace("orquestador", f"Agente asignado: {assigned_agent}")
    trace("orquestador", f"Descripción de tarea: {task_description}")

    updates = {
        "assigned_agent": assigned_agent,
        "task_description": task_description,
        "last_error": "",
    }

    # Para charla casual, el propio orquestador genera la respuesta amable
    # y la manda al crítico como filtro final, manteniendo la arquitectura.
    if assigned_agent == "none":
        direct_messages = [
            SystemMessage(
                content="Eres un asistente amable. Responde de forma breve y cordial en español."
            ),
            HumanMessage(content=state["user_input"]),
        ]
        direct_response = get_llm().invoke(direct_messages)
        trace_message("modelo LLM", "orquestador", str(direct_response.content))
        updates["agent_result"] = str(direct_response.content)
        updates["messages"] = [AIMessage(content=str(parsed.model_dump_json()))]
    else:
        updates["agent_result"] = ""
        updates["messages"] = [AIMessage(content=str(parsed.model_dump_json()))]

    return updates


def orchestrator_router(state: AgentState) -> str:
    """Devuelve el siguiente nodo según la decisión del orquestador."""
    agent = state.get("assigned_agent", "none")
    if agent in {"calculator", "organizer", "expert"}:
        return agent
    # "none" va al crítico para evaluar la respuesta directa generada
    return "critic"
