"""Construcción y compilación del grafo multiagente."""

from langgraph.graph import StateGraph, START, END
from state import AgentState
from agents import (
    orchestrator_node,
    orchestrator_router,
    calculator_node,
    organizer_node,
    expert_node,
    critic_node,
    critic_router,
)
from utils import trace, trace_state
from config import MAX_CRITIC_ITERATIONS


def increment_iteration_node(state: AgentState) -> dict:
    """Incrementa el contador de iteraciones al recibir feedback.

    Resetea la decisión del crítico para que el orquestador no la reinterprete
    como una aprobación pendiente y evita bucles infinitos.
    """
    new_count = state.get("iteration_count", 0) + 1
    trace("iteración", f"Preparando iteración de retroalimentación: {new_count}")
    trace_state(state)
    return {
        "iteration_count": new_count,
        "critic_decision": "pending",
        "critic_feedback": state.get("critic_feedback", ""),
    }


def ensure_final_response_node(state: AgentState) -> dict:
    """Nodo de guarda: asegura que haya una respuesta final al terminar.

    Si el crítico agotó las iteraciones sin aprobar, usa el último resultado
    disponible como respuesta final para no devolver un string vacío.
    """
    if not state.get("final_response"):
        fallback = state.get("agent_result", "No se pudo generar una respuesta.")
        trace("guarda", f"Asignando respuesta final de fallback: {fallback}")
        return {"final_response": fallback}
    return {}


def build_graph() -> StateGraph:
    """Construye el grafo completo y lo devuelve compilado."""
    builder = StateGraph(AgentState)

    # Nodos
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("calculator", calculator_node)
    builder.add_node("organizer", organizer_node)
    builder.add_node("expert", expert_node)
    builder.add_node("critic", critic_node)
    builder.add_node("increment_iteration", increment_iteration_node)
    builder.add_node("ensure_final_response", ensure_final_response_node)

    # Flujo principal
    builder.add_edge(START, "orchestrator")

    def route_from_orchestrator(state: AgentState) -> str:
        target = orchestrator_router(state)
        trace("router", f"Orquestador -> {target}")
        return target

    builder.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "calculator": "calculator",
            "organizer": "organizer",
            "expert": "expert",
            "critic": "critic",
            "__end__": "ensure_final_response",
        },
    )

    builder.add_edge("calculator", "critic")
    builder.add_edge("organizer", "critic")
    builder.add_edge("expert", "critic")

    def route_from_critic(state: AgentState) -> str:
        target = critic_router(state)
        trace("router", f"Crítico -> {target}")
        return target

    builder.add_conditional_edges(
        "critic",
        route_from_critic,
        {
            "orchestrator": "increment_iteration",
            "__end__": "ensure_final_response",
        },
    )

    builder.add_edge("increment_iteration", "orchestrator")
    builder.add_edge("ensure_final_response", END)

    return builder.compile()
