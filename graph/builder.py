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


def increment_iteration_node(state: AgentState) -> dict:
    """Incrementa el contador de iteraciones al recibir feedback."""
    return {"iteration_count": state.get("iteration_count", 0) + 1}


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

    # Edges
    builder.add_edge(START, "orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        orchestrator_router,
        {
            "calculator": "calculator",
            "organizer": "organizer",
            "expert": "expert",
            "critic": "critic",
        },
    )

    builder.add_edge("calculator", "critic")
    builder.add_edge("organizer", "critic")
    builder.add_edge("expert", "critic")

    builder.add_conditional_edges(
        "critic",
        critic_router,
        {
            "orchestrator": "increment_iteration",
            "__end__": END,
        },
    )

    builder.add_edge("increment_iteration", "orchestrator")

    return builder.compile()
