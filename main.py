"""Punto de entrada CLI del sistema multiagente."""

import sys

from langchain_core.messages import HumanMessage
from graph import build_graph
from utils import trace_header, trace_state


def process_input(graph, user_input: str) -> None:
    """Procesa una única entrada del usuario a través del grafo.

    Muestra en pantalla la traza completa de ejecución: activación de nodos,
    mensajes entre agentes, invocación de herramientas, evaluación del crítico
    y respuesta final consolidada.
    """
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "user_input": user_input,
        "assigned_agent": None,
        "task_description": user_input,
        "agent_result": "",
        "critic_decision": "pending",
        "critic_feedback": "",
        "final_response": "",
        "iteration_count": 0,
        "last_error": "",
    }

    trace_header("INICIO DE PROCESAMIENTO")
    print(f"  [TRAZA] [usuario] Entrada: {user_input}")

    final_state = None
    for event in graph.stream(initial_state, stream_mode="values"):
        final_state = event

    trace_header("FIN DE PROCESAMIENTO - RESPUESTA CONSOLIDADA")

    if final_state:
        trace_state(final_state)
        response = final_state.get(
            "final_response",
            final_state.get("agent_result", "No se generó respuesta."),
        )
        print(f"\nAsistente: {response}")
        print(f"(Agente asignado: {final_state.get('assigned_agent', 'ninguno')})")
        print(f"(Iteraciones: {final_state.get('iteration_count', 0)})")
    else:
        print("  [TRAZA] No se obtuvo estado final.")


def run_conversation() -> None:
    """Bucle interactivo de conversación con el sistema multiagente."""
    print("=" * 60)
    print("Sistema Multiagente con LangGraph + Ollama (gemma3:4b)")
    print("Escribe 'salir' para terminar.")
    print("=" * 60)

    graph = build_graph()

    while True:
        try:
            user_input = input("\nTú: ").strip()
        except EOFError:
            print("\nFin de entrada. ¡Hasta luego!")
            break

        if user_input.lower() in {"salir", "exit", "quit"}:
            print("¡Hasta luego!")
            break

        if not user_input:
            continue

        process_input(graph, user_input)


def run_single_query(query: str) -> None:
    """Ejecuta una única consulta y termina."""
    graph = build_graph()
    process_input(graph, query)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_single_query(query)
    else:
        run_conversation()
