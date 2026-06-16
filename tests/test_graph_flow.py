"""Tests de integración del grafo multiagente usando mocks.

Estos tests no requieren Ollama. Mockean los nodos del grafo para verificar
que el estado fluye correctamente entre orquestador, agentes y crítico, y
que los routers respetan la arquitectura especificada.
"""

import sys
from pathlib import Path
from importlib import reload

# Asegurar que el proyecto raíz está en el path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_orchestrator_result(agent: str, task: str, direct_response: str = ""):
    """Crea el dict de retorno de un nodo orquestador mock."""
    return {
        "assigned_agent": agent,
        "task_description": task,
        "agent_result": direct_response,
        "messages": [],
        "last_error": "",
    }


def _make_specialist_result(result: str):
    """Crea el dict de retorno de un agente especialista mock."""
    return {
        "agent_result": result,
        "messages": [],
        "last_error": "",
    }


def _make_critic_result(decision: str, feedback: str = "", result: str = ""):
    """Crea el dict de retorno del nodo crítico mock."""
    return {
        "critic_decision": decision,
        "critic_feedback": feedback,
        "final_response": result if decision == "approved" else "",
        "messages": [],
    }


def _build_initial_state():
    return {
        "messages": [],
        "user_input": "consulta de prueba",
        "assigned_agent": None,
        "task_description": "",
        "agent_result": "",
        "critic_decision": "pending",
        "critic_feedback": "",
        "final_response": "",
        "iteration_count": 0,
        "last_error": "",
    }


def _build_graph_with_mocks(agent_mocks: dict):
    """Carga el grafo reemplazando las funciones de nodo indicadas por mocks.

    Para que build_graph capture los mocks, se asignan sobre los módulos de
    agentes y sobre el namespace del paquete `agents`, y luego se recarga
    graph.builder para que su importación `from agents import ...` tome las
    funciones mockeadas.
    """
    import agents.orchestrator as orchestrator_mod
    import agents.calculator as calculator_mod
    import agents.organizer as organizer_mod
    import agents.expert as expert_mod
    import agents.critic as critic_mod
    import agents as agents_pkg

    for name, func in agent_mocks.items():
        if name == "orchestrator_node":
            orchestrator_mod.orchestrator_node = func
            agents_pkg.orchestrator_node = func
        elif name == "calculator_node":
            calculator_mod.calculator_node = func
            agents_pkg.calculator_node = func
        elif name == "organizer_node":
            organizer_mod.organizer_node = func
            agents_pkg.organizer_node = func
        elif name == "expert_node":
            expert_mod.expert_node = func
            agents_pkg.expert_node = func
        elif name == "critic_node":
            critic_mod.critic_node = func
            agents_pkg.critic_node = func

    import graph.builder as builder_mod
    reload(builder_mod)

    return builder_mod.build_graph()


def test_state_propagation_and_routing():
    """Verifica que el estado viaje orquestador → calculador → crítico → END."""
    graph = _build_graph_with_mocks(
        {
            "orchestrator_node": lambda state: _make_orchestrator_result(
                "calculator", "15 * 23 + sqrt(9)"
            ),
            "calculator_node": lambda state: _make_specialist_result("El resultado es 348."),
            "critic_node": lambda state: _make_critic_result(
                "approved", result="El resultado es 348."
            ),
        }
    )

    final_state = None
    for event in graph.stream(_build_initial_state(), stream_mode="values"):
        final_state = event

    assert final_state is not None
    assert final_state["assigned_agent"] == "calculator"
    assert final_state["task_description"] == "15 * 23 + sqrt(9)"
    assert final_state["agent_result"] == "El resultado es 348."
    assert final_state["final_response"] == "El resultado es 348."
    assert final_state["critic_decision"] == "approved"


def test_critic_feedback_loop():
    """Verifica que el feedback del crítico regrese al orquestador y luego termine."""
    calls = {"orchestrator": 0}

    def mock_orchestrator(state):
        calls["orchestrator"] += 1
        return _make_orchestrator_result("expert", "computación cuántica")

    def mock_expert(state):
        return _make_specialist_result("Respuesta mejorada.")

    def mock_critic(state):
        if state.get("iteration_count", 0) == 0:
            return _make_critic_result("feedback", "Falta precisión técnica.")
        return _make_critic_result(
            "approved", result=state.get("agent_result", "")
        )

    graph = _build_graph_with_mocks(
        {
            "orchestrator_node": mock_orchestrator,
            "expert_node": mock_expert,
            "critic_node": mock_critic,
        }
    )

    final_state = None
    for event in graph.stream(_build_initial_state(), stream_mode="values"):
        final_state = event

    assert final_state is not None
    assert calls["orchestrator"] == 2
    assert final_state["iteration_count"] == 1
    assert final_state["critic_decision"] == "approved"
    assert final_state["final_response"] == "Respuesta mejorada."


def test_fallback_on_none_agent():
    """Verifica que 'none' genere una respuesta directa y pase al crítico."""
    graph = _build_graph_with_mocks(
        {
            "orchestrator_node": lambda state: _make_orchestrator_result(
                "none", "", direct_response="¡Hola! ¿En qué puedo ayudarte?"
            ),
            "critic_node": lambda state: _make_critic_result(
                "approved", result=state.get("agent_result", "")
            ),
        }
    )

    final_state = None
    for event in graph.stream(_build_initial_state(), stream_mode="values"):
        final_state = event

    assert final_state is not None
    assert final_state["assigned_agent"] == "none"
    assert final_state["agent_result"] == "¡Hola! ¿En qué puedo ayudarte?"
    assert final_state["final_response"] == "¡Hola! ¿En qué puedo ayudarte?"


if __name__ == "__main__":
    test_state_propagation_and_routing()
    print("OK test_state_propagation_and_routing")
    test_critic_feedback_loop()
    print("OK test_critic_feedback_loop")
    test_fallback_on_none_agent()
    print("OK test_fallback_on_none_agent")
    print("\nTodos los tests pasaron.")
