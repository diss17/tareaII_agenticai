"""Tests de manejo de errores de herramientas.

Verifican que los agentes especialistas detecten cuando una herramienta falla
y reporten el error técnico en lugar de generar respuestas inventadas.
"""

import sys
from pathlib import Path
from importlib import reload

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tool_result import is_tool_error, make_tool_error, get_tool_error_detail


def test_tool_error_utilities():
    """Verifica la detección y construcción de mensajes de error."""
    error_msg = make_tool_error("fallo de red")
    assert is_tool_error(error_msg)
    assert not is_tool_error("resultado normal")
    assert not is_tool_error(None)
    assert get_tool_error_detail(error_msg) == "fallo de red"
    assert get_tool_error_detail("resultado normal") == "resultado normal"


def _mock_structured_llm(output_cls, **field_values):
    """Crea un mock de get_structured_llm que devuelve una instancia del esquema."""
    class MockLlm:
        def invoke(self, messages):
            return output_cls(**field_values)
    return MockLlm()


def test_expert_reports_search_error():
    """Si web_search falla, el experto reporta el error y no genera respuesta."""
    import models.llm as llm_mod
    import tools as tools_pkg
    import agents.expert as expert_mod

    # Mock de salida estructurada del experto
    from models.schemas import ExpertOutput
    llm_mod.get_structured_llm = lambda schema: _mock_structured_llm(
        ExpertOutput,
        tool="web_search",
        arguments={"query": "entropía termodinámica"},
    )

    # Mock de web_search que falla. Debe tener un método .invoke() como
    # las herramientas decoradas con @tool de LangChain.
    class FailingSearch:
        def invoke(self, arguments: dict) -> str:
            return make_tool_error("ddgs no instalado")

    failing_search = FailingSearch()
    tools_pkg.web_search = failing_search
    expert_mod.web_search = failing_search

    state = {
        "user_input": "¿Qué es la entropía?",
        "task_description": "Explicar entropía",
        "messages": [],
        "assigned_agent": "expert",
    }

    result = expert_mod.expert_node(state)

    assert "No pude realizar la búsqueda web" in result["agent_result"]
    assert is_tool_error(result["last_error"])
    assert "ddgs no instalado" in result["last_error"]


def test_calculator_reports_math_error():
    """Si safe_eval falla, el calculador reporta el error."""
    import models.llm as llm_mod
    import tools.math_tools as math_mod
    import agents.calculator as calculator_mod

    from models.schemas import CalculatorOutput
    llm_mod.get_structured_llm = lambda schema: _mock_structured_llm(
        CalculatorOutput,
        tool="safe_eval",
        arguments={"expression": "1/0"},
    )

    # safe_eval real fallará con 1/0
    reload(math_mod)
    reload(calculator_mod)

    state = {
        "user_input": "¿Cuánto es 1/0?",
        "task_description": "1/0",
        "messages": [],
        "assigned_agent": "calculator",
    }

    result = calculator_mod.calculator_node(state)

    assert "No pude resolver la operación matemática" in result["agent_result"]
    assert is_tool_error(result["last_error"])


def test_organizer_reports_calendar_error():
    """Si la herramienta de calendario falla, el organizador reporta el error."""
    import models.llm as llm_mod
    import agents.organizer as organizer_mod

    from models.schemas import OrganizerOutput
    llm_mod.get_structured_llm = lambda schema: _mock_structured_llm(
        OrganizerOutput,
        tool="get_events",
        arguments={},
    )

    # Mock de get_events dentro del diccionario TOOLS del organizador.
    # Debe tener un método .invoke() como las herramientas de LangChain.
    original_tool = organizer_mod.TOOLS["get_events"]

    class FailingGetEvents:
        def invoke(self, arguments: dict) -> str:
            return make_tool_error("base de datos bloqueada")

    organizer_mod.TOOLS["get_events"] = FailingGetEvents()

    state = {
        "user_input": "Muéstrame mis eventos",
        "task_description": "Consultar eventos",
        "messages": [],
        "assigned_agent": "organizer",
    }

    try:
        result = organizer_mod.organizer_node(state)
    finally:
        organizer_mod.TOOLS["get_events"] = original_tool

    assert "No pudo completarse la operación del calendario" in result["agent_result"]
    assert is_tool_error(result["last_error"])
    assert "base de datos bloqueada" in result["last_error"]


def test_critic_rejects_tool_error():
    """El crítico rechaza respuestas que contienen errores de herramienta."""
    import models.llm as llm_mod
    import agents.critic as critic_mod

    from models.schemas import CriticOutput
    llm_mod.get_structured_llm = lambda schema: _mock_structured_llm(
        CriticOutput,
        decision="approved",
        feedback="",
    )

    reload(critic_mod)

    state = {
        "user_input": "¿Qué es la entropía?",
        "agent_result": make_tool_error("ddgs no instalado"),
        "last_error": make_tool_error("ddgs no instalado"),
        "messages": [],
    }

    result = critic_mod.critic_node(state)

    assert result["critic_decision"] == "feedback"
    assert "falló" in result["critic_feedback"].lower() or "disponible" in result["critic_feedback"].lower()
    assert result["final_response"] == ""


if __name__ == "__main__":
    test_tool_error_utilities()
    print("OK test_tool_error_utilities")
    test_expert_reports_search_error()
    print("OK test_expert_reports_search_error")
    test_calculator_reports_math_error()
    print("OK test_calculator_reports_math_error")
    test_organizer_reports_calendar_error()
    print("OK test_organizer_reports_calendar_error")
    test_critic_rejects_tool_error()
    print("OK test_critic_rejects_tool_error")
    print("\nTodos los tests de manejo de errores pasaron.")
