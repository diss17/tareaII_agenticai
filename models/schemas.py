"""Esquemas Pydantic para salidas estructuradas de los agentes.

Usar `with_structured_output` con estos esquemas en modelos locales fuerza al
LLM a producir exactamente los campos esperados, evitando alucinaciones de
parámetros y reduciendo la fragilidad del parseo manual de JSON.
"""

from typing import Literal
from pydantic import BaseModel, Field


class OrchestratorOutput(BaseModel):
    """Decisión de enrutamiento del agente coordinador."""

    assigned_agent: Literal["calculator", "organizer", "expert", "out_of_scope"] = Field(
        ...,
        description=(
            "Agente especialista al que se delega la tarea. "
            "Usa 'out_of_scope' cuando la solicitud está fuera del alcance del sistema "
            "(charla casual, saludos, o temas que no pueden ser manejados por "
            "calculator, organizer o expert)."
        ),
    )
    task_description: str = Field(
        ...,
        description=(
            "Descripción clara y específica de la tarea a realizar. "
            "Debe contener toda la información necesaria para que el agente ejecute. "
            "Puede estar vacío si assigned_agent es 'out_of_scope'."
        ),
    )


class ToolCallOutput(BaseModel):
    """Salida estructurada para agentes que invocan herramientas."""

    tool: str = Field(
        ...,
        description="Nombre exacto de la herramienta a invocar.",
    )
    arguments: dict = Field(
        default_factory=dict,
        description="Diccionario con los argumentos concretos de la herramienta.",
    )


class CalculatorOutput(ToolCallOutput):
    """Salida estructurada del agente calculador."""

    tool: Literal["safe_eval"] = Field(
        ...,
        description="Siempre usa 'safe_eval' para operaciones matemáticas.",
    )
    arguments: dict = Field(
        ...,
        description=(
            "Debe contener el campo 'expression' con la expresión matemática a evaluar. "
            "Ejemplo: {'expression': 'sqrt(9) + 5'}"
        ),
    )


class OrganizerOutput(ToolCallOutput):
    """Salida estructurada del agente organizador."""

    tool: Literal["add_event", "get_events", "update_event", "delete_event"] = Field(
        ...,
        description="Nombre de la herramienta de calendario a usar.",
    )
    arguments: dict = Field(
        ...,
        description="Argumentos válidos para la herramienta de calendario seleccionada.",
    )


class ExpertOutput(ToolCallOutput):
    """Salida estructurada del agente experto."""

    tool: Literal["web_search"] = Field(
        ...,
        description="Siempre usa 'web_search' para consultas conceptuales.",
    )
    arguments: dict = Field(
        ...,
        description=(
            "Debe contener el campo 'query' con la consulta de búsqueda. "
            "Ejemplo: {'query': 'qué es la computación cuántica'}"
        ),
    )


class CriticOutput(BaseModel):
    """Decisión del agente crítico."""

    decision: Literal["approved", "feedback"] = Field(
        ...,
        description=(
            "'approved' si la respuesta responde adecuadamente a la solicitud; "
            "'feedback' solo si es claramente incorrecta o no responde la pregunta."
        ),
    )
    feedback: str = Field(
        default="",
        description=(
            "Retroalimentación constructiva y específica. "
            "Debe estar vacía cuando la decisión es 'approved'."
        ),
    )
