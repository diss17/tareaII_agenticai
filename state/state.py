"""Estado compartido del grafo multiagente."""

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Estado global compartido entre todos los nodos del grafo.

    Se usa `add_messages` para `messages` de forma que LangGraph acumule
    automáticamente el historial de conversación entre iteraciones y no
    se pierdan los mensajes producidos por los especialistas.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str
    assigned_agent: Literal["calculator", "organizer", "expert", "none"] | None
    task_description: str
    agent_result: str
    critic_decision: Literal["approved", "feedback", "pending"]
    critic_feedback: str
    final_response: str
    iteration_count: int
    last_error: str
