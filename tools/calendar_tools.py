"""Herramientas de calendario persistente usando SQLite."""

import sqlite3
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool
from config import CALENDAR_DB_PATH
from utils import make_tool_error


def _get_connection() -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(CALENDAR_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as exc:
        raise RuntimeError(f"No se pudo conectar a la base de datos del calendario: {exc}") from exc


def init_calendar_db() -> None:
    """Inicializa la tabla de eventos si no existe."""
    try:
        with _get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    event_datetime TEXT NOT NULL,
                    description TEXT DEFAULT ''
                )
                """
            )
            conn.commit()
    except sqlite3.Error as exc:
        raise RuntimeError(f"No se pudo inicializar la base de datos del calendario: {exc}") from exc


@tool
def add_event(title: str, event_datetime: str, description: str = "") -> str:
    """Agrega un evento al calendario.

    Args:
        title: Título del evento.
        event_datetime: Fecha y hora en formato ISO 8601, ej. "2026-06-16 10:00".
        description: Descripción opcional del evento.

    Returns:
        Confirmación del evento creado, o [TOOL_ERROR] si falla técnicamente.
    """
    try:
        dt = datetime.fromisoformat(event_datetime)
    except ValueError:
        return (
            "Error: formato de fecha inválido. "
            "Usa el formato 'YYYY-MM-DD HH:MM' o 'YYYY-MM-DD'."
        )

    try:
        with _get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO events (title, event_datetime, description) VALUES (?, ?, ?)",
                (title, dt.isoformat(), description),
            )
            conn.commit()
            event_id = cursor.lastrowid
    except sqlite3.Error as exc:
        return make_tool_error(f"Error de base de datos al crear el evento: {exc}")

    return f"Evento creado con ID {event_id}: '{title}' el {dt.isoformat()}"


@tool
def get_events(date: Optional[str] = None) -> str:
    """Obtiene eventos del calendario.

    Args:
        date: Fecha opcional en formato 'YYYY-MM-DD'. Si no se proporciona,
            devuelve todos los eventos futuros.

    Returns:
        Lista de eventos encontrados, o [TOOL_ERROR] si falla técnicamente.
    """
    query = "SELECT * FROM events"
    params: tuple = ()

    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return "Error: formato de fecha inválido. Usa 'YYYY-MM-DD'."
        query += " WHERE date(event_datetime) = date(?) ORDER BY event_datetime"
        params = (date,)
    else:
        query += " WHERE event_datetime >= datetime('now') ORDER BY event_datetime"

    try:
        with _get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
    except sqlite3.Error as exc:
        return make_tool_error(f"Error de base de datos al consultar eventos: {exc}")

    if not rows:
        return "No se encontraron eventos."

    lines = []
    for row in rows:
        lines.append(
            f"{row['title']} - {row['event_datetime']}"
            f"{' (' + row['description'] + ')' if row['description'] else ''}"
        )
    return "\n".join(lines)


@tool
def update_event(
    event_id: int,
    title: Optional[str] = None,
    event_datetime: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """Modifica un evento existente.

    Args:
        event_id: ID del evento a modificar.
        title: Nuevo título (opcional).
        event_datetime: Nueva fecha y hora en formato ISO (opcional).
        description: Nueva descripción (opcional).

    Returns:
        Confirmación de la actualización, o [TOOL_ERROR] si falla técnicamente.
    """
    updates = []
    params: list = []

    if title is not None:
        updates.append("title = ?")
        params.append(title)
    if event_datetime is not None:
        try:
            datetime.fromisoformat(event_datetime)
        except ValueError:
            return "Error: formato de fecha inválido. Usa 'YYYY-MM-DD HH:MM'."
        updates.append("event_datetime = ?")
        params.append(event_datetime)
    if description is not None:
        updates.append("description = ?")
        params.append(description)

    if not updates:
        return "No se proporcionaron campos para actualizar."

    params.append(event_id)
    query = f"UPDATE events SET {', '.join(updates)} WHERE id = ?"

    try:
        with _get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            if cursor.rowcount == 0:
                return f"No se encontró el evento con ID {event_id}."
    except sqlite3.Error as exc:
        return make_tool_error(f"Error de base de datos al actualizar el evento: {exc}")

    return f"Evento {event_id} actualizado correctamente."


@tool
def delete_event(event_id: int) -> str:
    """Elimina un evento del calendario.

    Args:
        event_id: ID del evento a eliminar.

    Returns:
        Confirmación de la eliminación, o [TOOL_ERROR] si falla técnicamente.
    """
    try:
        with _get_connection() as conn:
            cursor = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
            conn.commit()
            if cursor.rowcount == 0:
                return f"No se encontró el evento con ID {event_id}."
    except sqlite3.Error as exc:
        return make_tool_error(f"Error de base de datos al eliminar el evento: {exc}")

    return f"Evento {event_id} eliminado correctamente."


# Inicializar la base de datos al importar el módulo
init_calendar_db()
