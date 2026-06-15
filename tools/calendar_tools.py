"""Herramientas de calendario persistente usando SQLite."""

import sqlite3
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool
from config import CALENDAR_DB_PATH


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(CALENDAR_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_calendar_db() -> None:
    """Inicializa la tabla de eventos si no existe."""
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


@tool
def add_event(title: str, event_datetime: str, description: str = "") -> str:
    """Agrega un evento al calendario.

    Args:
        title: Título del evento.
        event_datetime: Fecha y hora en formato ISO 8601, ej. "2026-06-16 10:00".
        description: Descripción opcional del evento.

    Returns:
        Confirmación del evento creado.
    """
    try:
        dt = datetime.fromisoformat(event_datetime)
    except ValueError:
        return (
            "Error: formato de fecha inválido. "
            "Usa el formato 'YYYY-MM-DD HH:MM' o 'YYYY-MM-DD'."
        )

    with _get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO events (title, event_datetime, description) VALUES (?, ?, ?)",
            (title, dt.isoformat(), description),
        )
        conn.commit()
        event_id = cursor.lastrowid

    return f"Evento creado con ID {event_id}: '{title}' el {dt.isoformat()}"


@tool
def get_events(date: Optional[str] = None) -> str:
    """Obtiene eventos del calendario.

    Args:
        date: Fecha opcional en formato 'YYYY-MM-DD'. Si no se proporciona,
            devuelve todos los eventos futuros.

    Returns:
        Lista de eventos encontrados.
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

    with _get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return "No se encontraron eventos."

    lines = []
    for row in rows:
        lines.append(
            f"ID {row['id']}: {row['title']} - {row['event_datetime']}"
            f"{' (' + row['description'] + ')' if row['description'] else ''}"
        )
    return "\n".join(lines)


@tool
def update_event(event_id: int, title: Optional[str] = None,
                 event_datetime: Optional[str] = None,
                 description: Optional[str] = None) -> str:
    """Modifica un evento existente.

    Args:
        event_id: ID del evento a modificar.
        title: Nuevo título (opcional).
        event_datetime: Nueva fecha y hora en formato ISO (opcional).
        description: Nueva descripción (opcional).

    Returns:
        Confirmación de la actualización.
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

    with _get_connection() as conn:
        cursor = conn.execute(query, params)
        conn.commit()
        if cursor.rowcount == 0:
            return f"No se encontró el evento con ID {event_id}."

    return f"Evento {event_id} actualizado correctamente."


@tool
def delete_event(event_id: int) -> str:
    """Elimina un evento del calendario.

    Args:
        event_id: ID del evento a eliminar.

    Returns:
        Confirmación de la eliminación.
    """
    with _get_connection() as conn:
        cursor = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        conn.commit()
        if cursor.rowcount == 0:
            return f"No se encontró el evento con ID {event_id}."
    return f"Evento {event_id} eliminado correctamente."


# Inicializar la base de datos al importar el módulo
init_calendar_db()
