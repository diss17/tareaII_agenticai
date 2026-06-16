"""Detector de fechas en texto en español.

Este módulo usa expresiones regulares para extraer fechas del input del usuario
de forma determinista, sin depender del LLM. Esto es crítico porque el modelo
local (gemma3:4b) no es consistente extrayendo fechas de consultas en lenguaje
natural.

Formatos soportados:
- ISO: 2026-06-16, 2026/06/16
- Español: 16 de junio de 2026, 16/06/2026, 16-06-2026
- Relativos: hoy, mañana, pasado mañana
- Días de la semana: lunes, martes, etc. (próxima ocurrencia)
"""

import re
from datetime import datetime, timedelta
from typing import Optional


# Mapeo de meses en español
MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

# Mapeo de días de la semana
DIAS_SEMANA = {
    "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
    "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5,
    "domingo": 6,
}


def _format_date(dt: datetime) -> str:
    """Convierte datetime a string YYYY-MM-DD."""
    return dt.strftime("%Y-%m-%d")


def _parse_relative_day(text: str) -> Optional[datetime]:
    """Detecta días relativos: hoy, mañana, pasado mañana."""
    text_lower = text.lower()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if "pasado mañana" in text_lower or "pasado manana" in text_lower:
        return today + timedelta(days=2)
    if "mañana" in text_lower or "manana" in text_lower:
        return today + timedelta(days=1)
    if "hoy" in text_lower:
        return today
    if "anteayer" in text_lower:
        return today - timedelta(days=2)

    return None


def _parse_weekday(text: str) -> Optional[datetime]:
    """Detecta días de la semana: próximo lunes, viernes, etc."""
    text_lower = text.lower()

    # Buscar patrón: "próximo [día]" o "este [día]" o solo "[día]"
    for dia_nombre, dia_num in DIAS_SEMANA.items():
        if dia_nombre in text_lower:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            dias_adelante = (dia_num - today.weekday()) % 7
            if dias_adelante == 0:
                dias_adelante = 7  # Si es hoy, asumir el próximo
            return today + timedelta(days=dias_adelante)

    return None


def _parse_iso_format(text: str) -> Optional[datetime]:
    """Detecta formatos ISO: 2026-06-16, 2026/06/16."""
    # Patrón: YYYY-MM-DD o YYYY/MM/DD
    match = re.search(r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b', text)
    if match:
        try:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return datetime(year, month, day)
        except ValueError:
            return None
    return None


def _parse_dmy_format(text: str) -> Optional[datetime]:
    """Detecta formatos DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY."""
    # Patrón: DD/MM/YYYY o DD-MM-YYYY o DD.MM.YYYY
    match = re.search(r'\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b', text)
    if match:
        try:
            day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return datetime(year, month, day)
        except ValueError:
            return None
    return None


def _parse_spanish_text(text: str) -> Optional[datetime]:
    """Detecta fechas en texto español: '16 de junio de 2026', '16 junio 2026'."""
    text_lower = text.lower()

    # Patrón: "DD de MES de YYYY" o "DD de MES YYYY"
    for mes_nombre, mes_num in MESES.items():
        patron = rf'\b(\d{{1,2}})\s+de\s+{mes_nombre}(?:\s+de\s+(\d{{4}}))?\b'
        match = re.search(patron, text_lower)
        if match:
            try:
                day = int(match.group(1))
                year = int(match.group(2)) if match.group(2) else datetime.now().year
                return datetime(year, mes_num, day)
            except ValueError:
                continue

    # Patrón: "DD MES YYYY" o "DD MES"
    for mes_nombre, mes_num in MESES.items():
        patron = rf'\b(\d{{1,2}})\s+{mes_nombre}(?:\s+(\d{{4}}))?\b'
        match = re.search(patron, text_lower)
        if match:
            try:
                day = int(match.group(1))
                year = int(match.group(2)) if match.group(2) else datetime.now().year
                return datetime(year, mes_num, day)
            except ValueError:
                continue

    return None


def extract_date(text: str) -> Optional[str]:
    """Extrae una fecha del texto y la devuelve en formato YYYY-MM-DD.

    Intenta múltiples formatos en orden de especificidad:
    1. Texto en español (16 de junio de 2026)
    2. Formato ISO (2026-06-16)
    3. Formato DD/MM/YYYY (16/06/2026)
    4. Días relativos (hoy, mañana, pasado mañana)
    5. Días de la semana (lunes, martes, etc.)

    Args:
        text: Texto del usuario del cual extraer la fecha.

    Returns:
        Fecha en formato YYYY-MM-DD, o None si no se detecta ninguna fecha.
    """
    if not text:
        return None

    # Intentar cada parser en orden
    parsers = [
        _parse_spanish_text,
        _parse_iso_format,
        _parse_dmy_format,
        _parse_relative_day,
        _parse_weekday,
    ]

    for parser in parsers:
        result = parser(text)
        if result:
            return _format_date(result)

    return None
